// SPDX-License-Identifier: GPL-3.0-or-later
//
// Compiled Paper 0 oracle for selected native-81 TCV/Hermes 85604 frames.
// The radial xz and shifted-xy calculations are adapted from Hermes-3
// src/div_ops.cxx lines 128-229 and 273-326 at revision
// 920ba829cc78cdab0dbf6101c69fecc4689bd8dd (GPL-3.0). The launcher verifies
// that source revision and file hash before compilation.

#include <bout/bout.hxx>
#include <bout/boutcomm.hxx>
#include <bout/derivs.hxx>
#include <netcdf.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int FRAME_COUNT = 5;
constexpr int GLOBAL_X = 64;
constexpr int GLOBAL_Y = 32;
constexpr int GLOBAL_Z = 81;
constexpr int LOCAL_Y = 8;
constexpr std::array<int, FRAME_COUNT> EXPECTED_FRAMES{0, 156, 312, 467, 623};
constexpr std::array<const char*, 3> ADVECTED_FIELDS{"Ne", "Pe", "Pi"};

void nc_check(int status, const std::string& context) {
  if (status != NC_NOERR) {
    throw std::runtime_error(context + ": " + nc_strerror(status));
  }
}

std::string frame_label(int frame) {
  std::ostringstream label;
  label << "f" << std::setw(3) << std::setfill('0') << frame;
  return label.str();
}

BoutReal minmod3(BoutReal a, BoutReal b, BoutReal c) {
  if ((a * b <= 0.0) || (a * c <= 0.0)) {
    return 0.0;
  }
  const BoutReal magnitude = std::min(std::abs(a), std::min(std::abs(b), std::abs(c)));
  return std::copysign(magnitude, a);
}

BoutReal mc_slope(BoutReal minus, BoutReal center, BoutReal plus) {
  return minmod3(2.0 * (plus - center),
                 0.5 * (plus - minus),
                 2.0 * (center - minus));
}

class CanonicalFrames {
public:
  explicit CanonicalFrames(const std::string& path) {
    nc_check(nc_open(path.c_str(), NC_NOWRITE, &ncid), "open canonical frame file");
    validate_dimensions();
    validate_frames();
  }

  CanonicalFrames(const CanonicalFrames&) = delete;
  CanonicalFrames& operator=(const CanonicalFrames&) = delete;

  ~CanonicalFrames() {
    if (ncid >= 0) {
      nc_close(ncid);
    }
  }

  Field3D load(const std::string& name, std::size_t frame_position) const {
    using bout::globals::mesh;
    if (frame_position >= FRAME_COUNT) {
      throw std::out_of_range("canonical frame position is out of range");
    }
    if (mesh->getXProcIndex() != 0 || mesh->NXPE != 1) {
      throw std::runtime_error("native-frame oracle requires NXPE=1");
    }
    if (mesh->xend - mesh->xstart + 1 != GLOBAL_X ||
        mesh->yend - mesh->ystart + 1 != LOCAL_Y ||
        mesh->LocalNz != GLOBAL_Z) {
      throw std::runtime_error("BOUT physical domain does not match canonical input");
    }

    int variable = -1;
    nc_check(nc_inq_varid(ncid, name.c_str(), &variable), "find variable " + name);
    int dimensions = 0;
    int dimension_ids[NC_MAX_VAR_DIMS];
    nc_check(nc_inq_var(ncid, variable, nullptr, nullptr, &dimensions,
                        dimension_ids, nullptr),
             "inspect variable " + name);
    if (dimensions != 4 || dimension_ids[0] != frame_dim ||
        dimension_ids[1] != x_dim || dimension_ids[2] != y_dim ||
        dimension_ids[3] != z_dim) {
      throw std::runtime_error(name + " does not use [selected_frame,x,y,z]");
    }

    const std::size_t y_start =
        static_cast<std::size_t>(mesh->getYProcIndex() * LOCAL_Y);
    const std::array<std::size_t, 4> start{frame_position, 0, y_start, 0};
    const std::array<std::size_t, 4> count{1, GLOBAL_X, LOCAL_Y, GLOBAL_Z};
    std::vector<double> buffer(GLOBAL_X * LOCAL_Y * GLOBAL_Z);
    nc_check(nc_get_vara_double(ncid, variable, start.data(), count.data(),
                                buffer.data()),
             "read variable " + name);

    Field3D field{0.0};
    for (int x = 0; x < GLOBAL_X; ++x) {
      for (int y = 0; y < LOCAL_Y; ++y) {
        for (int z = 0; z < GLOBAL_Z; ++z) {
          const std::size_t offset =
              (static_cast<std::size_t>(x) * LOCAL_Y + y) * GLOBAL_Z + z;
          field(mesh->xstart + x, mesh->ystart + y, z) = buffer[offset];
        }
      }
    }
    field.applyBoundary("neumann");
    mesh->communicate(field);
    return field;
  }

private:
  int ncid{-1};
  int frame_dim{-1};
  int x_dim{-1};
  int y_dim{-1};
  int z_dim{-1};

  void require_dimension(const char* name, std::size_t expected, int& identifier) {
    nc_check(nc_inq_dimid(ncid, name, &identifier),
             std::string("find dimension ") + name);
    std::size_t length = 0;
    nc_check(nc_inq_dimlen(ncid, identifier, &length),
             std::string("inspect dimension ") + name);
    if (length != expected) {
      throw std::runtime_error(std::string(name) + " has unexpected length");
    }
  }

  void validate_dimensions() {
    require_dimension("selected_frame", FRAME_COUNT, frame_dim);
    require_dimension("x", GLOBAL_X, x_dim);
    require_dimension("y", GLOBAL_Y, y_dim);
    require_dimension("z", GLOBAL_Z, z_dim);
  }

  void validate_frames() {
    int variable = -1;
    nc_check(nc_inq_varid(ncid, "frame_index", &variable), "find frame_index");
    std::array<long long, FRAME_COUNT> frames{};
    nc_check(nc_get_var_longlong(ncid, variable, frames.data()), "read frame_index");
    for (std::size_t position = 0; position < frames.size(); ++position) {
      if (frames[position] != EXPECTED_FRAMES[position]) {
        throw std::runtime_error("canonical frame indices differ from frozen protocol");
      }
    }
  }
};

void evaluate_case(const std::string& name, const Field3D& q,
                   const Field3D& phi, Options& output) {
  using bout::globals::mesh;
  Coordinates* coord = mesh->getCoordinates();

  Field3D dfdy = DDY(phi);
  mesh->communicate(dfdy);
  dfdy.applyBoundary("neumann");

  Field3D xz_flow{0.0};
  Field3D xy_flow{0.0};
  Field3D total_radial_flow{0.0};
  Field3D radial_divergence{0.0};

  const int xs = mesh->xstart - 1;
  const int xe = mesh->xend;
  const int nz = mesh->LocalNz;
  for (int i = xs; i <= xe; ++i) {
    for (int j = mesh->ystart - 1; j <= mesh->yend; ++j) {
      for (int k = 0; k < nz; ++k) {
        const int kp = (k + 1) % nz;
        const int km = (k - 1 + nz) % nz;

        const BoutReal corner_plus =
            0.25 * (phi(i, j, k) + phi(i, j, kp) +
                    phi(i + 1, j, k) + phi(i + 1, j, kp));
        const BoutReal corner_minus =
            0.25 * (phi(i, j, k) + phi(i + 1, j, k) +
                    phi(i, j, km) + phi(i + 1, j, km));
        const BoutReal velocity_xz =
            0.5 * (coord->J(i, j) + coord->J(i + 1, j)) *
            (corner_plus - corner_minus) / coord->dz(i, j);

        const BoutReal left_state =
            q(i, j, k) +
            0.5 * mc_slope(q(i - 1, j, k), q(i, j, k), q(i + 1, j, k));
        const BoutReal right_state =
            q(i + 1, j, k) -
            0.5 * mc_slope(q(i, j, k), q(i + 1, j, k), q(i + 2, j, k));
        const BoutReal selected_xz_state =
            velocity_xz > 0.0 ? left_state : right_state;
        const BoutReal flow_xz = velocity_xz * selected_xz_state;

        const BoutReal metric_right =
            coord->g11(i + 1, j) * coord->g23(i + 1, j) /
            SQ(coord->Bxy(i + 1, j));
        const BoutReal metric_left =
            coord->g11(i, j) * coord->g23(i, j) / SQ(coord->Bxy(i, j));
        const BoutReal derivative_at_face =
            0.5 * (metric_right * dfdy(i + 1, j, k) +
                   metric_left * dfdy(i, j, k));
        const BoutReal velocity_xy =
            0.5 * (coord->J(i + 1, j) + coord->J(i, j)) * derivative_at_face;
        BoutReal selected_xy_state;
        if (velocity_xy > 0.0) {
          selected_xy_state =
              q(i, j, k) + 0.25 * (q(i + 1, j, k) - q(i - 1, j, k));
        } else {
          selected_xy_state =
              q(i + 1, j, k) - 0.25 * (q(i + 2, j, k) - q(i, j, k));
        }
        if (selected_xy_state < 0.0) {
          selected_xy_state = 0.0;
        }
        const BoutReal flow_xy = velocity_xy * selected_xy_state;

        xz_flow(i, j, k) = flow_xz;
        xy_flow(i, j, k) = flow_xy;
        total_radial_flow(i, j, k) = flow_xz + flow_xy;
      }
    }
  }

  for (int i = mesh->xstart; i <= mesh->xend; ++i) {
    for (int j = mesh->ystart - 1; j <= mesh->yend; ++j) {
      for (int k = 0; k < nz; ++k) {
        radial_divergence(i, j, k) =
            (total_radial_flow(i, j, k) -
             total_radial_flow(i - 1, j, k)) /
            (coord->J(i, j) * coord->dx(i, j));
      }
    }
  }

  output["q_" + name] = q;
  output["xz_flow_" + name] = xz_flow;
  output["xy_flow_" + name] = xy_flow;
  output["total_radial_flow_" + name] = total_radial_flow;
  output["radial_divergence_" + name] = radial_divergence;
}

} // namespace

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);
  if (BoutComm::size() != 4) {
    throw std::runtime_error("native-frame oracle requires exactly four MPI ranks");
  }

  const auto input_path =
      Options::root()["paper0"]["input_file"]
          .doc("Canonical native-81 Paper 0 frame file")
          .as<std::string>();
  if (input_path.empty()) {
    throw std::runtime_error("paper0:input_file must name the canonical frame file");
  }
  CanonicalFrames input(input_path);
  Options output;

  for (std::size_t position = 0; position < EXPECTED_FRAMES.size(); ++position) {
    const int frame = EXPECTED_FRAMES[position];
    const std::string frame_name = frame_label(frame);
    const Field3D phi = input.load("phi", position);
    output["phi_" + frame_name] = phi;
    output["canonical_frame_index_" + frame_name] = frame;
    for (const auto* field : ADVECTED_FIELDS) {
      const Field3D q = input.load(field, position);
      evaluate_case(std::string(field) + "_" + frame_name, q, phi, output);
    }
  }

  output["paper0_oracle_name"] = "hermes_native_85604_radial_flow";
  output["paper0_hermes_revision"] =
      "920ba829cc78cdab0dbf6101c69fecc4689bd8dd";
  output["paper0_zperiod"] = 5;
  bout::writeDefaultOutputFile(output);

  bout::checkForUnusedOptions();
  BoutFinalise();
  return 0;
}
