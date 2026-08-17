// SPDX-License-Identifier: GPL-3.0-or-later
//
// Paired potential reconstruction oracle for selected native-81 TCV/Hermes
// 85604 states. The metric-normalization and vorticity inversion blocks adapt
// Hermes-3 hermes-3.cxx and src/vorticity.cxx at revision
// 920ba829cc78cdab0dbf6101c69fecc4689bd8dd (GPL-3.0). The launcher locks
// those source files and the BOUT++ cyclic solver used by the executable.

#include <bout/bout.hxx>
#include <bout/boutcomm.hxx>
#include <bout/constants.hxx>
#include <bout/coordinates.hxx>
#include <bout/invert_laplace.hxx>
#include <netcdf.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <memory>
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
constexpr int INNER_SIDE = 0;
constexpr int OUTER_SIDE = 1;
constexpr BoutReal TNORM_EV = 50.0;
constexpr BoutReal BNORM_T = 1.0;
constexpr BoutReal FROZEN_RHO_S0_M = 0.0007224847664314034;
constexpr BoutReal PHI_TO_VOLTS = 50.0;
constexpr BoutReal ELECTRON_PRESSURE_DENOMINATOR = 3672.0;
constexpr BoutReal PRESSURE_DENSITY_FLOOR = 1.0e-7;
constexpr std::array<int, FRAME_COUNT> EXPECTED_FRAMES{0, 156, 312, 467, 623};
constexpr std::array<const char*, 5> INPUT_FIELDS{"Ne", "Pe", "Pi", "Vort",
                                                  "phi"};

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

class CanonicalInput {
public:
  explicit CanonicalInput(const std::string& path) {
    nc_check(nc_open(path.c_str(), NC_NOWRITE, &ncid),
             "open canonical potential input");
    validate_dimensions();
    validate_frames();
  }

  CanonicalInput(const CanonicalInput&) = delete;
  CanonicalInput& operator=(const CanonicalInput&) = delete;

  ~CanonicalInput() {
    if (ncid >= 0) {
      nc_close(ncid);
    }
  }

  Field3D load_field(const std::string& name,
                     std::size_t frame_position) const {
    using bout::globals::mesh;
    validate_mesh();
    if (frame_position >= FRAME_COUNT) {
      throw std::out_of_range("canonical frame position is out of range");
    }
    int variable = -1;
    nc_check(nc_inq_varid(ncid, name.c_str(), &variable),
             "find canonical field " + name);
    int dimensions = 0;
    int dimension_ids[NC_MAX_VAR_DIMS];
    nc_check(nc_inq_var(ncid, variable, nullptr, nullptr, &dimensions,
                        dimension_ids, nullptr),
             "inspect canonical field " + name);
    if (dimensions != 4 || dimension_ids[0] != frame_dim ||
        dimension_ids[1] != x_dim || dimension_ids[2] != y_dim ||
        dimension_ids[3] != z_dim) {
      throw std::runtime_error(name +
                               " does not use [selected_frame,x,y,z]");
    }

    const std::size_t y_start =
        static_cast<std::size_t>(mesh->getYProcIndex() * LOCAL_Y);
    const std::array<std::size_t, 4> start{frame_position, 0, y_start, 0};
    const std::array<std::size_t, 4> count{1, GLOBAL_X, LOCAL_Y, GLOBAL_Z};
    std::vector<double> buffer(GLOBAL_X * LOCAL_Y * GLOBAL_Z);
    nc_check(nc_get_vara_double(ncid, variable, start.data(), count.data(),
                                buffer.data()),
             "read canonical field " + name);

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

  std::array<BoutReal, LOCAL_Y>
  load_boundary(const std::string& name, std::size_t frame_position,
                int side) const {
    using bout::globals::mesh;
    validate_mesh();
    if (frame_position >= FRAME_COUNT) {
      throw std::out_of_range("canonical frame position is out of range");
    }
    if (side != INNER_SIDE && side != OUTER_SIDE) {
      throw std::out_of_range("canonical boundary side is out of range");
    }
    int variable = -1;
    nc_check(nc_inq_varid(ncid, name.c_str(), &variable),
             "find canonical boundary " + name);
    int dimensions = 0;
    int dimension_ids[NC_MAX_VAR_DIMS];
    nc_check(nc_inq_var(ncid, variable, nullptr, nullptr, &dimensions,
                        dimension_ids, nullptr),
             "inspect canonical boundary " + name);
    if (dimensions != 3 || dimension_ids[0] != frame_dim ||
        dimension_ids[1] != side_dim || dimension_ids[2] != y_dim) {
      throw std::runtime_error(
          name + " does not use [selected_frame,side,y]");
    }
    const std::size_t y_start =
        static_cast<std::size_t>(mesh->getYProcIndex() * LOCAL_Y);
    const std::array<std::size_t, 3> start{
        frame_position, static_cast<std::size_t>(side), y_start};
    const std::array<std::size_t, 3> count{1, 1, LOCAL_Y};
    std::array<double, LOCAL_Y> buffer{};
    nc_check(nc_get_vara_double(ncid, variable, start.data(), count.data(),
                                buffer.data()),
             "read canonical boundary " + name);
    std::array<BoutReal, LOCAL_Y> values{};
    for (int y = 0; y < LOCAL_Y; ++y) {
      values[y] = buffer[y];
    }
    return values;
  }

private:
  int ncid{-1};
  int frame_dim{-1};
  int x_dim{-1};
  int y_dim{-1};
  int z_dim{-1};
  int side_dim{-1};

  void require_dimension(const char* name, std::size_t expected,
                         int& identifier) {
    nc_check(nc_inq_dimid(ncid, name, &identifier),
             std::string("find dimension ") + name);
    std::size_t length = 0;
    nc_check(nc_inq_dimlen(ncid, identifier, &length),
             std::string("inspect dimension ") + name);
    if (length != expected) {
      throw std::runtime_error(std::string(name) +
                               " has unexpected length");
    }
  }

  void validate_dimensions() {
    require_dimension("selected_frame", FRAME_COUNT, frame_dim);
    require_dimension("x", GLOBAL_X, x_dim);
    require_dimension("y", GLOBAL_Y, y_dim);
    require_dimension("z", GLOBAL_Z, z_dim);
    require_dimension("side", 2, side_dim);
  }

  void validate_frames() {
    int variable = -1;
    nc_check(nc_inq_varid(ncid, "frame_index", &variable),
             "find canonical frame_index");
    std::array<long long, FRAME_COUNT> frames{};
    nc_check(nc_get_var_longlong(ncid, variable, frames.data()),
             "read canonical frame_index");
    for (std::size_t position = 0; position < frames.size(); ++position) {
      if (frames[position] != EXPECTED_FRAMES[position]) {
        throw std::runtime_error(
            "canonical frame indices differ from frozen protocol");
      }
    }
  }

  void validate_mesh() const {
    using bout::globals::mesh;
    if (mesh->getXProcIndex() != 0 || mesh->NXPE != 1) {
      throw std::runtime_error("potential oracle requires NXPE=1");
    }
    if (mesh->xend - mesh->xstart + 1 != GLOBAL_X ||
        mesh->yend - mesh->ystart + 1 != LOCAL_Y ||
        mesh->LocalNz != GLOBAL_Z) {
      throw std::runtime_error(
          "BOUT physical domain does not match canonical input");
    }
  }
};

BoutReal normalize_metric_exactly() {
  using bout::globals::mesh;
  Coordinates* coord = mesh->getCoordinates();
  const BoutReal sound_speed = std::sqrt(SI::qe * TNORM_EV / SI::Mp);
  const BoutReal cyclotron_frequency = SI::qe * BNORM_T / SI::Mp;
  const BoutReal rho_s0 = sound_speed / cyclotron_frequency;
  const BoutReal relative_error =
      std::abs(rho_s0 - FROZEN_RHO_S0_M) / FROZEN_RHO_S0_M;
  if (relative_error > 1.0e-14) {
    throw std::runtime_error(
        "source-computed rho_s0 differs from the frozen normalization");
  }

  coord->dx /= rho_s0 * rho_s0 * BNORM_T;
  coord->Bxy /= BNORM_T;
  coord->g11 /= SQ(BNORM_T * rho_s0);
  coord->g22 *= SQ(rho_s0);
  coord->g33 *= SQ(rho_s0);
  coord->g12 /= BNORM_T;
  coord->g13 /= BNORM_T;
  coord->g23 *= SQ(rho_s0);
  coord->J *= BNORM_T / rho_s0;
  coord->g_11 *= SQ(BNORM_T * rho_s0);
  coord->g_22 /= SQ(rho_s0);
  coord->g_33 /= SQ(rho_s0);
  coord->g_12 *= BNORM_T;
  coord->g_13 *= BNORM_T;
  coord->g_23 /= SQ(rho_s0);
  coord->geometry();
  return rho_s0;
}

Field2D boundary_echo(const std::array<BoutReal, LOCAL_Y>& values) {
  using bout::globals::mesh;
  Field2D field{0.0};
  for (int x = mesh->xstart; x <= mesh->xend; ++x) {
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      field(x, y) = values[y - mesh->ystart];
    }
  }
  return field;
}

void set_radial_phi_ghosts(
    Field3D& phi, const std::array<BoutReal, LOCAL_Y>& inner_midpoint,
    const std::array<BoutReal, LOCAL_Y>& outer_midpoint) {
  using bout::globals::mesh;
  if (mesh->firstX()) {
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      const BoutReal midpoint = inner_midpoint[y - mesh->ystart];
      for (int z = 0; z < mesh->LocalNz; ++z) {
        const BoutReal ghost =
            2.0 * midpoint - phi(mesh->xstart, y, z);
        phi(mesh->xstart - 1, y, z) = ghost;
        phi(mesh->xstart - 2, y, z) = ghost;
      }
    }
  }
  if (mesh->lastX()) {
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      const BoutReal midpoint = outer_midpoint[y - mesh->ystart];
      for (int z = 0; z < mesh->LocalNz; ++z) {
        const BoutReal ghost =
            2.0 * midpoint - phi(mesh->xend, y, z);
        phi(mesh->xend + 1, y, z) = ghost;
        phi(mesh->xend + 2, y, z) = ghost;
      }
    }
  }
}

std::unique_ptr<Laplacian> create_phi_solver(Options& options) {
  using bout::globals::mesh;
  const std::string solver_type =
      options["type"].doc("Frozen Paper 0 Laplacian type").as<std::string>();
  if (solver_type != "cyclic") {
    throw std::runtime_error("potential oracle requires cyclic Laplacian");
  }
  auto solver = Laplacian::create(&options);
  solver->setCoefC(2.0 / SQ(mesh->getCoordinates()->Bxy));
  solver->setInnerBoundaryFlags(INVERT_SET);
  solver->setOuterBoundaryFlags(INVERT_SET);
  return solver;
}

#ifdef PAPER0_RUNTIME_PRESSURE_CORRECTION
Field3D runtime_species_pressure(const Field3D& evolved_pressure,
                                 const Field3D& density) {
  using bout::globals::mesh;
  Field3D runtime_pressure{0.0};
  for (int x = 0; x < mesh->LocalNx; ++x) {
    for (int y = 0; y < mesh->LocalNy; ++y) {
      for (int z = 0; z < mesh->LocalNz; ++z) {
        const BoutReal raw_density = density(x, y, z);
        const BoutReal nonnegative_density = std::max(raw_density, 0.0);
        const BoutReal soft_density =
            nonnegative_density
            + PRESSURE_DENSITY_FLOOR
                  * std::exp(-nonnegative_density / PRESSURE_DENSITY_FLOOR);
        const BoutReal nonnegative_pressure =
            std::max(evolved_pressure(x, y, z), 0.0);
        const BoutReal temperature = nonnegative_pressure / soft_density;
        runtime_pressure(x, y, z) = raw_density * temperature;
        if (!std::isfinite(runtime_pressure(x, y, z))) {
          throw std::runtime_error(
              "runtime species pressure transformation is non-finite");
        }
      }
    }
  }
  return runtime_pressure;
}
#endif

Field3D solve_arm(
    Laplacian& solver, const Field3D& stored_phi, const Field3D& pi_hat,
    const Field3D& vorticity,
    const std::array<BoutReal, LOCAL_Y>& inner_midpoint,
    const std::array<BoutReal, LOCAL_Y>& outer_midpoint) {
  using bout::globals::mesh;
  Field3D phi_seed = stored_phi;
  set_radial_phi_ghosts(phi_seed, inner_midpoint, outer_midpoint);
  Field3D phi_plus_pi = phi_seed + pi_hat;

  if (mesh->firstX()) {
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      for (int z = 0; z < mesh->LocalNz; ++z) {
        phi_plus_pi(mesh->xstart - 1, y, z) =
            0.5 * (phi_plus_pi(mesh->xstart - 1, y, z) +
                   phi_plus_pi(mesh->xstart, y, z));
      }
    }
  }
  if (mesh->lastX()) {
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      for (int z = 0; z < mesh->LocalNz; ++z) {
        phi_plus_pi(mesh->xend + 1, y, z) =
            0.5 * (phi_plus_pi(mesh->xend + 1, y, z) +
                   phi_plus_pi(mesh->xend, y, z));
      }
    }
  }

  const Field2D bsq = SQ(mesh->getCoordinates()->Bxy);
  Field3D result = solver.solve(vorticity * (bsq / 2.0), phi_plus_pi) - pi_hat;
  mesh->communicate(result);
  if (mesh->firstX()) {
    for (int x = mesh->xstart - 2; x >= 0; --x) {
      for (int y = mesh->ystart; y <= mesh->yend; ++y) {
        for (int z = 0; z < mesh->LocalNz; ++z) {
          result(x, y, z) = result(x + 1, y, z);
        }
      }
    }
  }
  if (mesh->lastX()) {
    for (int x = mesh->xend + 2; x < mesh->LocalNx; ++x) {
      for (int y = mesh->ystart; y <= mesh->yend; ++y) {
        for (int z = 0; z < mesh->LocalNz; ++z) {
          result(x, y, z) = result(x - 1, y, z);
        }
      }
    }
  }
  return result;
}

} // namespace

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);
  using bout::globals::mesh;

  if (BoutComm::size() != 4) {
    throw std::runtime_error(
        "potential elliptic oracle requires exactly four MPI ranks");
  }
  if (mesh->NXPE != 1 || mesh->getNYPE() != 4 ||
      mesh->yend - mesh->ystart + 1 != LOCAL_Y) {
    throw std::runtime_error(
        "potential elliptic oracle requires NXPE=1,NYPE=4,MYSUB=8");
  }

  const auto input_path =
      Options::root()["paper0"]["input_file"]
          .doc("Canonical Paper 0 potential input")
          .as<std::string>();
  if (input_path.empty()) {
    throw std::runtime_error(
        "paper0:input_file must name the canonical input");
  }

  const BoutReal rho_s0 = normalize_metric_exactly();
  auto& laplace_options = Options::root()["paper0"]["laplacian"];
  auto retained_solver = create_phi_solver(laplace_options);
  auto instantaneous_solver = create_phi_solver(laplace_options);
  CanonicalInput input(input_path);
  Options output;

  for (std::size_t position = 0; position < EXPECTED_FRAMES.size();
       ++position) {
    const int frame = EXPECTED_FRAMES[position];
    const std::string label = frame_label(frame);
    const Field3D ne = input.load_field("Ne", position);
    const Field3D pe = input.load_field("Pe", position);
    const Field3D pi = input.load_field("Pi", position);
    const Field3D vorticity = input.load_field("Vort", position);
    const Field3D stored_phi = input.load_field("phi", position);

#ifdef PAPER0_RUNTIME_PRESSURE_CORRECTION
    const Field3D runtime_pe = runtime_species_pressure(pe, ne);
    const Field3D runtime_pi = runtime_species_pressure(pi, ne);
    Field3D pi_hat =
        runtime_pi - runtime_pe / ELECTRON_PRESSURE_DENOMINATOR;
#else
    Field3D pi_hat = pi - pe / ELECTRON_PRESSURE_DENOMINATOR;
#endif
    pi_hat.applyBoundary("neumann");
    mesh->communicate(pi_hat);

    const auto saved_inner =
        input.load_boundary("saved_midpoint", position, INNER_SIDE);
    const auto saved_outer =
        input.load_boundary("saved_midpoint", position, OUTER_SIDE);
    const auto target_inner =
        input.load_boundary("instantaneous_target", position, INNER_SIDE);
    const auto target_outer =
        input.load_boundary("instantaneous_target", position, OUTER_SIDE);

    const Field3D retained =
        solve_arm(*retained_solver, stored_phi, pi_hat, vorticity,
                  saved_inner, saved_outer);
    const Field3D instantaneous =
        solve_arm(*instantaneous_solver, stored_phi, pi_hat, vorticity,
                  target_inner, target_outer);

    output["input_Ne_" + label] = ne;
    output["input_Pe_" + label] = pe;
    output["input_Pi_" + label] = pi;
    output["input_Vort_" + label] = vorticity;
    output["input_phi_" + label] = stored_phi;
#ifdef PAPER0_RUNTIME_PRESSURE_CORRECTION
    output["runtime_Pe_" + label] = runtime_pe;
    output["runtime_Pi_" + label] = runtime_pi;
#endif
    output["pi_hat_" + label] = pi_hat;
    output["retained_phi_" + label] = retained;
    output["instantaneous_phi_" + label] = instantaneous;
    output["saved_midpoint_inner_" + label] = boundary_echo(saved_inner);
    output["saved_midpoint_outer_" + label] = boundary_echo(saved_outer);
    output["instantaneous_target_inner_" + label] =
        boundary_echo(target_inner);
    output["instantaneous_target_outer_" + label] =
        boundary_echo(target_outer);
    output["canonical_frame_index_" + label] = frame;
  }

  output["paper0_oracle_name"] =
      "phase2_potential_elliptic_85604_paired";
  output["paper0_hermes_revision"] =
      "920ba829cc78cdab0dbf6101c69fecc4689bd8dd";
  output["paper0_bout_revision"] =
      "7d28d67c3f12c24ec281c0982e870f5369c65a6f";
  output["paper0_solver_type"] = "cyclic";
  output["paper0_zperiod"] = 5;
  output["paper0_rho_s0_meters"] = rho_s0;
  output["paper0_phi_conversion_volts"] = PHI_TO_VOLTS;
  output["paper0_pressure_correction_denominator"] =
      ELECTRON_PRESSURE_DENOMINATOR;
#ifdef PAPER0_RUNTIME_PRESSURE_CORRECTION
  output["paper0_runtime_pressure_correction"] = 1;
  output["paper0_pressure_density_floor"] = PRESSURE_DENSITY_FLOOR;
#else
  output["paper0_runtime_pressure_correction"] = 0;
#endif
  bout::writeDefaultOutputFile(output);

  bout::checkForUnusedOptions();
  BoutFinalise();
  return 0;
}
