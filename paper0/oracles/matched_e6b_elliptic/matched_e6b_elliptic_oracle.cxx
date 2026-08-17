// SPDX-License-Identifier: GPL-3.0-or-later
//
// Exact potential reconstruction for matched Paper 0 E6B codec states.
// Metric normalization, runtime pressure publication, radial boundary handling,
// and cyclic inversion follow Hermes-3 revision
// 920ba829cc78cdab0dbf6101c69fecc4689bd8dd and the previously validated
// Paper 0 potential/vorticity oracles.  Candidate mode never reads interior
// truth phi: its numerical seed is zero and its only potential state is the
// retained two-side midpoint Bphi supplied by the model dataset.

#include <bout/bout.hxx>
#include <bout/boutcomm.hxx>
#include <bout/constants.hxx>
#include <bout/coordinates.hxx>
#include <bout/invert_laplace.hxx>
#include <hdf5.h>

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

constexpr int GLOBAL_X = 64;
constexpr int GLOBAL_Y = 32;
constexpr int GLOBAL_Z = 81;
constexpr int LOCAL_Y = 8;
constexpr int INNER_SIDE = 0;
constexpr int OUTER_SIDE = 1;
constexpr BoutReal TNORM_EV = 50.0;
constexpr BoutReal BNORM_T = 1.0;
constexpr BoutReal FROZEN_RHO_S0_M = 0.0007224847664314034;
constexpr BoutReal ELECTRON_PRESSURE_DENOMINATOR = 3672.0;
constexpr BoutReal PRESSURE_DENSITY_FLOOR = 1.0e-7;

void h5_check(herr_t status, const std::string& context) {
  if (status < 0) {
    throw std::runtime_error("HDF5 failure while " + context);
  }
}

hid_t h5_id(hid_t identifier, const std::string& context) {
  if (identifier < 0) {
    throw std::runtime_error("HDF5 failure while " + context);
  }
  return identifier;
}

std::string frame_label(long long frame) {
  std::ostringstream label;
  label << "f" << std::setw(3) << std::setfill('0') << frame;
  return label.str();
}

class CandidateInput {
public:
  CandidateInput(const std::string& path, bool truth_layout)
      : truth_layout(truth_layout) {
    file = h5_id(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
                 "opening matched elliptic input");
    frame_indices = read_frame_indices();
    if (frame_indices.empty() || frame_indices.size() > 624) {
      throw std::runtime_error("matched elliptic frame count is invalid");
    }
    for (std::size_t position = 0; position < frame_indices.size(); ++position) {
      const long long frame = frame_indices[position];
      if (frame < 0 || frame >= 624
          || (position > 0 && frame != frame_indices[position - 1] + 1)) {
        throw std::runtime_error(
            "matched elliptic frame indices must be contiguous in 0..623");
      }
    }
    for (const char* field : {"Ne", "Pe", "Pi", "Vort"}) {
      validate_volume(field_path(field));
    }
    validate_boundary(boundary_path());
    if (truth_layout) {
      validate_volume("phi");
    }
  }

  CandidateInput(const CandidateInput&) = delete;
  CandidateInput& operator=(const CandidateInput&) = delete;

  ~CandidateInput() {
    if (file >= 0) {
      H5Fclose(file);
    }
  }

  std::size_t frames() const { return frame_indices.size(); }
  long long frame(std::size_t position) const {
    if (position >= frame_indices.size()) {
      throw std::out_of_range("matched elliptic frame position is invalid");
    }
    return frame_indices[position];
  }
  bool has_truth_phi() const { return truth_layout; }

  Field3D load_field(const std::string& name, std::size_t position) const {
    using bout::globals::mesh;
    validate_mesh();
    if (position >= frame_indices.size()) {
      throw std::out_of_range("matched elliptic field position is invalid");
    }
    const std::string path = name == "phi" ? "phi" : field_path(name);
    const hid_t dataset = h5_id(H5Dopen2(file, path.c_str(), H5P_DEFAULT),
                                "opening field " + path);
    const hid_t file_space = h5_id(H5Dget_space(dataset),
                                   "getting field space " + path);
    const hsize_t y_start =
        static_cast<hsize_t>(mesh->getYProcIndex() * LOCAL_Y);
    const std::array<hsize_t, 4> start{
        static_cast<hsize_t>(position), 0, y_start, 0};
    const std::array<hsize_t, 4> count{1, GLOBAL_X, LOCAL_Y, GLOBAL_Z};
    h5_check(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start.data(),
                                 nullptr, count.data(), nullptr),
             "selecting field slab " + path);
    const hid_t memory_space = h5_id(
        H5Screate_simple(4, count.data(), nullptr),
        "creating field memory space " + path);
    std::vector<double> buffer(GLOBAL_X * LOCAL_Y * GLOBAL_Z);
    h5_check(H5Dread(dataset, H5T_NATIVE_DOUBLE, memory_space, file_space,
                     H5P_DEFAULT, buffer.data()),
             "reading field " + path);
    h5_check(H5Sclose(memory_space), "closing field memory space");
    h5_check(H5Sclose(file_space), "closing field file space");
    h5_check(H5Dclose(dataset), "closing field dataset");

    Field3D result{0.0};
    for (int x = 0; x < GLOBAL_X; ++x) {
      for (int y = 0; y < LOCAL_Y; ++y) {
        for (int z = 0; z < GLOBAL_Z; ++z) {
          const std::size_t offset =
              (static_cast<std::size_t>(x) * LOCAL_Y + y) * GLOBAL_Z + z;
          const double value = buffer[offset];
          if (!std::isfinite(value)) {
            throw std::runtime_error("matched elliptic input is non-finite");
          }
          result(mesh->xstart + x, mesh->ystart + y, z) = value;
        }
      }
    }
    result.applyBoundary("neumann");
    mesh->communicate(result);
    return result;
  }

  std::array<BoutReal, LOCAL_Y>
  load_boundary(std::size_t position, int side) const {
    using bout::globals::mesh;
    validate_mesh();
    if (position >= frame_indices.size()
        || (side != INNER_SIDE && side != OUTER_SIDE)) {
      throw std::out_of_range("matched elliptic boundary request is invalid");
    }
    const std::string path = boundary_path();
    const hid_t dataset = h5_id(H5Dopen2(file, path.c_str(), H5P_DEFAULT),
                                "opening boundary " + path);
    const hid_t file_space = h5_id(H5Dget_space(dataset),
                                   "getting boundary space " + path);
    const hsize_t y_start =
        static_cast<hsize_t>(mesh->getYProcIndex() * LOCAL_Y);
    const std::array<hsize_t, 3> start{
        static_cast<hsize_t>(position), static_cast<hsize_t>(side), y_start};
    const std::array<hsize_t, 3> count{1, 1, LOCAL_Y};
    h5_check(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start.data(),
                                 nullptr, count.data(), nullptr),
             "selecting boundary slab");
    const hid_t memory_space = h5_id(
        H5Screate_simple(3, count.data(), nullptr),
        "creating boundary memory space");
    std::array<double, LOCAL_Y> buffer{};
    h5_check(H5Dread(dataset, H5T_NATIVE_DOUBLE, memory_space, file_space,
                     H5P_DEFAULT, buffer.data()),
             "reading boundary");
    h5_check(H5Sclose(memory_space), "closing boundary memory space");
    h5_check(H5Sclose(file_space), "closing boundary file space");
    h5_check(H5Dclose(dataset), "closing boundary dataset");
    std::array<BoutReal, LOCAL_Y> result{};
    for (int y = 0; y < LOCAL_Y; ++y) {
      if (!std::isfinite(buffer[y])) {
        throw std::runtime_error("matched elliptic boundary is non-finite");
      }
      result[y] = buffer[y];
    }
    return result;
  }

private:
  hid_t file{-1};
  bool truth_layout{false};
  std::vector<long long> frame_indices;

  std::string field_path(const std::string& field) const {
    return truth_layout ? field : "candidate/" + field;
  }
  std::string boundary_path() const {
    return truth_layout ? "saved_midpoint" : "boundary/Bphi";
  }
  std::string frame_path() const {
    return truth_layout ? "frame_index" : "coordinates/frame_index";
  }

  std::vector<long long> read_frame_indices() const {
    const std::string path = frame_path();
    const hid_t dataset = h5_id(H5Dopen2(file, path.c_str(), H5P_DEFAULT),
                                "opening frame indices");
    const hid_t space = h5_id(H5Dget_space(dataset),
                              "getting frame-index space");
    if (H5Sget_simple_extent_ndims(space) != 1) {
      throw std::runtime_error("frame indices are not one-dimensional");
    }
    hsize_t dimension = 0;
    h5_check(H5Sget_simple_extent_dims(space, &dimension, nullptr),
             "reading frame-index dimension");
    std::vector<long long> result(static_cast<std::size_t>(dimension));
    h5_check(H5Dread(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, result.data()),
             "reading frame indices");
    h5_check(H5Sclose(space), "closing frame-index space");
    h5_check(H5Dclose(dataset), "closing frame-index dataset");
    return result;
  }

  void validate_shape(const std::string& path,
                      const std::vector<hsize_t>& expected) const {
    const hid_t dataset = h5_id(H5Dopen2(file, path.c_str(), H5P_DEFAULT),
                                "opening dataset " + path);
    const hid_t space = h5_id(H5Dget_space(dataset),
                              "getting dataset space " + path);
    const int rank = H5Sget_simple_extent_ndims(space);
    if (rank != static_cast<int>(expected.size())) {
      throw std::runtime_error(path + " has the wrong rank");
    }
    std::vector<hsize_t> actual(expected.size());
    h5_check(H5Sget_simple_extent_dims(space, actual.data(), nullptr),
             "reading dimensions for " + path);
    h5_check(H5Sclose(space), "closing dataset space " + path);
    h5_check(H5Dclose(dataset), "closing dataset " + path);
    if (actual != expected) {
      throw std::runtime_error(path + " has the wrong dimensions");
    }
  }

  void validate_volume(const std::string& path) const {
    validate_shape(path, {static_cast<hsize_t>(frame_indices.size()),
                          GLOBAL_X, GLOBAL_Y, GLOBAL_Z});
  }
  void validate_boundary(const std::string& path) const {
    validate_shape(path, {static_cast<hsize_t>(frame_indices.size()), 2,
                          GLOBAL_Y});
  }
  void validate_mesh() const {
    using bout::globals::mesh;
    if (mesh->getXProcIndex() != 0 || mesh->NXPE != 1
        || mesh->xend - mesh->xstart + 1 != GLOBAL_X
        || mesh->yend - mesh->ystart + 1 != LOCAL_Y
        || mesh->LocalNz != GLOBAL_Z) {
      throw std::runtime_error(
          "BOUT physical domain does not match matched elliptic input");
    }
  }
};

BoutReal normalize_metric_exactly() {
  using bout::globals::mesh;
  Coordinates* coordinates = mesh->getCoordinates();
  const BoutReal sound_speed = std::sqrt(SI::qe * TNORM_EV / SI::Mp);
  const BoutReal cyclotron_frequency = SI::qe * BNORM_T / SI::Mp;
  const BoutReal rho_s0 = sound_speed / cyclotron_frequency;
  if (std::abs(rho_s0 - FROZEN_RHO_S0_M) / FROZEN_RHO_S0_M > 1.0e-14) {
    throw std::runtime_error("source rho_s0 differs from the frozen value");
  }
  coordinates->dx /= rho_s0 * rho_s0 * BNORM_T;
  coordinates->Bxy /= BNORM_T;
  coordinates->g11 /= SQ(BNORM_T * rho_s0);
  coordinates->g22 *= SQ(rho_s0);
  coordinates->g33 *= SQ(rho_s0);
  coordinates->g12 /= BNORM_T;
  coordinates->g13 /= BNORM_T;
  coordinates->g23 *= SQ(rho_s0);
  coordinates->J *= BNORM_T / rho_s0;
  coordinates->g_11 *= SQ(BNORM_T * rho_s0);
  coordinates->g_22 /= SQ(rho_s0);
  coordinates->g_33 /= SQ(rho_s0);
  coordinates->g_12 *= BNORM_T;
  coordinates->g_13 *= BNORM_T;
  coordinates->g_23 /= SQ(rho_s0);
  coordinates->geometry();
  return rho_s0;
}

void set_radial_phi_ghosts(
    Field3D& phi, const std::array<BoutReal, LOCAL_Y>& inner_midpoint,
    const std::array<BoutReal, LOCAL_Y>& outer_midpoint) {
  using bout::globals::mesh;
  if (mesh->firstX()) {
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      const BoutReal midpoint = inner_midpoint[y - mesh->ystart];
      for (int z = 0; z < mesh->LocalNz; ++z) {
        const BoutReal ghost = 2.0 * midpoint - phi(mesh->xstart, y, z);
        phi(mesh->xstart - 1, y, z) = ghost;
        phi(mesh->xstart - 2, y, z) = ghost;
      }
    }
  }
  if (mesh->lastX()) {
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      const BoutReal midpoint = outer_midpoint[y - mesh->ystart];
      for (int z = 0; z < mesh->LocalNz; ++z) {
        const BoutReal ghost = 2.0 * midpoint - phi(mesh->xend, y, z);
        phi(mesh->xend + 1, y, z) = ghost;
        phi(mesh->xend + 2, y, z) = ghost;
      }
    }
  }
}

std::unique_ptr<Laplacian> create_phi_solver(Options& options) {
  using bout::globals::mesh;
  const std::string type =
      options["type"].doc("Frozen Paper 0 Laplacian type").as<std::string>();
  if (type != "cyclic") {
    throw std::runtime_error("matched elliptic oracle requires cyclic Laplacian");
  }
  auto solver = Laplacian::create(&options);
  solver->setCoefC(2.0 / SQ(mesh->getCoordinates()->Bxy));
  solver->setInnerBoundaryFlags(INVERT_SET);
  solver->setOuterBoundaryFlags(INVERT_SET);
  return solver;
}

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
        const BoutReal temperature =
            std::max(evolved_pressure(x, y, z), 0.0) / soft_density;
        runtime_pressure(x, y, z) = raw_density * temperature;
        if (!std::isfinite(runtime_pressure(x, y, z))) {
          throw std::runtime_error("runtime pressure is non-finite");
        }
      }
    }
  }
  return runtime_pressure;
}

Field3D solve_candidate(
    Laplacian& solver, const Field3D& pi_hat, const Field3D& vorticity,
    const std::array<BoutReal, LOCAL_Y>& inner_midpoint,
    const std::array<BoutReal, LOCAL_Y>& outer_midpoint) {
  using bout::globals::mesh;
  Field3D boundary_only_seed{0.0};
  set_radial_phi_ghosts(boundary_only_seed, inner_midpoint, outer_midpoint);
  Field3D phi_plus_pi = boundary_only_seed + pi_hat;
  if (mesh->firstX()) {
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      for (int z = 0; z < mesh->LocalNz; ++z) {
        phi_plus_pi(mesh->xstart - 1, y, z) =
            0.5 * (phi_plus_pi(mesh->xstart - 1, y, z)
                   + phi_plus_pi(mesh->xstart, y, z));
      }
    }
  }
  if (mesh->lastX()) {
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      for (int z = 0; z < mesh->LocalNz; ++z) {
        phi_plus_pi(mesh->xend + 1, y, z) =
            0.5 * (phi_plus_pi(mesh->xend + 1, y, z)
                   + phi_plus_pi(mesh->xend, y, z));
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
  if (BoutComm::size() != 4 || mesh->NXPE != 1 || mesh->getNYPE() != 4
      || mesh->yend - mesh->ystart + 1 != LOCAL_Y) {
    throw std::runtime_error(
        "matched elliptic oracle requires four ranks, NXPE=1, NYPE=4");
  }
  const std::string input_path =
      Options::root()["paper0"]["input_file"]
          .doc("Matched O1 native-81 candidate or truth input")
          .as<std::string>();
  const bool truth_layout =
      Options::root()["paper0"]["truth_layout"]
          .doc("Read a canonical truth shard instead of a codec candidate")
          .withDefault(false);
  if (input_path.empty()) {
    throw std::runtime_error("paper0:input_file is required");
  }

  const BoutReal rho_s0 = normalize_metric_exactly();
  auto& laplace_options = Options::root()["paper0"]["laplacian"];
  auto solver = create_phi_solver(laplace_options);
  CandidateInput input(input_path, truth_layout);
  Options output;

  for (std::size_t position = 0; position < input.frames(); ++position) {
    const long long frame = input.frame(position);
    const std::string label = frame_label(frame);
    const Field3D ne = input.load_field("Ne", position);
    const Field3D pe = input.load_field("Pe", position);
    const Field3D pi = input.load_field("Pi", position);
    const Field3D vorticity = input.load_field("Vort", position);
    const Field3D runtime_pe = runtime_species_pressure(pe, ne);
    const Field3D runtime_pi = runtime_species_pressure(pi, ne);
    Field3D pi_hat =
        runtime_pi - runtime_pe / ELECTRON_PRESSURE_DENOMINATOR;
    pi_hat.applyBoundary("neumann");
    mesh->communicate(pi_hat);
    const auto inner = input.load_boundary(position, INNER_SIDE);
    const auto outer = input.load_boundary(position, OUTER_SIDE);
    const Field3D derived =
        solve_candidate(*solver, pi_hat, vorticity, inner, outer);
    output["derived_phi_" + label] = derived;
    output["canonical_frame_index_" + label] = frame;
  }

  output["paper0_oracle_name"] = "phase2_matched_e6b_elliptic";
  output["paper0_hermes_revision"] =
      "920ba829cc78cdab0dbf6101c69fecc4689bd8dd";
  output["paper0_bout_revision"] =
      "7d28d67c3f12c24ec281c0982e870f5369c65a6f";
  output["paper0_solver_type"] = "cyclic";
  output["paper0_zperiod"] = 5;
  output["paper0_rho_s0_meters"] = rho_s0;
  output["paper0_pressure_correction_denominator"] =
      ELECTRON_PRESSURE_DENOMINATOR;
  output["paper0_runtime_pressure_correction"] = 1;
  output["paper0_pressure_density_floor"] = PRESSURE_DENSITY_FLOOR;
  output["paper0_boundary_only_zero_interior_seed"] = 1;
  output["paper0_truth_layout"] = truth_layout ? 1 : 0;
  output["paper0_frame_count"] = static_cast<int>(input.frames());
  bout::writeDefaultOutputFile(output);
  bout::checkForUnusedOptions();
  BoutFinalise();
  return 0;
}
