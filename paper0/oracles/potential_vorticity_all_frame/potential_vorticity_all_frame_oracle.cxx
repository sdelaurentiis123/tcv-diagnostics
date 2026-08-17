// SPDX-License-Identifier: GPL-3.0-or-later
//
// Potential-to-vorticity forward-closure oracle for one frozen 78-frame
// native-81 TCV/Hermes 85604 shard. The metric normalization, pressure, and
// cyclic-matrix blocks adapt Hermes-3 hermes-3.cxx and src/vorticity.cxx at
// revision
// 920ba829cc78cdab0dbf6101c69fecc4689bd8dd (GPL-3.0). The launcher locks
// those source files and the BOUT++ cyclic solver used by the executable.

#include <bout/bout.hxx>
#include <bout/boutcomm.hxx>
#include <bout/constants.hxx>
#include <bout/coordinates.hxx>
#include <bout/fft.hxx>
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

constexpr int FRAME_COUNT = 78;
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
constexpr BoutReal CONSTANT_NULL_VALUE = 3.25;
constexpr BoutReal GAUGE_SHIFT = 7.0;
constexpr int MANUFACTURED_TOROIDAL_MODE = 3;

#ifndef PAPER0_RUNTIME_PRESSURE_CORRECTION
#error "The forward-closure oracle requires the frozen runtime-pressure correction"
#endif

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
  CanonicalInput(const std::string& path, int expected_start)
      : shard_start(expected_start) {
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
      throw std::runtime_error(name + " does not use [frame,x,y,z]");
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
      throw std::runtime_error(name + " does not use [frame,side,y]");
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
  int shard_start;
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
    require_dimension("frame", FRAME_COUNT, frame_dim);
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
      if (frames[position] != shard_start + static_cast<int>(position)) {
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

Field3D retained_forward_field(
    const Field3D& stored_phi, const Field3D& pi_hat,
    const std::array<BoutReal, LOCAL_Y>& inner_midpoint,
    const std::array<BoutReal, LOCAL_Y>& outer_midpoint) {
  using bout::globals::mesh;
  Field3D phi_seed = stored_phi;
  set_radial_phi_ghosts(phi_seed, inner_midpoint, outer_midpoint);
  Field3D result = phi_seed + pi_hat;
  mesh->communicate(result);
  return result;
}

Field3D apply_cyclic_forward(Laplacian& solver, const Field3D& input) {
  using bout::globals::mesh;
  const int mode_count = mesh->LocalNz / 2 + 1;
  if (mode_count != 41) {
    throw std::runtime_error("forward oracle requires all 41 native modes");
  }

  Field3D communicated = input;
  mesh->communicate(communicated);
  const Field2D coefficient = 2.0 / SQ(mesh->getCoordinates()->Bxy);
  const Field2D d_coefficient{1.0};
  Field3D result{0.0};

  for (int y = mesh->ystart; y <= mesh->yend; ++y) {
    std::vector<dcomplex> spectrum(
        static_cast<std::size_t>(mesh->LocalNx) * mode_count);
    for (int x = 0; x < mesh->LocalNx; ++x) {
      rfft(communicated(x, y), mesh->LocalNz,
           spectrum.data() + static_cast<std::size_t>(x) * mode_count);
    }

    std::vector<dcomplex> forward_line(mode_count);
    for (int x = mesh->xstart; x <= mesh->xend; ++x) {
      for (int mode = 0; mode < mode_count; ++mode) {
        dcomplex lower;
        dcomplex diagonal;
        dcomplex upper;
        solver.tridagCoefs(x, y, mode, lower, diagonal, upper,
                           &coefficient, &d_coefficient);
        forward_line[mode] =
            lower
                * spectrum[static_cast<std::size_t>(x - 1) * mode_count
                           + mode]
            + diagonal
                  * spectrum[static_cast<std::size_t>(x) * mode_count + mode]
            + upper
                  * spectrum[static_cast<std::size_t>(x + 1) * mode_count
                             + mode];
      }
      irfft(forward_line.data(), mesh->LocalNz, result(x, y));
      for (int z = 0; z < mesh->LocalNz; ++z) {
        result(x, y, z) *= coefficient(x, y);
        if (!std::isfinite(result(x, y, z))) {
          throw std::runtime_error(
              "forward cyclic matrix produced non-finite data");
        }
      }
    }
  }
  result.applyBoundary("neumann");
  mesh->communicate(result);
  return result;
}

Field3D manufactured_solution() {
  using bout::globals::mesh;
  Field3D result{0.0};
  const BoutReal two_pi = 2.0 * PI;
  const int y_offset = mesh->getYProcIndex() * LOCAL_Y;
  for (int x = 0; x < mesh->LocalNx; ++x) {
    const BoutReal radial =
        (static_cast<BoutReal>(x - mesh->xstart) + 0.5) / GLOBAL_X;
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      const BoutReal poloidal =
          (static_cast<BoutReal>(y_offset + y - mesh->ystart) + 0.5)
          / GLOBAL_Y;
      for (int z = 0; z < mesh->LocalNz; ++z) {
        const BoutReal phase =
            two_pi * MANUFACTURED_TOROIDAL_MODE * z / GLOBAL_Z;
        result(x, y, z) =
            0.35 + 0.04 * radial + 0.02 * radial * radial
            + 0.01 * std::cos(two_pi * poloidal)
            + (0.06 + 0.01 * radial) * std::cos(phase)
            + 0.025 * std::sin(phase);
      }
    }
  }

  if (mesh->firstX()) {
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      for (int z = 0; z < mesh->LocalNz; ++z) {
        result(mesh->xstart - 2, y, z) =
            result(mesh->xstart - 1, y, z);
      }
    }
  }
  if (mesh->lastX()) {
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      for (int z = 0; z < mesh->LocalNz; ++z) {
        result(mesh->xend + 2, y, z) = result(mesh->xend + 1, y, z);
      }
    }
  }
  mesh->communicate(result);
  return result;
}

Field3D inversion_boundary_seed(const Field3D& solution) {
  using bout::globals::mesh;
  Field3D seed = solution;

  if (mesh->firstX()) {
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      for (int z = 0; z < mesh->LocalNz; ++z) {
        seed(mesh->xstart - 1, y, z) =
            0.5 * (solution(mesh->xstart - 1, y, z)
                   + solution(mesh->xstart, y, z));
      }
    }
  }
  if (mesh->lastX()) {
    for (int y = mesh->ystart; y <= mesh->yend; ++y) {
      for (int z = 0; z < mesh->LocalNz; ++z) {
        seed(mesh->xend + 1, y, z) =
            0.5 * (solution(mesh->xend + 1, y, z)
                   + solution(mesh->xend, y, z));
      }
    }
  }
  return seed;
}

Field3D invert_forward_result(Laplacian& solver, const Field3D& vorticity,
                              const Field3D& expected_solution) {
  using bout::globals::mesh;
  const Field2D bsq = SQ(mesh->getCoordinates()->Bxy);
  const Field3D seed = inversion_boundary_seed(expected_solution);
  Field3D result = solver.solve(vorticity * (bsq / 2.0), seed);
  mesh->communicate(result);
  return result;
}

} // namespace

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);
  using bout::globals::mesh;

  if (BoutComm::size() != 4) {
    throw std::runtime_error(
        "potential-vorticity forward oracle requires exactly four MPI ranks");
  }
  if (mesh->NXPE != 1 || mesh->getNYPE() != 4 ||
      mesh->yend - mesh->ystart + 1 != LOCAL_Y) {
    throw std::runtime_error(
        "potential-vorticity forward oracle requires NXPE=1,NYPE=4,MYSUB=8");
  }

  const auto input_path =
      Options::root()["paper0"]["input_file"]
          .doc("Canonical Paper 0 potential-vorticity input")
          .as<std::string>();
  if (input_path.empty()) {
    throw std::runtime_error(
        "paper0:input_file must name the canonical input");
  }
  const int shard_start =
      Options::root()["paper0"]["shard_start"]
          .doc("First global frame in the frozen 78-frame shard")
          .as<int>();
  if (shard_start < 0 || shard_start > 546 || shard_start % FRAME_COUNT != 0) {
    throw std::runtime_error(
        "paper0:shard_start must be one of 0,78,...,546");
  }

  const BoutReal rho_s0 = normalize_metric_exactly();
  auto& laplace_options = Options::root()["paper0"]["laplacian"];
  auto forward_solver = create_phi_solver(laplace_options);
  auto manufactured_inverse_solver = create_phi_solver(laplace_options);
  CanonicalInput input(input_path, shard_start);
  Options output;

  const Field3D constant_field{CONSTANT_NULL_VALUE};
  const Field3D constant_forward =
      apply_cyclic_forward(*forward_solver, constant_field);
  const Field3D manufactured = manufactured_solution();
  const Field3D manufactured_forward =
      apply_cyclic_forward(*forward_solver, manufactured);
  const Field3D manufactured_reconstruction =
      invert_forward_result(*manufactured_inverse_solver,
                            manufactured_forward, manufactured);
  output["constant_forward_vort"] = constant_forward;
  output["manufactured_u"] = manufactured;
  output["manufactured_forward_vort"] = manufactured_forward;
  output["manufactured_reconstructed_u"] = manufactured_reconstruction;

  for (std::size_t position = 0; position < FRAME_COUNT; ++position) {
    const int frame = shard_start + static_cast<int>(position);
    const std::string label = frame_label(frame);
    const Field3D ne = input.load_field("Ne", position);
    const Field3D pe = input.load_field("Pe", position);
    const Field3D pi = input.load_field("Pi", position);
    const Field3D vorticity = input.load_field("Vort", position);
    const Field3D stored_phi = input.load_field("phi", position);

    const Field3D runtime_pe = runtime_species_pressure(pe, ne);
    const Field3D runtime_pi = runtime_species_pressure(pi, ne);
    Field3D pi_hat =
        runtime_pi - runtime_pe / ELECTRON_PRESSURE_DENOMINATOR;
    pi_hat.applyBoundary("neumann");
    mesh->communicate(pi_hat);

    const auto saved_inner =
        input.load_boundary("saved_midpoint", position, INNER_SIDE);
    const auto saved_outer =
        input.load_boundary("saved_midpoint", position, OUTER_SIDE);
    const Field3D forward_input = retained_forward_field(
        stored_phi, pi_hat, saved_inner, saved_outer);
    const Field3D forward_vorticity =
        apply_cyclic_forward(*forward_solver, forward_input);

    output["input_Vort_" + label] = vorticity;
    output["runtime_Pe_" + label] = runtime_pe;
    output["runtime_Pi_" + label] = runtime_pi;
    output["forward_Vort_" + label] = forward_vorticity;
    output["canonical_frame_index_" + label] = frame;

    if (position == 0) {
      const Field3D gauge_shifted_input = forward_input + GAUGE_SHIFT;
      output["gauge_forward_base"] = forward_vorticity;
      output["gauge_forward_shifted"] =
          apply_cyclic_forward(*forward_solver, gauge_shifted_input);
    }
  }

  output["paper0_oracle_name"] =
      "phase2_potential_vorticity_all_frame_85604";
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
  output["paper0_forward_operator"] = "Laplacian::tridagCoefs";
  output["paper0_forward_mode_max"] = 40;
  output["paper0_constant_null_value"] = CONSTANT_NULL_VALUE;
  output["paper0_gauge_shift"] = GAUGE_SHIFT;
  output["paper0_manufactured_mode_k"] = MANUFACTURED_TOROIDAL_MODE;
  output["paper0_shard_start"] = shard_start;
  output["paper0_shard_stop"] = shard_start + FRAME_COUNT;
  output["paper0_shard_frame_count"] = FRAME_COUNT;
  bout::writeDefaultOutputFile(output);

  bout::checkForUnusedOptions();
  BoutFinalise();
  return 0;
}
