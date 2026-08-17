// SPDX-License-Identifier: GPL-3.0-or-later
//
// Compiled Paper 0 oracle for the radial x-face portion of Hermes-3's
// shifted-poloidal advection term. The face calculation is adapted from
// Hermes-3 src/div_ops.cxx lines 273-326 at revision
// 920ba829cc78cdab0dbf6101c69fecc4689bd8dd (GPL-3.0). The launcher verifies
// that exact source revision and file hash before compiling this executable.

#include <bout/bout.hxx>
#include <bout/derivs.hxx>
#include <bout/field_factory.hxx>

#include <array>
#include <string>

namespace {

Field3D manufactured_field(const std::string& option_name) {
  using bout::globals::mesh;
  const auto expression =
      Options::root()["mesh"][option_name].as<std::string>();
  Field3D field = FieldFactory::get()->create3D(
      expression, Options::getRoot(), mesh, CELL_CENTRE);
  field.applyBoundary("neumann");
  mesh->communicate(field);
  return field;
}

void evaluate_case(const std::string& name, Options& output) {
  using bout::globals::mesh;
  Coordinates* coord = mesh->getCoordinates();

  const Field3D q = manufactured_field("q_" + name);
  const Field3D phi = manufactured_field("phi_" + name);

  Field3D dfdy = DDY(phi);
  mesh->communicate(dfdy);
  dfdy.applyBoundary("neumann");

  Field3D face_velocity{0.0};
  Field3D face_state{0.0};
  Field3D face_flow{0.0};
  Field3D face_clipped{0.0};

  const int xs = mesh->xstart - 1;
  const int xe = mesh->xend;
  for (int i = xs; i <= xe; ++i) {
    for (int j = mesh->ystart - 1; j <= mesh->yend; ++j) {
      for (int k = 0; k < mesh->LocalNz; ++k) {
        const BoutReal metric_right =
            coord->g11(i + 1, j) * coord->g23(i + 1, j)
            / SQ(coord->Bxy(i + 1, j));
        const BoutReal metric_left =
            coord->g11(i, j) * coord->g23(i, j)
            / SQ(coord->Bxy(i, j));
        const BoutReal derivative_at_face =
            0.5 * (metric_right * dfdy(i + 1, j, k)
                   + metric_left * dfdy(i, j, k));
        const BoutReal velocity =
            0.5 * (coord->J(i + 1, j) + coord->J(i, j))
            * derivative_at_face;

        BoutReal state;
        if (velocity > 0.0) {
          state = q(i, j, k)
                  + 0.25 * (q(i + 1, j, k) - q(i - 1, j, k));
        } else {
          state = q(i + 1, j, k)
                  - 0.25 * (q(i + 2, j, k) - q(i, j, k));
        }
        const bool clipped = state < 0.0;
        if (clipped) {
          state = 0.0;
        }

        face_velocity(i, j, k) = velocity;
        face_state(i, j, k) = state;
        face_flow(i, j, k) = velocity * state;
        face_clipped(i, j, k) = clipped ? 1.0 : 0.0;
      }
    }
  }

  output["q_" + name] = q;
  output["phi_" + name] = phi;
  output["ddy_" + name] = dfdy;
  output["xy_velocity_" + name] = face_velocity;
  output["xy_state_" + name] = face_state;
  output["xy_flow_" + name] = face_flow;
  output["xy_clipped_" + name] = face_clipped;
}

} // namespace

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);

  Options output;
  constexpr std::array<const char*, 4> cases{
      "constant", "smooth", "signed", "clipping"};
  for (const auto* name : cases) {
    evaluate_case(name, output);
  }

  output["paper0_oracle_name"] =
      "hermes_shifted_xy_radial_face_85604_geometry";
  output["paper0_hermes_revision"] =
      "920ba829cc78cdab0dbf6101c69fecc4689bd8dd";
  output["paper0_positive_clipping"] = 1;
  bout::writeDefaultOutputFile(output);

  bout::checkForUnusedOptions();
  BoutFinalise();
  return 0;
}
