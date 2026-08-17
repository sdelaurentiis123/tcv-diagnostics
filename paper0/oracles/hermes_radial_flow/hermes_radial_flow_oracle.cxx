// SPDX-License-Identifier: GPL-3.0-or-later
//
// Compiled Paper 0 oracle for the radial-face portions of Hermes-3's
// Div_n_bxGrad_f_B_XPPM operator. The xz and shifted-xy calculations are
// adapted from Hermes-3 src/div_ops.cxx lines 128-229 and 273-326 at revision
// 920ba829cc78cdab0dbf6101c69fecc4689bd8dd (GPL-3.0). The launcher verifies
// that exact source revision and file hash before compilation.

#include <bout/bout.hxx>
#include <bout/derivs.hxx>
#include <bout/field_factory.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

namespace {

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

  Field3D xz_velocity{0.0};
  Field3D xz_state{0.0};
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
            0.25 * (phi(i, j, k) + phi(i, j, kp)
                    + phi(i + 1, j, k) + phi(i + 1, j, kp));
        const BoutReal corner_minus =
            0.25 * (phi(i, j, k) + phi(i + 1, j, k)
                    + phi(i, j, km) + phi(i + 1, j, km));
        const BoutReal velocity_xz =
            0.5 * (coord->J(i, j) + coord->J(i + 1, j))
            * (corner_plus - corner_minus) / coord->dz(i, j);

        const BoutReal left_state =
            q(i, j, k)
            + 0.5 * mc_slope(q(i - 1, j, k), q(i, j, k), q(i + 1, j, k));
        const BoutReal right_state =
            q(i + 1, j, k)
            - 0.5 * mc_slope(q(i, j, k), q(i + 1, j, k), q(i + 2, j, k));
        const BoutReal selected_xz_state =
            velocity_xz > 0.0 ? left_state : right_state;
        const BoutReal flow_xz = velocity_xz * selected_xz_state;

        const BoutReal metric_right =
            coord->g11(i + 1, j) * coord->g23(i + 1, j)
            / SQ(coord->Bxy(i + 1, j));
        const BoutReal metric_left =
            coord->g11(i, j) * coord->g23(i, j)
            / SQ(coord->Bxy(i, j));
        const BoutReal derivative_at_face =
            0.5 * (metric_right * dfdy(i + 1, j, k)
                   + metric_left * dfdy(i, j, k));
        const BoutReal velocity_xy =
            0.5 * (coord->J(i + 1, j) + coord->J(i, j))
            * derivative_at_face;
        BoutReal selected_xy_state;
        if (velocity_xy > 0.0) {
          selected_xy_state =
              q(i, j, k)
              + 0.25 * (q(i + 1, j, k) - q(i - 1, j, k));
        } else {
          selected_xy_state =
              q(i + 1, j, k)
              - 0.25 * (q(i + 2, j, k) - q(i, j, k));
        }
        if (selected_xy_state < 0.0) {
          selected_xy_state = 0.0;
        }
        const BoutReal flow_xy = velocity_xy * selected_xy_state;

        xz_velocity(i, j, k) = velocity_xz;
        xz_state(i, j, k) = selected_xz_state;
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
            (total_radial_flow(i, j, k) - total_radial_flow(i - 1, j, k))
            / (coord->J(i, j) * coord->dx(i, j));
      }
    }
  }

  output["q_" + name] = q;
  output["phi_" + name] = phi;
  output["xz_velocity_" + name] = xz_velocity;
  output["xz_state_" + name] = xz_state;
  output["xz_flow_" + name] = xz_flow;
  output["xy_flow_" + name] = xy_flow;
  output["total_radial_flow_" + name] = total_radial_flow;
  output["radial_divergence_" + name] = radial_divergence;
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
      "hermes_total_radial_face_and_divergence_85604_geometry";
  output["paper0_hermes_revision"] =
      "920ba829cc78cdab0dbf6101c69fecc4689bd8dd";
  bout::writeDefaultOutputFile(output);

  bout::checkForUnusedOptions();
  BoutFinalise();
  return 0;
}
