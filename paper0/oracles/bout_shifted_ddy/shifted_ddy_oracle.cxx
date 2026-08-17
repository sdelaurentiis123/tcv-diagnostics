// SPDX-License-Identifier: GPL-3.0-or-later
//
// Paper 0 executable oracle for the BOUT++ shifted-metric y derivative.
// The initialization/output pattern is adapted from BOUT++'s GPL-licensed
// tests/integrated/test-yupdown/test_yupdown.cxx at revision 7d28d67.

#include <bout/bout.hxx>
#include <bout/derivs.hxx>
#include <bout/field_factory.hxx>

#include <array>
#include <string>

namespace {

void evaluate_case(const std::string& name, Options& output) {
  using bout::globals::mesh;

  const auto expression =
      Options::root()["mesh"]["input_" + name].as<std::string>();
  Field3D input = FieldFactory::get()->create3D(
      expression, Options::getRoot(), mesh, CELL_CENTRE);

  // This mirrors the state of an evolved Hermes field before DDY: physical
  // boundary guards are present and logical/MPI connections are communicated.
  input.applyBoundary("neumann");
  mesh->communicate(input);

  output["input_" + name] = input;
  output["ddy_" + name] = DDY(input, CELL_CENTRE, "C2");
}

} // namespace

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);

  Options output;
  constexpr std::array<const char*, 4> cases{
      "constant", "zmode", "ycode", "mixed"};
  for (const auto* name : cases) {
    evaluate_case(name, output);
  }

  output["paper0_oracle_name"] = "shifted_ddy_single_null_85604_geometry";
  output["paper0_derivative_method"] = "C2";
  bout::writeDefaultOutputFile(output);

  bout::checkForUnusedOptions();
  BoutFinalise();
  return 0;
}
