cmake_minimum_required(VERSION 3.20)

project(paper0_hermes_radial_flow_oracle LANGUAGES CXX)

if(NOT TARGET bout++::bout++)
  find_package(bout++ REQUIRED)
endif()

add_executable(hermes_radial_flow_oracle hermes_radial_flow_oracle.cxx)
target_link_libraries(hermes_radial_flow_oracle PRIVATE bout++::bout++)
target_compile_features(hermes_radial_flow_oracle PRIVATE cxx_std_17)
