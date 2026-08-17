cmake_minimum_required(VERSION 3.20)

project(paper0_potential_elliptic_runtime_pressure_oracle LANGUAGES CXX)

if(NOT TARGET bout++::bout++)
  find_package(bout++ REQUIRED)
endif()
if(NOT TARGET netCDF::netcdf)
  find_package(netCDF CONFIG REQUIRED)
endif()

add_executable(
  potential_elliptic_runtime_pressure_oracle
  ../potential_elliptic/potential_elliptic_oracle.cxx
)
target_compile_definitions(
  potential_elliptic_runtime_pressure_oracle
  PRIVATE PAPER0_RUNTIME_PRESSURE_CORRECTION=1
)
target_link_libraries(
  potential_elliptic_runtime_pressure_oracle
  PRIVATE bout++::bout++ netCDF::netcdf
)
target_compile_features(
  potential_elliptic_runtime_pressure_oracle
  PRIVATE cxx_std_17
)
