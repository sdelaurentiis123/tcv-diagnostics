cmake_minimum_required(VERSION 3.20)

project(paper0_hermes_native_frames_oracle LANGUAGES CXX)

if(NOT TARGET bout++::bout++)
  find_package(bout++ REQUIRED)
endif()
if(NOT TARGET netCDF::netcdf)
  find_package(netCDF CONFIG REQUIRED)
endif()

add_executable(
  hermes_native_frames_oracle
  hermes_native_frames_oracle.cxx
)
target_link_libraries(
  hermes_native_frames_oracle
  PRIVATE bout++::bout++ netCDF::netcdf
)
target_compile_features(hermes_native_frames_oracle PRIVATE cxx_std_17)
