cmake_minimum_required(VERSION 3.20)

project(paper0_matched_e6b_elliptic LANGUAGES C CXX)

if(NOT TARGET bout++::bout++)
  find_package(bout++ REQUIRED)
endif()
find_package(HDF5 REQUIRED COMPONENTS C)

add_executable(
  matched_e6b_elliptic_oracle
  matched_e6b_elliptic_oracle.cxx
)
target_link_libraries(
  matched_e6b_elliptic_oracle
  PRIVATE bout++::bout++ HDF5::HDF5
)
target_compile_features(matched_e6b_elliptic_oracle PRIVATE cxx_std_17)
