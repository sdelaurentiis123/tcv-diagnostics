cmake_minimum_required(VERSION 3.20)

project(paper0_shifted_ddy_oracle LANGUAGES CXX)

if(NOT TARGET bout++::bout++)
  find_package(bout++ REQUIRED)
endif()

add_executable(shifted_ddy_oracle shifted_ddy_oracle.cxx)
target_link_libraries(shifted_ddy_oracle PRIVATE bout++::bout++)
target_compile_features(shifted_ddy_oracle PRIVATE cxx_std_17)

