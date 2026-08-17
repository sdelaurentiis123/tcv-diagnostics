cmake_minimum_required(VERSION 3.20)

project(paper0_hermes_xy_face_oracle LANGUAGES CXX)

if(NOT TARGET bout++::bout++)
  find_package(bout++ REQUIRED)
endif()

add_executable(hermes_xy_face_oracle hermes_xy_face_oracle.cxx)
target_link_libraries(hermes_xy_face_oracle PRIVATE bout++::bout++)
target_compile_features(hermes_xy_face_oracle PRIVATE cxx_std_17)
