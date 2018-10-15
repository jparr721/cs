cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(muscle_contraction LANGUAGES CXX CUDA)

find_package(CUDA)

add_library(muscle_lib STATIC
  muscle.cu
  muscle.hpp
)

target_compile_features(muscle_lib PUBLIC cxx_std_11)

set_target_properties(muscle_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(muscle main.cu)

set_property(TARGET muscle PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(muscle PRIVATE muscle_lib)
