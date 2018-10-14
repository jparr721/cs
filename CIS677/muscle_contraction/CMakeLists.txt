cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(muscle LANGUAGES CXX CUDA)

find_package(CUDA)

add_library(muscle STATIC
  src/muscle.cu
  include/muscle/muscle.h
  src/main.cu
)

target_compile_features(muscle PUBLIC cxx_std_11)

set_target_properties(muscle
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

