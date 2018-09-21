cmake_minimum_required(VERSION 3.9)
project(crystal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

find_package(OpenMP)

add_executable(crystal
  "${PROJECT_SOURCE_DIR}/src/main.cpp"
)

target_link_libraries(crystal)
