cmake_minimum_required(VERSION 3.9)
project(router LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED_ON)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_PATH})
set(LIBRARY_OUTOUT_PATH ${CMAKE_BINARY_DIR})

add_executable(router
  "${PROJECT_SOURCE_DIR}/src/*"
)

target_link_libraries(router)
