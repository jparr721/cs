cmake_minimum_required(VERSION 3.9)
project(swerv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

include_directories("${PROJECT_SOURCE_DIR}/include")

add_executable(swerver
  "${PROJECT_SOURCE_DIR}/src/main.cc"
  "${PROJECT_SOURCE_DIR}/src/core.cc"
)
