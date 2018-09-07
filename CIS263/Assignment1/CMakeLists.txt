cmake_minimum_required(VERSION 3.8)
project(CIS263)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp house.h parr.cpp)
add_executable(CIS263 ${SOURCE_FILES})