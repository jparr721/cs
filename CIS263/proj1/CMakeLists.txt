cmake_minimum_required(VERSION 3.8)
project(proj1)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES p1.c functions.c functions.h functions.c)
add_executable(proj1 ${SOURCE_FILES})
