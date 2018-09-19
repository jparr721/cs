cmake_minimum_required(VERSION 3.6)
project(server)

set(CMAKE_C_STANDARD 11)

set(SOURCE_FILES main.c)
add_executable(server ${SOURCE_FILES})