# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(mobilellama VERSION 1.0)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cc)
add_library(mobilellama SHARED ${SOURCES})
