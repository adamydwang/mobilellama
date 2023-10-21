# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the project name and version
project(mobilellama VERSION 1.0)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/deps/googletest/googletest/include
    ${PROJECT_SOURCE_DIR}/deps/spdlog/include
)

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cc)
add_subdirectory(${PROJECT_SOURCE_DIR}/deps)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
add_library(mobilellama SHARED ${SOURCES})