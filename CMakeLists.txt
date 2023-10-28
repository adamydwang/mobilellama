# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE DEBUG)
# Set the project name and version
project(mobilellama VERSION 1.0)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/deps/googletest/googletest/include
    ${PROJECT_SOURCE_DIR}/deps/spdlog/include
)
link_directories(${PROJECT_SOURCE_DIR}/lib)
file(
    GLOB_RECURSE SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cc
    ${PROJECT_SOURCE_DIR}/src/*.h
)

add_library(sentencepiece STATIC IMPORTED)
set_property(TARGET sentencepiece PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/lib/libsentencepiece.a)

add_library(mobilellama STATIC ${SOURCES})
target_link_libraries(mobilellama sentencepiece)

add_executable(demo ${PROJECT_SOURCE_DIR}/example/demo.cc)
target_link_libraries(demo mobilellama)

add_subdirectory(${PROJECT_SOURCE_DIR}/deps)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)