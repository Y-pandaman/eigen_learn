cmake_minimum_required(VERSION 3.5)
project(eigen_learn)

set( CMAKE_CXX_STANDARD 11 )
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(test src/test.cpp)
target_link_libraries(test)