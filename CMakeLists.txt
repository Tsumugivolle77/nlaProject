cmake_minimum_required(VERSION 3.29)
project(nlaProject)

set(CMAKE_CXX_STANDARD 14)

include_directories(/opt/homebrew/Cellar/armadillo/14.0.3/include)

link_directories(/opt/homebrew/Cellar/armadillo/14.0.3/lib)

add_executable(numerical_linear_algebra1 main.cpp)
target_link_libraries(numerical_linear_algebra1 armadillo)