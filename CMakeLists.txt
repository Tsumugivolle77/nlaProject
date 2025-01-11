cmake_minimum_required(VERSION 3.29)
project(nlaProject)

set(CMAKE_CXX_STANDARD 17)

include_directories(/opt/homebrew/Cellar/armadillo/14.0.3/include)

link_directories(/opt/homebrew/Cellar/armadillo/14.0.3/lib)

add_compile_options(-DDEBUG)

add_executable(nlaProject main.cpp)
target_link_libraries(nlaProject armadillo)