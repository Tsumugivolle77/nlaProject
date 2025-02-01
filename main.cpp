#pragma gcc optimize("-O3")
#include <armadillo>
#include <utils.hpp>

#include "examples.hpp"

// To enable printing additional information:
// #define DEBUG
// or
// `add_compile_option(-DDEBUG)` in CMakeLists.txt

using namespace arma;

int main() {
    int sizes[] = {30, 60, 90, 120, 500, 1000};
    for (auto size: sizes) {
        test_tridiag(size);
    }

    mat A = {
        {1.11, 2312, 123, 0.99},
        {1.31, 312, 231, 4.99},
        {3.11, 122, 3133, 8.99},
        {83.1, 12, 33.7, 156.99},
    };

    return 0;
}
