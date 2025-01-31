#pragma gcc optimize("-O3")
#include <armadillo>
#include "examples.hpp"

// To enable printing additional information:
// #define DEBUG
// or
// `add_compile_option(-DDEBUG)` in CMakeLists.txt

using namespace arma;

int main() {
    int sizes[] = {30, 60, 90, 120, 500, 1000};
    for (auto size : sizes) {
        test_tridiag(size);
    }

    return 0;
}
