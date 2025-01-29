#pragma gcc optimize("-O3")
#include <iostream>
#include <armadillo>
#include <complex>
#include <chrono>
#include "nebula.hpp"

// To enable printing additional information:
// #define DEBUG
// or
// `add_compile_option(-DDEBUG)` in CMakeLists.txt

void test() {
    using namespace std::complex_literals;
    using namespace arma;

    const int size = 200;

    cx_mat A(size, size, fill::zeros);

    for (int i = 0; i < size; ++i) {
        A(i, i) = i + 1;
    }

    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            std::complex<double> value = std::complex<double>(i + j + 1, i - j);
            A(i, j) = value;
            A(j, i) = std::conj(value);
        }
    }

    // A.print("The Elems of A:");

    eig_gen(A).print("Eigenvalues by Armadillo:");

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    auto eigs = nebula::qr::iteration_with_deflation(A);
    auto t2 = high_resolution_clock::now();

    std::cout << "Computation done after " << duration_cast<milliseconds>(t2 - t1).count() << "ms\n";
    std::cout << "Eigenvalues by my General Iteration with Deflation:\n";
    for (auto i : eigs) {
        std::cout << i << '\n';
    }
}

void test2() {
    using namespace arma;
    const int size = 5;

    arma_rng::set_seed(1919810);

    auto B = mat{ randu(size, size) };

    B.print("The Elems of B:");

    eig_gen(B).print("Eigenvalues by Armadillo:");

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    auto eigs = nebula::qr::general_iteration_with_deflation(B);
    auto t2 = high_resolution_clock::now();

    std::cout << "Computation done after " << duration_cast<milliseconds>(t2 - t1).count() << "ms\n";

    std::cout << "Eigenvalues by my General Iteration with Deflation:\n";
    for (auto i : eigs) {
        std::cout << i << '\n';
    }
}

int main() {
    // test();
    test2();

    return 0;
}
