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

using namespace arma;

void test(const int size, double tol = 1e-6) {
    using namespace std::complex_literals;
    using namespace arma;

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

    auto t0 = high_resolution_clock::now();
    auto M = nebula::hermitian_tridiag2sym_tridiag(nebula::to_hessenberg(A));
    auto t1 = high_resolution_clock::now();
    auto eigs = nebula::qr::iteration_with_deflation_for_tridiag_using_BFS(M, tol);
    // auto eigs = nebula::qr::iteration_with_shift_for_hermitian(A);
    auto t2 = high_resolution_clock::now();

    std::cout << "size=" << size << std::endl;
    std::cout << "Transformation done after " << duration_cast<milliseconds>(t1 - t0).count() << "ms\n";
    std::cout << "Iteration done after " << duration_cast<milliseconds>(t2 - t1).count() << "ms\n";
    std::cout << "Total time " << duration_cast<milliseconds>(t2 - t0).count() << "ms\n\n";
    // std::cout << "Eigenvalues by my Iteration with Deflation:\n";
    // eigs.print("Eigenvalues by My:");
    for (auto i : eigs) {
        std::cout << i << '\n';
    }
}

void test2() {
    using namespace arma;
    const int size = 200;

    mat B(size, size, fill::zeros);

    for (int i = 0; i < size; ++i) {
        B(i, i) = i + 1;
    }

    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            double value = i + j + 1;
            B(i, j) = value;
            B(j, i) = -value;
        }
    }

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
    int sizes[] = {30, 60, 90, 120, 500, /*1000*/};
    for (auto size : sizes) {
        test(size);
    }
    // test(10);
    // test(5);
    // test2();

    return 0;
}
