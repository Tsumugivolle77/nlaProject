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

    const int size = 100;

    cx_mat hermitian_matrix(size, size, fill::zeros);

    for (int i = 0; i < size; ++i) {
        hermitian_matrix(i, i) = i + 1;
    }

    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            std::complex<double> value = std::complex<double>(i + j + 1, i - j);
            hermitian_matrix(i, j) = value;
            hermitian_matrix(j, i) = std::conj(value);
        }
    }

    auto A = cx_mat { hermitian_matrix };

    arma_rng::set_seed(1919810);

    auto B = mat{ randu(200, 200) };

    eig_gen(B).print("Eigenvalues by Armadillo:");
        // << "eigs:\n" << nebula::qr::iteration_with_deflation(A) << std::endl
    ;

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    // nebula::qr::iteration(B).print("eigs");

    auto t1 = high_resolution_clock::now();
    auto eigs = nebula::qr::general_iteration_with_deflation(B);
    auto t2 = high_resolution_clock::now();

    std::cout << "Computation done after " << duration_cast<milliseconds>(t2 - t1).count() << "ms\n";

    std::cout << "Eigenvalues by my General Iteration with Deflation:\n";
    for (auto i : eigs) {
        std::cout << i << '\n';
    }

    // std::cout
    //     << "For B:" << std::endl
    // << "after QR iters with shift:\n" << nebula::qr::iteration_with_shift(B, 50) << std::endl
    //     << "after QR iters:\n" << nebula::qr::iteration(B, 500) << std::endl
    //     << "eigs:\n" << eig_gen(B.get_mat()) << std::endl
    ;

}

void test2() {
    using namespace std::complex_literals;

    arma::cx_mat A = {
        {-6.348e-01 + 3.770e-01i, -9.457e-01 + 9.748e-01i, -3.627e-01 + 5.396e-01i},
        {+6.734e-01 + 1.146e+00i, -4.490e-01 - 5.443e-01i, +2.160e-01 + 4.648e-01i},
        {0.0 + 0.0i,               -2.788e-02 + 1.173e-03i, -7.407e-01 - 1.331e+00i}
    };

    nebula::qr::general_iteration_with_deflation(A);
}

int main() {
    test();

    return 0;
}
