#pragma gcc optimize("-O3")
#include <iostream>
#include <armadillo>
#include <complex>
#include "nebula.hpp"

// To enable printing additional information:
// #define DEBUG
// or
// `add_compile_option(-DDEBUG)` in CMakeLists.txt

void test() {
    using namespace std::complex_literals;
    using namespace arma;

    const int size = 700;

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

    auto B = mat{ randu(48, 48) };

    eig_gen(B).print("eigs");
        // << "eigs:\n" << nebula::qr::iteration_with_deflation(A) << std::endl
    ;

    // nebula::qr::iteration(B).print("eigs");

    auto eigs = nebula::qr::general_iteration_with_deflation(B, 1e-6, nebula::qr::step_for_hessenberg<cx_mat>);

    std::cout << "eigs:\n";
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

class A {
    arma::vec x, y, z;
};

class B {
    arma::mat a;
};

int main() {
    test();

    return 0;
}
