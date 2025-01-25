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

    const int size = 100;

    arma::cx_mat hermitian_matrix(size, size, arma::fill::zeros);

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

    nebula::nla_mat A = cx_mat{ hermitian_matrix };

    // nebula::nla_mat B = mat {
    //         {1, 1.5, 4, 5.},
    //         {1, 4., 1, 9,},
    //         {4, 11., 9, 8,},
    //         {5, 9, 8., 0}
    // };

    // auto Ahermitri = A.to_hessenberg();
    // auto Asymmetri = hermitian_tridiag2sym_tridiag(Ahermitri);
    // // since A is real symmetric tridiagonal now, we can safely extract its real part
    // nebula::nla_mat Asymmetri_real = mat{Asymmetri.get_mat()};

    std::cout
        << "A Hermitian:\n" << A << '\n'
    // << "A after applying Householder Transform:\n" << Ahermitri << '\n'
    //     << "A transformed to real symmetric tridiagonal:\n" << Asymmetri << '\n'
    //     << "A transformed to real symmetric tridiagonal:\n" << Asymmetri_real << '\n'
    //     << "A eigenvalues:\n" << eig_sym(A.get_mat()) << '\n'
    // << "A Hermitri eigenvalues:\n" << eig_sym(Ahermitri.get_mat()) << '\n'
    // << "A Symmetri eigenvalues:\n" << eig_sym(Asymmetri.get_mat()) << '\n'
    //     << "A Symmetri eigenvalues:\n" << eig_sym(Asymmetri_real.get_mat()) << '\n'
    ;

    std::cout
        << "For A:" << std::endl
        // << "after QR iters with shift:\n" << nebula::qr::iteration_with_shift(A, 1000) << std::endl
        // << "after QR iters for hermi:\n" << nebula::qr::iteration_with_shift_for_hermitian(A, 400) << std::endl
        // << "after QR iters:\n" << nebula::qr::iteration(A, 20) << std::endl
        << "eigs:\n" << eig_gen(A.get_mat()) << std::endl
        // << "eigs:\n" << nebula::qr::iteration_with_deflation(A) << std::endl
    ;

    auto eigs = nebula::qr::iteration_with_deflation(A);

    std::cout << "eigs:\n";
    for (auto i : eigs) {
        std::cout << i << std::endl;
    }

    // std::cout
    //     << "For B:" << std::endl
    //     << "after QR iters with shift:\n" << nebula::qr::iteration_with_shift(B, 100) << std::endl
    //     << "after QR iters:\n" << nebula::qr::iteration(B, 500) << std::endl
    //     << "eigs:\n" << eig_gen(B.get_mat()) << std::endl
    // ;

}

int main() {
    test();

    return 0;
}
