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

    nebula::nla_mat A = {
        {cx_double(4, 0), cx_double(1, 2), cx_double(3, -1), cx_double(0, 1), cx_double(2, -3)},
        {cx_double(1, -2), cx_double(5, 0), cx_double(1, 4), cx_double(2, 1), cx_double(0, -1)},
        {cx_double(3, 1), cx_double(1, -4), cx_double(6, 0), cx_double(4, 2), cx_double(1, 1)},
        {cx_double(0, -1), cx_double(2, -1), cx_double(4, -2), cx_double(3, 0), cx_double(5, -2)},
        {cx_double(2, 3), cx_double(0, 1), cx_double(1, -1), cx_double(5, 2), cx_double(8, 0)}
    };

    nebula::nla_mat B = mat {
            {1, 1.5, 4, 5.},
            {1, 4., 1, 9,},
            {4, 11., 9, 8,},
            {5, 9, 8., 0}
    };

    // auto Ahermitri = A.to_hessenberg();
    // auto Asymmetri = hermitian_tridiag2sym_tridiag(Ahermitri);
    // // since A is real symmetric tridiagonal now, we can safely extract its real part
    // nebula::nla_mat Asymmetri_real = mat{Asymmetri.get_mat()};

    // std::cout
    //     << "A Hermitian:\n" << A << '\n'
    // << "A after applying Householder Transform:\n" << Ahermitri << '\n'
    //     << "A transformed to real symmetric tridiagonal:\n" << Asymmetri << '\n'
    //     << "A transformed to real symmetric tridiagonal:\n" << Asymmetri_real << '\n'
    //     << "A eigenvalues:\n" << eig_sym(A.get_mat()) << '\n'
    // << "A Hermitri eigenvalues:\n" << eig_sym(Ahermitri.get_mat()) << '\n'
    // << "A Symmetri eigenvalues:\n" << eig_sym(Asymmetri.get_mat()) << '\n'
    //     << "A Symmetri eigenvalues:\n" << eig_sym(Asymmetri_real.get_mat()) << '\n'
    // ;

    std::cout
        << "For A:" << std::endl
        << "after QR iters with shift:\n" << nebula::qr::iteration_with_shift(A, 600) << std::endl
        << "after QR iters for hermi:\n" << nebula::qr::iteration_with_shift_for_hermitian(A, 500) << std::endl
        << "after QR iters:\n" << nebula::qr::iteration(A, 1000) << std::endl
        << "eigs:\n" << eig_gen(A.get_mat()) << std::endl
    ;

    std::cout
        << "For B:" << std::endl
        << "after QR iters with shift:\n" << nebula::qr::iteration_with_shift(B, 500) << std::endl
        << "after QR iters:\n" << nebula::qr::iteration(B, 500) << std::endl
        << "eigs:\n" << eig_gen(B.get_mat()) << std::endl
    ;
}

int main() {
    test();

    return 0;
}
