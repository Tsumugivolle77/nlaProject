//
// Created by Tsumugi on 31.01.25.
//

#include "examples.hpp"
#include "armadillo"
#include "nebula.hpp"

using namespace arma;

void test(const int size, double tol) {
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

void test2(const int size) {
    using namespace arma;

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

void test_tridiag(const int size, double tol) {
    using namespace std::complex_literals;
    using namespace arma;

    // cx_mat A(size, size, fill::zeros);
    //
    // for (int i = 0; i < size; ++i) {
    //     A(i, i) = i + 1;
    // }
    //
    // for (int i = 0; i < size; ++i) {
    //     for (int j = i + 1; j < size; ++j) {
    //         std::complex<double> value = std::complex<double>(i + j + 1, i - j);
    //         A(i, j) = value;
    //         A(j, i) = std::conj(value);
    //     }
    // }

    cx_mat A = create_matrix(size);

    // A.print("The Elems of A:");

    eig_gen(A).print("Eigenvalues by Armadillo:");

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    auto t0 = high_resolution_clock::now();
    auto M = nebula::hermitian_to_tridiag_mat(nebula::to_hessenberg(A));
    auto t1 = high_resolution_clock::now();
    auto eigs = nebula::qr::iteration_with_deflation_for_specialized_tridiag(M, tol);
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

cx_mat create_matrix(const size_t &n, const vec &eigenvalues) {
    cx_mat A;
    if (eigenvalues.size() == 1) {
        cx_vec diag(n);
        for (int i = 0; i < n; i++) {
            diag(i) = cx_double{i + 1., 0};
        }
        A = diagmat(diag);
    } else {
        cx_vec diag(n);
        for (int i = 0; i < n; i++) {
            diag(i) = cx_double{eigenvalues[i], 0};
        }
        A = diagmat(diag);
    }

    for (int j = 0; j < n; ++j) {
        for (int k = j + 1; k < n; ++k) {
            const double phi = randu() * 2 * M_PI;
            const double c = cos(phi);
            const double s = sin(phi);

            for (int i = 0; i < n; ++i) {
                auto alpha = A(j, i);
                auto beta = A(k, i);

                A(j, i) = c * alpha - s * beta;
                A(k, i) = s * alpha + c * beta;
            }

            for (int i = 0; i < n; ++i) {
                auto alpha = A(i, j);
                auto beta = A(i, k);

                A(i, j) = c * alpha - s * beta;
                A(i, k) = s * alpha + c * beta;
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        const double phi = randu() * 2 * M_PI;
        const cx_double e_i_phi = exp(cx_double(0, phi));

        A.col(i) *= e_i_phi;
        A.row(i) *= conj(e_i_phi);
    }

    return A;
}