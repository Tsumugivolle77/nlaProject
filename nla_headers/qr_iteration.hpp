//
// Created by Tsumugi on 20.01.25.
//

#ifndef QR_ITERATION_HPP
#define QR_ITERATION_HPP

#include <armadillo>
#include "givens_matrix.hpp"

namespace nebula {
    namespace details {
        void __iteration_with_deflation_impl(std::shared_ptr<arma::mat> &, std::vector<double> &eigs, double tol);

        void __general_iteration_with_deflation_impl(std::shared_ptr<arma::mat> &,
                                                     std::vector<std::complex<double> > &eigs, double tol);

        template<typename M>
        bool doesConverge(const M &hess, double tol = 1e-6) { return norm(hess.diag(-1), 2) < tol; }

        template<typename M>
        bool nearZero(std::shared_ptr<M> &hess, int i, double tol) {
            return std::abs(hess->at(i, i - 1)) < tol * (std::abs(hess->at(i - 1, i - 1)) + std::abs(hess->at(i, i)));
        }

        // partition for real matrix
        void partition(std::shared_ptr<arma::mat> &hess, std::vector<double> &eigs, double tol = 1e-6);

        // partition for real nonsymmetric matrix
        void partition(std::shared_ptr<arma::mat> &hess, std::vector<std::complex<double> > &eigs, double tol = 1e-6);
    }

    namespace qr {
        inline void francis_step(arma::mat &);

        // function for perform a qr step
        template<typename M>
        void step_for_hessenberg(M &hess) {
            auto rows = hess.n_rows;

            for (uint j = 0; j < rows - 1; ++j) {
                auto a = hess.at(j, j);
                auto b = hess.at(j + 1, j);
                givens_matrix<typename M::elem_type> g{a, b, j, j + 1};
                std::vector<uint> applied_to(rows - j);
                std::iota(applied_to.begin(), applied_to.end(), j);
                hess = apply_givens(g, hess, applied_to);
                hess = apply_givens(hess, g.transpose(), applied_to);
            }
        }

        /*** !!! SUBTASK 2-1: QR iteration w\o deflation and shift
         **  @tparam M type of the Armadillo matrix, on which we are performing operations
         **  @param m input matrix
         **  @param maxiter maximum iteration steps
         **  @return quasi-upper-triangular matrix
        ***/
        template<typename M>
        arma::Col<typename M::elem_type> iteration(const M &m, uint maxiter = 1000) {
            auto hess = to_hessenberg(m);

            for (uint i = 0; i < maxiter; ++i) {
                step_for_hessenberg(hess);

                if (details::doesConverge(hess)) {
#ifdef DEBUG
            std::cout << "Converge after " << i << " Steps." << std::endl;
#endif
                    break;
                }
            }

#ifdef DEBUG
    std::cout << "Matrix after QR iteration:\n" << hess << std::endl;
#endif

            return {hess.diag()};
        }

        template<typename M>
        void step_with_wilkinson_shift(M &hess, const typename M::elem_type &shift) {
            auto row = hess.n_rows; {
                auto a = hess.at(0, 0) - shift;
                auto b = hess.at(1, 0);
                givens_matrix<typename M::elem_type> g{a, b, 0, 1};
                std::vector<uint> applied_to = {0, 1, 2};
                hess = apply_givens(g, hess, applied_to);
                hess = apply_givens(hess, g.transpose(), applied_to);
            }

            for (uint j = 1; j < row - 1; ++j) {
                auto a = hess.at(j, j - 1);
                auto b = hess.at(j + 1, j - 1);
                givens_matrix<typename M::elem_type> g{a, b, j, j + 1};
                std::vector<uint> applied_to = {};
                if (j < row - 2) applied_to = {j - 1, j, j + 1, j + 2};
                else applied_to = {j - 1, j, j + 1};
                hess = apply_givens(g, hess, applied_to);
                hess = apply_givens(hess, g.transpose(), applied_to);
            }
        }

        vec iteration_with_shift_for_real_symmetric_tridiagonal(const mat &tridiag, uint maxiter = 1000);

        vec iteration_with_shift_for_hermitian(const cx_mat &m, uint maxiter = 1000);

        vec iteration_with_shift_for_symmetric(const mat &m, uint maxiter = 1000);

        /*** !!! SUBTASK 2-2: QR Iteration with Francis QR Step
         **  @tparam M type of the Armadillo matrix, on which we are performing operations
         **  @param m input matrix
         **  @param maxiter maximum iteration steps
         **  @return quasi-upper-triangular matrix
         **/
        template<typename M>
        Col<typename M::elem_type> iteration_with_shift(const M &m, uint maxiter = 1000) {
            auto hess = to_hessenberg(m);

            for (uint i = 0; i < maxiter; ++i) {
                francis_step(hess);

                if (details::doesConverge(hess)) {
#ifdef DEBUG
            std::cout << "Converge after " << i << " Steps." << std::endl;
#endif
                    break;
                }
            }
#ifdef DEBUG
    std::cout << "Matrix after QR iteration with implicit double shift:\n" << hess.get_mat() << std::endl
        << "Computed eigs:\n";
#endif

            return {hess.diag()};
        }

        /*** !!! SUBTASK 2-3: QR Iteration with Deflation for Hermitian
         **  @param m complex hermitian matrix
         **  @param tol tolerance of error
         **  @return the real eigenvalues
         ***/
        std::vector<double> iteration_with_deflation(cx_mat &m, double tol = 1e-6);

        /*** !!! SUBTASK 2-3: QR Iteration with Deflation for Real Symmetric
         **  @param m real symmetric matrix
         **  @param tol tolerance of error
         **  @return the real eigenvalues
         ***/
        std::vector<double> iteration_with_deflation(mat &m, double tol = 1e-6);

        /*** !!! SUBTASK 2-3: QR Iteration with Deflation for Real Symmetric
         **  @param m real symmetric matrix
         **  @param tol tolerance of error
         **  @return the real eigenvalues
         ***/
        std::vector<double>
        iteration_with_deflation_for_tridiag(mat &m, double tol = 1e-6);

        /*** !!! SUBTASK 2-3: QR Iteration with Deflation for Real Symmetric
         **  @param m real symmetric matrix
         **  @param tol tolerance of error
         **  @return the real eigenvalues
         ***/
        std::vector<double>
        iteration_with_deflation_for_tridiag_using_BFS(mat &m, double tol = 1e-6);

        /*** !!! SUBTASK 2-3: QR Iteration with Deflation for Real Matrix
         **  @param m real matrix
         **  @param tol tolerance of error
         **  @return the complex eigenvalues
         ***/
        std::vector<std::complex<double> >
        general_iteration_with_deflation(mat &m, double tol = 1e-6);
    }
}

#endif //QR_ITERATION_HPP
