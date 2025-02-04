//
// Created by Tsumugi on 13.01.25.
//

#ifndef UTILS_H
#define UTILS_H

#include "tridiag_matrix.hpp"
#include <stdexcept>

namespace nebula {
    // for complex vector
    Mat<std::complex<double>>
    get_householder_mat(const Col<std::complex<double>> &x);

    // for real vector
    Mat<double>
    get_householder_mat(const Col<double> &x);

    Col<std::complex<double>>
    compute_householder_vec(const Col<std::complex<double>> &x);

    Col<double>
    compute_householder_vec(const Col<double> &x);

    // get the Hessenberg form of matx
    template<typename M>
    M to_hessenberg(const M &matx) {
        using elem_type = typename M::elem_type;

        if (matx.n_cols != matx.n_rows) { throw std::invalid_argument("to_hessenberg: not a square matrix"); }

        auto hess = matx;

        for (int i = 0; i < hess.n_cols - 2; ++i) {
            auto x = Col<elem_type>(hess.submat(i + 1, i, hess.n_rows - 1, i));
            auto H = get_householder_mat(x);
            auto Qi = Mat<elem_type>(hess.n_rows, hess.n_cols, fill::eye);

            Qi(span(i + 1, Qi.n_rows - 1), span(i + 1, Qi.n_cols - 1)) = H;

            hess = Qi * hess;
            hess = hess * Qi.ht();
        }

        for (int i = 2; i < hess.n_rows; ++i) {
            for (int j = 0; j < i - 1; ++j) {
                hess.at(i, j) = 0.;
            }
        }

        return hess;
    }

    // get the Hessenberg form of matx
    template<typename M>
    M to_hessenberg_optimized(const M &matx) {
        using elem_type = typename M::elem_type;

        if (matx.n_cols != matx.n_rows) { throw std::invalid_argument("to_hessenberg_optimized: not a square matrix"); }

        auto res = matx;

        for (int i = 0; i < res.n_cols - 2; ++i) {
            auto x = Col<elem_type>(res.submat(i + 1, i, res.n_rows - 1, i));
            auto w = compute_householder_vec(x);
            Col<elem_type> fullcol { res.n_rows, fill::zeros };
            fullcol.subvec(i + 1, res.n_rows - 1) = w;

            // derived from Algorithm 1.2.8 from Lecture Notes
            Mat<elem_type> Qres = res - 2. * fullcol * (res.ht() * fullcol).ht();
            res = Qres - 2. * (Qres * fullcol) * fullcol.ht();
        }

        return res;
    }

    /*** !!! FOR THE FIRST SUBTASK: converting Hermitian Tridiagonal resulting
     **  from `to_hessenberg` into Real Symmetric Tridiagonal
     **  @param H Hermitian Tridiagonal
     **  @return Real Symmetric Tridiagonal, similar to H
    ***/
    mat hermitian_tridiag2sym_tridiag(const cx_mat &H);

    tridiag_matrix hermitian_to_tridiag_mat(const cx_mat &H);
}

#endif //UTILS_H
