//
// Created by Tsumugi on 20.01.25.
//

#ifndef QR_ITERATION_HPP
#define QR_ITERATION_HPP
#include "givens_matrix.hpp"
#include "nla_mat.hpp"
#include <armadillo>
#include <complex>

namespace details {
    bool doesConverge(const arma::mat &A) {
        //...//
    }
};

namespace qr {
    /*** !!! SUBTASK 2: QR iteration w\o deflation and shift
     **  @tparam T type of the Armadillo matrix, on which we are performing operations
     **  @param m input matrix
     **  @param maxiter maximum iteration steps
     **  @return quasi-upper-triangular matrix
    ***/
    template <typename T>
    auto iteration(const nla_mat<T> &m, uint maxiter = 1000) {
        auto hess = m.to_hessenberg();
        auto row = hess.get_mat().n_rows;

        for (uint i = 0; i < maxiter; ++i) {
            for (uint j = 0; j < row - 1; ++j) {
                auto a = hess.get_mat().at(j, j);
                auto b = hess.get_mat().at(j + 1, j);
                givens_matrix<typename T::elem_type> g = {a, b, j, j + 1};
                hess = g * hess * g.transpose();
            }
        }

        return hess;
    }

    template <typename T>
    auto iteration_with_shift(const nla_mat<T> &m, uint maxiter = 1000) {
        auto hess = m.to_hessenberg();
        auto row = hess.get_mat().n_rows;

        for (uint i = 0; i < maxiter; ++i) {
            for (uint j = 0; j < row - 1; ++j) {
                auto a = hess.get_mat().at(j, j);
                auto b = hess.get_mat().at(j + 1, j);
                givens_matrix<typename T::elem_type> g = {a, b, j, j + 1};
                hess = g * hess * g.transpose();
            }
        }

        return hess;
    }

    template <typename T>
    auto iteration_with_deflation(const nla_mat<T> &m, uint maxiter = 1000) {
        auto hess = m.to_hessenberg();
        auto row = hess.get_mat().n_rows;

        for (uint i = 0; i < maxiter; ++i) {
            for (uint j = 0; j < row - 1; ++j) {
                auto a = hess.get_mat().at(j, j);
                auto b = hess.get_mat().at(j + 1, j);
                givens_matrix<typename T::elem_type> g = {a, b, j, j + 1};
                hess = g * hess * g.transpose();
            }
        }

        return hess;
    }

    // arma::rowvec qr_iteration(const nla_mat<arma::mat> &m, uint maxiter = 1000) {
    //     auto hess = m.to_hessenberg();
    //
    //     for (uint i = 0; i < maxiter; ++i) {
    //
    //     }
    // }
}

#endif //QR_ITERATION_HPP
