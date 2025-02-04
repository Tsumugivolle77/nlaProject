//
// Created by Tsumugi on 17.01.25.
//

#ifndef GIVENS_MATRIX_H
#define GIVENS_MATRIX_H

#include "tridiag_matrix.hpp"
#include <armadillo>

using namespace arma;

namespace nebula {
    template<typename T>
    class givens_matrix {
    public:
        uint j, k;
        T c, s;

        givens_matrix() = default;

        givens_matrix(uint j, uint k, T c, T s);

        givens_matrix(T a, T b, uint j, uint k);

        [[nodiscard]] givens_matrix transpose() const;
    };

    template<typename T>
    givens_matrix<T>::givens_matrix(uint j, uint k, T c, T s): j(j), k(k), c(c), s(s) {
    }

    template<>
    inline givens_matrix<double>::givens_matrix(double a, double b, const uint j, const uint k)
        : j(j), k(k) {
        double r = std::hypot(a, b);
        if (std::abs(r) < 1e-10) {
            c = 1.0;
            s = 0.0;
        } else {
            c = a / r;
            s = b / r;
        }
    }

    template<>
    inline givens_matrix<std::complex<double> >::givens_matrix(std::complex<double> a, std::complex<double> b, uint j,
                                                               uint k)
        : j(j), k(k) {
        using namespace std::complex_literals;

        auto anorm = std::abs(a);
        auto bnorm = std::abs(b);
        auto r = std::hypot(anorm, bnorm);

        if (bnorm < 1e-10) {
            c = 1.0;
            s = 0.0;
        }
        if (anorm < 1e-10) {
            c = 0.;
            s = 1.;
        } else {
            c = anorm / r;
            s = a / anorm * conj(b) / r;
        }
    }

    template<typename T>
    givens_matrix<T>
    givens_matrix<T>::transpose() const { return {j, k, c, -s}; }

    Row<std::complex<double> > operator*(const Row<std::complex<double> > &v,
                                         const givens_matrix<std::complex<double> > &g);

    Col<std::complex<double> > operator*(const givens_matrix<std::complex<double> > &g,
                                         const Col<std::complex<double> > &v);

    Col<double> operator*(const givens_matrix<double> &g, const Col<double> &v);

    Row<double> operator*(const Row<double> &v, const givens_matrix<double> &g);

    template<typename T>
    Mat<T> apply_givens(const givens_matrix<T> &g, const Mat<T> &m) {
        auto res = m;
        auto cols = m.n_cols;
        auto j = g.j, k = g.k;

        for (uint i = 0; i < cols; ++i) {
            auto gcol = g * res.col(i);
            res.at(j, i) = gcol(j);
            res.at(k, i) = gcol(k);
        }

        return res;
    }

    template<typename T>
    Mat<T> apply_givens(const givens_matrix<T> &g, const Mat<T> &m, const std::vector<uint> &cols) {
        auto res = m;
        auto j = g.j, k = g.k;

        for (auto col: cols) {
            auto gcol = g * res.col(col);
            res.at(j, col) = gcol(j);
            res.at(k, col) = gcol(k);
        }

        return res;
    }

    template<typename T>
    Mat<T> apply_givens(const Mat<T> &m, const givens_matrix<T> &g) {
        auto res = m;
        uint rows = res.n_rows;
        auto j = g.j, k = g.k;

        for (uint i = 0; i < rows; ++i) {
            auto rowg = res.row(i) * g;
            res.at(i, j) = rowg(j);
            res.at(i, k) = rowg(k);
        }

        return res;
    }

    template<typename T>
    Mat<T> apply_givens(const Mat<T> &m, const givens_matrix<T> &g, const std::vector<uint> &rows) {
        auto res = m;
        auto j = g.j, k = g.k;

        for (auto row: rows) {
            auto rowg = res.row(row) * g;
            res.at(row, j) = rowg(j);
            res.at(row, k) = rowg(k);
        }

        return res;
    }
}

#endif //GIVENS_MATRIX_H
