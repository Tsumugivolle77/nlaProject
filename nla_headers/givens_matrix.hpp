//
// Created by Tsumugi on 17.01.25.
//

#ifndef GIVENS_MATRIX_H
#define GIVENS_MATRIX_H

#include "tridiag_matrix.hpp"

template <typename T>
class givens_matrix {
public:
    uint j, k;
    T c, s;

    givens_matrix() = default;
    givens_matrix(uint j, uint k, T c, T s);
    givens_matrix(T a, T b, uint j, uint k);
    [[nodiscard]] givens_matrix transpose() const;
};

template <typename T>
givens_matrix<T>::givens_matrix(uint j, uint k, T c, T s): j(j), k(k), c(c), s(s)
{ }

template <>
inline givens_matrix<double>::givens_matrix(double a, double b, uint j, uint k)
    : j(j), k(k)
{
    double r = std::hypot(a, b);
    if (std::abs(r) < 1e-10) {
        c = 1.0;
        s = 0.0;
    } else {
        c = a / r;
        s = b / r;
    }
}

template <>
inline givens_matrix<std::complex<double>>::givens_matrix(std::complex<double> a, std::complex<double> b, uint j, uint k)
    : j(j), k(k)
{
    using namespace std::complex_literals;

    auto anorm = std::abs(a);
    auto bnorm = std::abs(b);
    double r = std::hypot(anorm, bnorm);

    if (bnorm < 1e-10) {
        c = 1.0;
        s = 0.0;
    } if (anorm < 1e-10) {
        c = 0.;
        s = 1.;
    } else {
        c = anorm / r;
        s = a / anorm * conj(b) / r;
    }
}

template <typename T>
givens_matrix<T>
givens_matrix<T>::transpose() const
{ return {j, k, c, -s}; }

inline Row<std::complex<double>> operator*(const Row<std::complex<double>> &v, const givens_matrix<std::complex<double>> &g) {
    auto res = v;
    uint j = g.j, k = g.k;
    std::complex<double> c = g.c, s = g.s;

    res[j] = c * v[j] - std::conj(s) * v[k];
    res[k] = s * v[j] + c * v[k];

    return res;
}

inline Col<std::complex<double>> operator*(const givens_matrix<std::complex<double>> &g, const Col<std::complex<double>> &v) {
    auto res = v;
    uint j = g.j, k = g.k;
    std::complex<double> c = g.c, s = g.s;

    res[j] = c * v[j] + s * v[k];
    res[k] = -std::conj(s) * v[j] + c * v[k];

    return res;
}

inline Col<double> operator*(const givens_matrix<double> &g, const Col<double> &v) {
    auto res = v;
    uint j = g.j, k = g.k;
    double c = g.c, s = g.s;

    res[j] = c * v[j] + s * v[k];
    res[k] = -s * v[j] + c * v[k];

    return res;
}


inline Row<double> operator*(const Row<double> &v, const givens_matrix<double> &g) {
    auto res = v;
    uint j = g.j, k = g.k;
    double c = g.c, s = g.s;

    res[j] = c * v[j] - s * v[k];
    res[k] = s * v[j] + c * v[k];

    return res;
}

template <typename T>
Mat<T> apply_givens(const givens_matrix<T> &g, const Mat<T> &m) {
    auto res = m;
    uint cols = res.n_cols;

    for (uint i = 0; i < cols; ++i) {
        res.col(i) = g * res.col(i);
    }

    return res;
}

template <typename T>
Mat<T> apply_givens(const givens_matrix<T> &g, const Mat<T> &m, const std::vector<uint> & cols) {
    auto res = m;

    for (auto col: cols) {
        res.col(col) = g * res.col(col);
    }

    return res;
}

template <typename T>
Mat<T> apply_givens(const Mat<T> &m, const givens_matrix<T> &g) {
    auto res = m;
    uint rows = res.n_rows;

    for (uint i = 0; i < rows; ++i) {
        res.row(i) = res.row(i) * g;
    }

    return res;
}

template <typename T>
Mat<T> apply_givens(const Mat<T> &m, const givens_matrix<T> &g, const std::vector<uint> & rows) {
    auto res = m;

    for (auto row: rows) {
        res.row(row) = res.row(row) * g;
    }

    return res;
}

// !!! NOT IMPLEMENTED
// inline tridiag_matrix apply_givens_tridiag(const givens_matrix<double> &g, const tridiag_matrix &m) {
//     auto res = m;
//     uint j = g.j, k = g.k;
//     double c = g.c, s = g.s;
//
//     double aj = res.at(j, j), ak = res.at(k, k);
//     double bj = res.at(j, k), bk = res.at(k, j);
//
//     res.set(j, j, c * aj + s * bj);
//     res.set(k, k, s * bk - c * ak);
//
//     if (k > 0) {
//         double sj = res.sub.at(k - 1);
//         res.sub.at(k - 1) = c * sj - s * res.super.at(k - 1);
//     }
//
//     if (j < res.n_rows - 1) {
//         double sk = res.super.at(j);
//         res.super.at(j) = s * res.sub.at(j) + c * sk;
//     }
//
//     return res;
// }
//
// inline tridiag_matrix apply_givens_tridiag(const tridiag_matrix &m, const givens_matrix<double> &g) {
//     auto res = m;
//     uint j = g.j, k = g.k;
//     double c = g.c, s = g.s;
//
//     double aj = res.at(j, j), ak = res.at(k, k);
//     double bj = res.at(j, k), bk = res.at(k, j);
//
//     res.set(j, j, c * aj + s * bk);
//     res.set(k, k, -s * bj + c * ak);
//
//     if (j > 0) {
//         double sj = res.sub.at(j - 1);
//         res.sub.at(j - 1) = c * sj - s * res.super.at(j - 1);
//     }
//
//     if (k < res.n_rows - 1) {
//         double sk = res.super.at(k);
//         res.super.at(k) = s * res.sub.at(k) + c * sk;
//     }
//
//     return res;
// }

#endif //GIVENS_MATRIX_H
