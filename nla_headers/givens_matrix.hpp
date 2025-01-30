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

inline Row<std::complex<double>> operator*(Row<std::complex<double>> &v, const givens_matrix<std::complex<double>> &g) {
    auto res = v;
    uint j = g.j, k = g.k;
    std::complex<double> c = g.c, s = g.s;
    std::complex<double> vj = v[j], vk = v[k];

    res[j] = c * vj - std::conj(s) * vk;
    res[k] = s * vj + c * vk;

    return res;
}

inline Col<std::complex<double>> operator*(const givens_matrix<std::complex<double>> &g, const Col<std::complex<double>> &v) {
    auto res = v;
    uint j = g.j, k = g.k;
    std::complex<double> c = g.c, s = g.s;
    std::complex<double> vj = v[j], vk = v[k];

    res[j] = c * vj + s * vk;
    res[k] = -std::conj(s) * vj + c * vk;

    return res;
}

inline Col<double> operator*(const givens_matrix<double> &g, Col<double> &v) {
    auto res = v;
    uint j = g.j, k = g.k;
    double c = g.c, s = g.s;
    double vj = v[j], vk = v[k];

    res[j] =  c * vj + s * vk;
    res[k] = -s * vj + c * vk;

    return res;
}


inline Row<double> operator*(Row<double> &v, const givens_matrix<double> &g) {
    auto res = v;
    uint j = g.j, k = g.k;
    double c = g.c, s = g.s;
    double vj = v[j], vk = v[k];

    res[j] = c * vj - s * vk;
    res[k] = s * vj + c * vk;

    return res;
}

template <typename T>
void apply_givens(const givens_matrix<T> &g, Mat<T> &m) {
    uint cols = m.n_cols;

    for (uint i = 0; i < cols; ++i) {
        uint j = g.j, k = g.k;
        double c = g.c, s = g.s;
        double vj = m.at(i, j), vk = m.at(i, k);

        m.at(i, j) = c * vj - s * vk;
        m.at(i, k) = s * vj + c * vk;
    }
}

template <typename T>
void apply_givens(const givens_matrix<T> &g, Mat<T> &m, const std::vector<uint> & cols) {
    for (auto i : cols) {
        uint j = g.j, k = g.k;
        double c = g.c, s = g.s;
        double vj = m.at(i, j), vk = m.at(i, k);

        m.at(i, j) = c * vj - s * vk;
        m.at(i, k) = s * vj + c * vk;
    }
}

template <typename T>
void apply_givens(Mat<T> &m, const givens_matrix<T> &g) {
    uint rows = m.n_rows;

    for (uint i = 0; i < rows; ++i) {
        uint j = g.j, k = g.k;
        double c = g.c, s = g.s;
        double vj = m.at(j, i), vk = m.at(k, i);

        m.at(j, i) =  c * vj + s * vk;
        m.at(k, i) = -s * vj + c * vk;
    }
}

template <typename T>
void apply_givens(Mat<T> &m, const givens_matrix<T> &g, const std::vector<uint> & rows) {
    for (auto i : rows) {
        uint j = g.j, k = g.k;
        double c = g.c, s = g.s;
        double vj = m.at(j, i), vk = m.at(k, i);

        m.at(j, i) =  c * vj + s * vk;
        m.at(k, i) = -s * vj + c * vk;
    }
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
