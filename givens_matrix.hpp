//
// Created by Tsumugi on 17.01.25.
//

#ifndef GIVENS_MATRIX_H
#define GIVENS_MATRIX_H
#include <complex>
#include <armadillo>
#include "nla_mat.hpp"

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
inline givens_matrix<T>::givens_matrix(uint j, uint k, T c, T s): j(j), k(k), c(c), s(s)
{ }

template <>
inline givens_matrix<double>::givens_matrix(double a, double b, uint j, uint k)
: j(j), k(k), c(1. / std::sqrt(1. + b * b / a / a)), s(1. / std::sqrt(a * a / b / b + 1.))
{ }

template <>
inline givens_matrix<std::complex<double>>::givens_matrix(std::complex<double> a, std::complex<double> b, uint j, uint k)
: j(j), k(k)
{
    using namespace std::complex_literals;

    auto alph = std::arg(a);
    auto beta = std::arg(b);
    auto phi = beta - alph;
    auto anorm = std::abs(a);
    auto bnorm = std::abs(b);
    auto r = std::sqrt(anorm * anorm + bnorm * bnorm);
    c = anorm / r;
    s = a / anorm * conj(b) / r;
}

template <typename T>
givens_matrix<T>
inline givens_matrix<T>::transpose() const
{ return {j, k, c, -s}; }

inline arma::Row<std::complex<double>> operator*(const arma::Row<std::complex<double>> &v, const givens_matrix<std::complex<double>> &g) {
    auto res = v;
    uint j = g.j, k = g.k;
    std::complex<double> c = g.c, s = g.s;

    res[j] = c * v[j] - std::conj(s) * v[k];
    res[k] = s * v[j] + c * v[k];

    return res;
}

inline arma::Col<std::complex<double>> operator*(const givens_matrix<std::complex<double>> &g, const arma::Col<std::complex<double>> &v) {
    auto res = v;
    uint j = g.j, k = g.k;
    std::complex<double> c = g.c, s = g.s;

    res[j] = c * v[j] + s * v[k];
    res[k] = -std::conj(s) * v[j] + c * v[k];

    return res;
}

inline arma::Col<double> operator*(const givens_matrix<double> &g, const arma::Col<double> &v) {
    auto res = v;
    uint j = g.j, k = g.k;
    double c = g.c, s = g.s;

    res[j] = c * v[j] + s * v[k];
    res[k] = -s * v[j] + c * v[k];

    return res;
}


inline arma::Row<double> operator*(const arma::Row<double> &v, const givens_matrix<double> &g) {
    auto res = v;
    uint j = g.j, k = g.k;
    double c = g.c, s = g.s;

    res[j] = c * v[j] - s * v[k];
    res[k] = s * v[j] + c * v[k];

    return res;
}

template <typename T>
nla_mat<arma::Mat<T>> operator*(const givens_matrix<T> &g, const nla_mat<arma::Mat<T>> &m) {
    using namespace arma;

    auto res = m.get_mat();
    uint cols = res.n_cols;

    for (uint i = 0; i < cols; ++i) {
        res.col(i) = g * res.col(i);
    }

    return res;
}

template <typename T, typename U = typename T::elem_type>
nla_mat<T> operator*(const nla_mat<T> &m, const givens_matrix<U> &g) {
    using namespace arma;

    auto res = m.get_mat();
    uint rows = res.n_rows;

    for (uint i = 0; i < rows; ++i) {
        res.row(i) = res.row(i) * g;
    }

    return res;
}


#endif //GIVENS_MATRIX_H
