//
// Created by Tsumugi on 13.01.25.
//

#ifndef UTILS_H
#define UTILS_H

// for complex vector
inline Mat<std::complex<double>>
get_householder_mat(const Col<std::complex<double>> &x) {
    using namespace std::complex_literals;

    auto x1 = x[0];
    auto phase = std::arg(x1);
    auto e1      = cx_colvec(x.n_rows);
    e1[0]        = 1.;
    const auto I = cx_mat(x.n_rows, x.n_rows, fill::eye);

    const cx_vec w = x + std::exp(1i * phase) * norm(x) * e1;
    const cx_rowvec wh{w.ht()};

    return I - 2 * w * wh / dot(wh, w);
}

// for real vector
inline Mat<double>
get_householder_mat(const Col<double> &x) {
    auto x1 = x[0];
    auto e1 = colvec(x.n_rows);
    e1[0] = 1.;
    const auto I = mat(x.n_rows, x.n_rows, fill::eye);

    auto sgn = [](auto v) -> auto { return v >= 0 ? 1 : -1; };
    const vec w = x + sgn(x1) * norm(x) * e1;
    const rowvec wh{w.t()};

    return I - 2 * w * wh / dot(wh, w);
}

// get the Hessenberg form of matx
template<typename M>
M to_hessenberg(const M &matx) {
    using elem_type = typename M::elem_type;

    if (matx.n_cols != matx.n_rows) { throw std::runtime_error("nla_mat: not a square matrix"); }

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

/*** !!! FOR THE FIRST SUBTASK: converting Hermitian Tridiagonal resulting
 **  from `to_hessenberg` into Real Symmetric Tridiagonal
 **  @param H Hermitian Tridiagonal
 **  @return Real Symmetric Tridiagonal, similar to H
***/
inline mat hermitian_tridiag2sym_tridiag(const cx_mat &H)
{
    using namespace std::complex_literals;

    auto hermitri = H;

    for (int i = 2; i < hermitri.n_cols; ++i) {
        for (int j = 0; j < i - 1; ++j) {
            hermitri.at(j, i) = 0.;
        }
    }

    auto rows = hermitri.n_rows;

    auto diag_entries = cx_colvec(rows);
    diag_entries[0] = 1.;
    auto totphase = 0.;

    for (int i = 1; i < rows; ++i) {
        auto subdiag_entry = hermitri.at(i, i - 1);
        auto phase = -std::arg(subdiag_entry);
        totphase += phase;
        diag_entries[i] = std::exp(1i * totphase);
    }

    for (int i = 0; i < rows; ++i) {
        int nonzero_beg = i > 0 ? i - 1 : i;
        int nonzero_end = i + 1 < rows ? i + 1 : i;
        for (int j = nonzero_beg; j <= nonzero_end; ++j) {
            hermitri.at(i, j) = diag_entries[i] * hermitri.at(i, j);
        }
    }

    for (int i = 0; i < rows; ++i) {
        int nonzero_beg = i > 0 ? i - 1 : i;
        int nonzero_end = i + 1 < rows ? i + 1 : i;
        for (int j = nonzero_beg; j <= nonzero_end; ++j) {
            hermitri.at(j, i) = std::conj(diag_entries[i]) * hermitri.at(j, i);
        }
    }

    return real(hermitri);
}

#endif //UTILS_H
