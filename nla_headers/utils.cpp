//
// Created by Tsumugi on 31.01.25.
//

#include "utils.hpp"

namespace nebula {
    Mat<std::complex<double>>
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
    Mat<double>
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

    mat hermitian_tridiag2sym_tridiag(const cx_mat &H)
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

    tridiag_matrix hermitian_to_tridiag_mat(const cx_mat &H)
    {
        using namespace std::complex_literals;

        auto hermitri = to_hessenberg(H);

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

        return tridiag_matrix {real(hermitri)};
    }
}