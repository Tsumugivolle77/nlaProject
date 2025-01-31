//
// Created by Tsumugi on 31.01.25.
//

#include "tridiag_matrix.hpp"
#include <armadillo>


namespace nebula {
    using namespace arma;

    tridiag_matrix::tridiag_matrix(const vec &diag, const vec &subdiag, const vec &superdiag)
        : diag(diag), subdiag(subdiag), superdiag(superdiag)
    { }

    tridiag_matrix::tridiag_matrix(const mat &m) {
        diag = vec(m.n_rows);
        subdiag = vec(m.n_rows - 1);
        superdiag = vec(m.n_rows - 1);

        for (uint i = 0; i < m.n_rows; ++i) {
            uint col_start = i == 0 ? i : i - 1;

            for (uint j = col_start; j <= i; ++j) {
                if (i == j) {
                    diag[i] = m(i, i);
                } else if (i == j + 1) {
                    subdiag[j] = m(i, j);
                    superdiag[j] = m(i, j);
                }
            }
        }
    }

    tridiag_matrix::tridiag_matrix(double only_elem)
        : diag({only_elem})
    { }

    tridiag_matrix::tridiag_matrix(const tridiag_matrix &sym_mat) = default;

    uint tridiag_matrix::size() const
    { return diag.n_rows; }

    vec &tridiag_matrix::diagonal() {
        return diag;
    }

    vec tridiag_matrix::diagonal() const
    { return diag; }

    vec &tridiag_matrix::subdiagonal()
    { return subdiag; }

    vec tridiag_matrix::subdiagonal() const
    { return subdiag; }

    vec &tridiag_matrix::superdiagonal()
    { return subdiag; }

    vec tridiag_matrix::superdiagonal() const
    { return subdiag; }

    double tridiag_matrix::operator()(const uint &i, const uint &j) const {
        if (i == j) {
            return diag[i];
        }

        if (i + 1 == j) {
            return superdiag[i];
        }

        if (i == j + 1) {
            return subdiag[j];
        }

        return 0;
    }

    double &tridiag_matrix::operator()(const uint &i, const uint &j) {
        if (i == j) {
            return diag[i];
        }

        if (i + 1 == j) {
            return superdiag[i];
        }

        if (i == j + 1) {
            return subdiag[j];
        }

        throw std::invalid_argument("index out of range.");
    }

    double tridiag_matrix::at(const uint &i, const uint &j) const {
        if (i == j) {
            return diag[i];
        }

        if (i + 1 == j) {
            return superdiag[i];
        }

        if (i == j + 1) {
            return subdiag[j];
        }

        return 0;
    }

   tridiag_matrix tridiag_matrix::partition(const uint &start, const uint &end) const {
        if (start == end) {
            return tridiag_matrix{diag.subvec(start, end)};
        }
        return tridiag_matrix{diag.subvec(start, end), subdiag.subvec(start, end - 1), superdiag.subvec(start, end - 1)};
    }

    void tridiag_matrix::print(const std::string &label) const {
        size_t n = diag.n_elem;
        std::cout << label << '\n';

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i == j) std::cout << std::setw(8) << diag(i) << " ";
                else if (i == j + 1) std::cout << std::setw(8) << subdiag(j) << " ";
                else if (i + 1 == j) std::cout << std::setw(8) << superdiag(i) << " ";
                else std::cout << std::setw(8) << "0" << " ";
            }
            std::cout << "\n";
        }

        std::cout << std::endl;
    }
}