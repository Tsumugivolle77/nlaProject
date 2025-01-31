//
// Created by Tsumugi on 31.01.25.
//

#ifndef TRIDIAG_MATRIX_HPP
#define TRIDIAG_MATRIX_HPP

#include <armadillo>
#include "givens_matrix.hpp"

namespace nebula {
    class tridiag_matrix {
        arma::vec diag;
        arma::vec subdiag;
        arma::vec superdiag;

    public:
        tridiag_matrix(const arma::vec &diag, const arma::vec &subdiag, const arma::vec &superdiag);

        explicit tridiag_matrix(const arma::mat &m);

        tridiag_matrix(const tridiag_matrix &sym_mat);

        explicit tridiag_matrix(double only_elem);

        [[nodiscard]] uint size() const;

        arma::vec &diagonal();

        [[nodiscard]] arma::vec diagonal() const;

        arma::vec &subdiagonal();

        [[nodiscard]] arma::vec subdiagonal() const;

        arma::vec &superdiagonal();

        [[nodiscard]] arma::vec superdiagonal() const;

        double operator()(const uint &i, const uint &j) const;

        double &operator()(const uint &i, const uint &j);

        double at(const uint &i, const uint &j) const;

        [[nodiscard]] tridiag_matrix partition(const uint &start, const uint &end) const;

        void print(const std::string &label = "") const;
    };
}

#endif //TRIDIAG_MATRIX_HPP
