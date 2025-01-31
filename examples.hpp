//
// Created by Tsumugi on 31.01.25.
//

#ifndef EXAMPLES_HPP
#define EXAMPLES_HPP
#include <armadillo>

void test(const int size, double tol = 1e-6);

void test2(const int size);

void test_tridiag(const int size, double tol = 1e-6);

arma::cx_mat create_matrix(const size_t &n, const arma::vec &eigenvalues = arma::vec{1, arma::fill::zeros});

#endif //EXAMPLES_HPP