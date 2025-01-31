//
// Created by Tsumugi on 30.01.25.
//

#ifndef QR_ITERATION_FOR_TRIDIAG_HPP
#define QR_ITERATION_FOR_TRIDIAG_HPP

#include "tridiag_matrix.hpp"

namespace nebula::details {
    bool nearZero(const tridiag_matrix &hess, int i, double tol = 1e-6);
}

namespace nebula::qr {
    void step_with_wilkinson_shift(tridiag_matrix &tridiag, const double &shift);

    std::vector<double> iteration_with_deflation_for_specialized_tridiag(const tridiag_matrix &m, double tol = 1e-6);
}

#endif //QR_ITERATION_FOR_TRIDIAG_HPP
