//
// Created by Tsumugi on 20.01.25.
//

#ifndef QR_ITERATION_HPP
#define QR_ITERATION_HPP

namespace details {
    template <typename T>
    bool doesConverge(const nla_mat<T> &hess, double tol = 1e-10)
    { return norm(hess.get_mat().diag(-1), 2) < tol; }
};

namespace qr {
using namespace details;
// function for perform a qr step
template <typename T>
void step_for_hessenberg(nla_mat<T> &hess, const typename T::elem_type &shift = 0.) {
    auto row = hess.get_mat().n_rows;

    {
        auto a = hess.get_mat().at(0, 0) - shift;
        auto b = hess.get_mat().at(1, 0);
        givens_matrix<typename T::elem_type> g {a, b, 0, 1};
        hess = g * hess * g.transpose();
    }

    for (uint j = 1; j < row - 1; ++j) {
        auto a = hess.get_mat().at(j, j);
        auto b = hess.get_mat().at(j + 1, j);
        givens_matrix<typename T::elem_type> g {a, b, j, j + 1};
        hess = g * hess * g.transpose();
    }
}

/*** !!! SUBTASK 2-1: QR iteration w\o deflation and shift
 **  @tparam T type of the Armadillo matrix, on which we are performing operations
 **  @param m input matrix
 **  @param maxiter maximum iteration steps
 **  @return quasi-upper-triangular matrix
***/
template <typename T>
Col<typename T::elem_type> iteration(const nla_mat<T> &m, uint maxiter = 1000) {
    auto hess = m.to_hessenberg();

    for (uint i = 0; i < maxiter; ++i) {
        step_for_hessenberg(hess);

        if (doesConverge(hess)) break;
    }

    std::cout << "Matrix after QR iteration:\n" << hess.get_mat() << std::endl;

    return {hess.get_mat().diag()};
}

inline vec iteration_with_shift_for_real_symmetric_tridiagonal(const nla_mat<mat> &tridiag, uint maxiter = 1000) {
    auto cols = tridiag.get_mat().n_cols;
    auto res  = tridiag;

    for (uint i = 0; i < maxiter; ++i) {
        auto sign = [] (const auto &num) { return num >= 0 ? 1 : -1; };
        const auto &r = res.get_mat();
        auto a = r.at(cols - 1, cols - 1);
        auto b = r.at(cols - 2, cols - 2);
        auto c = r.at(cols - 1, cols - 2);
        auto d = (b - a) / 2.;
        auto shift = a + d - sign(d) * std::sqrt(d * d + c * c);

        step_for_hessenberg(res, shift);

        if (doesConverge(res)) break;
    }

    return res.get_mat().diag();
}

inline vec iteration_with_shift_for_hermitian(const nla_mat<cx_mat> &m, uint maxiter = 1000) {
    auto hess = m.to_hessenberg();
    auto tridiag = hermitian_tridiag2sym_tridiag(hess);
    return iteration_with_shift_for_real_symmetric_tridiagonal(tridiag, maxiter);
}

inline vec iteration_with_shift_for_symmetric(const nla_mat<mat> &m, uint maxiter = 1000) {
    auto tridiag = m.to_hessenberg();
    return iteration_with_shift_for_real_symmetric_tridiagonal(tridiag, maxiter);
}

// Francis QR Step
template <typename T>
inline void francis_step(nla_mat<T> &hess) {
    {
        // set up the implicit shift
        using et = typename T::elem_type;
        et s      = trace(hess.get_mat());
        et t      = det(hess.get_mat());
        et h00    = hess.get_mat().at(0, 0);
        et h10    = hess.get_mat().at(1, 0);
        auto col0 = hess.get_mat().col(0);
        auto col1 = hess.get_mat().col(1);
        Col<et> w = h00 * col0 + h10 * col1 - s * col0;
        w[0] += t;


        auto Q = nla_mat<>::get_householder_mat(w);
        hess = {Q * hess.get_mat() * Q.ht()};
    }

    hess = {hess.to_hessenberg()};
}

// inline void francis_step(nla_mat<cx_mat> &hess) {
//     {
//         // set up the implicit shift
//         // using et = typename T::elem_type;
//         auto s      = trace(hess.get_mat());
//         auto t      = det(hess.get_mat());
//         auto h00    = hess.get_mat().at(0, 0);
//         auto h10    = hess.get_mat().at(1, 0);
//         auto col0 = hess.get_mat().col(0);
//         auto col1 = hess.get_mat().col(1);
//         cx_colvec w = h00 * col0 + h10 * col1 - s * col0;
//         w[0] += t;
//
//
//         auto Q = nla_mat<>::get_householder_mat(w);
//         hess = {Q * hess.get_mat() * Q.ht()};
//     }
//
//     hess = {hess.to_hessenberg()};
// }
//
// inline void francis_step(nla_mat<mat> &hess) {
//     {
//         // set up the implicit shift
//         auto s    = trace(hess.get_mat());
//         auto t    = det(hess.get_mat());
//         auto col0 = hess.get_mat().col(0);
//         auto col1 = hess.get_mat().col(1);
//         auto h00  = hess.get_mat().at(0, 0);
//         auto h10  = hess.get_mat().at(1, 0);
//         colvec w = h00 * col0 + h10 * col1 - s * col0;
//         w[0] += t;
//
//         auto Q = nla_mat<>::get_householder_mat(w);
//         hess = {Q * hess.get_mat() * Q.t()};
//     }
//
//     auto rows = hess.get_mat().n_rows;
//
//     hess = {hess.to_hessenberg()};
// }

/*** !!! SUBTASK 2-2: QR Iteration with Francis QR Step
 **  @tparam T type of the Armadillo matrix, on which we are performing operations
 **  @param m input matrix
 **  @param maxiter maximum iteration steps
 **  @return quasi-upper-triangular matrix
 **/
template <typename T>
Col<typename T::elem_type> iteration_with_shift(const nla_mat<T> &m, uint maxiter = 1000) {
    auto hess = m.to_hessenberg();

    for (uint i = 0; i < maxiter; ++i) {
        francis_step(hess);

        if (doesConverge(hess)) break;
    }
#ifdef DEBUG
    std::cout << "Matrix after QR iteration with implicit double shift:\n" << hess.get_mat() << std::endl;
#endif

    return {hess.get_mat().diag()};
}

// !!! NOT IMPLEMENTED
// template <typename T>
// Col<typename T::elem_type> iteration_with_deflation(const nla_mat<T> &m, uint maxiter = 1000) {
//     auto hess = m.to_hessenberg();
//     auto row = hess.get_mat().n_rows;
//
//     for (uint i = 0; i < maxiter; ++i) {
//         for (uint j = 0; j < row - 1; ++j) {
//             auto a = hess.get_mat().at(j, j);
//             auto b = hess.get_mat().at(j + 1, j);
//             givens_matrix<typename T::elem_type> g = {a, b, j, j + 1};
//             hess = g * hess * g.transpose();
//         }
//     }
//
//     return {hess.get_mat().diag()};
// }
}

#endif //QR_ITERATION_HPP
