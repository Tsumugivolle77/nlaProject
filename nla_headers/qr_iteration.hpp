//
// Created by Tsumugi on 20.01.25.
//

#ifndef QR_ITERATION_HPP
#define QR_ITERATION_HPP


namespace details {
template <typename M>
using __nm_ptr = std::shared_ptr<nla_mat<M>>;

inline void __iteration_with_deflation_impl(nla_mat<mat> &, std::vector<double> &eigs, double tol);

template <typename M>
bool doesConverge(const nla_mat<M> &hess, double tol = 1e-6)
{ return norm(hess.get_mat().diag(-1), 2) < tol; }


template <typename M, typename et = typename M::elem_type>
void partition(nla_mat<M> &hess, std::vector<double> &eigs, double tol = 1e-6) {
    auto &inhalt = hess.get_mat();
    auto cols = inhalt.n_cols;

    // deflate the matrix
    for (int i = cols - 1; i > 0; --i) {
        if (std::abs(inhalt.at(i, i - 1)) < tol * (std::abs(inhalt.at(i - 1, i - 1)) + std::abs(inhalt.at(i, i)))) {
                nla_mat<M> part1 = { inhalt(span(0, i - 1), span(0, i - 1)) };
                nla_mat<M> part2 = { inhalt(span(i, cols - 1), span(i, cols - 1)) };
#ifdef DEBUG
                std::cout << "First subpart:\n"  << part1.get_mat() << std::endl;
                std::cout << "Second subpart:\n" << part2.get_mat() << std::endl;
#endif
                __iteration_with_deflation_impl(part1, eigs, tol);
                __iteration_with_deflation_impl(part2, eigs, tol);

            return;
        }
    }

    // if no deflate happens, iterate with the original matrix
    __iteration_with_deflation_impl(hess, eigs, tol);
#ifdef DEBUG
    std::cout << "No deflation." << std::endl;
#endif
}
}

namespace qr {
// function for perform a qr step
template <typename M>
void step_for_hessenberg(nla_mat<M> &hess) {
    auto row = hess.get_mat().n_rows;

    for (uint j = 0; j < row - 1; ++j) {
        auto a = hess.get_mat().at(j, j);
        auto b = hess.get_mat().at(j + 1, j);
        givens_matrix<typename M::elem_type> g {a, b, j, j + 1};
        std::cout << g.c << ' ' << g.s << std::endl;
        hess = g * hess * g.transpose();
    }
}

/*** !!! SUBTASK 2-1: QR iteration w\o deflation and shift
 **  @tparam M type of the Armadillo matrix, on which we are performing operations
 **  @param m input matrix
 **  @param maxiter maximum iteration steps
 **  @return quasi-upper-triangular matrix
***/
template <typename M>
Col<typename M::elem_type> iteration(const nla_mat<M> &m, uint maxiter = 1000) {
    auto hess = m.to_hessenberg();

    for (uint i = 0; i < maxiter; ++i) {
        step_for_hessenberg(hess);

        if (details::doesConverge(hess)) {
#ifdef DEBUG
            std::cout << "Converge after " << i << " Steps." << std::endl;
#endif
            break;
        }
    }

#ifdef DEBUG
    std::cout << "Matrix after QR iteration:\n" << hess.get_mat() << std::endl;
#endif

    return { hess.get_mat().diag() };
}

template <typename M>
void step_with_wilkinson_shift(nla_mat<M> &hess, const typename M::elem_type &shift) {
    auto row = hess.get_mat().n_rows;

    {
        auto a = hess.get_mat().at(0, 0) - shift;
        auto b = hess.get_mat().at(1, 0);
        givens_matrix<typename M::elem_type> g {a, b, 0, 1};
        hess = g * hess * g.transpose();
    }

    for (uint j = 1; j < row - 1; ++j) {
        auto a = hess.get_mat().at(j, j - 1);
        auto b = hess.get_mat().at(j + 1, j - 1);
        givens_matrix<typename M::elem_type> g {a, b, j, j + 1};
        hess = g * hess * g.transpose();
    }
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
        auto shift = a + d - sign(d) * std::hypot(d, c);

        step_with_wilkinson_shift(res, shift);

        // std::cout << "Res after " << i << " Steps:\n" << res << std::endl;

        if (details::doesConverge(res)) {
#ifdef DEBUG
            std::cout << "Converge after " << i << " Steps." << std::endl;
#endif
            break;
        }
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
template <typename M>
inline void francis_step(nla_mat<M> &hess) {
    {
        // set up the implicit shift
        using et   = typename M::elem_type;
        uint cols  = hess.get_mat().n_cols;
        Mat<et> sm = { hess.get_mat().submat(cols - 2, cols - 1, cols - 2, cols - 1) };
        et s       = trace(sm);
        et t       = det(sm);
        et h00     = hess.get_mat().at(0, 0);
        et h10     = hess.get_mat().at(1, 0);
        auto col0  = hess.get_mat().col(0);
        auto col1  = hess.get_mat().col(1);
        Col<et> w  = h00 * col0 + h10 * col1 - s * col0;
        w[0]      += t;

        auto Q = nla_mat<>::get_householder_mat(w);
        hess = { Q * hess.get_mat() * Q.ht() };
    }

    hess = { hess.to_hessenberg() };
}

/*** !!! SUBTASK 2-2: QR Iteration with Francis QR Step
 **  @tparam M type of the Armadillo matrix, on which we are performing operations
 **  @param m input matrix
 **  @param maxiter maximum iteration steps
 **  @return quasi-upper-triangular matrix
 **/
template <typename M>
Col<typename M::elem_type> iteration_with_shift(const nla_mat<M> &m, uint maxiter = 1000) {
    auto hess = m.to_hessenberg();

    for (uint i = 0; i < maxiter; ++i) {
        francis_step(hess);

        if (details::doesConverge(hess)) {
#ifdef DEBUG
            std::cout << "Converge after " << i << " Steps." << std::endl;
#endif
            break;
        }
    }
#ifdef DEBUG
    std::cout << "Matrix after QR iteration with implicit double shift:\n" << hess.get_mat() << std::endl
        << "Computed eigs:\n";
#endif

    return { hess.get_mat().diag() };
}

/*** !!! SUBTASK 2-3: QR Iteration with Deflation for Hermitian
 **  @param m complex hermitian matrix
 **  @param tol tolerance of error
 **  @return the real eigenvalues
 ***/
inline std::vector<double> iteration_with_deflation(nla_mat<cx_mat> &m, double tol = 1e-6) {
    auto tridiag = hermitian_tridiag2sym_tridiag(m.to_hessenberg());
    std::vector<double> eigs = {};

    details::__iteration_with_deflation_impl(tridiag, eigs, tol);

    return eigs;
}

/*** !!! SUBTASK 2-3: QR Iteration with Deflation for Real Symmetric
 **  @param m real symmetric matrix
 **  @param tol tolerance of error
 **  @return the real eigenvalues
 ***/
inline std::vector<double> iteration_with_deflation(nla_mat<mat> &m, double tol = 1e-6) {
    auto tridiag = m.to_hessenberg();
    std::vector<double> eigs = {};

    details::__iteration_with_deflation_impl(tridiag, eigs, tol);

    return eigs;
}
}

namespace details {
inline void __iteration_with_deflation_impl(nla_mat<mat> &tridiag, std::vector<double> &eigs, double tol) {
    // return the eigen value directly for the 1x1 block
    if (tridiag.get_mat().n_cols == 1) {
        eigs.emplace_back(tridiag.get_mat().at(0, 0));
        return;
    }

    // for 2x2 matrix we have simple formula for it
    if (tridiag.get_mat().n_cols == 2) {
        auto &h = tridiag.get_mat();
        double a = h.at(0, 0), b = h.at(0, 1),
               c = h.at(1, 0), d = h.at(1, 1);
        if (c != 0) {
            double delta = std::sqrt((a + d) * (a + d) - 4 * (a * d - b * c));
            double lambda1 = (a + d + delta) / 2., lambda2 = (a + d - delta) / 2.;
            eigs.emplace_back(lambda1);
            eigs.emplace_back(lambda2);
        } else {
            eigs.emplace_back(a);
            eigs.emplace_back(d);
        }
        return;
    }

    auto sign = [] (const auto &num) { return num >= 0 ? 1 : -1; };
    auto &h = tridiag.get_mat();
    auto cols = h.n_cols;
    auto a = h.at(cols - 1, cols - 1);
    auto b = h.at(cols - 2, cols - 2);
    auto c = h.at(cols - 1, cols - 2);
    auto d = (b - a) / 2.;
    auto shift = a + d - sign(d) * std::hypot(d, c);
    qr::step_with_wilkinson_shift(tridiag, shift);

    details::partition(tridiag, eigs, tol);
}
}

#endif //QR_ITERATION_HPP
