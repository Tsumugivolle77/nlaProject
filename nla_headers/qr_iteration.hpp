//
// Created by Tsumugi on 20.01.25.
//

#ifndef QR_ITERATION_HPP
#define QR_ITERATION_HPP

namespace qr {
template <typename M> void francis_step(M &);
}

namespace details {
template <typename M>
using __nm_ptr = std::shared_ptr<M>;

inline void __iteration_with_deflation_impl(__nm_ptr<mat> &, std::vector<double> &eigs, double tol);

inline void __general_iteration_with_deflation_impl(__nm_ptr<cx_mat> &, std::vector<std::complex<double>> &eigs, double tol);

template <typename M>
bool doesConverge(const M &hess, double tol = 1e-6)
{ return norm(hess.diag(-1), 2) < tol; }

template <typename M>
bool nearZero(__nm_ptr<M> &hess, int i, double tol) {
    return std::abs(hess->at(i, i - 1)) < tol * (std::abs(hess->at(i - 1, i - 1)) + std::abs(hess->at(i, i)));
}

// partition for real matrix
inline void partition(__nm_ptr<mat> &hess, std::vector<double> &eigs, double tol = 1e-6) {
    auto cols = hess->n_cols;

    // deflate the matrix
    for (int i = cols - 1; i > 0; --i) {
        if (nearZero(hess, i, tol)) {
            int j = i - 1;
            for (; j > 0; --j) {
                if (!nearZero(hess, j, tol)) break;
                eigs.emplace_back(hess->at(j, j));
            }
            auto part1 = std::make_shared<mat>(
                (*hess)(span(0, j), span(0, j)));
            auto part2 = std::make_shared<mat>(
                (*hess)(span(i, cols - 1), span(i, cols - 1)));
#ifdef DEBUG
            std::cout << "First subpart:\n"  << *part1 << std::endl;
            std::cout << "Second subpart:\n" << *part2 << std::endl;
#endif
            hess.reset();
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

// partition for complex matrix
inline void
partition(__nm_ptr<cx_mat> &hess, std::vector<std::complex<double>> &eigs, double tol = 1e-6) {
    auto cols = hess->n_cols;

    // For the 4x4 matrix, deflate them as two 2x2 matrices, regardless of how they look like
    if (cols == 4) {
        auto part1 = std::make_shared<cx_mat>(
            (*hess)(span(0, 1), span(0, 1)));
        auto part2 = std::make_shared<cx_mat>(
            (*hess)(span(2, cols - 1), span(2, cols - 1)));
#ifdef DEBUG
        std::cout << "Full matrix:\n" << *hess << std::endl;
        std::cout << "First subpart:\n"  << *part1 << std::endl;
        std::cout << "Second subpart:\n" << *part2 << std::endl;
#endif
        hess.reset();
        __general_iteration_with_deflation_impl(part1, eigs, tol);
        __general_iteration_with_deflation_impl(part2, eigs, tol);
        return;
    };

    // deflate the matrix
    for (int i = cols - 1; i > 0; --i) {
        if (nearZero(hess, i, tol)) {
            int j = i - 1;
            for (; j > 0; --j) {
                if (!nearZero(hess, j, tol)) break;
                eigs.emplace_back(hess->at(j, j));
            }
            auto part1 = std::make_shared<cx_mat>(
                (*hess)(span(0, j), span(0, j)));
            auto part2 = std::make_shared<cx_mat>(
                (*hess)(span(i, cols - 1), span(i, cols - 1)));
#ifdef DEBUG
            std::cout << "Full matrix:\n" << *hess << std::endl;
            std::cout << "First subpart:\n"  << *part1 << std::endl;
            std::cout << "Second subpart:\n" << *part2 << std::endl;
#endif
            hess.reset();
            __general_iteration_with_deflation_impl(part1, eigs, tol);
            __general_iteration_with_deflation_impl(part2, eigs, tol);

            return;
        }
    }

    // if no deflate happens, iterate with the original matrix
    __general_iteration_with_deflation_impl(hess, eigs, tol);
#ifdef DEBUG
    std::cout << "No deflation." << std::endl;
#endif
}
}

namespace qr {
// function for perform a qr step
template <typename M>
void step_for_hessenberg(M &hess) {
    auto rows = hess.n_rows;

    for (uint j = 0; j < rows - 1; ++j) {
        auto a = hess.at(j, j);
        auto b = hess.at(j + 1, j);
        givens_matrix<typename M::elem_type> g {a, b, j, j + 1};
        std::vector<uint> applied_to(rows - j);
        std::iota(applied_to.begin(), applied_to.end(), j);
        hess = apply_givens(g, hess, applied_to);
        hess = apply_givens(hess, g.transpose(), applied_to);
    }
}

/*** !!! SUBTASK 2-1: QR iteration w\o deflation and shift
 **  @tparam M type of the Armadillo matrix, on which we are performing operations
 **  @param m input matrix
 **  @param maxiter maximum iteration steps
 **  @return quasi-upper-triangular matrix
***/
template <typename M>
Col<typename M::elem_type> iteration(const M &m, uint maxiter = 1000) {
    auto hess = to_hessenberg(m);

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
    std::cout << "Matrix after QR iteration:\n" << hess << std::endl;
#endif

    return { hess.diag() };
}

template <typename M>
void step_with_wilkinson_shift(M &hess, const typename M::elem_type &shift) {
    auto row = hess.n_rows;

    {
        auto a = hess.at(0, 0) - shift;
        auto b = hess.at(1, 0);
        givens_matrix<typename M::elem_type> g {a, b, 0, 1};
        std::vector<uint> applied_to = {0, 1, 2};
        hess = apply_givens(g, hess, applied_to);
        hess = apply_givens(hess, g.transpose(), applied_to);
    }

    for (uint j = 1; j < row - 1; ++j) {
        auto a = hess.at(j, j - 1);
        auto b = hess.at(j + 1, j - 1);
        givens_matrix<typename M::elem_type> g {a, b, j, j + 1};
        std::vector<uint> applied_to = {};
        if (j < row - 2) applied_to = {j - 1, j, j + 1, j + 2};
        else applied_to = {j - 1, j, j + 1};
        hess = apply_givens(g, hess, applied_to);
        hess = apply_givens(hess, g.transpose(), applied_to);
    }
}

inline vec iteration_with_shift_for_real_symmetric_tridiagonal(const mat &tridiag, uint maxiter = 1000) {
    auto cols = tridiag.n_cols;
    auto res  = tridiag;

    for (uint i = 0; i < maxiter; ++i) {
        auto sign = [] (const auto &num) { return num >= 0 ? 1 : -1; };
        const auto &r = res;
        auto a = r.at(cols - 1, cols - 1);
        auto b = r.at(cols - 2, cols - 2);
        auto c = r.at(cols - 1, cols - 2);
        auto d = (b - a) / 2.;
        auto shift = a + d - sign(d) * std::hypot(d, c);

        step_with_wilkinson_shift(res, shift);

        if (details::doesConverge(res)) {
#ifdef DEBUG
            std::cout << "Converge after " << i << " Steps." << std::endl;
#endif
            break;
        }
    }

    return res.diag();
}

inline vec iteration_with_shift_for_hermitian(const cx_mat &m, uint maxiter = 1000) {
    auto hess = to_hessenberg(m);
    auto tridiag = hermitian_tridiag2sym_tridiag(hess);

    return iteration_with_shift_for_real_symmetric_tridiagonal(tridiag, maxiter);
}

inline vec iteration_with_shift_for_symmetric(const mat &m, uint maxiter = 1000) {
    auto tridiag = to_hessenberg(m);

    return iteration_with_shift_for_real_symmetric_tridiagonal(tridiag, maxiter);
}

// Francis QR Step
template <typename M>
void francis_step(M &hess) {
    {
        // set up the implicit shift
        using et   = typename M::elem_type;
        uint cols  = hess.n_cols;
        Mat<et> sm = { hess.submat(cols - 2, cols - 1, cols - 2, cols - 1) };
        et s       = trace(sm);
        et t       = det(sm);
        et h00     = hess.at(0, 0);
        et h10     = hess.at(1, 0);
        auto col0  = hess.col(0);
        auto col1  = hess.col(1);
        Col<et> w  = h00 * col0 + h10 * col1 - s * col0;
        w[0]      += t;

        auto Q = get_householder_mat(w);
        hess = { Q * hess * Q.ht() };
    }

    hess = { to_hessenberg(hess) };
}

/*** !!! SUBTASK 2-2: QR Iteration with Francis QR Step
 **  @tparam M type of the Armadillo matrix, on which we are performing operations
 **  @param m input matrix
 **  @param maxiter maximum iteration steps
 **  @return quasi-upper-triangular matrix
 **/
template <typename M>
Col<typename M::elem_type> iteration_with_shift(const M &m, uint maxiter = 1000) {
    auto hess = to_hessenberg(m);

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

    return { hess.diag() };
}

/*** !!! SUBTASK 2-3: QR Iteration with Deflation for Hermitian
 **  @param m complex hermitian matrix
 **  @param tol tolerance of error
 **  @return the real eigenvalues
 ***/
inline std::vector<double> iteration_with_deflation(cx_mat &m, double tol = 1e-6) {
    auto tridiag = std::make_shared<mat>(hermitian_tridiag2sym_tridiag(to_hessenberg(m)));
    std::vector<double> eigs = {};

    details::__iteration_with_deflation_impl(tridiag, eigs, tol);

    return eigs;
}

/*** !!! SUBTASK 2-3: QR Iteration with Deflation for Real Symmetric
 **  @param m real symmetric matrix
 **  @param tol tolerance of error
 **  @return the real eigenvalues
 ***/
inline std::vector<double> iteration_with_deflation(mat &m, double tol = 1e-6) {
    auto tridiag = std::make_shared<mat>(to_hessenberg(m));
    std::vector<double> eigs = {};

    details::__iteration_with_deflation_impl(tridiag, eigs, tol);

    return eigs;
}

/*** !!! SUBTASK 2-3: QR Iteration with Deflation for Complex Matrix
 **  @param m complex matrix
 **  @param tol tolerance of error
 **  @return the complex eigenvalues
 ***/
inline std::vector<std::complex<double>>
general_iteration_with_deflation(cx_mat &m, double tol = 1e-6, const std::function<void(cx_mat &)> &iteration_step = qr::francis_step<cx_mat>) {
    std::vector<std::complex<double>> eigs = {};
    auto cx = std::make_shared<cx_mat>(m);

    details::__general_iteration_with_deflation_impl(cx, eigs, tol);

    return eigs;
}

/*** !!! SUBTASK 2-3: QR Iteration with Deflation for Real Matrix
 **  @param m real matrix
 **  @param tol tolerance of error
 **  @param iteration_step function used for perform a QR Step
 **  @return the complex eigenvalues
 ***/
inline std::vector<std::complex<double>>
general_iteration_with_deflation(mat &m, double tol = 1e-6) {
    auto cx = std::make_shared<cx_mat>(cx_mat{ m, mat(m.n_rows, m.n_cols, fill::zeros) });
    std::vector<std::complex<double>> eigs = {};

    details::__general_iteration_with_deflation_impl(cx, eigs, tol);

    return eigs;
}
}

namespace details {
inline void __iteration_with_deflation_impl(__nm_ptr<mat> &tridiag, std::vector<double> &eigs, double tol) {
    // return the eigen value directly for the 1x1 block
    if (tridiag->n_cols == 1) {
        eigs.emplace_back(tridiag->at(0, 0));
        tridiag.reset();
        return;
    }

    // for 2x2 matrix we have simple formula for it
    if (tridiag->n_cols == 2) {
        auto &h = *tridiag;
        double a = h.at(0, 0), b = h.at(0, 1),
               c = h.at(1, 0), d = h.at(1, 1);

        if (std::abs(c) > tol * (std::abs(a) + std::abs(d))) {
            double trace = a + d;
            double determinant = a * d - b * c;
            double delta = trace * trace - 4 * determinant;

            double sqrt_delta = std::sqrt(delta);
            double lambda1 = (trace + sqrt_delta) / 2.0;
            double lambda2 = (trace - sqrt_delta) / 2.0;

            eigs.emplace_back(lambda1);
            eigs.emplace_back(lambda2);
        } else {
            eigs.emplace_back(a);
            eigs.emplace_back(d);
        }
        tridiag.reset();
        return;
    }

    {
        auto sign = [] (const auto &num) { return num >= 0 ? 1 : -1; };
        auto &h = *tridiag;
        auto cols = h.n_cols;
        auto a = h.at(cols - 1, cols - 1);
        auto b = h.at(cols - 2, cols - 2);
        auto c = h.at(cols - 1, cols - 2);
        auto d = (b - a) / 2.;
        auto shift = a + d - sign(d) * std::hypot(d, c);
        qr::step_with_wilkinson_shift(*tridiag, shift);
    }

    details::partition(tridiag, eigs, tol);
}

inline void __general_iteration_with_deflation_impl(
    __nm_ptr<cx_mat> &m,
    std::vector<std::complex<double>> &eigs,
    double tol)
{
    // return the eigen value directly for the 1x1 block
    if (m->n_cols == 1) {
        eigs.emplace_back(m->at(0, 0));
        return;
    }

    // for 2x2 matrix we have simple formula for it
    if (m->n_cols == 2) {
        auto &h = *m;
        std::complex<double> a = h.at(0, 0), b = h.at(0, 1),
                             c = h.at(1, 0), d = h.at(1, 1);
        if (std::abs(c) > tol * (std::abs(a) + std::abs(d))) {
            std::complex<double> delta = std::sqrt((a + d) * (a + d) - 4. * (a * d - b * c));
            std::complex<double> lambda1 = (a + d + delta) / 2., lambda2 = (a + d - delta) / 2.;
            eigs.emplace_back(lambda1);
            eigs.emplace_back(lambda2);
        } else {
            eigs.emplace_back(a);
            eigs.emplace_back(d);
        }
        return;
    }

    qr::francis_step(*m);

    details::partition(m, eigs, tol);
}
}

#endif //QR_ITERATION_HPP
