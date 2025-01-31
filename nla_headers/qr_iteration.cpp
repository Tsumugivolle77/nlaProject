//
// Created by Tsumugi on 31.01.25.
//

#include "qr_iteration.hpp"
#include "utils.hpp"

using namespace arma;

namespace nebula {
    namespace details {
        // partition for real matrix
        void partition(std::shared_ptr<mat> &hess, std::vector<double> &eigs, double tol) {
            auto cols = hess->n_cols;

            // deflate the matrix
            for (int i = cols - 1; i > 0; --i) {
                if (nearZero(hess, i, tol)) {
                    auto part1 = std::make_shared<mat>(
                        (*hess)(span(0, i - 1), span(0, i - 1)));
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

        void partition(std::shared_ptr<mat> &hess, std::vector<std::complex<double>> &eigs, double tol) {
            auto cols = hess->n_cols;

            // deflate the matrix
            for (int i = cols - 1; i > 0; --i) {
                if (nearZero(hess, i, tol)) {
                    auto part1 = std::make_shared<mat>(
                        (*hess)(span(0, i - 1), span(0, i - 1)));
                    auto part2 = std::make_shared<mat>(
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

#ifdef DEBUG
            hess->print("No deflation:");
#endif
            // if no deflate happens, iterate with the original matrix
            __general_iteration_with_deflation_impl(hess, eigs, tol);
        }
    }

    namespace qr {
        vec iteration_with_shift_for_real_symmetric_tridiagonal(const mat &tridiag, uint maxiter) {
            auto cols = tridiag.n_cols;
            auto res  = tridiag;

            for (uint i = 0; i < maxiter; ++i) {
                auto sign = [] (const auto &num) { return num >= 0 ? 1 : -1; };
                auto a = res.at(cols - 1, cols - 1);
                auto b = res.at(cols - 2, cols - 2);
                auto c = res.at(cols - 1, cols - 2);
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

        vec iteration_with_shift_for_hermitian(const cx_mat &m, uint maxiter) {
            auto hess = to_hessenberg(m);
            auto tridiag = hermitian_tridiag2sym_tridiag(hess);

            return iteration_with_shift_for_real_symmetric_tridiagonal(tridiag, maxiter);
        }

        vec iteration_with_shift_for_symmetric(const mat &m, uint maxiter) {
            auto tridiag = to_hessenberg(m);

            return iteration_with_shift_for_real_symmetric_tridiagonal(tridiag, maxiter);
        }

        void francis_step(mat &hess) {
            {
                // set up the implicit shift
                uint cols  = hess.n_cols;
                double a   = hess.at(cols - 2, cols - 2);
                double b   = hess.at(cols - 1, cols - 2);
                double c   = hess.at(cols - 2, cols - 1);
                double d   = hess.at(cols - 1, cols - 1);
                double s   = a + d;
                double t   = a * d - b * c;
                double h00 = hess.at(0, 0);
                double h10 = hess.at(1, 0);
                auto col0  = hess.col(0);
                auto col1  = hess.col(1);
                colvec w   = h00 * col0 + h10 * col1 - s * col0;
                w[0]      += t;

                auto Q = get_householder_mat(w);
                hess = { Q * hess * Q.ht() };
            }

            hess = { to_hessenberg(hess) };
        }

        std::vector<double> iteration_with_deflation(cx_mat &m, double tol) {
            auto tridiag = std::make_shared<mat>(hermitian_tridiag2sym_tridiag(to_hessenberg(m)));
            std::vector<double> eigs = {};

            details::__iteration_with_deflation_impl(tridiag, eigs, tol);

            return eigs;
        }

        std::vector<double> iteration_with_deflation(mat &m, double tol) {
            auto tridiag = std::make_shared<mat>(to_hessenberg(m));
            std::vector<double> eigs = {};

            details::__iteration_with_deflation_impl(tridiag, eigs, tol);

            return eigs;
        }

        std::vector<double> iteration_with_deflation_for_tridiag(mat &m, double tol) {
            auto tridiag = std::make_shared<mat>(m);
            std::vector<double> eigs = {};

            details::__iteration_with_deflation_impl(tridiag, eigs, tol);

            return eigs;
        }

        std::vector<double> iteration_with_deflation_for_tridiag_using_BFS(mat &m, double tol) {
            std::vector<double> eigs = {};
            std::queue<std::shared_ptr<mat>> submats;
            submats.push(std::make_shared<mat>(m));

            uint maxiter = m.n_rows * m.n_cols;
            uint i = 0;

            while (!submats.empty() && i++ < maxiter) {
                auto &tridiag = submats.front();

                // return the eigen value directly for the 1x1 block
                if (tridiag->n_cols == 1) {
                    eigs.emplace_back(tridiag->at(0, 0));
                    submats.pop();
                    continue;
                }

                // for 2x2 matrix we have simple formula for it
                if (tridiag->n_cols == 2) {
                    double a = tridiag->at(0, 0), b = tridiag->at(0, 1),
                           c = tridiag->at(1, 0), d = tridiag->at(1, 1);

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
                    submats.pop();
                    continue;
                }

                {
                    auto sign = [] (const auto &num) { return num >= 0 ? 1 : -1; };
                    auto cols = tridiag->n_cols;
                    auto a = tridiag->at(cols - 2, cols - 2);
                    auto b = tridiag->at(cols - 2, cols - 1);
                    auto d = tridiag->at(cols - 1, cols - 1);
                    auto sigma = (a - d) / 2.;
                    auto shift = d + sigma - sign(sigma) * hypot(sigma, b);
                    step_with_wilkinson_shift(*tridiag, shift);
                }

                auto cols = tridiag->n_cols;

                std::vector<uint> deflatedIndices = {};

                // deflate the matrix
                for (int j = cols - 1; j > 0; --j) {
                    if (details::nearZero(tridiag, j, tol)) {
                        deflatedIndices.emplace_back(j);
                    }
                }

                if (!deflatedIndices.empty()) {
                    for (uint j = 0; j < deflatedIndices.size(); ++j) {
                        uint index = deflatedIndices[j];

                        auto copy_tridiag = [&] (std::shared_ptr<mat> &to, const std::shared_ptr<mat> &from, int row_start, int row_end)
                        {
                            for (uint row = row_start; row <= row_end; ++row) {
                                uint col_start = row == row_start ? row : row - 1;
                                uint col_end   = row == row_end   ? row : row + 1;
                                for (uint col = col_start; col <= col_end; ++col) {
                                    to->at(row - row_start, col - row_start) = from->at(row, col);
                                }
                            }
                        };

                        if (j == 0) {
                            uint start = index;
                            uint last = deflatedIndices.size() != 1 ? deflatedIndices[1] : 0;

                            auto submat1 = std::make_shared<mat>(cols - start, cols - start, fill::zeros);
                            copy_tridiag(submat1, tridiag, start, cols - 1);
                            auto submat2 = std::make_shared<mat>(start - last, start - last, fill::zeros);
                            copy_tridiag(submat2, tridiag, last, start - 1);

                            submats.push(submat1);
                            submats.push(submat2);
                        } else if (j == deflatedIndices.size() - 1) {
                            uint start = 0;
                            uint end = index - 1;

                            auto submat = std::make_shared<mat>(end - start + 1, end - start + 1, fill::zeros);
                            copy_tridiag(submat, tridiag, start, end);
                            submats.push(submat);
                        } else {
                            uint start = deflatedIndices[j + 1];
                            uint end   = index - 1;

                            auto submat = std::make_shared<mat>(end - start + 1, end - start + 1, fill::zeros);
                            copy_tridiag(submat, tridiag, start, end);
                            submats.push(submat);
                        }
                    }
                    submats.pop();
                }

                // if no deflate happens, iterate with the original matrix, don't pop
            }

            return eigs;
        }

        std::vector<std::complex<double>>
        general_iteration_with_deflation(mat &m, double tol) {
            auto mp = std::make_shared<mat>(m);
            std::vector<std::complex<double>> eigs = {};

            details::__general_iteration_with_deflation_impl(mp, eigs, tol);

            return eigs;
        }
    }

    namespace details {
        void __iteration_with_deflation_impl(std::shared_ptr<mat> &tridiag, std::vector<double> &eigs, double tol) {
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

        void __general_iteration_with_deflation_impl(
            std::shared_ptr<mat> &m,
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
}