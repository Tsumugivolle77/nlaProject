//
// Created by Tsumugi on 31.01.25.
//

#include "qr_iteration_for_tridiag.hpp"

namespace nebula::details {
    bool nearZero(const tridiag_matrix &hess, int i, double tol)
    { return std::abs(hess(i, i - 1)) < tol * (std::abs(hess(i - 1, i - 1)) + std::abs(hess(i, i))); }
}

namespace nebula::qr {
    void step_with_wilkinson_shift(tridiag_matrix &tridiag, const double &shift) {
        std::queue<double> bulge_low{};
        std::queue<double> bulge_upp{};

        auto size = tridiag.size();

        {
            auto a = tridiag(0, 0) - shift;
            auto b = tridiag(1, 0);

            givens_matrix<double> g{a, b, 0, 1};
            auto gt = g.transpose();
            std::vector<uint> applied_to = {0, 1, 2};

            for (auto &col: applied_to) {
                double c = g.c, s = g.s;
                uint j = g.j, k = g.k;
                double aj = tridiag.at(j, col), ak = tridiag.at(k, col);

                if (col != j + 2) {
                    tridiag(j, col) = c * aj + s * ak;
                    tridiag(k, col) = -s * aj + c * ak;
                } else {
                    bulge_upp.push(c * aj + s * ak);
                    tridiag(k, col) = -s * aj + c * ak;
                }
            }

            for (auto &row: applied_to) {
                double c = gt.c, s = gt.s;
                uint j = gt.j, k = gt.k;
                double aj = tridiag.at(row, j), ak = tridiag.at(row, k);

                if (row != j + 2) {
                    tridiag(row, j)  = c * aj - s * ak;
                    tridiag(row, k) = s * aj + c * ak;
                } else {
                    bulge_low.push(c * aj - s * ak);
                    tridiag(row, k) = s * aj + c * ak;
                }
            }
        }


        for (uint j = 1; j < size - 1; ++j) {
            auto a = tridiag.at(j, j - 1);
            auto b = bulge_low.front();
            givens_matrix<double> g{a, b, j, j + 1};
            auto gt = g.transpose();
            std::vector<uint> applied_to = {};

            if (j < size - 2) applied_to = {j - 1, j, j + 1, j + 2};
            else applied_to = {j - 1, j, j + 1};

            for (auto &col: applied_to) {
                double c = g.c, s = g.s;
                uint _j = g.j, k = g.k;

                if (col == j - 1) {
                    double aj = tridiag.at(_j, col), ak = bulge_low.front();
                    bulge_low.pop();
                    tridiag(_j, col) = c * aj + s * ak;
                } else if (col != j + 2) {
                    double aj = tridiag.at(_j, col), ak = tridiag.at(k, col);
                    tridiag(_j, col) = c * aj + s * ak;
                    tridiag(k, col) = -s * aj + c * ak;
                } else {
                    double aj = tridiag.at(_j, col), ak = tridiag.at(k, col);
                    bulge_upp.push(c * aj + s * ak);
                    tridiag(k, col) = -s * aj + c * ak;
                }
            }

            for (auto &row: applied_to) {
                double c = gt.c, s = gt.s;
                uint _j = gt.j, k = gt.k;

                if (row == j - 1) {
                    double aj = tridiag.at(row, _j), ak = bulge_upp.front();
                    bulge_upp.pop();
                    tridiag(row, _j) = c * aj - s * ak;
                } else if (row != j + 2) {
                    double aj = tridiag.at(row, _j), ak = tridiag.at(row, k);
                    tridiag(row, _j) = c * aj - s * ak;
                    tridiag(row, k) = s * aj + c * ak;
                } else {
                    double aj = tridiag.at(row, _j), ak = tridiag.at(row, k);
                    bulge_low.push(c * aj - s * ak);
                    tridiag(row, k) = s * aj + c * ak;
                }
            }
        }
    }

    std::vector<double> iteration_with_deflation_for_specialized_tridiag(const tridiag_matrix &m, double tol) {
        std::vector<double> eigs = {};
        std::queue<tridiag_matrix> submats;
        submats.push(m);

        uint maxiter = m.size() * m.size();
        uint i = 0;

        while (!submats.empty() && i++ < maxiter) {
            auto &tridiag = submats.front();

            // return the eigen value directly for the 1x1 block
            if (tridiag.size() == 1) {
                eigs.emplace_back(tridiag(0, 0));
                submats.pop();
                continue;
            }

            // for 2x2 matrix we have simple formula for it
            if (tridiag.size() == 2) {
                double a = tridiag(0, 0), b = tridiag(0, 1),
                       c = tridiag(1, 0), d = tridiag(1, 1);

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

            auto cols = tridiag.size();

            {
                auto sign = [](const auto &num) { return num >= 0 ? 1 : -1; };
                auto a = tridiag(cols - 2, cols - 2);
                auto b = tridiag(cols - 2, cols - 1);
                auto d = tridiag(cols - 1, cols - 1);
                auto sigma = (a - d) / 2.;
                auto shift = d + sigma - sign(sigma) * hypot(sigma, b);
                step_with_wilkinson_shift(tridiag, shift);
            }

            std::vector<uint> deflatedIndices = {};

            // deflate the matrix
            for (uint j = cols - 1; j > 0; --j) {
                if (details::nearZero(tridiag, j, tol)) {
                    deflatedIndices.emplace_back(j);
                }
            }

            if (!deflatedIndices.empty()) {
                for (uint j = 0; j < deflatedIndices.size(); ++j) {
                    uint index = deflatedIndices[j];

                    if (j == 0) {
                        uint start = index;
                        uint last = deflatedIndices.size() != 1 ? deflatedIndices[1] : 0;

                        auto submat1 = tridiag.partition(start, cols - 1);
                        auto submat2 = tridiag.partition(last, start - 1);

                        submats.push(submat1);
                        submats.push(submat2);
                    } else if (j == deflatedIndices.size() - 1) {
                        uint start = 0;
                        uint end = index - 1;

                        auto submat = tridiag.partition(start, end);

                        submats.push(submat);
                    } else {
                        uint start = deflatedIndices[j + 1];
                        uint end = index - 1;

                        auto submat = tridiag.partition(start, end);

                        submats.push(submat);
                    }
                }
                submats.pop();
            }

            // if no deflate happens, iterate with the original matrix, don't pop
        }

        return eigs;
    }
}
