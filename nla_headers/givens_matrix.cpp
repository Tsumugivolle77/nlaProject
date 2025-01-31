//
// Created by Tsumugi on 31.01.25.
//

#include "givens_matrix.hpp"

namespace nebula {
    Col<std::complex<double> > operator*(const givens_matrix<std::complex<double> > &g,
                                         const Col<std::complex<double> > &v) {
        auto res = v;
        uint j = g.j, k = g.k;
        std::complex<double> c = g.c, s = g.s;

        res[j] = c * v[j] + s * v[k];
        res[k] = -std::conj(s) * v[j] + c * v[k];

        return res;
    }

    Col<double> operator*(const givens_matrix<double> &g, const Col<double> &v) {
        auto res = v;
        uint j = g.j, k = g.k;
        double c = g.c, s = g.s;

        res[j] = c * v[j] + s * v[k];
        res[k] = -s * v[j] + c * v[k];

        return res;
    }


    Row<double> operator*(const Row<double> &v, const givens_matrix<double> &g) {
        auto res = v;
        uint j = g.j, k = g.k;
        double c = g.c, s = g.s;

        res[j] = c * v[j] - s * v[k];
        res[k] = s * v[j] + c * v[k];

        return res;
    }
}
