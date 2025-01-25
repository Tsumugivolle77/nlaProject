//
// Created by Tsumugi on 13.01.25.
//

#ifndef NLA_MAT_H
#define NLA_MAT_H

/***  This class is designed for
 **  @tparam M Type of matrix this object uses
 **/
template<typename M = cx_mat>
class nla_mat
{
public:
    using elem_type = typename M::elem_type;
    using list      = std::initializer_list<elem_type>;
    using lists     = std::initializer_list<list>;

    nla_mat(M &&m);
    nla_mat(lists &&li);

    M       &get_mat();
    const M &get_mat() const;
    nla_mat to_hessenberg() const;

    // for complex vector
    static Mat<std::complex<double>>
    get_householder_mat(const Col<std::complex<double>> &x) {
        using namespace std::complex_literals;

        auto x1 = x[0];
        auto phase = std::arg(x1);
        auto e1      = cx_colvec(x.n_rows);
        e1[0]        = 1.;
        const auto I = cx_mat(x.n_rows, x.n_rows, fill::eye);

        const cx_vec w = x + std::exp(1i * phase) * norm(x) * e1;
        const cx_rowvec wh{w.ht()};

        return I - 2 * w * wh / dot(wh, w);
    }

    // for real vector
    static Mat<double>
    get_householder_mat(const Col<double> &x) {
        using namespace std::complex_literals;

        auto x1 = x[0];
        auto e1 = colvec(x.n_rows);
        e1[0] = 1.;
        const auto I = mat(x.n_rows, x.n_rows, fill::eye);

        auto sgn = [](auto v) -> auto { return v >= 0 ? 1 : -1; };
        const vec w = x + sgn(x1) * norm(x) * e1;
        const rowvec wh{w.t()};

        return I - 2 * w * wh / dot(wh, w);
    }

private:
    M matx;
};

// printing nla_mat
template <typename M>
std::ostream &operator<<(std::ostream &os, const nla_mat<M> &mat) {
    return os << mat.get_mat();
}

template<typename M>
nla_mat<M>::nla_mat(M &&m): matx(m) { }

template<typename M>
nla_mat<M>::nla_mat(lists &&li): matx(li) { }

template<typename M>
M       &nla_mat<M>::get_mat() { return matx; }

template<typename M>
const M &nla_mat<M>::get_mat() const { return matx; }

// get the Hessenberg form of matx
template<typename M>
nla_mat<M> nla_mat<M>::to_hessenberg() const {
    if (matx.n_cols != matx.n_rows) { throw std::runtime_error("nla_mat: not a square matrix"); }

    auto hess = matx;

    for (int i = 0; i < hess.n_cols - 2; ++i) {
        auto x = Col<elem_type>(hess.submat(i + 1, i, hess.n_rows - 1, i));
        auto H = get_householder_mat(x);
        auto Qi = Mat<elem_type>(hess.n_rows, hess.n_cols, fill::eye);

        Qi(span(i + 1, Qi.n_rows - 1), span(i + 1, Qi.n_cols - 1)) = H;

        hess = Qi * hess;
        hess = hess * Qi.ht();
    }

    for (int i = 2; i < hess.n_rows; ++i) {
        for (int j = 0; j < i - 1; ++j) {
            hess.at(i, j) = {0.};
        }
    }

    return hess;
}

/*** !!! FOR THE FIRST SUBTASK: converting Hermitian Tridiagonal resulting
 **  from `A.to_hessenberg()` into Real Symmetric Tridiagonal
 **  @param A Hermitian Tridiagonal
 **  @return Real Symmetric Tridiagonal, similar to A
***/
nla_mat<mat>
inline hermitian_tridiag2sym_tridiag(const nla_mat<> &A)
{
    using namespace std::complex_literals;

    auto hermitri = A.get_mat();

    for (int i = 2; i < hermitri.n_cols; ++i) {
        for (int j = 0; j < i - 1; ++j) {
            hermitri.at(j, i) = 0.;
        }
    }

    auto rows = hermitri.n_rows;

    auto diag_entries = cx_colvec(rows);
    diag_entries[0] = 1.;
    auto totphase = 0.;

    for (int i = 1; i < rows; ++i) {
        auto subdiag_entry = hermitri.at(i, i - 1);
        auto phase = -std::arg(subdiag_entry);
        totphase += phase;
        diag_entries[i] = std::exp(1i * totphase);
    }

    auto D = diagmat(diag_entries);

    return { real(D * hermitri * D.ht()) };
}

#endif //NLA_MAT_H
