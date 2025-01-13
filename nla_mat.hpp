//
// Created by Tsumugi on 13.01.25.
//

#ifndef NLA_MAT_H
#define NLA_MAT_H

/**  This class is designed for
 **  @tparam Mat Type of matrix this object uses
 **/
template<typename Mat = arma::cx_mat>
class nla_mat
{
public:
    using elem_type = typename Mat::elem_type;
    using list      = std::initializer_list<elem_type>;
    using lists     = std::initializer_list<list>;

    nla_mat(Mat &&m);
    nla_mat(lists &&li);

    Mat       &get_mat();
    const Mat &get_mat() const;
    nla_mat to_hessenberg();

#ifndef DEBUG
private:
#endif
    Mat matx;

    // for complex vector
    template<typename ET>
    static std::enable_if_t<std::is_same_v<ET, std::complex<double> >, arma::Mat<ET> >
    get_householder_mat(arma::Col<ET> x) {
        using namespace arma;
        using namespace std::complex_literals;

        auto x1 = x[0];
        auto phase = std::arg(x1);
        auto e1      = cx_colvec(x.n_rows);
        e1[0]        = 1.;
        const auto I = cx_mat(x.n_rows, x.n_rows, fill::eye);

        const cx_vec w = x + std::exp(1i * phase) * norm(x) * e1;
        // const cx_vec w = x + norm(x) * e1;
        const cx_rowvec wh{w.ht()};

        return I - 2 * w * wh / dot(wh, w);
    }

    // for real vector
    template<typename ET>
    static std::enable_if_t<!std::is_same_v<ET, std::complex<double> >, arma::Mat<ET> >
    get_householder_mat(arma::Col<ET> x) {
        using namespace arma;
        using namespace std::complex_literals;

        auto x1 = x[0];
        auto e1 = colvec(x.n_rows);
        e1[0] = 1.;
        const auto I = mat(x.n_rows, x.n_rows, fill::eye);

        auto sgn = [](auto v) -> auto { return v >= 0 ? 1 : -1; };
        const vec w = x + sgn(x1) * norm(x) * e1;
        // const cx_vec w = x + norm(x) * e1;
        const rowvec wh{w.t()};

        return I - 2 * w * wh / dot(wh, w);
    }
};

// printing nla_mat
template <typename Mat>
std::ostream &operator<<(std::ostream &os, const nla_mat<Mat> &mat) {
    return os << mat.get_mat();
}

template<typename Mat>
nla_mat<Mat>::nla_mat(Mat &&m): matx(m) { }

template<typename Mat>
nla_mat<Mat>::nla_mat(lists &&li): matx(li) { }

template<typename Mat>
Mat       &nla_mat<Mat>::get_mat() { return matx; }

template<typename Mat>
const Mat &nla_mat<Mat>::get_mat() const { return matx; }

// get the Hessenberg form of matx
template<typename Mat>
nla_mat<Mat> nla_mat<Mat>::to_hessenberg() {
    using namespace arma;
    auto hess = matx;

    for (int i = 0; i < hess.n_cols - 2; ++i) {
        auto x = Col<elem_type>(hess.submat(i + 1, i, hess.n_rows - 1, i));
        auto H = get_householder_mat(x);
        auto Qi = Mat(hess.n_rows, hess.n_cols, fill::eye);

        Qi(span(i + 1, Qi.n_rows - 1), span(i + 1, Qi.n_cols - 1)) = H;

        hess = Qi * hess;

#ifdef DEBUG
            std::cout << "apply householder left\n" << hess << std::endl;
#endif

        hess = hess * Qi.ht();

#ifdef DEBUG
            std::cout << "apply householder right\n" << hess << std::endl;
#endif
    }

    return hess;
}

/*** !!! FOR THE FIRST SUBTASK: converting Hermitian Tridiagonal resulting
 **  from `A.to_hessenberg()` into Real Symmetric Tridiagonal
 **  @param A Hermitian Tridiagonal
 *   @return Real Symmetric Tridiagonal, similar to A
***/
nla_mat<>
inline hermitian_tridiag2sym_tridiag(const nla_mat<> &A)
{
    using namespace arma;
    using namespace std::complex_literals;

    const auto &hermitri = A.get_mat();
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

    return {D * hermitri * D.ht()};
}

#endif //NLA_MAT_H
