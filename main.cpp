#include <iostream>
#include <armadillo>
#include <complex>
#include <utility>

// To enable printing additional information:
// #define DEBUG
// or
// `add_compile_option(-DDEBUG)` in CMakeLists.txt

template<typename Mat = arma::cx_mat>
class nla_mat {
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
        auto e1 = cx_colvec(x.n_rows);
        e1[0] = 1.;
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
nla_mat<Mat>::nla_mat(Mat &&m): matx(m) {
}

template<typename Mat>
nla_mat<Mat>::nla_mat(lists &&li): matx(li) {
}

template<typename Mat>
Mat &nla_mat<Mat>::get_mat() { return matx; }

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


// arma::cx_mat QR_iteration(arma::cx_mat A, uint maxiter = 1000) {
//     using namespace arma;
//
//     for (uint i = 0; i < maxiter; ++i) {
//         auto [Q, R] = QR_decomp(A);
//         A = Q.ht() * A * Q;
//     }
//
//     return A;
// }

int main() {
    using namespace std::complex_literals;
    using namespace arma;
    nla_mat A = {
        {cx_double(4, 0), cx_double(1, 2), cx_double(3, -1), cx_double(0, 1), cx_double(2, -3)},
        {cx_double(1, -2), cx_double(5, 0), cx_double(1, 4), cx_double(2, 1), cx_double(0, -1)},
        {cx_double(3, 1), cx_double(1, -4), cx_double(6, 0), cx_double(4, 2), cx_double(1, 1)},
        {cx_double(0, -1), cx_double(2, -1), cx_double(4, -2), cx_double(3, 0), cx_double(5, -2)},
        {cx_double(2, 3), cx_double(0, 1), cx_double(1, -1), cx_double(5, 2), cx_double(8, 0)}
    };

    nla_mat B = mat{
        {1, 1, 4, 5},
        {1, 4, 1, 9,},
        {4, 1, 9, 8,},
        {5, 9, 8, 0}
    };

    std::cout << A.to_hessenberg() << B.to_hessenberg() << std::endl;

    return 0;
}
