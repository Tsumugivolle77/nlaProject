#include <iostream>
#include <armadillo>
#include <complex>
#include <utility>

// To enable printing additional information:
// #define DEBUG
// or
// `add_compile_option(-DDEBUG)` in CMakeLists.txt

template <typename Mat = arma::cx_mat>
class nla_mat
{
private:
    Mat mat;
    using elem_type = typename Mat::elem_type;
    using list      = std::initializer_list<elem_type>;
    using lists     = std::initializer_list<list>;

    static Mat get_householder_mat(arma::Col<elem_type> x) {
        using namespace arma;
        using namespace std::complex_literals;

        auto x1 = x[0];
        auto phase = std::arg(x1);
        auto e1 = cx_colvec(x.n_rows);
        e1[0] = 1.;
        const auto I = cx_mat(x.n_rows, x.n_rows, fill::eye);

        const cx_vec w = x + std::exp(1i * phase) * norm(x) * e1;
        const cx_rowvec wh {w.ht()};

        return I - 2 * w * wh / dot(wh, w);
    }

public:
    explicit nla_mat(Mat &&m): mat(m) { }

    nla_mat(lists &&li): mat(li) { }

    Mat       &get_mat()       { return mat; }
    const Mat &get_mat() const { return mat; }

    // get the Hessenberg form of mat
    nla_mat to_hessenberg() {
        using namespace arma;
        auto hess = mat;

        for (int i = 0; i < hess.n_cols - 1; ++i) {
            auto x = cx_colvec(hess.submat(i + 1, i, hess.n_rows -1 , i));
            auto H = get_householder_mat(x);
            cx_mat Qi = cx_mat(hess.n_rows, hess.n_cols, fill::eye);

            Qi(span(i + 1, Qi.n_rows - 1), span(i + 1, Qi.n_cols - 1)) = H;

            hess = Qi * hess;

#ifdef DEBUG
            std::cout << "apply householder left\n" << mat << std::endl;
#endif

            hess = hess * Qi.ht();

#ifdef DEBUG
            std::cout << "apply householder right\n" << mat << std::endl;
#endif
        }

        return hess;
    }

};



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

int main()
{
    using namespace std::complex_literals;
    using namespace arma;
    nla_mat A = {
        {arma::cx_double(4, 0), arma::cx_double(1, 2), arma::cx_double(3, -1), arma::cx_double(0, 1), arma::cx_double(2, -3)},
        {arma::cx_double(1, -2), arma::cx_double(5, 0), arma::cx_double(1, 4), arma::cx_double(2, 1), arma::cx_double(0, -1)},
        {arma::cx_double(3, 1), arma::cx_double(1, -4), arma::cx_double(6, 0), arma::cx_double(4, 2), arma::cx_double(1, 1)},
        {arma::cx_double(0, -1), arma::cx_double(2, -1), arma::cx_double(4, -2), arma::cx_double(3, 0), arma::cx_double(5, -2)},
        {arma::cx_double(2, 3), arma::cx_double(0, 1), arma::cx_double(1, -1), arma::cx_double(5, 2), arma::cx_double(8, 0)}
    };

    // A = QR_iteration(A);
    std::cout << A.to_hessenberg() << std::endl;

    return 0;
}
