Project `NebuLA` by Tsumugivolle77 Copyright (C) reserved.

I name this project after **N**ebu**LA** instead of **NLA** (Numerical Linear Algebra) since it looks cool.

README file for my project and also for the final document submission.

Github repo: https://github.com/Tsumugivolle77/nlaProject

It will be temporarily invisible until submission deadline comes.

# Prerequisites for compiling project successfully
User should set `-std=c++17` since I used some `c++17` features such as CTAD, which allows the compiler deduce template arguments from the constructor arguments. It would make my code look better.

If it's OK, I would also want to use some features from `c++20` or `c++23` like `concepts` and `std::fmt`.

To enable printing additional information or make the code more debug-friendly:

`#define DEBUG` at the beginning of the `main.cpp`

or

`add_compile_option(-DDEBUG)` in `CMakeLists.txt`

or 

`g++ <blabla> -DDEBUG` in your terminal, etc.

# Overview of the Project Structure
The structure of the project looks like:
```
.
└── nlaProject/
    ├── nla_headers/
    │   ├── nla_mat.hpp
    │   ├── givens_matrix.hpp
    │   └── README.md (this file)
    ├── main.cpp
    └── nebula.hpp
```
All files of `./nlaProject/nla_headers/` are included in `nla.hpp` in `namespace nebula`.

I write test codes in `main.cpp` and implement the various functions for doing tridiagonalization and QR iteration with shift and deflation in the other headers or sources.

# `nla_mat.hpp`
## class `nla_mat`
So I wrote a class to make my codes look better (more professional :D).
```cpp
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
    nla_mat to_hessenberg() const;

#ifndef DEBUG
private:
#endif
    Mat matx;

    // for complex vector
    static arma::Mat<std::complex<double>>
    get_householder_mat(const arma::Col<std::complex<double>> &x) {
        using namespace arma;
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
    static arma::Mat<double>
    get_householder_mat(const arma::Col<double> &x) {
        using namespace arma;
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
};
```
At the beginning of the class I defined alias for types and declared some member functions including constructors and so on. The implementation would be introduced above.

Then I introduced a static member function `get_householder_mat` to compute the Householder Matrix from a given vector `x`. It has two overloads. For the complex vector it will call the first overload, otherwise the second will be called.

I used `std::enable_if_t` for deciding which version to be called, which is based upon `SFINAE` in C++. But factually in this case, we can simply use function overload (they are equivalent this time). It anyway serves as a watermark for my own version :P.

### Utility and member functions
This part is trivial and has nothing notable to talk about. We can just skip to the next part.
#### `operator<<()`
This operator overload is for applying like `std::cout` or `fout` on our `nla_mat` object.
```cpp
// printing nla_mat
template <typename Mat>
std::ostream &operator<<(std::ostream &os, const nla_mat<Mat> &mat) {
    return os << mat.get_mat();
}
```

#### constructors
```cpp
template<typename Mat>
nla_mat<Mat>::nla_mat(Mat &&m): matx(m) { }

template<typename Mat>
nla_mat<Mat>::nla_mat(lists &&li): matx(li) { }
```

### SUBTASK ONE: Hermitian to Symmetric Tridiagonal
One of the most important and interesting property of Hermitian Matrix is that all its eigenvalues are real.

We can also derive from this another statement: Hermitian Matrix is similar to a Symmetric Tridiagonal Matrix by using some Unitary Transform.

Transforming into Hermitian Tridiagonal Matrix is straightforward by applying Householder. However, imaginary parts of the sub- and superdiagonal entries still remain.
```cpp
// get the Hessenberg form of matx
template<typename Mat>
nla_mat<Mat> nla_mat<Mat>::to_hessenberg() const {
    using namespace arma;
    auto hess = matx;
    
    if (matx.n_cols != matx.n_rows) { throw std::runtime_error("nla_mat: not a square matrix"); }

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
```

In order to eliminate these imaginary part we need the help of another Diagonal Unitary Matrix. In the following code, you will see how the Diagonal Unitary Matrix is constructed.

```cpp
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
```

# `givens_matrix.hpp`
## class `givens_matrix`: Our class for performing Givens Rotation
Applying the whole Givens matrix on a dense matrix is costly, since it only affects very limited rows and columns of the other dense matrix, and only has 4 "interesting" entries.

To improve the time and space complexity, we only store the affected rows and the sine and cosine value of the rotation. Moreover, I rewrite the `operator*` to apply our Givens Rotation from both left and right.

The class looks like:
```cpp
template <typename T>
class givens_matrix {
public:
    uint j, k;
    T c, s;
    
    givens_matrix() = default;
    givens_matrix(uint j, uint k, T c, T s);
    givens_matrix(T a, T b, uint j, uint k);
    [[nodiscard]] givens_matrix transpose() const;
};
```

## implementation of member functions
Constructors and the function for `transpose()` is implemented here.
```cpp
template <typename T>
inline givens_matrix<T>::givens_matrix(uint j, uint k, T c, T s): j(j), k(k), c(c), s(s)
{ }

template <>
inline givens_matrix<double>::givens_matrix(double a, double b, uint j, uint k)
: j(j), k(k), c(1. / std::sqrt(1. + b * b / a / a)), s(1. / std::sqrt(a * a / b / b + 1.))
{ }

template <>
inline givens_matrix<std::complex<double>>::givens_matrix(std::complex<double> a, std::complex<double> b, uint j, uint k)
: j(j), k(k)
{
    using namespace std::complex_literals;

    auto alph = std::arg(a);
    auto beta = std::arg(b);
    auto phi = beta - alph;
    auto anorm = std::abs(a);
    auto bnorm = std::abs(b);
    auto r = std::sqrt(anorm * anorm + bnorm * bnorm);
    c = anorm / r;
    s = a / anorm * conj(b) / r;
}

template <typename T>
givens_matrix<T>
inline givens_matrix<T>::transpose() const
{ return {j, k, c, -s}; }
```

## `operator*()`
I overload the `operator*()` to perform Givens Rotation both for complex and real matrix, and from left and right. It's much less costly since I use references as the parameter and perform changes on the affected entries.
```cpp
inline arma::Col<std::complex<double>> operator*(const givens_matrix<std::complex<double>> &g, const arma::Col<std::complex<double>> &v) {
    auto res = v;
    uint j = g.j, k = g.k;
    std::complex<double> c = g.c, s = g.s;

    res[j] = c * v[j] + s * v[k];
    res[k] = -std::conj(s) * v[j] + c * v[k];

    return res;
}

inline arma::Col<double> operator*(const givens_matrix<double> &g, const arma::Col<double> &v) {
    auto res = v;
    uint j = g.j, k = g.k;
    double c = g.c, s = g.s;

    res[j] = c * v[j] + s * v[k];
    res[k] = -s * v[j] + c * v[k];

    return res;
}


inline arma::Row<double> operator*(const arma::Row<double> &v, const givens_matrix<double> &g) {
    auto res = v;
    uint j = g.j, k = g.k;
    double c = g.c, s = g.s;

    res[j] = c * v[j] - s * v[k];
    res[k] = s * v[j] + c * v[k];

    return res;
}

template <typename T>
nla_mat<arma::Mat<T>> operator*(const givens_matrix<T> &g, const nla_mat<arma::Mat<T>> &m) {
    using namespace arma;

    auto res = m.get_mat();
    uint cols = res.n_cols;

    for (uint i = 0; i < cols; ++i) {
        res.col(i) = g * res.col(i);
    }

    return res;
}

template <typename T, typename U = typename T::elem_type>
nla_mat<T> operator*(const nla_mat<T> &m, const givens_matrix<U> &g) {
    using namespace arma;

    auto res = m.get_mat();
    uint rows = res.n_rows;

    for (uint i = 0; i < rows; ++i) {
        res.row(i) = res.row(i) * g;
    }

    return res;
}
```

# `qr_iteration.hpp`
## `namespace details`
## `namespace qr`
Different types of QR iterations are given in `namespace qr`: the most fundamental one, one with shifts and one with both shifts and deflations.

### function `qr::iteration`
This function implements the very basic requirements for computing the eigenvalues of a matrix.

It first turns the input matrix `m` into a Hessenberg one by `nla_mat<T>::to_hessenberg`, and then perform QR steps with Givens Rotation on it.
```cpp
/*** !!! SUBTASK 2: QR iteration w\o deflation and shift
 **  @tparam T type of the Armadillo matrix, on which we are performing operations
 **  @param m input matrix
 **  @param maxiter maximum iteration steps
 **  @return quasi-upper-triangular matrix
***/
template <typename T>
auto qr_iteration(const nla_mat<T> &m, uint maxiter = 1000) {
    auto hess = m.to_hessenberg();
    auto row = hess.get_mat().n_rows;

    for (uint i = 0; i < maxiter; ++i) {
        for (uint j = 0; j < row - 1; ++j) {
            auto a = hess.get_mat().at(j, j);
            auto b = hess.get_mat().at(j + 1, j);
            givens_matrix<typename T::elem_type> g = {a, b, j, j + 1};
            hess = g * hess * g.transpose();
        }
    }

    return hess;
}
```

### function `qr::iteration_with_shift`

### function `qr::iteration_with_deflation`