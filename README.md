# Forewords

Project `NebuLA` by Tsumugivolle77 (My GitHub Username) Copyright (C) Reserved.

I name this project after **N**ebu**LA** instead of **NLA** (Numerical Linear Algebra) since it looks cool.

README file for my project and also for the final document submission.

GitHub repo: [https://github.com/Tsumugivolle77/nlaProject](https://github.com/Tsumugivolle77/nlaProject)

It will be temporarily invisible until submission deadline comes.

Since the document is written as Markdown, and rendered as PDF afterward, some of the code blocks cannot be fully viewed.

For displaying the complete code block when it causes hardship for reading, I also will attach the original Markdown file `README.md`.

## Prerequisites for compiling project successfully

Install Armadillo(14.0.3) before you start and configure it properly.

Armadillo Web Page: [https://arma.sourceforge.net](https://arma.sourceforge.net)

User should set `-std=c++17` since I used some `c++17` features such as CTAD, which allows the compiler deduce template arguments from the constructor arguments. It would make my code look better.

If it's OK, I would also want to use some features from `c++20` or `c++23` like `concepts` and `std::format`.

To enable printing additional information or make the code more debug-friendly:

`#define DEBUG` at the beginning of the `main.cpp`

or

`add_compile_option(-DDEBUG)` in `CMakeLists.txt`

or

`g++/clang/.. <blabla> -DDEBUG` in your terminal, etc.

## Overview of the Project Structure

Initially, the structure of the project looks like:

```
.
└── nlaProject/
    ├── nla_headers/
    │   ├── utils.hpp
    │   ├── givens_matrix.hpp
    │   ├── qr_iteration.hpp
    │   └── README.md (this file)
    ├── main.cpp
    └── nebula.hpp
```

To solve some specific performance issues in chapter 3, and make the project more readable, I:

- introduced new classes and functions to refactor my code;
- separate non-template declarations from definitions in .cpp.

The new structure looks like:

```
.
└── nlaProject/
    ├── nla_headers/
    │   ├── utils.hpp
    │   ├── utils.cpp
    │   ├── tridiag_matrix.hpp
    │   ├── tridiag_matrix.cpp
    │   ├── nla_headers/qr_iteration_for_tridiag.hpp
    │   ├── nla_headers/qr_iteration_for_tridiag.cpp
    │   ├── givens_matrix.hpp
    │   ├── givens_matrix.cpp
    │   ├── qr_iteration.hpp
    │   ├── qr_iteration.cpp
    │   └── README.md (this file)
    ├── main.cpp
    ├── examples.hpp (including functions for test my code)
    ├── examples.cpp
    └── nebula.hpp
```

All .hpp files in `./nlaProject/nla_headers/` are included in `nla.hpp` in `namespace nebula`:

```cpp
#include "nla_headers/nla_mat.hpp"
#include "nla_headers/givens_matrix.hpp"
#include "nla_headers/qr_iteration.hpp"
...
```

I write test codes in `main.cpp` and implement the various functions for doing tridiagonalization and QR iteration with shift and deflation in the other headers or sources.

In the following parts, I will:

1. Introduce the initial version of code, file by file (a bit verbose);
2. Test the feasibility and performance of my work given in chapter 1.;
3. How I attempted to improve the performance of the code, and to make it better than that of chapter 1.;
4. Give comments and retrospectives.

# Code (initial version) Introduction

## `utils.hpp`

Some useful functions are defined here.

### Get Householder Transform

From a input row vector, compute the corresponding Householder Matrix, that eliminates all the entries below the first entry.

I overload the function for both real vector and complex vector.

```cpp
// for complex vector
inline Mat<std::complex<double>>
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
inline Mat<double>
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
```

### From square matrix of any type to its Hessenberg Form

With the aid of `get_householder_mat`, I am equipped with the necessary tools to convert a square matrix to its Hessenberg Form.

By applying Householder Transform on almost each column of the input matrix, we can easily get the Hessenberg Form.

```cpp
// get the Hessenberg form of matx
template<typename M>
M to_hessenberg(const M &matx) {
    using elem_type = typename M::elem_type;

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
            hess.at(i, j) = 0.;
        }
    }

    return hess;
}
```

### SUBTASK ONE: Hermitian to Symmetric Tridiagonal

Transforming Hermitian Matrix into Hermitian Tridiagonal Matrix is straightforward by applying Householder, after which we manually set the lower parts zero to diminish computation errors.

However, imaginary parts of the sub- and superdiagonal entries still remain.

We note that, all eigenvalues of Hermitian Matrix are real. We can also derive from this another statement: Hermitian Matrix is similar to a Symmetric Tridiagonal Matrix by using some Unitary Transform.

In order to eliminate these imaginary part we need the help of another Diagonal Unitary Matrix. In the following code, you will see how the Diagonal Unitary Matrix is constructed.

```cpp
/*** !!! FOR THE FIRST SUBTASK: converting Hermitian Tridiagonal resulting
 **  from `to_hessenberg` into Real Symmetric Tridiagonal
 **  @param H Hermitian Tridiagonal
 **  @return Real Symmetric Tridiagonal, similar to H
***/
inline mat hermitian_tridiag2sym_tridiag(const cx_mat &H)
{
    using namespace std::complex_literals;

    auto hermitri = H;

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

    for (int i = 0; i < rows; ++i) {
        int nonzero_beg = i > 0 ? i - 1 : i;
        int nonzero_end = i + 1 < rows ? i + 1 : i;
        for (int j = nonzero_beg; j <= nonzero_end; ++j) {
            hermitri.at(i, j) = diag_entries[i] * hermitri.at(i, j);
        }
    }

    for (int i = 0; i < rows; ++i) {
        int nonzero_beg = i > 0 ? i - 1 : i;
        int nonzero_end = i + 1 < rows ? i + 1 : i;
        for (int j = nonzero_beg; j <= nonzero_end; ++j) {
            hermitri.at(j, i) = std::conj(diag_entries[i]) * hermitri.at(j, i);
        }
    }

    return real(hermitri);
}
```

## `givens_matrix.hpp`

### class `givens_matrix`: Our class for performing Givens Rotation

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

### implementation of member functions

Constructors and the function for `transpose()` is implemented here.

```cpp
template <typename T>
inline givens_matrix<T>::givens_matrix(uint j, uint k, T c, T s): j(j), k(k), c(c), s(s)
{ }

template <>
inline givens_matrix<double>::givens_matrix(double a, double b, uint j, uint k)
    : j(j), k(k)
{
    double r = std::hypot(a, b);
    if (std::abs(r) < 1e-10) {
        c = 1.0;
        s = 0.0;
    } else {
        c = a / r;
        s = b / r;
    }
}

template <>
inline givens_matrix<std::complex<double>>::givens_matrix(std::complex<double> a, std::complex<double> b, uint j, uint k)
    : j(j), k(k)
{
    using namespace std::complex_literals;

    auto anorm = std::abs(a);
    auto bnorm = std::abs(b);
    double r = std::hypot(anorm, bnorm);

    if (std::abs(r) < 1e-10) {
        c = 1.0;
        s = 0.0;
    } else {
        c = anorm / r;
        s = a / anorm * conj(b) / r;
    }
}

template <typename T>
givens_matrix<T>
inline givens_matrix<T>::transpose() const
{ return { j, k, c, -s }; }
```

### `operator*()` and `apply_givens`

I overload the `operator*()` to perform Givens Rotation on Row and Col. It's much less costly since I use references as the parameter and perform changes on the affected entries.

For perform Givens Rotation on Full Matrix, I write the function `apply_givens`. I intended to use `operator*()` too but it caused error for some reasons.

```cpp
inline Col<std::complex<double>> operator*(const givens_matrix<std::complex<double>> &g, const Col<std::complex<double>> &v) {
    auto res = v;
    uint j = g.j, k = g.k;
    std::complex<double> c = g.c, s = g.s;

    res[j] = c * v[j] + s * v[k];
    res[k] = -std::conj(s) * v[j] + c * v[k];

    return res;
}

inline Col<double> operator*(const givens_matrix<double> &g, const Col<double> &v) {
    auto res = v;
    uint j = g.j, k = g.k;
    double c = g.c, s = g.s;

    res[j] = c * v[j] + s * v[k];
    res[k] = -s * v[j] + c * v[k];

    return res;
}


inline Row<double> operator*(const Row<double> &v, const givens_matrix<double> &g) {
    auto res = v;
    uint j = g.j, k = g.k;
    double c = g.c, s = g.s;

    res[j] = c * v[j] - s * v[k];
    res[k] = s * v[j] + c * v[k];

    return res;
}

template <typename T>
Mat<T> apply_givens(const givens_matrix<T> &g, const Mat<T> &m) {
    auto res = m;
    auto cols = m.n_cols;
    auto j = g.j, k = g.k;

    for (uint i = 0; i < cols; ++i) {
        auto gcol = g * res.col(i);
        res.at(j, i) = gcol(j);
        res.at(k, i) = gcol(k);
    }

    return res;
}

template <typename T>
Mat<T> apply_givens(const givens_matrix<T> &g, const Mat<T> &m, const std::vector<uint> & cols) {
    auto res = m;
    auto j = g.j, k = g.k;

    for (auto col: cols) {
        auto gcol = g * res.col(col);
        res.at(j, col) = gcol(j);
        res.at(k, col) = gcol(k);
    }

    return res;
}

template <typename T>
Mat<T> apply_givens(const Mat<T> &m, const givens_matrix<T> &g) {
    auto res = m;
    uint rows = res.n_rows;
    auto j = g.j, k = g.k;

    for (uint i = 0; i < rows; ++i) {
        auto rowg = res.row(i) * g;
        res.at(i, j) = rowg(j);
        res.at(i, k) = rowg(k);
    }

    return res;
}

template <typename T>
Mat<T> apply_givens(const Mat<T> &m, const givens_matrix<T> &g, const std::vector<uint> & rows) {
    auto res = m;
    auto j = g.j, k = g.k;

    for (auto row: rows) {
        auto rowg = res.row(row) * g;
        res.at(row, j) = rowg(j);
        res.at(row, k) = rowg(k);
    }

    return res;
}
```

## `qr_iteration.hpp`

```cpp
template <typename M>
bool doesConverge(const M &hess, double tol = 1e-6)
{ return norm(hess.diag(-1), 2) < tol; }
```

### QR Iteration

Different types of QR iterations are given in `namespace qr`: the most fundamental one, one with shifts and one with both shifts and deflations.

#### function `qr::step_for_hessenberg`

This function performs exactly one QR step for a Hessenberg form by Givens Rotation, which is O(n).

```cpp
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
```

#### function `qr::iteration`

This function implements the very basic requirements for computing the eigenvalues of a matrix.

It first turns the input matrix `m` into a Hessenberg one by `to_hessenberg`, and then perform QR steps by calling `qr::step_with_hessenberg` given above, until convergence.

```cpp
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
```

#### Iteration with shift for matrices with special structures

In `NebuLA`, matrices with special structures are:

- Tridiagonal Real Symmetric;
- Real Symmetric;
- Hermitian.

The last two are similar to the first one by performing Unitary Transform, whilst the first one can be applied **Wilkinson Shift**.

Thus, I write specialized QR iteration versions for these categories of matrices. As **Thm. 2.5.11** states, the convergence rate for this algorithm is quadratic and in many cases even cubic. (The proof is too long and technical)

```cpp
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

        // std::cout << "Res after " << i << " Steps:\n" << res << std::endl;

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
```

As you can see, the last two functions above will first turn the matrices into real symmetric tridiagonal form, then call the iteration specialized for real symmetric tridiagonal matrices on them.

The specialized iteration step `qr::iteration_with_wilkinson_shift` is given below. The first scope performs **implicit shift**, whereas the following `for-loop`s do **Bulge Chasing** on the resulting matrix.

```cpp
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
```

#### functions `qr::francis_step` and  `qr::iteration_with_shift`

For this part, we will use Francis QR Step, which is given by Algorithm 2.5.20. Note that we need to compute the first colum of $M = H^2 - sH + tI$, which will be consuming if we compute the full matrix.

To simplify we note that the first column of $AB$ is $A$ multiply the first column of $B$. We also find the special structure of the first colum our $H$, which has only 2 nonzero entries. Hence, we can rewrite the first colum of $H^2$ as the linear combination of $H$'s first two columns.

Below, the implementation of Francis QR Step is given.

```cpp
// Francis QR Step
inline void francis_step(mat &hess) {
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
```

With the help of `qr::francis_step` we can implement the QR iteration with implicit double shift as:

```cpp
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
```

This implementation is still imperfect, since:

- Real matrices can have complex conjugate eigen pairs, which could not be computed in this iteration.

#### Iteration with Deflation for Matrix with Special Structures

My implementation of QR iteration with Deflation for Real Symmetric and Hermitian Matrix are each consisted of 3 parts, including 2 shared parts and 2 function overloads.

I use smart pointer to manage the storage, but the effect is not ideal. Only a very small improvement.

##### function `qr::iteration_with_deflation`

The two overloads of this function first turn the input matrix into Real Symmetric Tridiagonal Form, then delegates the computation of eigenvalues to `details::__iteration_with_deflation_impl`, which only accepts Real Symmetric Tridiagonal Form as input.

```cpp
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
```

##### function `details::partition`

To deflate the matrix, we need to find the '0' in the subdiagonal after each iteration step, and split it into at least 2 submatrices. In my implementation, either no deflation happens or the martix is divided into 2 parts. If we want even more parts, we could also maintain a list of submatrices.

In either case, the resulting matrix (or matrices) will be used as the input of `details::__iteration_with_deflation_impl`, which performs one QR Step with **Wilkinson Shift**, then send the new matrix back to `details::partition`.

```cpp
// partition for real matrix
inline void partition(std::shared_ptr<mat> &hess, std::vector<double> &eigs, double tol = 1e-6) {
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

#ifdef DEBUG
    std::cout << "No deflation." << std::endl;
#endif
    // if no deflate happens, iterate with the original matrix
    __iteration_with_deflation_impl(hess, eigs, tol);
}
```

##### function `details::__iteration_with_deflation_impl`

Till now, we are *dividing* the problem. To conquer the partitioned submatrices, we need to find some condition, where the branches of the recursions shall end.

Thankfully, this question is easy to solve:

- For the 1x1 matrix, the only element is the eigenvalue. Simply put it into the resulting eigenvalues;
- For the 2x2 matrix, the eigenvalues are easily computed with some formula;
- For any larger matrix, we cannot reduce the scale of the problem, but only do QR Step on the matrix and take the resulting matrix to `details::partition`.

The implementation are given below:

```cpp
inline void __iteration_with_deflation_impl(std::shared_ptr<mat> &tridiag, std::vector<double> &eigs, double tol) {
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

    auto sign = [] (const auto &num) { return num >= 0 ? 1 : -1; };
    auto &h = *tridiag;
    auto cols = h.n_cols;
    auto a = h.at(cols - 1, cols - 1);
    auto b = h.at(cols - 2, cols - 2);
    auto c = h.at(cols - 1, cols - 2);
    auto d = (b - a) / 2.;
    auto shift = a + d - sign(d) * std::hypot(d, c);
    qr::step_with_wilkinson_shift(*tridiag, shift);

    details::partition(tridiag, eigs, tol);
}
```

#### Iteration with Deflation on General Matrices

The implementation of this part is very similar to the part above. However, when I tested the code with some even very small real matrices, it failed to converge.

My implementation crashed very often at first.

To be honest I was at a loss and didn't figure out how to fix the bug, until I find that I called `arma::trace` and `arma::det` in my `qr::francis_step`, however, the result of this 2 functions are ridiculous when I print them out in console. As a result I manually computed them and it works well since then.

##### function `qr::general_iteration_with_deflation`

```cpp
/*** !!! SUBTASK 2-3: QR Iteration with Deflation for Real Matrix
 **  @param m real matrix
 **  @param tol tolerance of error
 **  @return the complex eigenvalues
 ***/
inline std::vector<std::complex<double>>
general_iteration_with_deflation(mat &m, double tol = 1e-6) {
    auto mp = std::make_shared<mat>(m);
    std::vector<std::complex<double>> eigs = {};

    details::__general_iteration_with_deflation_impl(mp, eigs, tol);

    return eigs;
}
```

##### function `details::partition`

```cpp
// partition for real nonsymmetric matrix
inline void
partition(std::shared_ptr<mat> &hess, std::vector<std::complex<double>> &eigs, double tol = 1e-6) {
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
```

##### function `details::__general_iteration_with_deflation_impl`

```cpp
inline void __general_iteration_with_deflation_impl(
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
```

# Experiment for the previous codes

The test results of my implementation will be shown in this chapter.

I verify the results by compare them with the result computed by Armadillo. My results are at least very close to the Armadillo results. But when it comes to performance, the previous code only gives a bad outcome. In the next chapter, I will talk about that.

I run the code on `Apple M3 Pro` with `18 GB` RAM.

## Test QR Iteration for Real Symmetric Tridiagonal Matrix (start with those matrices similar to them)

### Real Symmetric Matrix

### Hermitian Matrix

The Matrix is generated by this for-loop:

```cpp
for (int i = 0; i < size; ++i) {
    A(i, i) = i + 1;
}

for (int i = 0; i < size; ++i) {
    for (int j = i + 1; j < size; ++j) {
        std::complex<double> value = std::complex<double>(i + j + 1, i - j);
        A(i, j) = value;
        A(j, i) = std::conj(value);
    }
}
```

The time consumed (both **Transformation to Real Symmetric Tridiagonal** and **QR Iteration**) are listed below:


| size    | time consumption (w\ deflation) | time consumption (w\o deflation) |
| ------- | ------------------------------- | -------------------------------- |
| 30x30   | 2ms                             | 62ms                             |
| 50x50   | 15ms                            | 148ms                            |
| 70x70   | 34ms                            | 272ms                            |
| 90x90   | 58ms                            | 495ms                            |
| 100x100 | 90ms                            | 777ms                            |
| 200x200 | 698ms                           | 5290ms                           |
| 500x500 | 20556ms                         | 103990ms                         |

Strange is, **even if I separate the 2 stages and compute the duration once again**, the estimated Time Complexity of QR Iteration with deflation is still $O(n^3)$, whereas the version without deflation is approx. $O(n^2)$ (with a much larger constant).

**However, theoretically, QR Iteration with deflation is still $O(n^2)$, and in addition with a smaller constant.** This means there's something wrong with my deflation algorithm. Maybe I should use the **reference to the submatrix** of the original matrix, though I don't know how to make it in Armadillo.

## Test QR Iteration for General Matrix

For this part, I generate a matrix with following form:

```cpp
mat B(size, size, fill::zeros);

for (int i = 0; i < size; ++i) {
    B(i, i) = i + 1;
}

for (int i = 0; i < size; ++i) {
    for (int j = i + 1; j < size; ++j) {
        double value = i + j + 1;
        B(i, j) = value;
        B(j, i) = -value;
    }
}
```

At beginning, I used a random matrix for it. But I think this would be unfair since the random matrix may have very tricky structure, which is prone to complicated computations.

The time consumed are listed below. For this time I will not give the time consumption by iteration by Francis Step without deflation, as they will not give the conjugate eigenvalues and lack an effective convergence criterion.


| size    | time consumption (w\ deflation) |
| ------- | ------------------------------- |
| 30x30   | 2ms                             |
| 50x50   | 16ms                            |
| 70x70   | 60ms                            |
| 90x90   | 304ms                           |
| 100x100 | 488ms                           |
| 200x200 | 8615ms                          |
| 500x500 | UNKNOWN                         |

The ascending speed of the time consumption is horrible, which means my code fails to handle the deflation very efficiently.

# Improve the Performance of Deflation

Since the tested performance was not optimistic, I tried to find ways to fix those issues. I will put down what I did, in this chapter.

## First Thought: Full Matrix to Specilized Tridiagonal Matrix?

For the code above, I used the full matrix instead of a specialized tridiagonal matrix, which only stores the nonzero 3 or 2 diagonals. In fact, I did consider the probable issues in the implementation and only did the necessary operations on my tridiagonal full matrices.

When I read my code over and over, I noticed a possible performance pitfall: Could it be the **copy of the submatrices in deflation**, which is possibly $O(n^2)$, hindering my code from running faster, since the copy of 2 or 3 diagonals is $O(n)$? For that, I also rewrite the code to only copy the (sub/super)diagonal elements.

This attempt was a failure. Changing the copy scheme also did not affect the performance. I guess the copy of array-like objects is optimized by Armadillo or the compiler, as they occupy a **contiguous storage space**. (This might be proven wrong)

```cpp
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
```

## TOO MUCH Recursions?

I used **Depth First Pattern**, when I implemented my QR Iteration with Deflation:

1. Deflate the matrix;
2. Deflate the submatrix of the matrix;
3. Deflate the submatrix of the submatrix...

Following this pattern, the **depth of the runtime stack could be insanely large**. What if I use a **queue** and perform **Breadth First Pattern** on the matrix? Like:

1. Deflate the matrix until there's no matrix in the **queue** to deflate;
2. Pop the matrix out of **queue**;
3. Add the deflated submatrices to the **queue**...

This shall tremendously reduce the depth of the runtime stack. The implementation is given below. However, this time I was also not lucky enough:

* 30x30: 1ms
* 60x60: 8ms
* 90x90: 26ms
* ...

This is again absolutely $O(n^3)$ and these running times are all familiar numbers to me, in that they have never changed much since I test my codes.

```cpp
/*** !!! SUBTASK 2-3: QR Iteration with Deflation for Real Symmetric
 **  @param m real symmetric matrix
 **  @param tol tolerance of error
 **  @return the real eigenvalues
 ***/
inline std::vector<double> iteration_with_deflation_for_tridiag_using_BFS(mat &m, double tol = 1e-6) {
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
```

## Last Resort: Refactor with a specialized tridiagonal class

Till now, I have tried almost every possible ways to reduce Time Complexity:

- Checked the Wilkinson Shift;
- Avoid the possible redundant operations on zero entries;
- Breadth First Pattern to evade recursions;
- Copy the tridiagonal matrix.

And I also used smart pointer manage the storage to prevent the possible heap overflow, and reduce Space Complexity.

But it appears none of them did help. It looks like the only thing I can do is rewrite my code with a specialized tridiagonal class. In fact, all my struggles were to save myself from this worst situation.

After I discussed with one of my friend, I am very sure it can only be those inevitable but numerous copy costs that contribute to the bad performance.

The revision of those important parts of my code are listed below:

- A class `tridiag_matrix`, with several constructors (See `tridiag_matrix.hpp/.cpp`);
- Specialized QR Step with Wilkinson Shift for this new class (See `qr_iteration.hpp/.cpp`. This part is interesting to read, I used 2 queues for storing the bulges);
- Specialized QR Iteration with Deflation for this new class (Also see `qr_iteration.hpp/.cpp`);
- Improved transformation into Hessenberg Matrix `to_hessenberg_optimized`, which do not need to apply the full matrix multiplication. (See `utils.hpp/.cpp`)

The full code will be too long to display in this section, but I will show you how fast it is right now:


| size      | transformation time consumption | iteration time consumption |
| --------- |---------------------------------|----------------------------|
| 30x30     | 2ms                             | 0ms                        |
| 60x60     | 15ms                            | 1ms                        |
| 90x90     | 65ms                            | 3ms                        |
| 120x120   | 116ms                           | 6ms                        |
| 500x500   | 12425ms                         | 98ms                       |
| 1000x1000 | 153856ms                        | 409ms                      |

The iteration is at first $O(n^2)$ and moreover superfast compared with the previous outcomes, whereby the transformation is not $O(n^3)$ vs. previous $O(n^4)$.

# Retrospective

At the first glance, doing a project will be much easier than preparing for an oral test. Since, the project only involves the knowledge from the first half of our lectures.

But in fact it's not that case. Writing an efficient algorithm for performing the matrix operations, iterations etc. (esp. for those very large matrix) is a earnest work full with details.

By doing the project, I reviewed the important algorithms once again and learned many interesting things that I've missed out during this semester.

Even so, I failed to make my implementation as good as possible. The Time Complexities of some of my codes are not that optimal.

I didn't implement the computation of eigenvectors, since I think the performance will also be very unsatisfactory if I simply use the inverse power method.

I feel that I still have many things to learn in this field, especially when I witness the marvellous speed of Armadillo giving me the result, while at the same time I have to wait a century for my own results. (It might have used multi-thread and some optimization tricks)
