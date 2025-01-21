Project `NebuLA` by Tsumugivolle77 Copyright (C) reserved.

I name this project after **N**ebu**LA** instead of **NLA** (Numerical Linear Algebra) since it looks cool.

README file for my project and also for the final document submission.

GitHub repo: https://github.com/Tsumugivolle77/nlaProject

It will be temporarily invisible until submission deadline comes.

# Prerequisites for compiling project successfully
Install Armadillo(14.0.3) before you start and configure it properly.

Armadillo Web Page: https://arma.sourceforge.net

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

All files in `./nlaProject/nla_headers/` are included in `nla.hpp` in `namespace nebula`:
```cpp
namespace nebula {
using namespace arma;
#include "nla_headers/nla_mat.hpp"
#include "nla_headers/givens_matrix.hpp"
#include "nla_headers/qr_iteration.hpp"
}
```

I write test codes in `main.cpp` and implement the various functions for doing tridiagonalization and QR iteration with shift and deflation in the other headers or sources.

# `nla_mat.hpp`
## class `nla_mat`
So I write a class to make my codes look better.
```cpp
/**  This class is designed for
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
```
At the beginning of the class I defined alias for types and declared some member functions including constructors and so on. The implementation would be introduced above.

Then I introduced a static member function `get_householder_mat` to compute the Householder Matrix from a given vector `x`. It has two overloads. For the complex vector it will call the first overload, otherwise the second will be called.

I function overload (they are equivalent this time) for real and complex vectors.

### Utility and member functions
This part is trivial and has nothing notable to talk about. We can just skip to the next part.
#### `operator<<()`
This operator overload is for applying like `std::cout` or `fout` on our `nla_mat` object.
```cpp
// printing nla_mat
template <typename M>
std::ostream &operator<<(std::ostream &os, const nla_mat<M> &mat) {
    return os << mat.get_mat();
}
```

#### constructors
```cpp
template<typename M>
nla_mat<M>::nla_mat(M &&m): matx(m) { }

template<typename M>
nla_mat<M>::nla_mat(lists &&li): matx(li) { }
```

### SUBTASK ONE: Hermitian to Symmetric Tridiagonal
```cpp
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
```

Transforming into Hermitian Tridiagonal Matrix is straightforward by applying Householder, after which we manually set the lower parts zero to diminish computation errors.

However, imaginary parts of the sub- and superdiagonal entries still remain.

We note that, all eigenvalues of Hermitian Matrix are real. We can also derive from this another statement: Hermitian Matrix is similar to a Symmetric Tridiagonal Matrix by using some Unitary Transform.

In order to eliminate these imaginary part we need the help of another Diagonal Unitary Matrix. In the following code, you will see how the Diagonal Unitary Matrix is constructed.

(**NOTE:** THE CODES BELOW COULD AND SHOULD BE OPTIMIZED BY `class givens_matrix`)
```cpp
/*** !!! FOR THE FIRST SUBTASK: converting Hermitian Tridiagonal resulting
 **  from `A.to_hessenberg()` into Real Symmetric Tridiagonal
 **  @param A Hermitian Tridiagonal
 *   @return Real Symmetric Tridiagonal, similar to A
***/
nla_mat<mat>
inline hermitian_tridiag2sym_tridiag(const nla_mat<> &A)
{
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

    return {real(D * hermitri * D.ht())};
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
{ return {j, k, c, -s}; }
```

## `operator*()`
I overload the `operator*()` to perform Givens Rotation both for complex and real matrix, and from left and right. It's much less costly since I use references as the parameter and perform changes on the affected entries.
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
nla_mat<Mat<T>> operator*(const givens_matrix<T> &g, const nla_mat<Mat<T>> &m) {
    auto res = m.get_mat();
    uint cols = res.n_cols;

    for (uint i = 0; i < cols; ++i) {
        res.col(i) = g * res.col(i);
    }

    return res;
}

template <typename M, typename U = typename M::elem_type>
nla_mat<M> operator*(const nla_mat<M> &m, const givens_matrix<U> &g) {
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

### function `qr::step_for_hessenberg`
This function performs exactly one QR step for a Hessenberg form by Givens Rotation, which is O(n).

It contains the **implicit shift** if it's needed for accelerating convergence and saving storage.

It is frequently called by the other QR iteration algorithms in the same namespace. 
```cpp
// function for perform a qr step
template <typename M>
void qr_step_for_hessenberg(nla_mat<M> &hess, const typename M::elem_type &shift = 0.) {
    auto row = hess.get_mat().n_rows;
    
    // performs implicit shift
    {
        auto a = hess.get_mat().at(0, 0) - shift;
        auto b = hess.get_mat().at(1, 0);
        givens_matrix<typename M::elem_type> g {a, b, 0, 1};
        hess = g * hess * g.transpose();
    }

    for (uint j = 0; j < row - 1; ++j) {
        auto a = hess.get_mat().at(j, j);
        auto b = hess.get_mat().at(j + 1, j);
        givens_matrix<typename M::elem_type> g = {a, b, j, j + 1};
        hess = g * hess * g.transpose();
    }
}
```

### function `qr::iteration`
This function implements the very basic requirements for computing the eigenvalues of a matrix.

It first turns the input matrix `m` into a Hessenberg one by `nla_mat<M>::to_hessenberg`, and then perform QR steps by calling `qr::step_with_hessenberg` given above, until convergence.
```cpp
/*** !!! SUBTASK 2-1: QR iteration w\o deflation and shift
 **  @tparam M type of the Armadillo matrix, on which we are performing operations
 **  @param m input matrix
 **  @param maxiter maximum iteration steps
 **  @return quasi-upper-triangular matrix
***/
template <typename M>
Col<typename M::elem_type> iteration(const nla_mat<M> &m, uint maxiter = 1000) {
    auto hess = m.to_hessenberg();

    for (uint i = 0; i < maxiter; ++i) {
        step_for_hessenberg(hess);
    }

    return {hess.get_mat().diag()};
}
```

### Iteration with shift for matrices with special structures
In `NebuLA`, matrices with special structures are:
- Tridiagonal Real Symmetric;
- Real Symmetric;
- Hermitian.

The last two are similar to the first one by performing Unitary Transform, whilst the first one can be applied **Perfect Shift** (Lemma 2.5.4).

Thus, I write specialized QR iteration versions for these categories of matrices. As Thm. 2.5.11 states, the convergence rate for this algorithm is quadratic and in many cases even cubic. (The proof is too long and technical)
```cpp
inline vec iteration_with_shift_for_real_symmetric_tridiagonal(const nla_mat<mat> &tridiag, uint maxiter = 1000) {
    auto cols = tridiag.get_mat().n_cols;
    auto res  = tridiag;

    for (uint i = 0; i < maxiter; ++i) {
        auto sign = [] (const auto &num) { return num >= 0 ? 1 : -1; };
        const auto &r = res.get_mat();
        auto a = r.at(cols - 1, cols - 1);
        auto b = r.at(cols - 2, cols - 2);
        auto c = r.at(cols - 1, cols - 2);
        auto d = (b - a) / 2.;
        auto shift = a + d - sign(d) * std::hypot(d, c);

        step_for_hessenberg(res, shift);
        
        // break if it converges to quasi upper triangular form
        if (doesConverge(hess)) break;
    }

    return res.get_mat().diag();
}

inline vec iteration_with_shift_for_hermitian(const nla_mat<cx_mat> &m, uint maxiter = 1000) {
    auto hess = m.to_hessenberg();
    auto tridiag = hermitian_tridiag2sym_tridiag(hess);
    return iteration_with_shift_for_real_symmetric_tridiagonal(tridiag, maxiter);
}

inline vec iteration_with_shift_for_symmetric(const nla_mat<mat> &m, uint maxiter = 1000) {
    auto tridiag = m.to_hessenberg();
    return iteration_with_shift_for_real_symmetric_tridiagonal(tridiag, maxiter);
}
```
As you can see, the last two functions above will first turn the matrices into real symmetric tridiagonal form, then call the iteration specialized for real symmetric tridiagonal matrices on them.

```
Hermitian -- Householder --> Hermitian Tridiagonal -- A Diagonal Unitary Matrix -->
                                                                                  |---> Real Symmetric Tridiagonal -- QR STEPS --> Quasi Upper Triangular
Real Symmetric ----------------------------- Householder ------------------------->
```

### functions `qr::francis_step` and  `qr::iteration_with_shift`
For this part, we will use Francis QR Step, which is given by Algorithm 2.5.20. Note that we need to compute the first colum of $M = H^2 - sH + tI$, which will be consuming if we compute the full matrix.

To simplify we note that the first column of $AB$ is $A$ multiply the first column of $B$. We also find the special structure of the first colum our $H$, which has only 2 nonzero entries. Hence, we can rewrite the first colum of $H^2$ as the linear combination of $H$'s first two columns.

Below, the implementation of Francis QR Step is given.
```cpp
// Francis QR Step
template <typename M>
inline void francis_step(nla_mat<M> &hess) {
    {
        // set up the implicit shift
        using et   = typename M::elem_type;
        uint cols  = hess.get_mat().n_cols;
        Mat<et> sm = {hess.get_mat().submat(cols - 2, cols - 1, cols - 2, cols - 1)};
        et s       = trace(sm);
        et t       = det(sm);
        et h00     = hess.get_mat().at(0, 0);
        et h10     = hess.get_mat().at(1, 0);
        auto col0  = hess.get_mat().col(0);
        auto col1  = hess.get_mat().col(1);
        Col<et> w  = h00 * col0 + h10 * col1 - s * col0;
        w[0]      += t;
        
        auto Q = nla_mat<>::get_householder_mat(w);
        hess = {Q * hess.get_mat() * Q.ht()};
    }

    hess = {hess.to_hessenberg()};
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
Col<typename M::elem_type> iteration_with_shift(const nla_mat<M> &m, uint maxiter = 1000) {
    auto hess = m.to_hessenberg();

    for (uint i = 0; i < maxiter; ++i) {
        francis_step(hess);

        if (doesConverge(hess)) break;
    }
#ifdef DEBUG
    std::cout << "Matrix after QR iteration with implicit double shift:\n" << hess.get_mat() << std::endl;
#endif

    return {hess.get_mat().diag()};
}
```

This implementation is still imperfect, since:
- Real matrices can have complex conjugate eigen pairs 

### function `qr::iteration_with_deflation`