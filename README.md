Record dev. breakthroughs and my thoughts in this file.

## Something unimportant
To enable printing additional information or make the code more debug-friendly:

`#define DEBUG` at the beginning of the `main.cpp`

or

`add_compile_option(-DDEBUG)` in `CMakeLists.txt`

or 

`g++ <blabla> -DDEBUG` in your terminal, etc.

## class `nla_mat`
So I wrote a class to make my codes look better (more professional :D), hiding the details behind member functions.
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
```
At the beginning of the class I defined alias for types and declared some member functions including constructors and so on. The implementation would be introduced above.

Then I introduced a static member function `get_householder_mat` to compute the Householder Matrix from a given vector `x`. It has two overloads. For the complex vector it will call the first overload, otherwise the second will be called.

I used `std::enable_if_t` for deciding which version to be called, which is based upon `SFINAE` in C++. But factually in this case, we can simply use function overload (they are equivalent this time). It anyway serves as a watermark for my own version :P.

### Utility and member functions
#### operator<<()
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

```