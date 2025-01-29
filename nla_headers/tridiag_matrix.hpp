//
// Created by Tsumugi on 29.01.25.
//

#ifndef TRIDIAG_HPP
#define TRIDIAG_HPP

class tridiag_matrix {
public:
    tridiag_matrix(colvec, colvec, colvec);
    double at(uint, uint);
    void set(uint, uint, double);
    void print(const std::string &label) const;
    uint n_cols, n_rows;
    colvec diag, super, sub;
};

inline tridiag_matrix::tridiag_matrix(colvec diag, colvec super, colvec sub)
    : diag(std::move(diag)), super(std::move(super)), sub(std::move(sub)), n_cols(diag.n_cols), n_rows(diag.n_cols)
{ }

inline double tridiag_matrix::at(uint i, uint j) {
    if (i >= n_rows || j >= n_cols) {
        throw std::out_of_range("tridiag_matrix::at");
        return 0.;
    }
    if (i == j) return diag.at(j);
    if (i == j + 1) return sub.at(j);
    if (i + 1 == j) return super.at(i);
    return 0.;
}

inline void tridiag_matrix::set(uint i, uint j, double value) {
    if (i >= n_rows || j >= n_cols) {
        throw std::out_of_range("tridiag_matrix::at");
    }
    if (i == j)  diag.at(j) = value;
    if (i == j + 1) sub.at(j) = value;
    if (i + 1 == j) super.at(i) = value;
}

inline void tridiag_matrix::print(const std::string &label) const {
    std::cout << label << " (" << n_rows << "x" << n_cols << "):\n";

    for (uint i = 0; i < n_rows; ++i) {
        for (uint j = 0; j < n_cols; ++j) {
            if (i == j)
                std::cout << std::setw(8) << diag.at(i) << " ";
            else if (i == j + 1)
                std::cout << std::setw(8) << sub.at(j) << " ";
            else if (i + 1 == j)
                std::cout << std::setw(8) << super.at(i) << " ";
            else
                std::cout << std::setw(8) << "0" << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

#endif //TRIDIAG_HPP
