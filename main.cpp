#pragma gcc optimize("-O3")
#include <utils.hpp>

// examples of running the code
#include "examples.hpp"

// To enable printing additional information:
// #define DEBUG
// or
// `add_compile_option(-DDEBUG)` in CMakeLists.txt

/*** | Main features:
 **  --------------------------------
 **  | 1.    Transformation of a Hermitian matrix to a Real Symmetric Tridiagonal matrix
 **  | 1.1   Hermitian -> Hermitian Tridiagonal (Hessenberg)
 **  | 1.2   Hermitian Tridiagonal -> Real Symmetric Tridiagonal
 **
 **  | 2.    QR Iteration
 **  | 2.1   Oridinary QR Iteration with nothing notable (sometimes fail to converge for ill-formed matrices)
 **  | 2.2   QR Step with Shift (including Francis Step for nonsymmetric matrices)
 **  | 2.3   QR Iteration with Deflation (not optimized)
 **  | 2.3.1 Interface exposed to Users
 **  | 2.3.2 Partition
 **  | 2.3.3 Implementation of the Interface
 **  | 2.4   QR Iteration with Deflation (optimized): BFS + Specialized Tridiagonal Matrix
 **/
int main() {
    // At first the performance of my code was really bad
    // test of the specialized version of QR Iteration with Deflation
    int sizes[] = {30, 60, 90, 120, 500};
    for (auto size: sizes) {
        test_tridiag(size);
    }

    // You can get the eigenvalues of the matrix, but the time complexity is insanely large
    test_nonsymmetric_iteration(30);

    return 0;
}
