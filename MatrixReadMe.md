# Matrix API Development Documentation

## Matrix_T API Development Documentation

### Overview

`SimpFlex::Matrix<T, Rows, Cols>` is a **compile‑time fixed‑size matrix class** that provides high‑performance linear algebra operations. It is designed for embedded, real‑time, and game development where stack allocation, `constexpr` evaluation, and zero overhead are critical.

The library includes fully unrolled specializations for 2×2, 3×3, and 4×4 matrices, delivering optimal performance for small matrices. Larger matrices fall back to a generic loop‑based implementation.

All operations are `constexpr`‑compatible (except I/O and functions that throw exceptions), enabling compile‑time computation of matrix expressions.

---

### Header

```cpp
#include "Matrix_T.hpp"
```

### Namespace

```cpp
using namespace SimpFlex;
```

---

### Type Aliases

| Alias | Definition |
|-------|------------|
| `Matrix2f` | `Matrix<float, 2, 2>` |
| `Matrix3f` | `Matrix<float, 3, 3>` |
| `Matrix4f` | `Matrix<float, 4, 4>` |
| `Matrix2d` | `Matrix<double, 2, 2>` |
| `Matrix3d` | `Matrix<double, 3, 3>` |
| `Matrix4d` | `Matrix<double, 4, 4>` |
| `Matrix2i` | `Matrix<int, 2, 2>` |
| `Matrix3i` | `Matrix<int, 3, 3>` |
| `Matrix4i` | `Matrix<int, 4, 4>` |
| `Vector2f` / `Vector3f` / `Vector4f` | `Matrix<float, N, 1>` |
| `Vector2d` / `Vector3d` / `Vector4d` | `Matrix<double, N, 1>` |
| `Vector2i` / `Vector3i` / `Vector4i` | `Matrix<int, N, 1>` |

---

### Class Template `Matrix<T, Rows, Cols>`

### Constructors

| Constructor | Description |
|-------------|-------------|
| `Matrix() noexcept` | Default constructor, value‑initializes all elements to `0`. |
| `explicit Matrix(T value) noexcept` | Fills all elements with `value`. |
| `Matrix(std::initializer_list<T> list) noexcept` | Constructs from an initializer list. The list must contain at most `Rows * Cols` elements; missing entries are zero‑filled. |
| `explicit Matrix(const T(&arr)[Rows * Cols]) noexcept` | Constructs from a C‑array. |

---

### Static Functions

| Function | Returns |
|----------|---------|
| `static Matrix zero() noexcept` | Zero‑initialized matrix. |
| `static Matrix identity() noexcept` (square only) | Identity matrix (1 on diagonal, 0 elsewhere). |
| `static Matrix constant(T value) noexcept` | Matrix filled with `value`. |

---

### Access

| Function | Signature | Description |
|----------|-----------|-------------|
| `operator()` | `T& operator()(int row, int col)`<br>`const T& operator()(int row, int col) const` | Access element by row and column. |
| `operator[]` | `T& operator[](int idx)`<br>`const T& operator[](int idx) const` | Access element by linear index (row‑major). |
| `data()` | `T* data()`<br>`const T* data() const` | Pointer to underlying raw storage. |
| `rows()` / `cols()` / `size()` | `static constexpr int rows()`, `cols()`, `size()` | Static dimension queries. |

---

### Operators

| Operator | Semantics |
|----------|-----------|
| `+` (unary) | Unary plus (returns `*this`) |
| `-` (unary) | Negation (element‑wise) |
| `+` / `+=` | Element‑wise addition |
| `-` / `-=` | Element‑wise subtraction |
| `*` / `*=` (scalar) | Element‑wise multiplication by scalar |
| `/` / `/=` (scalar) | Element‑wise division by scalar |
| `*` (matrix multiplication) | **Matrix product** – dimensions must match: `A(R×K) * B(K×C) -> Matrix<R,C>` |
| `*` (matrix‑vector) | Matrix‑vector product (specialized for vectors) |

---

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `transpose()` | `Matrix<T, Cols, Rows> transpose() const noexcept` | Returns transposed matrix. |
| `trace()` | `T trace() const noexcept` (square only) | Sum of diagonal elements. |
| `determinant()` | `T determinant() const noexcept` (square only) | Computes determinant (LU decomposition for generic, closed‑form for 2×2/3×3). |
| `inverse()` | `Matrix inverse() const` (square only) | Returns inverse matrix. Throws `std::runtime_error` if singular. |
| `cwiseProduct()` | `Matrix cwiseProduct(const Matrix& rhs) const noexcept` | Hadamard product (element‑wise multiplication). |
| `cwiseQuotient()` | `Matrix cwiseQuotient(const Matrix& rhs) const noexcept` | Element‑wise division. |
| `row()` | `Matrix<T, 1, Cols> row(int r) const noexcept` | Returns the `r`‑th row as a row vector. |
| `col()` | `Matrix<T, Rows, 1> col(int c) const noexcept` | Returns the `c`‑th column as a column vector. |
| `block<SR, SC>()` | `Matrix<T, SR, SC> block(int start_row, int start_col) const noexcept` | Returns a sub‑matrix of size `SR×SC` starting at `(start_row, start_col)`. |
| `diagonal()` | `Matrix<T, min(Rows,Cols), 1> diagonal() const noexcept` | Returns a column vector containing the diagonal elements. |
| `sum()` | `T sum() const noexcept` | Sum of all elements. |
| `prod()` | `T prod() const noexcept` | Product of all elements. |
| `minCoeff()` / `maxCoeff()` | `T minCoeff() const noexcept`<br>`T maxCoeff() const noexcept` | Minimum / maximum element. |
| `squaredNorm()` / `norm()` | `T squaredNorm() const noexcept`<br>`T norm() const noexcept` | Frobenius norm (squared and regular). |
| `toString()` | `std::string toString() const` | Readable matrix representation. |
| `dot` | `T dot(const Matrix<T, N, 1>& a, const Matrix<T, N, 1>& b) noexcept` | Dot product of two vectors. |
| `cross` | `Matrix<T, 3, 1> cross(const Matrix<T, 3, 1>& a, const Matrix<T, 3, 1>& b) noexcept` | Cross product of two 3D vectors. |
| `normalized` | `Matrix<T, N, 1> normalized(const Matrix<T, N, 1>& v) noexcept` | Returns a unit vector; returns zero vector if norm is zero. |
| `operator<<` | `std::ostream& operator<<(std::ostream& os, const Matrix& m)` | Stream insertion operator (calls `toString()`). |

---

### Example

```cpp
#include "Matrix_T.hpp"
#include <iostream>

int main() {
    using namespace SimpFlex;
    
    Matrix2f A(1, 2, 3, 4);
    Matrix2f B(5, 6, 7, 8);
    auto C = A * B;
    float det = A.determinant();
    auto inv = A.inverse();

    Matrix3d M = Matrix3d::identity();
    Vector3d v{1, 2, 3};
    auto w = M * v;
    auto c = cross(v, w);
    auto d = dot(v, w);

    Matrix<float, 6, 6> big;
    auto big_t = big.transpose();
    auto det_big = big.determinant();

    constexpr Matrix2i m1(1, 2, 3, 4);
    constexpr Matrix2i m2(5, 6, 7, 8);
    constexpr auto m3 = m1 + m2;
    static_assert(m3(0, 0) == 6, "m3(0,0) must be 6");

    std::cout << "A = \n" << A << "\n";
    std::cout << "C = \n" << C << "\n";
    std::cout << "det(A) = " << det << "\n";
    std::cout << "w = " << w.transpose() << "\n";
    std::cout << "big_t = \n" << big_t << "\n";
}
```

## Matrix_S API Development Documentation

### Overview

`SimpFlex::Matrix<T, Rows, Cols>` is a compile‑time fixed‑size dense matrix class designed for high‑performance numerical computing, providing full linear algebra operations, matrix decompositions, and linear solvers; it includes fully unrolled specializations for 2×2, 3×3, and 4×4 matrices to deliver maximum performance, while also offering a generic implementation for any matrix size. The library provides a complete numerical linear algebra suite (LU, Cholesky, QR, eigenvalues, SVD) along with linear system solvers (LU, Cholesky, QR least‑squares). All matrices are stack‑allocated with zero heap allocation, and most operations are constexpr compatible, making it suitable for embedded, real‑time, and performance‑critical applications.



---

### Header

```cpp
#include "Matrix_S.hpp"
```

### Namespace

```cpp
using namespace SimpFlex;
```

---

### Type Aliases

| Alias | Definition |
|-------|------------|
| `Matrix2f` | `Matrix<float, 2, 2>` |
| `Matrix3f` | `Matrix<float, 3, 3>` |
| `Matrix4f` | `Matrix<float, 4, 4>` |
| `Matrix2d` | `Matrix<double, 2, 2>` |
| `Matrix3d` | `Matrix<double, 3, 3>` |
| `Matrix4d` | `Matrix<double, 4, 4>` |
| `Vector2f` / `Vector3f` / `Vector4f` | `Matrix<float, N, 1>` |
| `Vector2d` / `Vector3d` / `Vector4d` | `Matrix<double, N, 1>` |

### Constructors

| Constructor | Description |
|-------------|-------------|
| `Matrix() noexcept` | Default, value‑initializes all elements to `0`. |
| `explicit Matrix(T value) noexcept` | Fills all elements with `value`. |
| `Matrix(std::initializer_list<T> list) noexcept` | Constructs from an initializer list (row‑major order). |
| `explicit Matrix(const T(&arr)[Rows * Cols]) noexcept` | Constructs from a C‑array. |

---

### Static Functions

| Function | Description |
|----------|-------------|
| `static Matrix zero() noexcept` | Returns a zero‑initialized matrix. |
| `static Matrix identity() noexcept` (square only) | Returns the identity matrix. |
| `static Matrix constant(T value) noexcept` | Returns a matrix filled with `value`. |

### Access

| Function | Signature | Description |
|----------|-----------|-------------|
| `operator()` | `T& operator()(int r, int c)`<br>`const T& operator()(int r, int c) const` | Access element by (row, col). **No bounds checking.** |
| `operator[]` | `T& operator[](int idx)`<br>`const T& operator[](int idx) const` | Access by linear index (row‑major). |
| `data()` | `T* data()`<br>`const T* data() const` | Raw pointer to underlying storage. |
| `rows()` / `cols()` / `size()` | `static constexpr int rows()`, `cols()`, `size()` | Dimension queries. |

---

### Operators

| Operator | Description |
|----------|-------------|
| `+` (unary) | Returns `*this` |
| `-` (unary) | Element‑wise negation |
| `+` / `+=` | Element‑wise addition |
| `-` / `-=` | Element‑wise subtraction |
| `*` / `*=` (scalar) | Element‑wise scalar multiplication |
| `/` / `/=` (scalar) | Element‑wise scalar division |
| `*` (matrix) | Matrix multiplication (`A(R×K) * B(K×C) -> Matrix<R,C>`) |
| `*` (matrix‑vector) | Matrix‑vector product |

---

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `transpose()` | `Matrix<T, Cols, Rows> transpose() const noexcept` | Returns transposed matrix. |
| `trace()` | `T trace() const noexcept` (square only) | Sum of diagonal elements. |
| `determinant()` | `T determinant() const noexcept` (square only) | Computes determinant. |
| `inverse()` | `Matrix inverse() const` (square only) | Returns inverse. Throws `std::runtime_error` if singular. |
| `cwiseProduct()` | `Matrix cwiseProduct(const Matrix& rhs) const noexcept` | Element‑wise multiplication. |
| `row()` | `Matrix<T, 1, Cols> row(int r) const noexcept` | Returns the `r`‑th row as a row vector. |
| `col()` | `Matrix<T, Rows, 1> col(int c) const noexcept` | Returns the `c`‑th column as a column vector. |
| `block<SR, SC>()` | `Matrix<T, SR, SC> block(int r0, int c0) const noexcept` | Returns sub‑matrix of size `SR×SC` at `(r0, c0)`. |
| `diagonal()` | `Matrix<T, min(Rows,Cols), 1> diagonal() const noexcept` | Returns a column vector of diagonal elements. |
| `sum()` | `T sum() const noexcept` | Sum of all elements. |
| `prod()` | `T prod() const noexcept` | Product of all elements (may overflow). |
| `minCoeff()` / `maxCoeff()` | `T minCoeff() const noexcept`<br>`T maxCoeff() const noexcept` | Minimum / maximum element. |
| `squaredNorm()` / `norm()` | `T squaredNorm() const noexcept`<br>`T norm() const noexcept` | Squared / regular Frobenius norm. |
| `toString()` | `std::string toString() const` | Readable string representation. |
| `dot` | `T dot(const Vector<T,N>& a, const Vector<T,N>& b) noexcept` | Dot product of two vectors. |
| `cross` | `Vector3<T> cross(const Vector3<T>& a, const Vector3<T>& b) noexcept` | Cross product of two 3D vectors. |
| `normalized` | `Vector<T,N> normalized(const Vector<T,N>& v) noexcept` | Returns unit vector (zero if norm is zero). |
| `operator<<` | `std::ostream& operator<<(std::ostream& os, const Matrix& m)` | Stream output (calls `toString()`). |

---

### Algorithm

#### LU Decomposition (Partial Pivoting)

```cpp
template <typename T, int N>
bool lu(const Matrix<T, N, N>& A,
        Matrix<T, N, N>& L,
        Matrix<T, N, N>& U,
        int* perm) noexcept;
```

Computes `PA = LU`. Returns `true` on success, `false` if singular.

---

#### Cholesky Decomposition

```cpp
template <typename T, int N>
bool cholesky(const Matrix<T, N, N>& A,
              Matrix<T, N, N>& L) noexcept;
```

Computes `A = L * L^T` for symmetric positive‑definite matrices. Returns `true` on success, `false` if not SPD.

---

#### QR Decomposition (Householder)

```cpp
template <typename T, int M, int N>
void qr(const Matrix<T, M, N>& A,
        Matrix<T, M, M>& Q,
        Matrix<T, M, N>& R) noexcept;
```

Computes `A = Q * R`. **Requires explicit template arguments for M,N.**

---

#### Eigenvalue Decomposition

##### Jacobi Method

```cpp
template <typename T, int N>
int jacobi_eigen(const Matrix<T, N, N>& A,
                 Matrix<T, N, N>& V,
                 T* eigvals,
                 T tol = T(1e-10),
                 int max_sweeps = 50) noexcept;
```

Returns eigenvector matrix `V` and eigenvalues in `eigvals`. Returns iteration count (`-1` if not converged).

##### QR Algorithm

```cpp
template <typename T, int N>
int eigenvalues_qr(const Matrix<T, N, N>& A,
                   T* eigvals,
                   T tol = T(1e-12),
                   int max_iter = 100) noexcept;
```

Returns eigenvalues only. Returns iteration count.

---

#### SVD Decomposition

```cpp
template <typename T, int M, int N>
int svd_jacobi(const Matrix<T, M, N>& A,
               Matrix<T, M, M>& U,
               Matrix<T, M, N>& S,
               Matrix<T, N, N>& V,
               T tol = T(1e-10),
               int max_sweeps = 50) noexcept;
```

Computes `A = U * S * V^T`. Returns iteration count (`-1` if not converged).

---

### Linear System Solvers [BOOL]

| Function | Description |
|----------|-------------|
| `solve_lu<T,N>(A, b, x)` | Solves `Ax = b` using LU (square systems). |
| `solve_cholesky<T,N>(A, b, x)` | Solves `Ax = b` using Cholesky (SPD systems). |
| `solve_qr<T,M,N>(A, b, x)` | Solves `Ax ≈ b` in least‑squares sense (overdetermined systems). |

---

### Example

```cpp
#include "Matrix_S.hpp"
#include <iostream>
#include <iomanip>

int main() {
    using namespace SimpFlex;
    std::cout << std::fixed << std::setprecision(4);

    {
        Matrix3d A{ 2, -1, 0, -1, 2, -1, 0, -1, 2 };
        Matrix3d L, U;
        int perm[3];
        bool ok = decomp::lu(A, L, U, perm);

        if (ok) {
            std::cout << "L =\n" << L << "\n";
            std::cout << "U =\n" << U << "\n";
        }
    }

    {
        Matrix3d A{ 4, 12, -16, 12, 37, -43, -16, -43, 98 };
        Matrix3d L;
        bool ok = decomp::cholesky(A, L);

        if (ok) {
            std::cout << "L =\n" << L << "\n";
            std::cout << "L * L^T =\n" << L * L.transpose() << "\n";
        }
    }

    {
        Matrix<double, 4, 3> A{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        Matrix4d Q;
        Matrix<double, 4, 3> R;
        decomp::qr<double, 4, 3>(A, Q, R);
    }

    {
        Matrix3d A{ 2, -1, 0, -1, 2, -1, 0, -1, 2 };
        double eig[3];
        decomp::eigenvalues_qr(A, eig);
    }

    {
        Matrix<double, 3, 2> A{ 1, 2, 3, 4, 5, 6 };
        Matrix3d U;
        Matrix<double, 3, 2> S;
        Matrix2d V;
        decomp::svd_jacobi<double, 3, 2>(A, U, S, V);
    }

    {
        Matrix3d A{ 3, 2, -1, 2, -2, 4, -1, 0.5, -1 };
        Vector3d b{ 1, -2, 0 };
        Vector3d x;
        if (decomp::solve_lu(A, b, x)) {
            std::cout << "x = [" << x(0, 0) << ", " << x(1, 0) << ", " << x(2, 0) << "]\n";
        }
    }

    {
        Matrix3d A{ 4, 12, -16, 12, 37, -43, -16, -43, 98 };
        Vector3d b{ 1, 2, 3 };
        Vector3d x;
        decomp::solve_cholesky(A, b, x);
    }

    {
        Matrix<double, 4, 3> A{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13 };
        Matrix<double, 4, 1> b{ 1, 2, 3, 4 };
        Matrix<double, 3, 1> x;
        decomp::solve_qr(A, b, x);
    }
    return 0;
}
```

## Matrix_BI API Development Documentation

### Overview

`SimpFlex::Matrix<T>` is a runtime‑sized dense matrix class designed for high‑performance numerical computing with dynamic dimensions, providing full linear algebra operations, matrix decompositions, and linear solvers, with memory allocated on the heap using a custom 64‑byte aligned allocator for cache‑friendly and SIMD‑optimized access. Runtime‑sized matrices, AVX2/FMA‑accelerated GEMM via a GotoBLAS micro‑kernel (8×6 for double, 8×12 for float), a complete numerical linear algebra suite (LU with partial pivoting, Cholesky, QR via Householder, eigenvalues via Jacobi and QR algorithms, and SVD via the Jacobi method), linear system solvers (LU, Cholesky, QR least‑squares), 64‑byte aligned memory for optimal SIMD performance, optional OpenMP parallelization for large matrix multiplications, and zero external dependencies, relying solely on the C++ standard library and compiler intrinsics.


### Header

```cpp
#include "Matrix_BI.hpp"
```

### Namespace

```cpp
using namespace SimpFlex;
```

---

### Class Template `Matrix<T>`

### Template

| Parameter | Description |
|-----------|-------------|
| `T` | Element type (e.g., `float`, `double`, `int`). Must support arithmetic operations. |

### Variables

| Member | Type | Description |
|--------|------|-------------|
| `rows` | `size_t` | Number of rows. |
| `cols` | `size_t` | Number of columns. |
| `data` | `std::vector<T, AlignedAllocator<T, 64>>` | Underlying aligned storage (row‑major order). |

---

### Constructors

| Constructor | Description |
|-------------|-------------|
| `Matrix() noexcept` | Default constructor – creates an empty matrix (0 rows, 0 cols). |
| `Matrix(size_t r, size_t c)` | Constructs an `r×c` matrix with zero‑initialized elements. |
| `Matrix(size_t r, size_t c, T init)` | Constructs an `r×c` matrix with all elements set to `init`. |

---

### Access

| Function | Signature | Description |
|----------|-----------|-------------|
| `operator()` | `T& operator()(size_t i, size_t j)`<br>`const T& operator()(size_t i, size_t j) const` | Access element at (row, col) ; no bounds checking (asserts in debug). |
| `operator[]` | `T* operator[](size_t i)`<br>`const T* operator[](size_t i) const` | Returns pointer to the `i`‑th row (row‑major). |
| `ptr()` | `T* ptr()`<br>`const T* ptr() const` | Returns raw pointer to the data array. |
| `size()` | `size_t size() const noexcept` | Total number of elements (`rows * cols`). |
| `empty()` | `bool empty() const noexcept` | Checks if matrix has zero elements. |

---

### Modifiers

| Function | Description |
|----------|-------------|
| `resize(size_t r, size_t c)` | Resizes matrix to `r×c` (existing data lost). |
| `resize(size_t r, size_t c, T v)` | Resizes and initializes new elements to `v`. |
| `fill(T v)` | Sets all elements to `v`. |

---

### Util Functions

| Function | Description |
|----------|-------------|
| `static Matrix eye(size_t n)` | Returns an `n×n` identity matrix. |
| `transpose() const` | Returns a transposed copy of the matrix. |
| `matmul(const Matrix<T>& A, const Matrix<T>& B)` | Matrix multiplication – uses optimized GEMM (AVX2/FMA if available). |
| `operator+(const Matrix<T>& a, const Matrix<T>& b)` | Element‑wise addition. |
| `operator-(const Matrix<T>& a, const Matrix<T>& b)` | Element‑wise subtraction. |
| `operator*(const Matrix<T>& m, T s)` | Scalar multiplication (right side). |

---

### Decomposition

#### LU

```cpp
template <typename T>
std::pair<Matrix<T>, LUResult> lu_decomp(const Matrix<T>& A);
```

Returns a packed LU matrix and an `LUResult` structure containing pivots and info.And uses partial pivoting; `info == 0` indicates success.

---

#### Cholesky

```cpp
template <typename T>
Matrix<T> cholesky(const Matrix<T>& A);
```

Returns the lower triangular Cholesky factor `L` such that `A = L * L^T`.Throws `std::runtime_error` if the matrix is not positive‑definite.

---

#### QR

```cpp
QRResult qr_decomp(const Matrix<double>& A);
```

Returns a packed QR matrix (upper triangular `R` in the upper part, Householder vectors in the lower part) and a `tau` vector.Only implemented for `double`.

---

#### Eigenvalue

```cpp
EigResult eig_sym(Matrix<double> A, double tol = 1e-12, int max_iter = 1000);
```

Computes eigenvalues and eigenvectors of a symmetric matrix using the Jacobi method.And returns eigenvalues sorted in descending order, and eigenvectors as columns.

---

#### SVD

```cpp
SVDResult svd(const Matrix<double>& A);
```

Computes `A = U * S * V^T` using Jacobi method on `A^T A`.And returns `U`, singular values `S`, and `V^T`.

---

### Util Functions

| Function | Description |
|----------|-------------|
| `print(const Matrix<T>& m, size_t max_rows = 10, size_t max_cols = 8)` | Prints a matrix with formatting (limited rows/cols). |
| `rand_matrix(size_t r, size_t c, T lo = T(-1), T hi = T(1))` | Generates a random matrix with uniform distribution. |
| `rand_spd(size_t n)` | Generates a random symmetric positive‑definite matrix. |
| `max_abs_diff(const Matrix<T>& a, const Matrix<T>& b)` | Computes the maximum absolute difference between two matrices. |
| `frobenius_norm(const Matrix<T>& m)` | Computes the Frobenius norm. |

### Example

```cpp
#include "Matrix_BI.hpp"
#include <iostream>

int main() {
    using namespace SimpFlex;

    Matrix<double> A(3, 3);
    A(0, 0) = 2; A(0, 1) = -1; A(0, 2) = 0;
    A(1, 0) = -1; A(1, 1) = 2; A(1, 2) = -1;
    A(2, 0) = 0; A(2, 1) = -1; A(2, 2) = 2;

    Matrix<double> B = A;
    B(0, 0) += 1;

    auto C = A + B;
    auto D = A * 2.0;

    auto E = matmul(A, B);

    auto lu = lu_decomp(A);
    std::cout << "LU info: " << lu.second.info << "\n";

    auto SPD = rand_spd<double>(4);
    auto L = cholesky(SPD);

    auto qr = qr_decomp(A);

    auto eig = eig_sym(A);
    for (auto v : eig.eigenvalues) std::cout << v << " ";

    auto svd_res = svd(A);

    return 0;
}
```

### Matrix_MS API Development Documentation

### Overview

`Matrix_MS` provides high-performance sparse matrix operations for scientific computing, graph processing, and machine learning applications. It offers three storage formats optimized for different access patterns and hardware capabilities: **CSR** (general-purpose), **ELL** (GPU-friendly), and **SELL-C-sigma** (CPU cache-optimized with AVX2 acceleration). All formats use 64-byte aligned memory and support OpenMP parallelization.

---

### Header

```cpp
#include "Matrix_MS.hpp"
```

### Namespace

```cpp
using namespace SimpFlex;
```

---

### CSRMatrix<T, Idx>

### Overview

`CSRMatrix<T, Idx>` is the **Compressed Sparse Row** format – the most general-purpose sparse matrix format, ideal for matrices with irregular sparsity patterns.

---

### Constructors

| Constructor | Description |
|-------------|-------------|
| `CSRMatrix() noexcept` | Default constructor, empty matrix. |
| `CSRMatrix(Idx r, Idx c, size_t nz)` | Constructs an `r×c` matrix with `nz` non-zeros (allocates storage). |

---

### Factories

| Function | Description |
|----------|-------------|
| `static CSRMatrix from_coo(rows, cols, coo, sorted)` | Builds CSR from COO (triplet) list. If `sorted=false`, entries are automatically sorted by row then column. |
| `static CSRMatrix from_dense(dense, rows, cols, eps)` | Extracts non-zeros from a dense matrix with tolerance `eps`. |

---

### Members

| Member | Type | Description |
|--------|------|-------------|
| `rows` | `Idx` | Number of rows. |
| `cols` | `Idx` | Number of columns. |
| `nnz` | `size_t` | Number of non-zero elements. |
| `row_ptr` | `std::vector<Idx>` | Row offsets, length `rows + 1`. |
| `col_idx` | `std::vector<Idx>` | Column indices, length `nnz`. |
| `values` | `std::vector<T>` | Non-zero values, length `nnz`. |

---

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `row_nnz` | `Idx row_nnz(Idx i) const noexcept` | Returns non-zero count in row `i`. |
| `density` | `double density() const noexcept` | Returns density as `nnz / (rows * cols)`. |
| `print_info` | `void print_info(const char* name = "CSR") const` | Prints matrix statistics (dimensions, nnz, density). |

---

### ELLMatrix<T, Idx>

### Overview

`ELLMatrix<T, Idx>` is the **ELLPACK** format – stores all rows with the same maximum length using padding. Ideal for matrices with similar row lengths and GPU-friendly access patterns.

---

### Members

| Member | Type | Description |
|--------|------|-------------|
| `rows` / `cols` | `Idx` | Dimensions. |
| `max_nz_per_row` | `Idx` | Maximum non-zeros per row. |
| `nnz` | `size_t` | Total non-zeros (excluding padding). |
| `col_idx` | `std::vector<Idx>` | Column indices, size `max_nz_per_row * rows`. |
| `values` | `std::vector<T>` | Values, same layout as `col_idx`. |

---

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `print_info` | `void print_info(const char* name = "ELL") const` | Prints format statistics (dimensions, max_nz/row, fill ratio). |

---

### SELLMatrix<T, C, Idx>

### Overview

`SELLMatrix<T, C, Idx>` is the **SELL-C-sigma** format – groups rows into chunks of `C` rows and reorders rows by density within each sigma block. Optimized for AVX2 acceleration.

---

### Template Parameters

| Parameter | Description |
|-----------|-------------|
| `T` | Element type (`double` or `float`). |
| `C` | Number of rows per chunk (4 for double, 8 for float). |
| `Idx` | Index type (must be `uint32_t` for AVX2). |

---

### Members

| Member | Type | Description |
|--------|------|-------------|
| `rows` / `cols` | `Idx` | Dimensions. |
| `nnz` | `size_t` | Non-zero count. |
| `sigma` | `int` | Row grouping size for reordering (default 256). |
| `num_chunks` | `Idx` | Number of chunks (`ceil(rows / C)`). |
| `chunk_ptr` | `std::vector<Idx>` | Offsets into column-major storage. |
| `chunk_width` | `std::vector<Idx>` | Padded width per chunk. |
| `col_idx` / `values` | `std::vector` | Column indices and values in column-major order. |
| `perm` / `inv_perm` | `std::vector<Idx>` | Row permutation and its inverse. |

---

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `print_info` | `void print_info(const char* name = "SELL") const` | Prints format statistics (chunks, nnz, fill ratio). |

---

## Conversions

| Function | Description |
|----------|-------------|
| `csr_to_ell(A)` | Converts CSR to ELL format. |
| `csr_to_sell<T, C, Idx>(A, sigma)` | Converts CSR to SELL format with chunk size `C` and sigma grouping. |

---

### SpMV Functions

### CSR SpMV

| Function | Description |
|----------|-------------|
| `spmv_csr(A, x, y)` | CSR SpMV with 8-way unrolling and prefetch. OpenMP-aware. |
| `spmv_csr_avx2(A, x, y)` | CSR SpMV with AVX2 gather + FMA (double/float only). |
| `spmv_transpose_csr(A, x, y)` | Computes `y = A^T * x`. OpenMP-aware. |

### ELL SpMV

| Function | Description |
|----------|-------------|
| `spmv_ell_avx2(E, x, y)` | ELL SpMV with AVX2 acceleration (double/float only). |

### SELL SpMV

| Function | Description |
|----------|-------------|
| `spmv_sell_avx2(S, x, y)` | SELL SpMV with AVX2 acceleration – specialized for `C=4` (double) and `C=8` (float). |

---

### Generators

| Function | Description |
|----------|-------------|
| `random_sparse<T, Idx>(rows, cols, density, seed)` | Random sparse matrix. |
| `diag_matrix<T, Idx>(n, diag_val)` | Diagonal matrix. |
| `tridiagonal<T, Idx>(n, sub, diag, super)` | Tridiagonal matrix. |
| `band_matrix<T, Idx>(n, bandwidth, diag_val, off_val)` | Banded matrix. |
| `random_spd_sparse<T, Idx>(n, density, seed)` | Random symmetric positive-definite matrix. |

---

### Util Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `spmv_max_error` | `T spmv_max_error(const CSRMatrix<T, Idx>& A, const T* x, const T* y_sparse)` | Compares SpMV result against dense reference. |
| `vec_max_abs_diff` | `T vec_max_abs_diff(const T* a, const T* b, size_t n)` | Maximum absolute difference between two vectors. |
| `vec_norm2` | `T vec_norm2(const T* x, size_t n)` | Euclidean norm of a vector. |

### Example

```cpp
#include "Matrix_MS.hpp"
#include <iostream>
#include <vector>

int main() {
    using namespace SimpFlex;
    using Real = double;
    using Idx = uint32_t;

    auto A = random_sparse<Real, Idx>(10000, 10000, 0.01);
    A.print_info("CSR A");

    std::vector<Real> x(10000, 1.0);
    std::vector<Real> y(10000);

    auto E = csr_to_ell(A);
    E.print_info("ELL A");

    std::vector<Real> y_ell(10000);
#if SFX_HAVE_AVX2_FMA
    spmv_ell_avx2(E, x.data(), y_ell.data());
    Real err_ell = vec_max_abs_diff(y.data(), y_ell.data(), 10000);
    std::cout << "ELL AVX2 error vs CSR: " << err_ell << "\n";
#endif

    auto S = csr_to_sell<Real, 4, Idx>(A, 256);
    S.print_info("SELL A");

    std::vector<Real> y_sell(10000);
#if SFX_HAVE_AVX2_FMA
    spmv_sell_avx2(S, x.data(), y_sell.data());
    Real err_sell = vec_max_abs_diff(y.data(), y_sell.data(), 10000);
    std::cout << "SELL AVX2 error vs CSR: " << err_sell << "\n";
#endif

    auto T = tridiagonal<Real, Idx>(10000);
    std::vector<Real> x_jacobi(10000, 1.0);
    std::vector<Real> b(10000, 0.0);
    std::vector<Real> x_new(10000);

    for (int iter = 0; iter < 100; ++iter) {
        spmv_csr(T, x_jacobi.data(), x_new.data());
        for (Idx i = 0; i < 10000; ++i) {
            Real lu_x = x_new[i] - 2.0 * x_jacobi[i];
            x_new[i] = (b[i] - lu_x) * 0.5;
        }
        x_jacobi.swap(x_new);
    }
    std::cout << "Jacobi solution norm: " << vec_norm2(x_jacobi.data(), 10000) << "\n";

    return 0;
}
```
