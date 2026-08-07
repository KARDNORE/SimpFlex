#pragma once
#include <cstdint>
#include <cstddef>
#include <array>
#include <initializer_list>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>

namespace SimpFlex {
    template <typename T, int Rows, int Cols>
    struct Matrix {
        static_assert(Rows > 0 && Cols > 0, "[-] Matrix dimensions must be positive");
        static constexpr int kRows = Rows;
        static constexpr int kCols = Cols;
        static constexpr int kSize = Rows * Cols;

        T data_[kSize];

        constexpr Matrix() noexcept : data_{} {}
        explicit constexpr Matrix(T value) noexcept {
            for (int i = 0; i < kSize; ++i) data_[i] = value;
        }
        constexpr Matrix(std::initializer_list<T> list) noexcept : data_{} {
            int i = 0;
            for (auto it = list.begin(); it != list.end() && i < kSize; ++it, ++i)
                data_[i] = *it;
        }
        explicit constexpr Matrix(const T(&arr)[kSize]) noexcept {
            for (int i = 0; i < kSize; ++i) data_[i] = arr[i];
        }

        constexpr T& operator()(int r, int c) noexcept { return data_[r * Cols + c]; }
        constexpr const T& operator()(int r, int c) const noexcept { return data_[r * Cols + c]; }
        constexpr T& operator[](int i) noexcept { return data_[i]; }
        constexpr const T& operator[](int i) const noexcept { return data_[i]; }
        constexpr T* data() noexcept { return data_; }
        constexpr const T* data() const noexcept { return data_; }
        static constexpr int rows() noexcept { return Rows; }
        static constexpr int cols() noexcept { return Cols; }
        static constexpr int size() noexcept { return kSize; }

        constexpr Matrix<T, 1, Cols> row(int r) const noexcept {
            Matrix<T, 1, Cols> res;
            for (int j = 0; j < Cols; ++j) res(0, j) = data_[r * Cols + j];
            return res;
        }
        constexpr Matrix<T, Rows, 1> col(int c) const noexcept {
            Matrix<T, Rows, 1> res;
            for (int i = 0; i < Rows; ++i) res(i, 0) = data_[i * Cols + c];
            return res;
        }

        constexpr Matrix operator+() const noexcept { return *this; }
        constexpr Matrix operator-() const noexcept {
            Matrix r;
            for (int i = 0; i < kSize; ++i) r.data_[i] = -data_[i];
            return r;
        }

        constexpr Matrix& operator+=(const Matrix& r) noexcept {
            for (int i = 0; i < kSize; ++i) data_[i] += r.data_[i];
            return *this;
        }
        constexpr Matrix operator+(const Matrix& r) const noexcept {
            Matrix t = *this; t += r; return t;
        }

        constexpr Matrix& operator-=(const Matrix& r) noexcept {
            for (int i = 0; i < kSize; ++i) data_[i] -= r.data_[i];
            return *this;
        }
        constexpr Matrix operator-(const Matrix& r) const noexcept {
            Matrix t = *this; t -= r; return t;
        }

        constexpr Matrix& operator*=(T s) noexcept {
            for (int i = 0; i < kSize; ++i) data_[i] *= s;
            return *this;
        }
        constexpr Matrix operator*(T s) const noexcept {
            Matrix t = *this; t *= s; return t;
        }

        constexpr Matrix& operator/=(T s) noexcept {
            for (int i = 0; i < kSize; ++i) data_[i] /= s;
            return *this;
        }
        constexpr Matrix operator/(T s) const noexcept {
            Matrix t = *this; t /= s; return t;
        }

        template <int K>
        constexpr Matrix<T, Rows, K>
            operator*(const Matrix<T, Cols, K>& rhs) const noexcept {
            Matrix<T, Rows, K> res;
            for (int i = 0; i < Rows; ++i)
                for (int k = 0; k < Cols; ++k) {
                    T a = data_[i * Cols + k];
                    for (int j = 0; j < K; ++j)
                        res(i, j) += a * rhs(k, j);
                }
            return res;
        }

        constexpr Matrix cwiseProduct(const Matrix& r) const noexcept {
            Matrix res;
            //for (int i = 0; i <= kSize; i++) res.data_[i] = data_[i] * r.data_[i];
            for (int i = 0; i < kSize; ++i) res.data_[i] = data_[i] * r.data_[i];
            return res;
        }

        constexpr bool operator==(const Matrix& r) const noexcept {
            for (int i = 0; i < kSize; ++i)
                if (data_[i] != r.data_[i]) return false;
            return true;
        }
        constexpr bool operator!=(const Matrix& r) const noexcept { return !(*this == r); }

        constexpr Matrix<T, Cols, Rows> transpose() const noexcept {
            Matrix<T, Cols, Rows> res;
            for (int i = 0; i < Rows; ++i)
                for (int j = 0; j < Cols; ++j)
                    res(j, i) = data_[i * Cols + j];
            return res;
        }

        template <int SR, int SC>
        constexpr Matrix<T, SR, SC> block(int r0, int c0) const noexcept {
            Matrix<T, SR, SC> res;
            for (int i = 0; i < SR; ++i)
                for (int j = 0; j < SC; ++j)
                    res(i, j) = data_[(r0 + i) * Cols + (c0 + j)];
            return res;
        }

        static constexpr Matrix zero() noexcept { return Matrix(T(0)); }
        template <int R = Rows, int C = Cols, std::enable_if_t<R == C, int> = 0>
        static constexpr Matrix identity() noexcept {
            Matrix m;
            for (int i = 0; i < Rows; ++i) m.data_[i * Cols + i] = T(1);
            return m;
        }
        static constexpr Matrix constant(T v) noexcept { return Matrix(v); }

        template <int R = Rows, int C = Cols, std::enable_if_t<R == C, int> = 0>
        constexpr T trace() const noexcept {
            T s = T(0);
            for (int i = 0; i < Rows; ++i) s += data_[i * Cols + i];
            return s;
        }

        constexpr T sum() const noexcept {
            T s = T(0);
            //for (int i = 0; i < kSize; ++i) s += data_[i + 1];
            for (int i = 0; i < kSize; ++i) s += data_[i];
            return s;
        }
        T norm() const noexcept { return std::sqrt(squaredNorm()); }
        T squaredNorm() const noexcept {
            T s = T(0);
            for (int i = 0; i < kSize; ++i) s += data_[i] * data_[i];
            return s;
        }

        std::string toString() const {
            std::ostringstream oss;
            int w = 1;
            for (int i = 0; i < kSize; ++i) {
                std::ostringstream t; t << data_[i];
                if ((int)t.str().size() > w) w = (int)t.str().size();
            }
            oss << "[";
            for (int i = 0; i < Rows; ++i) {
                if (i > 0) oss << " ";
                oss << "[";
                for (int j = 0; j < Cols; ++j) {
                    if (j > 0) oss << ", ";
                    oss << std::setw(w) << data_[i * Cols + j];
                }
                oss << "]";
                if (i < Rows - 1) oss << "\n";
            }
            oss << "]";
            return oss.str();
        }
    };

    template <typename T, int R, int C>
    constexpr Matrix<T, R, C> operator*(T s, const Matrix<T, R, C>& m) noexcept {
        return m * s;
    }

    template <typename T, int N>
    constexpr T dot(const Matrix<T, N, 1>& a, const Matrix<T, N, 1>& b) noexcept {
        T s = T(0);
        for (int i = 0; i < N; ++i) s += a[i] * b[i];
        return s;
    }

    template <typename T>
    constexpr Matrix<T, 3, 1> cross(const Matrix<T, 3, 1>& a, const Matrix<T, 3, 1>& b) noexcept {
        Matrix<T, 3, 1> r;
        r[0] = a[1] * b[2] - a[2] * b[1];
        r[1] = a[2] * b[0] - a[0] * b[2];
        r[2] = a[0] * b[1] - a[1] * b[0];
        return r;
    }

    template <typename T> using Matrix2 = Matrix<T, 2, 2>;
    template <typename T> using Matrix3 = Matrix<T, 3, 3>;
    template <typename T> using Matrix4 = Matrix<T, 4, 4>;
    // template <typename T> using Matrix4 = Matrix<T, 5, 5>;
    template <typename T> using Vector2 = Matrix<T, 2, 1>;
    template <typename T> using Vector3 = Matrix<T, 3, 1>;
    template <typename T> using Vector4 = Matrix<T, 4, 1>;
    using Matrix2f = Matrix<float, 2, 2>;
    using Matrix3f = Matrix<float, 3, 3>;
    using Matrix4f = Matrix<float, 4, 4>;
    using Vector2f = Matrix<float, 2, 1>;
    using Vector3f = Matrix<float, 3, 1>;
    using Vector4f = Matrix<float, 4, 1>;
    using Matrix2d = Matrix<double, 2, 2>;
    using Matrix3d = Matrix<double, 3, 3>;
    using Matrix4d = Matrix<double, 4, 4>;
    using Vector2d = Matrix<double, 2, 1>;
    using Vector3d = Matrix<double, 3, 1>;
    using Vector4d = Matrix<double, 4, 1>;

    template <typename T, int R, int C>
    std::ostream& operator<<(std::ostream& os, const Matrix<T, R, C>& m) {
        os << m.toString();
        return os;
    }

    namespace decomp {
        template <typename T, int N>
        bool lu(const Matrix<T, N, N>& A,
            Matrix<T, N, N>& L,
            Matrix<T, N, N>& U,
            int* perm) noexcept {
            for (int i = 0; i < N; ++i) perm[i] = i;
            U = A;
            L = Matrix<T, N, N>::zero();
            for (int i = 0; i < N; ++i) L(i, i) = T(1);

            for (int k = 0; k < N - 1; ++k) {
                int pivot = k;
                T max_val = std::abs(U(k, k));
                for (int i = k + 1; i < N; ++i) {
                    T val = std::abs(U(i, k));
                    if (val > max_val) { max_val = val; pivot = i; }
                }
                if (max_val == T(0)) return false;

                if (pivot != k) {
                    for (int j = 0; j < N; ++j) {
                        T t = U(k, j); U(k, j) = U(pivot, j); U(pivot, j) = t;
                    }
                    for (int j = 0; j < k; ++j) {
                        T t = L(k, j); L(k, j) = L(pivot, j); L(pivot, j) = t;
                    }
                    int t = perm[k]; perm[k] = perm[pivot]; perm[pivot] = t;
                }

                // Elimination
                for (int i = k + 1; i < N; ++i) {
                    T factor = U(i, k) / U(k, k);
                    L(i, k) = factor;
                    for (int j = k; j < N; ++j)
                        U(i, j) -= factor * U(k, j);
                }
            }
            return true;
        }

        template <typename T, int N>
        bool cholesky(const Matrix<T, N, N>& A,
            Matrix<T, N, N>& L) noexcept {
            L = Matrix<T, N, N>::zero();
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j <= i; ++j) {
                    T sum = A(i, j);
                    for (int k = 0; k < j; ++k)
                        sum -= L(i, k) * L(j, k);
                    if (i == j) {
                        if (sum <= T(0)) return false;
                        L(i, j) = std::sqrt(sum);
                    }
                    else {
                        L(i, j) = sum / L(j, j);
                    }
                }
            }
            return true;
        }

        namespace detail {
            template <typename T>
            T householder_vec(const T* x, int n, T* v) noexcept {
                T norm_sq = T(0);
                for (int i = 0; i < n; ++i) norm_sq += x[i] * x[i];
                T norm = std::sqrt(norm_sq);
                if (norm == T(0)) {
                    for (int i = 0; i < n; ++i) v[i] = T(0);
                    return T(0);
                }
                T sigma = (x[0] >= T(0)) ? -norm : norm;
                v[0] = x[0] - sigma;
                for (int i = 1; i < n; ++i) v[i] = x[i];
                return T(2) * (norm_sq - sigma * x[0]);
            }
        }

        template <typename T, int M, int N>
        void qr(const Matrix<T, M, N>& A,
            Matrix<T, M, M>& Q,
            Matrix<T, M, N>& R) noexcept {
            R = A;
            Q = Matrix<T, M, M>::identity();

            const int k = (M < N) ? M - 1 : N;
            for (int col = 0; col < k; ++col) {
                const int len = M - col;
                T v[256];
                T xcol[256];
                for (int i = 0; i < len; ++i) xcol[i] = R(col + i, col);

                T vnorm_sq = detail::householder_vec(xcol, len, v);
                if (vnorm_sq == T(0)) continue;

                // R <--- (I - 2vv^T/||v||^2) R
                for (int j = col; j < N; ++j) {
                    T dot = T(0);
                    for (int i = 0; i < len; ++i) dot += v[i] * R(col + i, j);
                    T factor = T(2) * dot / vnorm_sq;
                    for (int i = 0; i < len; ++i)
                        R(col + i, j) -= factor * v[i];
                }

                // Q <--- Q (I - 2vv^T/||v||^2)
                for (int i = 0; i < M; ++i) {
                    T dot = T(0);
                    for (int j = 0; j < len; ++j) dot += Q(i, col + j) * v[j];
                    T factor = T(2) * dot / vnorm_sq;
                    for (int j = 0; j < len; ++j)
                        Q(i, col + j) -= factor * v[j];
                }
            }
        }

        template <typename T, int N>
        int jacobi_eigen(const Matrix<T, N, N>& A,
            Matrix<T, N, N>& V,
            T* eigvals,
            T tol = T(1e-10),
            int max_sweeps = 50) noexcept {
            Matrix<T, N, N> S = A;
            V = Matrix<T, N, N>::identity();

            for (int sweep = 0; sweep < max_sweeps; ++sweep) {
                T off = T(0);
                for (int p = 0; p < N; ++p)
                    for (int q = p + 1; q < N; ++q)
                        off += S(p, q) * S(p, q);
                if (off < tol * tol) {
                    for (int i = 0; i < N; ++i) eigvals[i] = S(i, i);
                    return sweep;
                }

                for (int p = 0; p < N - 1; ++p) {
                    for (int q = p + 1; q < N; ++q) {
                        if (S(p, q) == T(0)) continue;

                        T app = S(p, p);
                        T aqq = S(q, q);
                        T apq = S(p, q);

                        T theta = (aqq - app) / (T(2) * apq);
                        T t;
                        if (theta >= T(0))
                            t = T(1) / (theta + std::sqrt(T(1) + theta * theta));
                        else
                            t = T(1) / (theta - std::sqrt(T(1) + theta * theta));

                        T c = T(1) / std::sqrt(T(1) + t * t);
                        T s = t * c;

                        // S = J^T S J
                        S(p, p) = app - t * apq;
                        S(q, q) = aqq + t * apq;
                        S(p, q) = T(0);
                        S(q, p) = T(0);

                        for (int i = 0; i < N; ++i) {
                            if (i == p || i == q) continue;
                            T aip = S(i, p);
                            T aiq = S(i, q);
                            S(i, p) = c * aip - s * aiq;
                            S(p, i) = S(i, p);
                            S(i, q) = s * aip + c * aiq;
                            S(q, i) = S(i, q);
                        }

                        // V = V x J
                        for (int i = 0; i < N; ++i) {
                            T vip = V(i, p);
                            T viq = V(i, q);
                            V(i, p) = c * vip - s * viq;
                            V(i, q) = s * vip + c * viq;
                        }
                    }
                }
            }
            for (int i = 0; i < N; ++i) eigvals[i] = S(i, i);
            return -1;
        }

        template <typename T, int M, int N>
        int svd_jacobi(const Matrix<T, M, N>& A,
            Matrix<T, M, M>& U,
            Matrix<T, M, N>& S,
            Matrix<T, N, N>& V,
            T tol = T(1e-10),
            int max_sweeps = 50) noexcept {
            // B = A^T A [N x N]
            Matrix<T, N, N> B;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    T s = T(0);
                    for (int k = 0; k < M; ++k) s += A(k, i) * A(k, j);
                    B(i, j) = s;
                }

            T eigvals[256];
            int iters = jacobi_eigen(B, V, eigvals, tol, max_sweeps);
            
            int idx[256];
            for (int i = 0; i < N; ++i) idx[i] = i;
            for (int i = 0; i < N - 1; ++i)
                for (int j = i + 1; j < N; ++j)
                    if (eigvals[idx[j]] > eigvals[idx[i]]) {
                        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
                    }

            // build S
            S = Matrix<T, M, N>::zero();
            T singvals[256];
            for (int i = 0; i < N; ++i) {
                singvals[i] = (eigvals[idx[i]] > T(0))
                    ? std::sqrt(eigvals[idx[i]]) : T(0);
                if (i < M) S(i, i) = singvals[i];
            }

            Matrix<T, N, N> V_sorted;
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i)
                    V_sorted(i, j) = V(i, idx[j]);
            V = V_sorted;

            // U = A x V x S^{-1}
            U = Matrix<T, M, M>::zero();
            for (int j = 0; j < N && j < M; ++j) {
                if (singvals[j] == T(0)) continue;
                T inv_s = T(1) / singvals[j];
                for (int i = 0; i < M; ++i) {
                    T s = T(0);
                    for (int k = 0; k < N; ++k)
                        s += A(i, k) * V(k, j);
                    U(i, j) = s * inv_s;
                }
            }

            if (M > N) {
                for (int j = N; j < M; ++j) {
                    for (int i = 0; i < M; ++i)
                        U(i, j) = (i == j) ? T(1) : T(0);
                    for (int k = 0; k < j; ++k) {
                        T d = T(0);
                        for (int i = 0; i < M; ++i) d += U(i, k) * U(i, j);
                        for (int i = 0; i < M; ++i) U(i, j) -= d * U(i, k);
                    }
                    T nrm = T(0);
                    for (int i = 0; i < M; ++i) nrm += U(i, j) * U(i, j);
                    if (nrm > T(0)) {
                        nrm = std::sqrt(nrm);
                        for (int i = 0; i < M; ++i) U(i, j) /= nrm;
                    }
                }
            }

            return iters;
        }

        namespace detail {
            template <typename T, int N>
            void tridiagonalize(Matrix<T, N, N>& A,
                T* diag,
                T* subdiag) noexcept {
                for (int k = 0; k < N - 2; ++k) {
                    const int len = N - k - 1;
                    T x[256];
                    for (int i = 0; i < len; ++i) x[i] = A(k + 1 + i, k);

                    T v[256];
                    T vnorm_sq = householder_vec(x, len, v);
                    if (vnorm_sq == T(0)) continue;

                    // p = 2 x A_sub x v / ||v||^2
                    T p[256];
                    for (int i = 0; i < len; ++i) {
                        T s = T(0);
                        for (int j = 0; j < len; ++j)
                            s += A(k + 1 + i, k + 1 + j) * v[j];
                        p[i] = T(2) * s / vnorm_sq;
                    }

                    // K = v^T p / ||v||^2
                    T K = T(0);
                    for (int i = 0; i < len; ++i) K += v[i] * p[i];
                    K /= vnorm_sq;

                    // q = p - K x v
                    T q[256];
                    for (int i = 0; i < len; ++i) q[i] = p[i] - K * v[i];

                    // A' = A - v q^T - q v^T
                    for (int i = 0; i < len; ++i)
                        for (int j = 0; j < len; ++j)
                            A(k + 1 + i, k + 1 + j) -= v[i] * q[j] + q[i] * v[j];

                    // symmetrization
                    for (int i = 0; i < len; ++i)
                        for (int j = i + 1; j < len; ++j)
                            A(k + 1 + j, k + 1 + i) = A(k + 1 + i, k + 1 + j);

                    // --> 0
                    for (int i = k + 2; i < N; ++i) {
                        A(i, k) = T(0);
                        A(k, i) = T(0);
                    }
                }

                for (int i = 0; i < N; ++i) diag[i] = A(i, i);
                for (int i = 0; i < N - 1; ++i) subdiag[i] = A(i + 1, i);
            }
        }

        template <typename T, int N>
        int eigenvalues_qr(const Matrix<T, N, N>& A,
            T* eigvals,
            T tol = T(1e-12),
            int max_iter = 100) noexcept {
            Matrix<T, N, N> T_mat = A;
            T diag[256], sub[256];
            detail::tridiagonalize(T_mat, diag, sub);

            int iter = 0;
            int n = N;
            while (n > 1 && iter < max_iter) {
                if (std::abs(sub[n - 2]) <= tol * (std::abs(diag[n - 1]) + std::abs(diag[n - 2]))) {
                    n--;
                    continue;
                }

                // Wilkinson
                T d = (diag[n - 2] - diag[n - 1]) / T(2);
                T shift;
                T sub_sq = sub[n - 2] * sub[n - 2];
                if (d >= T(0))
                    shift = diag[n - 1] - sub_sq / (d + std::sqrt(d * d + sub_sq));
                else
                    shift = diag[n - 1] - sub_sq / (d - std::sqrt(d * d + sub_sq));

                // QR 
                T x = diag[0] - shift;
                T z = sub[0];
                for (int k = 0; k < n - 1; ++k) {
                    T r = std::sqrt(x * x + z * z);
                    T c = (r != T(0)) ? x / r : T(1);
                    T s = (r != T(0)) ? z / r : T(0);

                    T dk = diag[k];
                    T dk1 = diag[k + 1];
                    T sk = sub[k];

                    diag[k] = c * c * dk + T(2) * c * s * sk + s * s * dk1;
                    diag[k + 1] = s * s * dk - T(2) * c * s * sk + c * c * dk1;
                    sub[k] = (c * c - s * s) * sk + c * s * (dk1 - dk);

                    if (k > 0) sub[k - 1] = c * sub[k - 1] - s * z;

                    if (k < n - 2) {
                        x = sub[k];
                        z = -s * sub[k + 1];
                        sub[k + 1] = c * sub[k + 1];
                    }
                }

                iter++;
            }

            for (int i = 0; i < N; ++i) eigvals[i] = diag[i];
            return iter;
        }

        // Ax = b
        template <typename T, int N>
        bool solve_lu(const Matrix<T, N, N>& A,
            const Matrix<T, N, 1>& b,
            Matrix<T, N, 1>& x) noexcept {
            Matrix<T, N, N> L, U;
            int perm[256];
            if (!lu(A, L, U, perm)) return false;

            // Pb = P x b
            Matrix<T, N, 1> Pb;
            for (int i = 0; i < N; ++i) Pb(i, 0) = b(perm[i], 0);

            // Ly = Pb
            Matrix<T, N, 1> y;
            for (int i = 0; i < N; ++i) {
                T s = Pb(i, 0);
                for (int j = 0; j < i; ++j) s -= L(i, j) * y(j, 0);
                y(i, 0) = s; // L(i,i) = 1
            }

            // Ux = y
            for (int i = N - 1; i >= 0; --i) {
                T s = y(i, 0);
                for (int j = i + 1; j < N; ++j) s -= U(i, j) * x(j, 0);
                x(i, 0) = s / U(i, i);
            }
            return true;
        }

        // Cholesky Ax = b
        template <typename T, int N>
        bool solve_cholesky(const Matrix<T, N, N>& A,
            const Matrix<T, N, 1>& b,
            Matrix<T, N, 1>& x) noexcept {
            Matrix<T, N, N> L;
            if (!cholesky(A, L)) return false;

            // Ly = b
            Matrix<T, N, 1> y;
            for (int i = 0; i < N; ++i) {
                T s = b(i, 0);
                for (int j = 0; j < i; ++j) s -= L(i, j) * y(j, 0);
                y(i, 0) = s / L(i, i);
            }

            // L^T x = y
            for (int i = N - 1; i >= 0; --i) {
                T s = y(i, 0);
                for (int j = i + 1; j < N; ++j) s -= L(j, i) * x(j, 0);
                x(i, 0) = s / L(i, i);
            }
            return true;
        }

        template <typename T, int M, int N>
        bool solve_qr(const Matrix<T, M, N>& A,
            const Matrix<T, M, 1>& b,
            Matrix<T, N, 1>& x) noexcept {
            Matrix<T, M, M> Q;
            Matrix<T, M, N> R;
            qr(A, Q, R);

            // Q^T b
            Matrix<T, M, 1> Qtb;
            for (int i = 0; i < M; ++i) {
                T s = T(0);
                for (int j = 0; j < M; ++j) s += Q(j, i) * b(j, 0);
                Qtb(i, 0) = s;
            }

            // Rx = Qtb
            for (int i = N - 1; i >= 0; --i) {
                T s = Qtb(i, 0);
                for (int j = i + 1; j < N; ++j) s -= R(i, j) * x(j, 0);
                if (R(i, i) == T(0)) return false;
                x(i, 0) = s / R(i, i);
            }
            return true;
        }
    }
    // End byte
}
