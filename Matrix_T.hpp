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

        constexpr Matrix(std::initializer_list<T> list) noexcept {
            int i = 0;
            for (auto it = list.begin(); it != list.end() && i < kSize; ++it, ++i)
                data_[i] = *it;
            for (; i < kSize; ++i) data_[i] = T(0);
        }

        explicit constexpr Matrix(const T(&arr)[kSize]) noexcept {
            for (int i = 0; i < kSize; ++i) data_[i] = arr[i];
        }

        constexpr T& operator()(int row, int col) noexcept {
            return data_[row * Cols + col];
        }
        constexpr const T& operator()(int row, int col) const noexcept {
            return data_[row * Cols + col];
        }

        constexpr T& operator[](int idx) noexcept { return data_[idx]; }
        constexpr const T& operator[](int idx) const noexcept { return data_[idx]; }

        constexpr T* data() noexcept { return data_; }
        constexpr const T* data() const noexcept { return data_; }

        static constexpr int rows() noexcept { return Rows; }
        static constexpr int cols() noexcept { return Cols; }
        static constexpr int size() noexcept { return kSize; }

        constexpr Matrix<T, 1, Cols> row(int r) const noexcept {
            Matrix<T, 1, Cols> result;
            for (int j = 0; j < Cols; ++j) result(0, j) = data_[r * Cols + j];
            return result;
        }

        constexpr Matrix<T, Rows, 1> col(int c) const noexcept {
            Matrix<T, Rows, 1> result;
            for (int i = 0; i < Rows; ++i) result(i, 0) = data_[i * Cols + c];
            return result;
        }

        constexpr Matrix operator+() const noexcept { return *this; }
        constexpr Matrix operator-() const noexcept {
            Matrix result;
            for (int i = 0; i < kSize; ++i) result.data_[i] = -data_[i];
            return result;
        }

        constexpr Matrix& operator+=(const Matrix& rhs) noexcept {
            for (int i = 0; i < kSize; ++i) data_[i] += rhs.data_[i];
            return *this;
        }
        constexpr Matrix operator+(const Matrix& rhs) const noexcept {
            Matrix tmp = *this;
            tmp += rhs;
            return tmp;
        }

        constexpr Matrix& operator-=(const Matrix& rhs) noexcept {
            for (int i = 0; i < kSize; ++i) data_[i] -= rhs.data_[i];
            return *this;
        }
        constexpr Matrix operator-(const Matrix& rhs) const noexcept {
            Matrix tmp = *this;
            tmp -= rhs;
            return tmp;
        }

        constexpr Matrix& operator*=(T scalar) noexcept {
            for (int i = 0; i < kSize; ++i) data_[i] *= scalar;
            return *this;
        }
        constexpr Matrix operator*(T scalar) const noexcept {
            Matrix tmp = *this;
            tmp *= scalar;
            return tmp;
        }

        constexpr Matrix& operator/=(T scalar) noexcept {
            for (int i = 0; i < kSize; ++i) data_[i] /= scalar;
            return *this;
        }
        constexpr Matrix operator/(T scalar) const noexcept {
            Matrix tmp = *this;
            tmp /= scalar;
            return tmp;
        }

        template <int K>
        constexpr Matrix<T, Rows, K>
            operator*(const Matrix<T, Cols, K>& rhs) const noexcept {
            Matrix<T, Rows, K> result;
            for (int i = 0; i < Rows; ++i) {
                for (int k = 0; k < Cols; ++k) {
                    T a_ik = data_[i * Cols + k];
                    for (int j = 0; j < K; ++j) {
                        result(i, j) += a_ik * rhs(k, j);
                    }
                }
            }
            return result;
        }

        // Hadamard product
        constexpr Matrix cwiseProduct(const Matrix& rhs) const noexcept {
            Matrix result;
            for (int i = 0; i < kSize; ++i)
                result.data_[i] = data_[i] * rhs.data_[i];
            return result;
        }

        constexpr Matrix cwiseQuotient(const Matrix& rhs) const noexcept {
            Matrix result;
            for (int i = 0; i < kSize; ++i)
                result.data_[i] = data_[i] / rhs.data_[i];
            return result;
        }

        constexpr bool operator==(const Matrix& rhs) const noexcept {
            for (int i = 0; i < kSize; ++i)
                if (data_[i] != rhs.data_[i]) return false;
            return true;
        }
        constexpr bool operator!=(const Matrix& rhs) const noexcept {
            return !(*this == rhs);
        }

        constexpr Matrix<T, Cols, Rows> transpose() const noexcept {
            Matrix<T, Cols, Rows> result;
            for (int i = 0; i < Rows; ++i)
                for (int j = 0; j < Cols; ++j)
                    result(j, i) = data_[i * Cols + j];
            return result;
        }

        template <int SubRows, int SubCols>
        constexpr Matrix<T, SubRows, SubCols>
            block(int start_row, int start_col) const noexcept {
            Matrix<T, SubRows, SubCols> result;
            for (int i = 0; i < SubRows; ++i)
                for (int j = 0; j < SubCols; ++j)
                    result(i, j) = data_[(start_row + i) * Cols + (start_col + j)];
            return result;
        }

        constexpr Matrix<T, (Rows < Cols ? Rows : Cols), 1> diagonal() const noexcept {
            Matrix<T, (Rows < Cols ? Rows : Cols), 1> result;
            constexpr int n = (Rows < Cols ? Rows : Cols);
            for (int i = 0; i < n; ++i)
                result(i, 0) = data_[i * Cols + i];
            return result;
        }

        template <int R = Rows, int C = Cols,
            std::enable_if_t<R == C, int> = 0>
        constexpr T trace() const noexcept {
            T sum = T(0);
            for (int i = 0; i < Rows; ++i) sum += data_[i * Cols + i];
            return sum;
        }

        template <int R = Rows, int C = Cols,
            std::enable_if_t<R == C, int> = 0>
        T determinant() const noexcept {
            Matrix< T, Rows, Cols> m = *this;
            T det = T(1);
            for (int i = 0; i < Rows; ++i) {
                int pivot = i;
                T max_val = m(i, i) < T(0) ? -m(i, i) : m(i, i);
                for (int k = i + 1; k < Rows; ++k) {
                    T val = m(k, i) < T(0) ? -m(k, i) : m(k, i);
                    if (val > max_val) { max_val = val; pivot = k; }
                }
                if (max_val == T(0)) return T(0);
                if (pivot != i) {
                    for (int j = 0; j < Cols; ++j) {
                        T tmp = m(i, j);
                        m(i, j) = m(pivot, j);
                        m(pivot, j) = tmp;
                    }
                    det = -det;
                }
                det *= m(i, i);
                for (int k = i + 1; k < Rows; ++k) {
                    T factor = m(k, i) / m(i, i);
                    for (int j = i; j < Cols; ++j) {
                        m(k, j) -= factor * m(i, j);
                    }
                }
            }
            return det;
        }

        template <int R = Rows, int C = Cols,
            std::enable_if_t<R == C, int> = 0>
        Matrix inverse() const {
            Matrix<T, Rows, Cols> m = *this;
            Matrix<T, Rows, Cols> inv = identity();

            for (int i = 0; i < Rows; ++i) {
                int pivot = i;
                T max_val = m(i, i) < T(0) ? -m(i, i) : m(i, i);
                for (int k = i + 1; k < Rows; ++k) {
                    T val = m(k, i) < T(0) ? -m(k, i) : m(k, i);
                    if (val > max_val) { max_val = val; pivot = k; }
                }
                if (max_val == T(0))
                    throw std::runtime_error("[-] Matrix is singular");
                if (pivot != i) {
                    for (int j = 0; j < Cols; ++j) {
                        T tmp = m(i, j); m(i, j) = m(pivot, j); m(pivot, j) = tmp;
                        tmp = inv(i, j); inv(i, j) = inv(pivot, j); inv(pivot, j) = tmp;
                    }
                }
                T piv = m(i, i);
                for (int j = 0; j < Cols; ++j) {
                    m(i, j) /= piv;
                    inv(i, j) /= piv;
                }
                for (int k = 0; k < Rows; ++k) {
                    if (k == i) continue;
                    T factor = m(k, i);
                    for (int j = 0; j < Cols; ++j) {
                        m(k, j) -= factor * m(i, j);
                        inv(k, j) -= factor * inv(i, j);
                    }
                }
            }
            return inv;
        }

        static constexpr Matrix zero() noexcept {
            return Matrix(T(0));
        }

        template <int R = Rows, int C = Cols,
            std::enable_if_t<R == C, int> = 0>
        static constexpr Matrix identity() noexcept {
            Matrix m;
            for (int i = 0; i < Rows; ++i)
                m.data_[i * Cols + i] = T(1);
            return m;
        }

        static constexpr Matrix constant(T value) noexcept {
            return Matrix(value);
        }

        constexpr T sum() const noexcept {
            T s = T(0);
            for (int i = 0; i < kSize; ++i) s += data_[i];
            return s;
        }

        constexpr T prod() const noexcept {
            T p = T(1);
            for (int i = 0; i < kSize; ++i) p *= data_[i];
            return p;
        }

        constexpr T minCoeff() const noexcept {
            T m = data_[0];
            for (int i = 1; i < kSize; ++i)
                if (data_[i] < m) m = data_[i];
            return m;
        }

        constexpr T maxCoeff() const noexcept {
            T m = data_[0];
            for (int i = 1; i < kSize; ++i)
                if (data_[i] > m) m = data_[i];
            return m;
        }

        T norm() const noexcept {
            T s = T(0);
            for (int i = 0; i < kSize; ++i) s += data_[i] * data_[i];
            return std::sqrt(s);
        }

        T squaredNorm() const noexcept {
            T s = T(0);
            for (int i = 0; i < kSize; ++i) s += data_[i] * data_[i];
            return s;
        }

        std::string toString() const {
            std::ostringstream oss;
            int max_w = 1;
            for (int i = 0; i < kSize; ++i) {
                std::ostringstream tmp;
                tmp << data_[i];
                if ((int)tmp.str().size() > max_w) max_w = (int)tmp.str().size();
            }
            oss << "[";
            for (int i = 0; i < Rows; ++i) {
                if (i > 0) oss << " ";
                oss << "[";
                for (int j = 0; j < Cols; ++j) {
                    if (j > 0) oss << ", ";
                    oss << std::setw(max_w) << data_[i * Cols + j];
                }
                oss << "]";
                if (i < Rows - 1) oss << "\n";
            }
            oss << "]";
            return oss.str();
        }
    };

    template <typename T, int R, int C>
    constexpr Matrix<T, R, C> operator*(T scalar, const Matrix<T, R, C>& m) noexcept {
        return m * scalar;
    }

    template <typename T> using Matrix2 = Matrix<T, 2, 2>;
    template <typename T> using Matrix3 = Matrix<T, 3, 3>;
    template <typename T> using Matrix4 = Matrix<T, 4, 4>;
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

    using Matrix2i = Matrix<int, 2, 2>;
    using Matrix3i = Matrix<int, 3, 3>;
    using Matrix4i = Matrix<int, 4, 4>;
    using Vector2i = Matrix<int, 2, 1>;
    using Vector3i = Matrix<int, 3, 1>;
    using Vector4i = Matrix<int, 4, 1>;

    template <typename T>
    struct Matrix<T, 2, 2> {
        static constexpr int kRows = 2;
        static constexpr int kCols = 2;
        static constexpr int kSize = 4;

        T data_[4];

        constexpr Matrix() noexcept : data_{} {}
        explicit constexpr Matrix(T v) noexcept : data_{ v, v, v, v } {}
        constexpr Matrix(T m00, T m01, T m10, T m11) noexcept
            : data_{ m00, m01, m10, m11 } {
        }
        constexpr Matrix(std::initializer_list<T> list) noexcept : data_{} {
            int i = 0;
            for (auto it = list.begin(); it != list.end() && i < 4; ++it, ++i)
                data_[i] = *it;
        }

        constexpr T& operator()(int r, int c) noexcept { return data_[r * 2 + c]; }
        constexpr const T& operator()(int r, int c) const noexcept { return data_[r * 2 + c]; }
        constexpr T& operator[](int i) noexcept { return data_[i]; }
        constexpr const T& operator[](int i) const noexcept { return data_[i]; }
        constexpr T* data() noexcept { return data_; }
        constexpr const T* data() const noexcept { return data_; }
        static constexpr int rows() noexcept { return 2; }
        static constexpr int cols() noexcept { return 2; }
        static constexpr int size() noexcept { return 4; }

        constexpr Matrix operator+() const noexcept { return *this; }
        constexpr Matrix operator-() const noexcept {
            return Matrix(-data_[0], -data_[1], -data_[2], -data_[3]);
        }

        constexpr Matrix& operator+=(const Matrix& r) noexcept {
            data_[0] += r.data_[0]; data_[1] += r.data_[1];
            data_[2] += r.data_[2]; data_[3] += r.data_[3];
            return *this;
        }
        constexpr Matrix operator+(const Matrix& r) const noexcept {
            return Matrix(data_[0] + r.data_[0], data_[1] + r.data_[1],
                data_[2] + r.data_[2], data_[3] + r.data_[3]);
        }

        constexpr Matrix& operator-=(const Matrix& r) noexcept {
            data_[0] -= r.data_[0]; data_[1] -= r.data_[1];
            data_[2] -= r.data_[2]; data_[3] -= r.data_[3];
            return *this;
        }
        constexpr Matrix operator-(const Matrix& r) const noexcept {
            return Matrix(data_[0] - r.data_[0], data_[1] - r.data_[1],
                data_[2] - r.data_[2], data_[3] - r.data_[3]);
        }

        constexpr Matrix& operator*=(T s) noexcept {
            data_[0] *= s; data_[1] *= s; data_[2] *= s; data_[3] *= s;
            return *this;
        }
        constexpr Matrix operator*(T s) const noexcept {
            return Matrix(data_[0] * s, data_[1] * s, data_[2] * s, data_[3] * s);
        }

        constexpr Matrix& operator/=(T s) noexcept {
            data_[0] /= s; data_[1] /= s; data_[2] /= s; data_[3] /= s;
            return *this;
        }
        constexpr Matrix operator/(T s) const noexcept {
            return Matrix(data_[0] / s, data_[1] / s, data_[2] / s, data_[3] / s);
        }

        constexpr Matrix operator*(const Matrix& r) const noexcept {
            return Matrix(
                data_[0] * r.data_[0] + data_[1] * r.data_[2],
                data_[0] * r.data_[1] + data_[1] * r.data_[3],
                data_[2] * r.data_[0] + data_[3] * r.data_[2],
                data_[2] * r.data_[1] + data_[3] * r.data_[3]
            );
        }

        constexpr Matrix<T, 2, 1> operator*(const Matrix<T, 2, 1>& v) const noexcept {
            Matrix<T, 2, 1> r;
            r[0] = data_[0] * v[0] + data_[1] * v[1];
            r[1] = data_[2] * v[0] + data_[3] * v[1];
            return r;
        }

        constexpr bool operator==(const Matrix& r) const noexcept {
            return data_[0] == r.data_[0] && data_[1] == r.data_[1]
                && data_[2] == r.data_[2] && data_[3] == r.data_[3];
        }
        constexpr bool operator!=(const Matrix& r) const noexcept { return !(*this == r); }

        constexpr Matrix transpose() const noexcept {
            return Matrix(data_[0], data_[2], data_[1], data_[3]);
        }

        constexpr T trace() const noexcept { return data_[0] + data_[3]; }

        constexpr T determinant() const noexcept {
            return data_[0] * data_[3] - data_[1] * data_[2];
        }

        Matrix inverse() const {
            T det = determinant();
            if (det == T(0)) throw std::runtime_error("[-] Singular matrix");
            T inv_det = T(1) / det;
            return Matrix(data_[3] * inv_det, -data_[1] * inv_det,
                -data_[2] * inv_det, data_[0] * inv_det);
        }

        static constexpr Matrix zero() noexcept { return Matrix(T(0)); }
        static constexpr Matrix identity() noexcept {
            return Matrix(T(1), T(0), T(0), T(1));
        }
        static constexpr Matrix constant(T v) noexcept { return Matrix(v); }

        constexpr T sum() const noexcept {
            return data_[0] + data_[1] + data_[2] + data_[3];
        }
        constexpr T prod() const noexcept {
            return data_[0] * data_[1] * data_[2] * data_[3];
        }
        constexpr T minCoeff() const noexcept {
            T m = data_[0];
            if (data_[1] < m) m = data_[1];
            if (data_[2] < m) m = data_[2];
            if (data_[3] < m) m = data_[3];
            return m;
        }
        constexpr T maxCoeff() const noexcept {
            T m = data_[0];
            if (data_[1] > m) m = data_[1];
            if (data_[2] > m) m = data_[2];
            if (data_[3] > m) m = data_[3];
            return m;
        }

        T norm() const noexcept { return std::sqrt(squaredNorm()); }
        T squaredNorm() const noexcept {
            return data_[0] * data_[0] + data_[1] * data_[1]
                + data_[2] * data_[2] + data_[3] * data_[3];
        }

        std::string toString() const {
            std::ostringstream oss;
            oss << "[[" << data_[0] << ", " << data_[1] << "]\n"
                << " [" << data_[2] << ", " << data_[3] << "]]";
            return oss.str();
        }
    };

    template <typename T>
    struct Matrix<T, 3, 3> {
        static constexpr int kRows = 3;
        static constexpr int kCols = 3;
        static constexpr int kSize = 9;

        T data_[9];

        constexpr Matrix() noexcept : data_{} {}
        explicit constexpr Matrix(T v) noexcept
            : data_{ v,v,v,v,v,v,v,v,v } {
        }
        constexpr Matrix(T m00, T m01, T m02,
            T m10, T m11, T m12,
            T m20, T m21, T m22) noexcept
            : data_{ m00,m01,m02, m10,m11,m12, m20,m21,m22 } {
        }
        constexpr Matrix(std::initializer_list<T> list) noexcept : data_{} {
            int i = 0;
            for (auto it = list.begin(); it != list.end() && i < 9; ++it, ++i)
                data_[i] = *it;
        }

        constexpr T& operator()(int r, int c) noexcept { return data_[r * 3 + c]; }
        constexpr const T& operator()(int r, int c) const noexcept { return data_[r * 3 + c]; }
        constexpr T& operator[](int i) noexcept { return data_[i]; }
        constexpr const T& operator[](int i) const noexcept { return data_[i]; }
        constexpr T* data() noexcept { return data_; }
        constexpr const T* data() const noexcept { return data_; }
        static constexpr int rows() noexcept { return 3; }
        static constexpr int cols() noexcept { return 3; }
        static constexpr int size() noexcept { return 9; }

        constexpr Matrix operator+() const noexcept { return *this; }
        constexpr Matrix operator-() const noexcept {
            Matrix r;
            for (int i = 0; i < 9; ++i) r.data_[i] = -data_[i];
            return r;
        }

        constexpr Matrix& operator+=(const Matrix& r) noexcept {
            for (int i = 0; i < 9; ++i) data_[i] += r.data_[i];
            return *this;
        }
        constexpr Matrix operator+(const Matrix& r) const noexcept {
            Matrix m;
            for (int i = 0; i < 9; ++i) m.data_[i] = data_[i] + r.data_[i];
            return m;
        }

        constexpr Matrix& operator-=(const Matrix& r) noexcept {
            for (int i = 0; i < 9; ++i) data_[i] -= r.data_[i];
            return *this;
        }
        constexpr Matrix operator-(const Matrix& r) const noexcept {
            Matrix m;
            for (int i = 0; i < 9; ++i) m.data_[i] = data_[i] - r.data_[i];
            return m;
        }

        constexpr Matrix& operator*=(T s) noexcept {
            for (int i = 0; i < 9; ++i) data_[i] *= s;
            return *this;
        }
        constexpr Matrix operator*(T s) const noexcept {
            Matrix m;
            for (int i = 0; i < 9; ++i) m.data_[i] = data_[i] * s;
            return m;
        }

        constexpr Matrix& operator/=(T s) noexcept {
            for (int i = 0; i < 9; ++i) data_[i] /= s;
            return *this;
        }
        constexpr Matrix operator/(T s) const noexcept {
            Matrix m;
            for (int i = 0; i < 9; ++i) m.data_[i] = data_[i] / s;
            return m;
        }

        constexpr Matrix operator*(const Matrix& r) const noexcept {
            return Matrix(
                data_[0] * r.data_[0] + data_[1] * r.data_[3] + data_[2] * r.data_[6],
                data_[0] * r.data_[1] + data_[1] * r.data_[4] + data_[2] * r.data_[7],
                data_[0] * r.data_[2] + data_[1] * r.data_[5] + data_[2] * r.data_[8],
                data_[3] * r.data_[0] + data_[4] * r.data_[3] + data_[5] * r.data_[6],
                data_[3] * r.data_[1] + data_[4] * r.data_[4] + data_[5] * r.data_[7],
                data_[3] * r.data_[2] + data_[4] * r.data_[5] + data_[5] * r.data_[8],
                data_[6] * r.data_[0] + data_[7] * r.data_[3] + data_[8] * r.data_[6],
                data_[6] * r.data_[1] + data_[7] * r.data_[4] + data_[8] * r.data_[7],
                data_[6] * r.data_[2] + data_[7] * r.data_[5] + data_[8] * r.data_[8]
            );
        }

        constexpr Matrix<T, 3, 1> operator*(const Matrix<T, 3, 1>& v) const noexcept {
            Matrix<T, 3, 1> r;
            r[0] = data_[0] * v[0] + data_[1] * v[1] + data_[2] * v[2];
            r[1] = data_[3] * v[0] + data_[4] * v[1] + data_[5] * v[2];
            r[2] = data_[6] * v[0] + data_[7] * v[1] + data_[8] * v[2];
            return r;
        }

        constexpr bool operator==(const Matrix& r) const noexcept {
            for (int i = 0; i < 9; ++i)
                if (data_[i] != r.data_[i]) return false;
            return true;
        }
        constexpr bool operator!=(const Matrix& r) const noexcept { return !(*this == r); }

        constexpr Matrix transpose() const noexcept {
            return Matrix(
                data_[0], data_[3], data_[6],
                data_[1], data_[4], data_[7],
                data_[2], data_[5], data_[8]
            );
        }

        constexpr T trace() const noexcept { return data_[0] + data_[4] + data_[8]; }

        constexpr T determinant() const noexcept {
            return data_[0] * (data_[4] * data_[8] - data_[5] * data_[7])
                - data_[1] * (data_[3] * data_[8] - data_[5] * data_[6])
                + data_[2] * (data_[3] * data_[7] - data_[4] * data_[6]);
        }

        Matrix inverse() const {
            T det = determinant();
            if (det == T(0)) throw std::runtime_error("Singular matrix");
            T inv_det = T(1) / det;
            return Matrix(
                (data_[4] * data_[8] - data_[5] * data_[7]) * inv_det,
                (data_[2] * data_[7] - data_[1] * data_[8]) * inv_det,
                (data_[1] * data_[5] - data_[2] * data_[4]) * inv_det,
                (data_[5] * data_[6] - data_[3] * data_[8]) * inv_det,
                (data_[0] * data_[8] - data_[2] * data_[6]) * inv_det,
                (data_[2] * data_[3] - data_[0] * data_[5]) * inv_det,
                (data_[3] * data_[7] - data_[4] * data_[6]) * inv_det,
                (data_[1] * data_[6] - data_[0] * data_[7]) * inv_det,
                (data_[0] * data_[4] - data_[1] * data_[3]) * inv_det
            );
        }

        static constexpr Matrix zero() noexcept { return Matrix(T(0)); }
        static constexpr Matrix identity() noexcept {
            return Matrix(1, 0, 0, 0, 1, 0, 0, 0, 1);
        }
        static constexpr Matrix constant(T v) noexcept { return Matrix(v); }

        constexpr T sum() const noexcept {
            T s = T(0);
            for (int i = 0; i < 9; ++i) s += data_[i];
            return s;
        }
        constexpr T prod() const noexcept {
            T p = T(1);
            for (int i = 0; i < 9; ++i) p *= data_[i];
            return p;
        }
        constexpr T minCoeff() const noexcept {
            T m = data_[0];
            for (int i = 1; i < 9; ++i) if (data_[i] < m) m = data_[i];
            return m;
        }
        constexpr T maxCoeff() const noexcept {
            T m = data_[0];
            for (int i = 1; i < 9; ++i) if (data_[i] > m) m = data_[i];
            return m;
        }

        T norm() const noexcept { return std::sqrt(squaredNorm()); }
        T squaredNorm() const noexcept {
            T s = T(0);
            for (int i = 0; i < 9; ++i) s += data_[i] * data_[i];
            return s;
        }

        std::string toString() const {
            std::ostringstream oss;
            oss << "[[" << data_[0] << ", " << data_[1] << ", " << data_[2] << "]\n"
                << " [" << data_[3] << ", " << data_[4] << ", " << data_[5] << "]\n"
                << " [" << data_[6] << ", " << data_[7] << ", " << data_[8] << "]]";
            return oss.str();
        }
    };

    template <typename T>
    struct Matrix<T, 4, 4> {
        static constexpr int kRows = 4;
        static constexpr int kCols = 4;
        static constexpr int kSize = 16;

        T data_[16];

        constexpr Matrix() noexcept : data_{} {}
        explicit constexpr Matrix(T v) noexcept : data_{} {
            for (int i = 0; i < 16; ++i) data_[i] = v;
        }
        constexpr Matrix(std::initializer_list<T> list) noexcept : data_{} {
            int i = 0;
            for (auto it = list.begin(); it != list.end() && i < 16; ++it, ++i)
                data_[i] = *it;
        }

        constexpr T& operator()(int r, int c) noexcept { return data_[r * 4 + c]; }
        constexpr const T& operator()(int r, int c) const noexcept { return data_[r * 4 + c]; }
        constexpr T& operator[](int i) noexcept { return data_[i]; }
        constexpr const T& operator[](int i) const noexcept { return data_[i]; }
        constexpr T* data() noexcept { return data_; }
        constexpr const T* data() const noexcept { return data_; }
        static constexpr int rows() noexcept { return 4; }
        static constexpr int cols() noexcept { return 4; }
        static constexpr int size() noexcept { return 16; }

        constexpr Matrix operator+() const noexcept { return *this; }
        constexpr Matrix operator-() const noexcept {
            Matrix r;
            for (int i = 0; i < 16; ++i) r.data_[i] = -data_[i];
            return r;
        }

        constexpr Matrix& operator+=(const Matrix& r) noexcept {
            for (int i = 0; i < 16; ++i) data_[i] += r.data_[i];
            return *this;
        }
        constexpr Matrix operator+(const Matrix& r) const noexcept {
            Matrix m;
            for (int i = 0; i < 16; ++i) m.data_[i] = data_[i] + r.data_[i];
            return m;
        }

        constexpr Matrix& operator-=(const Matrix& r) noexcept {
            for (int i = 0; i < 16; ++i) data_[i] -= r.data_[i];
            return *this;
        }
        constexpr Matrix operator-(const Matrix& r) const noexcept {
            Matrix m;
            for (int i = 0; i < 16; ++i) m.data_[i] = data_[i] - r.data_[i];
            return m;
        }

        constexpr Matrix& operator*=(T s) noexcept {
            for (int i = 0; i < 16; ++i) data_[i] *= s;
            return *this;
        }
        constexpr Matrix operator*(T s) const noexcept {
            Matrix m;
            for (int i = 0; i < 16; ++i) m.data_[i] = data_[i] * s;
            return m;
        }

        constexpr Matrix& operator/=(T s) noexcept {
            for (int i = 0; i < 16; ++i) data_[i] /= s;
            return *this;
        }
        constexpr Matrix operator/(T s) const noexcept {
            Matrix m;
            for (int i = 0; i < 16; ++i) m.data_[i] = data_[i] / s;
            return m;
        }

        constexpr Matrix operator*(const Matrix& r) const noexcept {
            Matrix m;
            for (int i = 0; i < 4; ++i) {
                for (int k = 0; k < 4; ++k) {
                    T a = data_[i * 4 + k];
                    m.data_[i * 4 + 0] += a * r.data_[k * 4 + 0];
                    m.data_[i * 4 + 1] += a * r.data_[k * 4 + 1];
                    m.data_[i * 4 + 2] += a * r.data_[k * 4 + 2];
                    m.data_[i * 4 + 3] += a * r.data_[k * 4 + 3];
                }
            }
            return m;
        }

        constexpr Matrix<T, 4, 1> operator*(const Matrix<T, 4, 1>& v) const noexcept {
            Matrix<T, 4, 1> r;
            r[0] = data_[0] * v[0] + data_[1] * v[1] + data_[2] * v[2] + data_[3] * v[3];
            r[1] = data_[4] * v[0] + data_[5] * v[1] + data_[6] * v[2] + data_[7] * v[3];
            r[2] = data_[8] * v[0] + data_[9] * v[1] + data_[10] * v[2] + data_[11] * v[3];
            r[3] = data_[12] * v[0] + data_[13] * v[1] + data_[14] * v[2] + data_[15] * v[3];
            return r;
        }

        constexpr bool operator==(const Matrix& r) const noexcept {
            for (int i = 0; i < 16; ++i)
                if (data_[i] != r.data_[i]) return false;
            return true;
        }
        constexpr bool operator!=(const Matrix& r) const noexcept { return !(*this == r); }

        constexpr Matrix transpose() const noexcept {
            Matrix r;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    r.data_[j * 4 + i] = data_[i * 4 + j];
            return r;
        }

        constexpr T trace() const noexcept {
            return data_[0] + data_[5] + data_[10] + data_[15];
        }

        T determinant() const noexcept {
            T m00 = data_[0], m01 = data_[1], m02 = data_[2], m03 = data_[3];
            T m10 = data_[4], m11 = data_[5], m12 = data_[6], m13 = data_[7];
            T m20 = data_[8], m21 = data_[9], m22 = data_[10], m23 = data_[11];
            T m30 = data_[12], m31 = data_[13], m32 = data_[14], m33 = data_[15];

            T a2323 = m22 * m33 - m23 * m32;
            T a1323 = m21 * m33 - m23 * m31;
            T a1223 = m21 * m32 - m22 * m31;
            T a0323 = m20 * m33 - m23 * m30;
            T a0223 = m20 * m32 - m22 * m30;
            T a0123 = m20 * m31 - m21 * m30;

            return m00 * (m11 * a2323 - m12 * a1323 + m13 * a1223)
                - m01 * (m10 * a2323 - m12 * a0323 + m13 * a0223)
                + m02 * (m10 * a1323 - m11 * a0323 + m13 * a0123)
                - m03 * (m10 * a1223 - m11 * a0223 + m12 * a0123);
        }

        Matrix inverse() const {
            Matrix<T, 4, 4> inv;
            // Gauss-Jordan with partial pivoting
            Matrix<T, 4, 4> m = *this;
            inv = identity();

            for (int i = 0; i < 4; ++i) {
                int pivot = i;
                T max_val = m(i, i) < T(0) ? -m(i, i) : m(i, i);
                for (int k = i + 1; k < 4; ++k) {
                    T val = m(k, i) < T(0) ? -m(k, i) : m(k, i);
                    if (val > max_val) { max_val = val; pivot = k; }
                }
                if (max_val == T(0)) throw std::runtime_error("[-] Singular matrix");
                if (pivot != i) {
                    for (int j = 0; j < 4; ++j) {
                        T t = m(i, j); m(i, j) = m(pivot, j); m(pivot, j) = t;
                        t = inv(i, j); inv(i, j) = inv(pivot, j); inv(pivot, j) = t;
                    }
                }
                T piv = m(i, i);
                for (int j = 0; j < 4; ++j) {
                    m(i, j) /= piv;
                    inv(i, j) /= piv;
                }
                for (int k = 0; k < 4; ++k) {
                    if (k == i) continue;
                    T f = m(k, i);
                    for (int j = 0; j < 4; ++j) {
                        m(k, j) -= f * m(i, j);
                        inv(k, j) -= f * inv(i, j);
                    }
                }
            }
            return inv;
        }

        static constexpr Matrix zero() noexcept { return Matrix(T(0)); }
        static constexpr Matrix identity() noexcept {
            Matrix m;
            m.data_[0] = m.data_[5] = m.data_[10] = m.data_[15] = T(1);
            return m;
        }
        static constexpr Matrix constant(T v) noexcept { return Matrix(v); }

        constexpr T sum() const noexcept {
            T s = T(0);
            for (int i = 0; i < 16; ++i) s += data_[i];
            return s;
        }
        constexpr T prod() const noexcept {
            T p = T(1);
            for (int i = 0; i < 16; ++i) p *= data_[i];
            return p;
        }
        constexpr T minCoeff() const noexcept {
            T m = data_[0];
            for (int i = 1; i < 16; ++i) if (data_[i] < m) m = data_[i];
            return m;
        }
        constexpr T maxCoeff() const noexcept {
            T m = data_[0];
            for (int i = 1; i < 16; ++i) if (data_[i] > m) m = data_[i];
            return m;
        }

        T norm() const noexcept { return std::sqrt(squaredNorm()); }
        T squaredNorm() const noexcept {
            T s = T(0);
            for (int i = 0; i < 16; ++i) s += data_[i] * data_[i];
            return s;
        }

        std::string toString() const {
            std::ostringstream oss;
            oss << "[";
            for (int i = 0; i < 4; ++i) {
                if (i > 0) oss << " ";
                oss << "[";
                for (int j = 0; j < 4; ++j) {
                    if (j > 0) oss << ", ";
                    oss << data_[i * 4 + j];
                }
                oss << "]";
                if (i < 3) oss << "\n";
            }
            oss << "]";
            return oss.str();
        }
    };

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

    template <typename T, int N>
    Matrix<T, N, 1> normalized(const Matrix<T, N, 1>& v) noexcept {
        T n = v.norm();
        return n == T(0) ? v : v / n;
    }

    template <typename T, int R, int C>
    std::ostream& operator<<(std::ostream& os, const Matrix<T, R, C>& m) {
        os << m.toString();
        return os;
    }
    // end
}
