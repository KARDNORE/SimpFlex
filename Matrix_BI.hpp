#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <memory>
#include <cstring>
#include <cassert>
#include <numbers>
#include <type_traits>

#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#pragma intrinsic(_mm_prefetch)
#if defined(_OPENMP)
#include <omp.h>
#endif
#endif

#if defined(__AVX2__) && defined(__FMA__)
#define SFX_HAVE_AVX2_FMA 1
#elif defined(_MSC_VER) && defined(__AVX2__)
#define SFX_HAVE_AVX2_FMA 1
#else
#define SFX_HAVE_AVX2_FMA 0
#endif

namespace SimpFlex {

    template <typename T, size_t Align = 64>
    class AlignedAllocator {
    public:
        using value_type = T;
        using size_type = size_t;
        using difference_type = ptrdiff_t;

        template <typename U>
        struct rebind { using other = AlignedAllocator<U, Align>; };

        AlignedAllocator() noexcept = default;
        template <typename U>
        AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

        [[nodiscard]] T* allocate(size_t n) {
            if (n == 0) return nullptr;
            size_t bytes = n * sizeof(T);
            size_t aligned_bytes = (bytes + Align - 1) & ~(Align - 1);
            void* p = _aligned_malloc(aligned_bytes, Align);
            if (!p) throw std::bad_alloc();
            return static_cast<T*>(p);
        }

        void deallocate(T* p, size_t) noexcept {
            if (p) _aligned_free(p);
        }

        template <typename U>
        bool operator==(const AlignedAllocator<U, Align>&) const noexcept { return true; }
        template <typename U>
        bool operator!=(const AlignedAllocator<U, Align>&) const noexcept { return false; }
    };

    template <typename T>
    class Matrix {
    public:
        size_t rows{ 0 };
        size_t cols{ 0 };
        std::vector<T, AlignedAllocator<T, 64>> data;

        Matrix() noexcept = default;

        Matrix(size_t r, size_t c)
            : rows(r), cols(c), data(r* c) {
        }

        Matrix(size_t r, size_t c, T init)
            : rows(r), cols(c), data(r* c, init) {
        }

        [[nodiscard]] size_t size()  const noexcept { return rows * cols; }
        [[nodiscard]] bool   empty() const noexcept { return size() == 0; }

        void resize(size_t r, size_t c) {
            rows = r; cols = c;
            data.resize(r * c);
        }
        void resize(size_t r, size_t c, T v) {
            rows = r; cols = c;
            data.resize(r * c, v);
        }

        [[nodiscard]] T* ptr()       noexcept { return data.data(); }
        [[nodiscard]] const T* ptr() const noexcept { return data.data(); }

        [[nodiscard]] T* operator[](size_t i)       noexcept { return data.data() + i * cols; }
        [[nodiscard]] const T* operator[](size_t i) const noexcept { return data.data() + i * cols; }

        [[nodiscard]] T& operator()(size_t i, size_t j) {
            assert(i < rows && j < cols);
            return data[i * cols + j];
        }
        [[nodiscard]] const T& operator()(size_t i, size_t j) const {
            assert(i < rows && j < cols);
            return data[i * cols + j];
        }

        Matrix& fill(T v) {
            std::fill(data.begin(), data.end(), v);
            return *this;
        }

        [[nodiscard]] Matrix transpose() const;

        static Matrix eye(size_t n) {
            Matrix m(n, n, T(0));
            for (size_t i = 0; i < n; ++i) m(i, i) = T(1);
            return m;
        }
    };

    namespace detail {
        // MC: L2 Cahce
        // KC: L1 Cache
        // NC: L3 Cache
        // MR x NR: Micro Kernel
        constexpr size_t GEMM_MC = 256;
        constexpr size_t GEMM_KC = 256;
        //constexpr size_t GEMM_NC = 2048;
        constexpr size_t GEMM_NC = 4096;
        constexpr size_t GEMM_MR = 8;
        constexpr size_t GEMM_NR = 6;
#if SFX_HAVE_AVX2_FMA

        inline void gemm_micro_kernel_8x6(
            size_t k,
            const double* __restrict A,
            const double* __restrict B,
            double* __restrict C, size_t ldc)
        {
            __m256d c00 = _mm256_setzero_pd();  // col 0, rows 0-3
            __m256d c10 = _mm256_setzero_pd();  // col 0, rows 4-7
            __m256d c01 = _mm256_setzero_pd();
            __m256d c11 = _mm256_setzero_pd();
            __m256d c02 = _mm256_setzero_pd();
            __m256d c12 = _mm256_setzero_pd();
            __m256d c03 = _mm256_setzero_pd();
            __m256d c13 = _mm256_setzero_pd();
            __m256d c04 = _mm256_setzero_pd();
            __m256d c14 = _mm256_setzero_pd();
            __m256d c05 = _mm256_setzero_pd();
            __m256d c15 = _mm256_setzero_pd();

            for (size_t p = 0; p < k; ++p) {
                __m256d a0 = _mm256_load_pd(A + p * 8);
                __m256d a1 = _mm256_load_pd(A + p * 8 + 4);

                __m256d b0 = _mm256_broadcast_sd(B + p * 6 + 0);
                __m256d b1 = _mm256_broadcast_sd(B + p * 6 + 1);
                __m256d b2 = _mm256_broadcast_sd(B + p * 6 + 2);
                __m256d b3 = _mm256_broadcast_sd(B + p * 6 + 3);
                __m256d b4 = _mm256_broadcast_sd(B + p * 6 + 4);
                __m256d b5 = _mm256_broadcast_sd(B + p * 6 + 5);

                c00 = _mm256_fmadd_pd(a0, b0, c00);
                c10 = _mm256_fmadd_pd(a1, b0, c10);
                c01 = _mm256_fmadd_pd(a0, b1, c01);
                c11 = _mm256_fmadd_pd(a1, b1, c11);
                c02 = _mm256_fmadd_pd(a0, b2, c02);
                c12 = _mm256_fmadd_pd(a1, b2, c12);
                c03 = _mm256_fmadd_pd(a0, b3, c03);
                c13 = _mm256_fmadd_pd(a1, b3, c13);
                c04 = _mm256_fmadd_pd(a0, b4, c04);
                c14 = _mm256_fmadd_pd(a1, b4, c14);
                c05 = _mm256_fmadd_pd(a0, b5, c05);
                c15 = _mm256_fmadd_pd(a1, b5, c15);
                //c15 = _mm256_fmadd_pd(a0, b5, c15);
            }

            auto store_col = [&](size_t j, __m256d v0, __m256d v1) {
                double* col = C + j;
                __m256d cur0 = _mm256_loadu_pd(col + 0 * ldc);
                __m256d cur1 = _mm256_loadu_pd(col + 4 * ldc);
                cur0 = _mm256_add_pd(cur0, v0);
                cur1 = _mm256_add_pd(cur1, v1);
                _mm256_storeu_pd(col + 0 * ldc, cur0);
                _mm256_storeu_pd(col + 4 * ldc, cur1);
                };

            store_col(0, c00, c10);
            store_col(1, c01, c11);
            store_col(2, c02, c12);
            store_col(3, c03, c13);
            store_col(4, c04, c14);
            store_col(5, c05, c15);
        }

        inline void gemm_micro_kernel_8x12(
            size_t k,
            const float* __restrict A,
            const float* __restrict B,
            float* __restrict C, size_t ldc)
        {
            __m256 c0[12], c1[12];
            for (int j = 0; j < 12; ++j) {
                c0[j] = _mm256_setzero_ps();
                c1[j] = _mm256_setzero_ps();
            }
            for (size_t p = 0; p < k; ++p) {
                __m256 a0 = _mm256_load_ps(A + p * 8);
                __m256 a1 = _mm256_load_ps(A + p * 8 + 8);
                for (int j = 0; j < 12; ++j) {
                    __m256 b = _mm256_broadcast_ss(B + p * 12 + j);
                    c0[j] = _mm256_fmadd_ps(a0, b, c0[j]);
                    c1[j] = _mm256_fmadd_ps(a1, b, c1[j]);
                }
            }
            for (int j = 0; j < 12; ++j) {
                float* col = C + j;
                __m256 cur0 = _mm256_loadu_ps(col + 0 * ldc);
                __m256 cur1 = _mm256_loadu_ps(col + 8 * ldc);
                cur0 = _mm256_add_ps(cur0, c0[j]);
                cur1 = _mm256_add_ps(cur1, c1[j]);
                _mm256_storeu_ps(col + 0 * ldc, cur0);
                _mm256_storeu_ps(col + 8 * ldc, cur1);
            }
        }

#endif // SFX_HAVE_AVX2_FMA

        template <typename T>
        inline void gemm_micro_kernel_scalar(
            size_t MR, size_t NR, size_t k,
            const T* __restrict A,   // MR x k, packed column-major within block
            const T* __restrict B,   // k x NR, packed
            T* __restrict C, size_t ldc)
        {
            for (size_t i = 0; i < MR; ++i) {
                for (size_t j = 0; j < NR; ++j) {
                    T sum = T(0);
                    for (size_t p = 0; p < k; ++p) {
                        sum += A[p * MR + i] * B[p * NR + j];
                    }
                    C[i * ldc + j] += sum;
                }
            }
        }

        template <typename T>
        void pack_A(size_t m, size_t k,
            const T* __restrict A, size_t lda,
            T* __restrict A_pack)
        {
            constexpr size_t MR = (std::is_same_v<T, double>) ? GEMM_MR : 8;
            size_t i = 0;
            for (; i + MR <= m; i += MR) {
                for (size_t p = 0; p < k; ++p) {
                    for (size_t ii = 0; ii < MR; ++ii) {
                        A_pack[p * MR + ii] = A[(i + ii) * lda + p];
                    }
                }
                A_pack += k * MR;
            }
            // line
            if (i < m) {
                size_t m_rem = m - i;
                for (size_t p = 0; p < k; ++p) {
                    size_t ii = 0;
                    for (; ii < m_rem; ++ii)
                        A_pack[p * MR + ii] = A[(i + ii) * lda + p];
                    for (; ii < MR; ++ii)
                        A_pack[p * MR + ii] = T(0);
                }
            }
        }

        template <typename T>
        void pack_B(size_t k, size_t n,
            const T* __restrict B, size_t ldb,
            T* __restrict B_pack)
        {
            constexpr size_t NR = (std::is_same_v<T, double>) ? GEMM_NR : 12;
            size_t j = 0;
            for (; j + NR <= n; j += NR) {
                for (size_t p = 0; p < k; ++p) {
                    for (size_t jj = 0; jj < NR; ++jj) {
                        B_pack[p * NR + jj] = B[p * ldb + j + jj];
                    }
                }
                B_pack += k * NR;
            }
            if (j < n) {
                size_t n_rem = n - j;
                for (size_t p = 0; p < k; ++p) {
                    size_t jj = 0;
                    for (; jj < n_rem; ++jj)
                        B_pack[p * NR + jj] = B[p * ldb + j + jj];
                    for (; jj < NR; ++jj)
                        B_pack[p * NR + jj] = T(0);
                }
            }
        }

        template <typename T>
        inline void gemm_micro_kernel_dispatch(
            size_t k,
            const T* __restrict A,
            const T* __restrict B,
            T* __restrict C, size_t ldc)
        {
            if constexpr (std::is_same_v<T, double>) {
#if SFX_HAVE_AVX2_FMA
                gemm_micro_kernel_8x6(k, A, B, C, ldc);
#else
                gemm_micro_kernel_scalar(GEMM_MR, GEMM_NR, k, A, B, C, ldc);
#endif
            }
            else {
#if SFX_HAVE_AVX2_FMA
                gemm_micro_kernel_8x12(k, A, B, C, ldc);
#else
                gemm_micro_kernel_scalar(8, 12, k, A, B, C, ldc);
#endif
            }
        }

        template <typename T>
        void gebp(size_t mc, size_t nc, size_t kc,
            const T* __restrict A_pack,   // mc x kc, packed
            const T* __restrict B_pack,   // kc x nc, packed
            T* __restrict C, size_t ldc)
        {
            constexpr size_t MR = (std::is_same_v<T, double>) ? GEMM_MR : 8;
            constexpr size_t NR = (std::is_same_v<T, double>) ? GEMM_NR : 12;

            size_t i = 0;
            for (; i + MR <= mc; i += MR) {
                size_t j = 0;
                for (; j + NR <= nc; j += NR) {
                    gemm_micro_kernel_dispatch<T>(
                        kc,
                        A_pack + i * kc,
                        B_pack + j * kc,
                        C + i * ldc + j,
                        ldc);
                }
                if (j < nc) {
                    size_t n_rem = nc - j;
                    const T* Ap = A_pack + i * kc;
                    const T* Bp = B_pack + j * kc;
                    // T* Ap = A_pack + i * kc;
                    // T* Bp = B_pack + j * kc;
                    T* Cp = C + i * ldc + j;
                    for (size_t ii = 0; ii < MR; ++ii) {
                        for (size_t jj = 0; jj < n_rem; ++jj) {
                            T sum = T(0);
                            for (size_t p = 0; p < kc; ++p) {
                                sum += Ap[p * MR + ii] * Bp[p * NR + jj];
                            }
                            Cp[ii * ldc + jj] += sum;
                        }
                    }
                }
            }
            if (i < mc) {
                size_t m_rem = mc - i;
                size_t j = 0;
                for (; j + NR <= nc; j += NR) {
                    const T* Ap = A_pack + i * kc;
                    const T* Bp = B_pack + j * kc;
                    T* Cp = C + i * ldc + j;
                    for (size_t ii = 0; ii < m_rem; ++ii) {
                        for (size_t jj = 0; jj < NR; ++jj) {
                            T sum = T(0);
                            for (size_t p = 0; p < kc; ++p) {
                                sum += Ap[p * MR + ii] * Bp[p * NR + jj];
                            }
                            Cp[ii * ldc + jj] += sum;
                        }
                    }
                }
                if (j < nc) {
                    size_t n_rem = nc - j;
                    const T* Ap = A_pack + i * kc;
                    const T* Bp = B_pack + j * kc;
                    T* Cp = C + i * ldc + j;
                    for (size_t ii = 0; ii < m_rem; ++ii) {
                        for (size_t jj = 0; jj < n_rem; ++jj) {
                            T sum = T(0);
                            for (size_t p = 0; p < kc; ++p) {
                                sum += Ap[p * MR + ii] * Bp[p * NR + jj];
                            }
                            Cp[ii * ldc + jj] += sum;
                        }
                    }
                }
            }
        }

    }

    template <typename T>
    [[nodiscard]] Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B)
    {
        if (A.cols != B.rows)
            throw std::invalid_argument("[-] matmul: dimension mismatch");

        const size_t M = A.rows;
        const size_t N = B.cols;
        const size_t K = A.cols;

        Matrix<T> C(M, N, T(0));
        if (M == 0 || N == 0 || K == 0) return C;

        constexpr size_t MC = detail::GEMM_MC;
        constexpr size_t KC = detail::GEMM_KC;
        constexpr size_t NC = detail::GEMM_NC;
        constexpr size_t MR = (std::is_same_v<T, double>) ? detail::GEMM_MR : 8;
        constexpr size_t NR = (std::is_same_v<T, double>) ? detail::GEMM_NR : 12;

        // A_pack: MC x KC
        // B_pack: KC x NC
        const size_t a_pack_size = MC * KC;
        const size_t b_pack_size = KC * ((NC + NR - 1) / NR * NR);

        int nthreads = 1;
#if defined(_OPENMP)
        nthreads = omp_get_max_threads();
#endif

        for (size_t jc = 0; jc < N; jc += NC) {
            size_t nc = std::min(NC, N - jc);

            for (size_t pc = 0; pc < K; pc += KC) {
                size_t kc = std::min(KC, K - pc);

                std::vector<T, AlignedAllocator<T, 64>> B_pack(kc * ((nc + NR - 1) / NR * NR));
                detail::pack_B<T>(kc, nc, B.ptr() + pc * B.cols + jc, B.cols, B_pack.data());

#if defined(_OPENMP)
#pragma omp parallel
#endif
                {
                    std::vector<T, AlignedAllocator<T, 64>> A_pack(a_pack_size);

#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 1)
#endif
                    for (int64_t ic_int = 0; ic_int < (int64_t)M; ic_int += (int64_t)MC) {
                        size_t ic = (size_t)ic_int;
                        size_t mc = std::min(MC, M - ic);

                        // A_pack (mc x kc)
                        detail::pack_A<T>(mc, kc,
                            A.ptr() + ic * A.cols + pc, A.cols,
                            A_pack.data());

                        // GEBP: C(ic:ic+mc, jc:jc+nc) += A_pack x B_pack
                        detail::gebp<T>(mc, nc, kc,
                            A_pack.data(),
                            B_pack.data(),
                            C.ptr() + ic * C.cols + jc, C.cols);
                    }
                }
            }
        }
        return C;
    }

    template <typename T>
    Matrix<T> Matrix<T>::transpose() const
    {
        Matrix out(cols, rows);
        constexpr size_t BLK = 64;
        const T* src = data.data();
        T* dst = out.data.data();

        for (size_t i = 0; i < rows; i += BLK) {
            size_t ie = std::min(i + BLK, rows);
            for (size_t j = 0; j < cols; j += BLK) {
                size_t je = std::min(j + BLK, cols);
                for (size_t ii = i; ii < ie; ++ii) {
                    const T* srow = src + ii * cols + j;
                    for (size_t jj = j; jj < je; ++jj) {
                        dst[jj * rows + ii] = srow[jj - j];
                    }
                }
            }
        }
        return out;
    }

    template <typename T>
    [[nodiscard]] Matrix<T> operator+(const Matrix<T>& a, const Matrix<T>& b)
    {
        if (a.rows != b.rows || a.cols != b.cols)
            throw std::invalid_argument("[-] operator+: dimension mismatch");
        Matrix<T> r(a.rows, a.cols);
        const size_t n = a.size();
        const T* pa = a.ptr(); const T* pb = b.ptr(); T* pr = r.ptr();
        size_t i = 0;
#if SFX_HAVE_AVX2_FMA
        if constexpr (std::is_same_v<T, double>) {
            for (; i + 4 <= n; i += 4) {
                __m256d va = _mm256_loadu_pd(pa + i);
                __m256d vb = _mm256_loadu_pd(pb + i);
                _mm256_storeu_pd(pr + i, _mm256_add_pd(va, vb));
            }
        }
        else {
            for (; i + 8 <= n; i += 8) {
                __m256 va = _mm256_loadu_ps(pa + i);
                __m256 vb = _mm256_loadu_ps(pb + i);
                _mm256_storeu_ps(pr + i, _mm256_add_ps(va, vb));
            }
        }
#endif
        for (; i < n; ++i) pr[i] = pa[i] + pb[i];
        return r;
    }

    template <typename T>
    [[nodiscard]] Matrix<T> operator-(const Matrix<T>& a, const Matrix<T>& b)
    {
        if (a.rows != b.rows || a.cols != b.cols)
            throw std::invalid_argument("[-] operator-: dimension mismatch");
        Matrix<T> r(a.rows, a.cols);
        const size_t n = a.size();
        const T* pa = a.ptr(); const T* pb = b.ptr(); T* pr = r.ptr();
        size_t i = 0;
#if SFX_HAVE_AVX2_FMA
        if constexpr (std::is_same_v<T, double>) {
            for (; i + 4 <= n; i += 4) {
                __m256d va = _mm256_loadu_pd(pa + i);
                __m256d vb = _mm256_loadu_pd(pb + i);
                _mm256_storeu_pd(pr + i, _mm256_sub_pd(va, vb));
            }
        }
        else {
            for (; i + 8 <= n; i += 8) {
                __m256 va = _mm256_loadu_ps(pa + i);
                __m256 vb = _mm256_loadu_ps(pb + i);
                _mm256_storeu_ps(pr + i, _mm256_sub_ps(va, vb));
            }
        }
#endif
        for (; i < n; ++i) pr[i] = pa[i] - pb[i];
        return r;
    }

    template <typename T>
    [[nodiscard]] Matrix<T> operator*(const Matrix<T>& m, T s)
    {
        Matrix<T> r(m.rows, m.cols);
        const size_t n = m.size();
        const T* pm = m.ptr(); T* pr = r.ptr();
        size_t i = 0;
#if SFX_HAVE_AVX2_FMA
        if constexpr (std::is_same_v<T, double>) {
            __m256d vs = _mm256_set1_pd(s);
            for (; i + 4 <= n; i += 4) {
                __m256d vm = _mm256_loadu_pd(pm + i);
                _mm256_storeu_pd(pr + i, _mm256_mul_pd(vm, vs));
            }
        }
        else {
            __m256 vs = _mm256_set1_ps(s);
            for (; i + 8 <= n; i += 8) {
                __m256 vm = _mm256_loadu_ps(pm + i);
                _mm256_storeu_ps(pr + i, _mm256_mul_ps(vm, vs));
            }
        }
#endif
        for (; i < n; ++i) pr[i] = pm[i] * s;
        return r;
    }

    struct LUResult {
        std::vector<size_t> pivots;
        int info{ 0 };  // 0=ok, >0 = 1-based
    };

    template <typename T>
    LUResult lu_decomp_inplace(Matrix<T>& A)
    {
        LUResult res;
        if (A.rows != A.cols)
            throw std::invalid_argument("[-] LU: square only");
        const size_t n = A.rows;
        res.pivots.resize(n);
        for (size_t i = 0; i < n; ++i) res.pivots[i] = i;

        T* a = A.ptr();
        const size_t lda = A.cols;

        for (size_t k = 0; k < n; ++k) {
            size_t piv = k;
            T max_val = std::abs(a[k * lda + k]);
            for (size_t i = k + 1; i < n; ++i) {
                T v = std::abs(a[i * lda + k]);
                if (v > max_val) { max_val = v; piv = i; }
            }
            if (max_val < std::numeric_limits<T>::epsilon() * T(10)) {
                res.info = (int)(k + 1);
                return res;
            }
            if (piv != k) {
                std::swap(res.pivots[k], res.pivots[piv]);
                for (size_t j = 0; j < n; ++j)
                    std::swap(a[k * lda + j], a[piv * lda + j]);
            }

            T pivot = a[k * lda + k];
            T inv_pivot = T(1) / pivot;

            size_t i = k + 1;
#if SFX_HAVE_AVX2_FMA
            if constexpr (std::is_same_v<T, double>) {
                __m256d vinv = _mm256_set1_pd(inv_pivot);
                for (; i + 4 <= n; i += 4) {
                    __m256d v = _mm256_loadu_pd(&a[i * lda + k]);
                    _mm256_storeu_pd(&a[i * lda + k], _mm256_mul_pd(v, vinv));
                }
            }
#endif
            for (; i < n; ++i) {
                a[i * lda + k] *= inv_pivot;
            }

            // Schur
            constexpr size_t BLK = 64;
            for (size_t jj = k + 1; jj < n; jj += BLK) {
                size_t je = std::min(jj + BLK, n);
                for (size_t ii = k + 1; ii < n; ++ii) {
                    T factor = a[ii * lda + k];
                    T* dst = a + ii * lda + jj;
                    const T* src = a + k * lda + jj;
                    size_t j = jj;
#if SFX_HAVE_AVX2_FMA
                    if constexpr (std::is_same_v<T, double>) {
                        __m256d vf = _mm256_set1_pd(factor);
                        for (; j + 4 <= je; j += 4) {
                            __m256d vd = _mm256_loadu_pd(dst + (j - jj));
                            __m256d vs = _mm256_loadu_pd(src + (j - jj));
                            _mm256_storeu_pd(dst + (j - jj),
                                _mm256_fnmadd_pd(vf, vs, vd));
                        }
                    }
#endif
                    for (; j < je; ++j) {
                        dst[j - jj] -= factor * src[j - jj];
                    }
                }
            }
        }
        return res;
    }

    template <typename T>
    [[nodiscard]] std::pair<Matrix<T>, LUResult> lu_decomp(const Matrix<T>& A)
    {
        Matrix<T> LU = A;
        auto res = lu_decomp_inplace(LU);
        return { std::move(LU), res };
    }

    template <typename T>
    Matrix<T> cholesky(const Matrix<T>& A)
    {
        if (A.rows != A.cols)
            throw std::invalid_argument("[-] Cholesky: square only");
        const size_t n = A.rows;
        Matrix<T> L(n, n, T(0));
        const T* a = A.ptr();
        T* l = L.ptr();
        const size_t lda = A.cols;
        const size_t ldl = L.cols;

        for (size_t j = 0; j < n; ++j) {
            // dot = L(j, 0..j-1) · L(j, 0..j-1)
            T dot = T(0);
            for (size_t k = 0; k < j; ++k)
                dot += l[j * ldl + k] * l[j * ldl + k];
            T d = a[j * lda + j] - dot;
            if (d <= T(0))
                throw std::runtime_error("[-] Cholesky: not positive definite");
            l[j * ldl + j] = std::sqrt(d);
            T inv_diag = T(1) / l[j * ldl + j];

            // L(i, j) = (A(i,j) - sum_k L(i,k) x L(j,k)) / L(j,j)
            for (size_t i = j + 1; i < n; ++i) {
                T s = T(0);
                size_t k = 0;
#if SFX_HAVE_AVX2_FMA
                if constexpr (std::is_same_v<T, double>) {
                    __m256d vs = _mm256_setzero_pd();
                    for (; k + 4 <= j; k += 4) {
                        __m256d vi = _mm256_loadu_pd(l + i * ldl + k);
                        __m256d vj = _mm256_loadu_pd(l + j * ldl + k);
                        vs = _mm256_fmadd_pd(vi, vj, vs);
                    }
                    __m128d lo = _mm256_castpd256_pd128(vs);
                    __m128d hi = _mm256_extractf128_pd(vs, 1);
                    lo = _mm_add_pd(lo, hi);
                    lo = _mm_hadd_pd(lo, lo);
                    s += _mm_cvtsd_f64(lo);
                }
#endif
                for (; k < j; ++k)
                    s += l[i * ldl + k] * l[j * ldl + k];
                l[i * ldl + j] = (a[i * lda + j] - s) * inv_diag;
            }
        }
        return L;
    }

    struct QRResult {
        Matrix<double> QR;
        std::vector<double> tau;
        int info{ 0 };
    };

    inline QRResult qr_decomp(const Matrix<double>& A)
    {
        QRResult res;
        const size_t m = A.rows;
        const size_t n = A.cols;
        res.QR = A;
        res.tau.resize(std::min(m, n));
        double* a = res.QR.ptr();
        const size_t lda = res.QR.cols;

        const size_t k_max = std::min(m, n);
        for (size_t k = 0; k < k_max; ++k) {
            double norm = 0.0;
            for (size_t i = k; i < m; ++i)
                norm += a[i * lda + k] * a[i * lda + k];
            norm = std::sqrt(norm);
            if (norm < 1e-300) {
                res.tau[k] = 0.0;
                res.info = (int)(k + 1);
                continue;
            }
            double alpha = -std::copysign(norm, a[k * lda + k]);
            res.tau[k] = (alpha - a[k * lda + k]) / alpha;
            double inv_beta = 1.0 / (a[k * lda + k] - alpha);
            a[k * lda + k] -= alpha;

            // v (v[0]=1)
            for (size_t i = k + 1; i < m; ++i)
                a[i * lda + k] *= inv_beta;

            // A(k:m, k+1:n) -= tau x v x (v^T x A(k:m, k+1:n))
            for (size_t j = k + 1; j < n; ++j) {
                double dot = 0.0;
                for (size_t i = k; i < m; ++i)
                    dot += a[i * lda + k] * a[i * lda + j];
                dot *= res.tau[k];
                for (size_t i = k; i < m; ++i)
                    a[i * lda + j] -= dot * a[i * lda + k];
            }
        }
        return res;
    }

    struct EigResult {
        std::vector<double> eigenvalues;
        Matrix<double> eigenvectors;
        int iterations{ 0 };
        bool converged{ false };
    };

    inline EigResult eig_sym(Matrix<double> A, double tol = 1e-12, int max_iter = 1000)
    {
        EigResult res;
        if (A.rows != A.cols)
            throw std::invalid_argument("[-] eig_sym: square only");
        const size_t n = A.rows;
        res.eigenvalues.resize(n);
        res.eigenvectors = Matrix<double>::eye(n);
        double* a = A.ptr();
        double* v = res.eigenvectors.ptr();
        const size_t lda = A.cols;
        const size_t ldv = res.eigenvectors.cols;

        for (int iter = 0; iter < max_iter; ++iter) {
            double off = 0.0;
            size_t p = 0, q = 1;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    double v_ij = std::abs(a[i * lda + j]);
                    if (v_ij > off) { off = v_ij; p = i; q = j; }
                }
            }
            if (off < tol) {
                res.converged = true;
                res.iterations = iter;
                break;
            }

            double app = a[p * lda + p];
            double aqq = a[q * lda + q];
            double apq = a[p * lda + q];

            double theta = (aqq - app) / (2.0 * apq);
            double t;
            if (std::abs(theta) > 1e10) {
                t = 1.0 / (2.0 * theta);
            }
            else {
                t = std::copysign(1.0, theta) / (std::abs(theta) + std::sqrt(1.0 + theta * theta));
            }
            double c = 1.0 / std::sqrt(1.0 + t * t);
            double s = t * c;

            a[p * lda + p] = app - t * apq;
            a[q * lda + q] = aqq + t * apq;
            a[p * lda + q] = 0.0;
            a[q * lda + p] = 0.0;

            for (size_t i = 0; i < n; ++i) {
                if (i != p && i != q) {
                    double aip = a[i * lda + p];
                    double aiq = a[i * lda + q];
                    a[i * lda + p] = c * aip - s * aiq;
                    a[p * lda + i] = a[i * lda + p];
                    a[i * lda + q] = s * aip + c * aiq;
                    a[q * lda + i] = a[i * lda + q];
                }
            }
            for (size_t i = 0; i < n; ++i) {
                double vip = v[i * ldv + p];
                double viq = v[i * ldv + q];
                v[i * ldv + p] = c * vip - s * viq;
                v[i * ldv + q] = s * vip + c * viq;
            }
            res.iterations = iter + 1;
        }
        for (size_t i = 0; i < n; ++i)
            res.eigenvalues[i] = a[i * lda + i];

        std::vector<size_t> idx(n);
        for (size_t i = 0; i < n; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
            return res.eigenvalues[x] > res.eigenvalues[y];
            });
        std::vector<double> sorted_vals(n);
        Matrix<double> sorted_vecs(n, n);
        for (size_t i = 0; i < n; ++i) {
            sorted_vals[i] = res.eigenvalues[idx[i]];
            for (size_t j = 0; j < n; ++j)
                sorted_vecs(j, i) = v[j * ldv + idx[i]];
        }
        res.eigenvalues = std::move(sorted_vals);
        res.eigenvectors = std::move(sorted_vecs);
        return res;
    }

    struct SVDResult {
        Matrix<double> U;
        std::vector<double> S;
        Matrix<double> Vt;
    };

    inline SVDResult svd(const Matrix<double>& A)
    {
        SVDResult res;
        const size_t m = A.rows;
        const size_t n = A.cols;
        const size_t k = std::min(m, n);

        // AtA = A^T x A  (n x n)
        auto At = A.transpose();
        auto AtA = matmul(At, A);

        auto eig = eig_sym(AtA);
        res.S.resize(k);
        res.Vt = Matrix<double>(k, n);
        for (size_t i = 0; i < k; ++i) {
            double val = eig.eigenvalues[i];
            res.S[i] = val > 0 ? std::sqrt(val) : 0.0;
            for (size_t j = 0; j < n; ++j)
                res.Vt(i, j) = eig.eigenvectors(j, i);
        }

        // U = A x V x S^{-1}
        Matrix<double> V(n, k);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < k; ++j)
                V(i, j) = res.Vt(j, i);

        auto AV = matmul(A, V);
        res.U = Matrix<double>(m, k);
        for (size_t j = 0; j < k; ++j) {
            double inv_s = (res.S[j] > 1e-15) ? (1.0 / res.S[j]) : 0.0;
            for (size_t i = 0; i < m; ++i)
                res.U(i, j) = AV(i, j) * inv_s;
        }
        return res;
    }

    template <typename T>
    void print(const Matrix<T>& m, size_t max_rows = 10, size_t max_cols = 8)
    {
        size_t r = std::min(m.rows, max_rows);
        size_t c = std::min(m.cols, max_cols);
        std::cout << "Matrix(" << m.rows << "x" << m.cols << ")\n";
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j) {
                std::cout << std::setw(12) << std::fixed << std::setprecision(4) << m(i, j);
            }
            if (m.cols > max_cols) std::cout << " ...";
            std::cout << '\n';
        }
        if (m.rows > max_rows)
            std::cout << "... (" << (m.rows - max_rows) << " more rows)\n";
    }

    template <typename T>
    [[nodiscard]] Matrix<T> rand_matrix(size_t r, size_t c, T lo = T(-1), T hi = T(1))
    {
        std::mt19937_64 rng{ 12345ULL };
        std::uniform_real_distribution<T> dist(lo, hi);
        Matrix<T> m(r, c);
        for (auto& v : m.data) v = dist(rng);
        return m;
    }

    template <typename T>
    [[nodiscard]] Matrix<T> rand_spd(size_t n)
    {
        auto X = rand_matrix<T>(n, n);
        return matmul(X.transpose(), X) + Matrix<T>::eye(n) * T(n * T(0.1));
    }

    template <typename T>
    [[nodiscard]] T max_abs_diff(const Matrix<T>& a, const Matrix<T>& b)
    {
        if (a.rows != b.rows || a.cols != b.cols)
            return std::numeric_limits<T>::max();
        T err = T(0);
        size_t n = a.size();
        const T* pa = a.ptr(); const T* pb = b.ptr();
        for (size_t i = 0; i < n; ++i)
            err = std::max(err, std::abs(pa[i] - pb[i]));
        return err;
    }

    template <typename T>
    [[nodiscard]] T frobenius_norm(const Matrix<T>& m)
    {
        T s = T(0);
        const T* p = m.ptr();
        size_t n = m.size();
        for (size_t i = 0; i < n; ++i) s += p[i] * p[i];
        return std::sqrt(s);
    }

}
