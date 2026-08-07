#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstring>
#include <cassert>
#include <type_traits>
#include <numeric>
#include <chrono>

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

    template <typename T, typename Idx = uint32_t>
    class CSRMatrix {
    public:
        Idx rows{ 0 };
        Idx cols{ 0 };
        size_t nnz{ 0 };
        std::vector<Idx, AlignedAllocator<Idx, 64>> row_ptr;   // [rows + 1]
        std::vector<Idx, AlignedAllocator<Idx, 64>> col_idx;   // [nnz]
        std::vector<T, AlignedAllocator<T, 64>> values;    // [nnz]

        CSRMatrix() noexcept = default;

        CSRMatrix(Idx r, Idx c, size_t nz)
            : rows(r), cols(c), nnz(nz),
            row_ptr(r + 1, 0), col_idx(nz), values(nz) {
        }

        struct COOEntry {
            Idx row, col;
            T val;
        };

        static CSRMatrix from_coo(Idx rows, Idx cols,
            std::vector<COOEntry>& coo, bool sorted = false)
        {
            CSRMatrix A;
            A.rows = rows;
            A.cols = cols;
            A.nnz = coo.size();

            if (!sorted) {
                std::sort(coo.begin(), coo.end(),
                    [](const COOEntry& a, const COOEntry& b) {
                        if (a.row != b.row) return a.row < b.row;
                        return a.col < b.col;
                    });
            }

            A.row_ptr.resize(rows + 1, 0);
            A.col_idx.resize(A.nnz);
            A.values.resize(A.nnz);

            for (size_t k = 0; k < A.nnz; ++k) {
                A.col_idx[k] = coo[k].col;
                A.values[k] = coo[k].val;
                A.row_ptr[coo[k].row + 1]++;
            }
            for (Idx i = 0; i < rows; ++i)
                A.row_ptr[i + 1] += A.row_ptr[i];

            return A;
        }

        static CSRMatrix from_dense(const T* dense, Idx rows, Idx cols, T eps = T(0))
        {
            std::vector<COOEntry> coo;
            coo.reserve(rows * cols / 10);
            for (Idx i = 0; i < rows; ++i) {
                for (Idx j = 0; j < cols; ++j) {
                    T v = dense[i * cols + j];
                    if (std::abs(v) > eps) {
                        coo.push_back({ i, j, v });
                    }
                }
            }
            return from_coo(rows, cols, coo, false);
        }

        [[nodiscard]] Idx row_nnz(Idx i) const noexcept {
            return static_cast<Idx>(row_ptr[i + 1] - row_ptr[i]);
        }

        [[nodiscard]] double density() const noexcept {
            return static_cast<double>(nnz) / (static_cast<double>(rows) * cols);
        }

        void print_info(const char* name = "CSR") const {
            std::cout << name << ": " << rows << " x " << cols
                << ", nnz = " << nnz
                << ", density = " << (density() * 100.0) << "%\n";
        }
    };

    template <typename T, typename Idx>
    void spmv_csr(const CSRMatrix<T, Idx>& A,
        const T* __restrict x,
        T* __restrict y)
    {
        const Idx n = A.rows;
        const Idx* __restrict row_ptr = A.row_ptr.data();
        const Idx* __restrict col_idx = A.col_idx.data();
        const T* __restrict values = A.values.data();

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 64)
#endif
        for (int64_t i64 = 0; i64 < (int64_t)n; ++i64) {
            const Idx i = static_cast<Idx>(i64);
            const size_t start = row_ptr[i];
            const size_t end = row_ptr[i + 1];

            T s0 = T(0), s1 = T(0), s2 = T(0), s3 = T(0);
            size_t j = start;

            if (start < end) {
                _mm_prefetch(reinterpret_cast<const char*>(x + col_idx[start]), _MM_HINT_T0);
                if (start + 8 < end)
                    _mm_prefetch(reinterpret_cast<const char*>(x + col_idx[start + 8]), _MM_HINT_T0);
            }

            // 8-way unrolled main loop
            for (; j + 8 <= end; j += 8) {
                _mm_prefetch(reinterpret_cast<const char*>(x + col_idx[j + 16]), _MM_HINT_T0);
                s0 += values[j] * x[col_idx[j]];
                s1 += values[j + 1] * x[col_idx[j + 1]];
                //s2 += values[j + 2] * x[col_idx[j + 1]];
                s2 += values[j + 2] * x[col_idx[j + 2]];
                s3 += values[j + 3] * x[col_idx[j + 3]];
                s0 += values[j + 4] * x[col_idx[j + 4]];
                s1 += values[j + 5] * x[col_idx[j + 5]];
                s2 += values[j + 6] * x[col_idx[j + 6]];
                s3 += values[j + 7] * x[col_idx[j + 7]];
            }
            T sum = s0 + s1 + s2 + s3;
            for (; j < end; ++j) {
                sum += values[j] * x[col_idx[j]];
            }
            y[i] = sum;
        }
    }

#if SFX_HAVE_AVX2_FMA

    inline void spmv_csr_avx2(const CSRMatrix<double, uint32_t>& A,
        const double* __restrict x,
        double* __restrict y)
    {
        const uint32_t n = A.rows;
        const uint32_t* __restrict row_ptr = A.row_ptr.data();
        const uint32_t* __restrict col_idx = A.col_idx.data();
        const double* __restrict values = A.values.data();

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 64)
#endif
        for (int64_t i64 = 0; i64 < (int64_t)n; ++i64) {
            const uint32_t i = static_cast<uint32_t>(i64);
            const size_t start = row_ptr[i];
            const size_t end = row_ptr[i + 1];

            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            size_t j = start;

            for (; j + 8 <= end; j += 8) {
                __m128i idx0 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(col_idx + j));
                __m128i idx1 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(col_idx + j + 4));

                __m256d xv0 = _mm256_i32gather_pd(x, idx0, 8);
                __m256d xv1 = _mm256_i32gather_pd(x, idx1, 8);

                __m256d v0 = _mm256_loadu_pd(values + j);
                __m256d v1 = _mm256_loadu_pd(values + j + 4);

                sum0 = _mm256_fmadd_pd(v0, xv0, sum0);
                sum1 = _mm256_fmadd_pd(v1, xv1, sum1);
            }
            if (j + 4 <= end) {
                __m128i idx = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(col_idx + j));
                __m256d xv = _mm256_i32gather_pd(x, idx, 8);
                __m256d v = _mm256_loadu_pd(values + j);
                sum0 = _mm256_fmadd_pd(v, xv, sum0);
                j += 4;
            }
            __m256d s = _mm256_add_pd(sum0, sum1);
            __m128d lo = _mm256_castpd256_pd128(s);
            __m128d hi = _mm256_extractf128_pd(s, 1);
            lo = _mm_add_pd(lo, hi);
            lo = _mm_hadd_pd(lo, lo);
            double sum = _mm_cvtsd_f64(lo);

            for (; j < end; ++j) {
                sum += values[j] * x[col_idx[j]];
            }
            y[i] = sum;
        }
    }

    // float ver.
    inline void spmv_csr_avx2(const CSRMatrix<float, uint32_t>& A,
        const float* __restrict x,
        float* __restrict y)
    {
        const uint32_t n = A.rows;
        const uint32_t* __restrict row_ptr = A.row_ptr.data();
        const uint32_t* __restrict col_idx = A.col_idx.data();
        const float* __restrict values = A.values.data();

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 64)
#endif
        for (int64_t i64 = 0; i64 < (int64_t)n; ++i64) {
            const uint32_t i = static_cast<uint32_t>(i64);
            const size_t start = row_ptr[i];
            const size_t end = row_ptr[i + 1];

            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            size_t j = start;

            for (; j + 16 <= end; j += 16) {
                __m256i idx0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(col_idx + j));
                __m256i idx1 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(col_idx + j + 8));

                __m256 xv0 = _mm256_i32gather_ps(x, idx0, 4);
                __m256 xv1 = _mm256_i32gather_ps(x, idx1, 4);

                __m256 v0 = _mm256_loadu_ps(values + j);
                __m256 v1 = _mm256_loadu_ps(values + j + 8);

                sum0 = _mm256_fmadd_ps(v0, xv0, sum0);
                sum1 = _mm256_fmadd_ps(v1, xv1, sum1);
            }
            if (j + 8 <= end) {
                __m256i idx = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(col_idx + j));
                __m256 xv = _mm256_i32gather_ps(x, idx, 4);
                __m256 v = _mm256_loadu_ps(values + j);
                sum0 = _mm256_fmadd_ps(v, xv, sum0);
                //j += 4;
                j += 8;
            }
            __m256 s = _mm256_add_ps(sum0, sum1);
            __m128 lo = _mm256_castps256_ps128(s);
            __m128 hi = _mm256_extractf128_ps(s, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            float sum = _mm_cvtss_f32(lo);

            for (; j < end; ++j) {
                //sum = values[j] * x[col_idx[j]];
                sum += values[j] * x[col_idx[j]];
            }
            y[i] = sum;
        }
    }

#endif // SFX_HAVE_AVX2_FMA

    template <typename T, typename Idx = uint32_t>
    class ELLMatrix {
    public:
        Idx rows{ 0 };
        Idx cols{ 0 };
        Idx max_nz_per_row{ 0 };
        size_t nnz{ 0 };

        // col_idx[k x rows + i], values[k x rows + i]
        std::vector<Idx, AlignedAllocator<Idx, 64>> col_idx;
        std::vector<T, AlignedAllocator<T, 64>> values;

        ELLMatrix() noexcept = default;

        void print_info(const char* name = "ELL") const {
            std::cout << name << ": " << rows << " x " << cols
                << ", max_nz/row = " << max_nz_per_row
                << ", nnz = " << nnz
                << ", fill ratio = "
                << (static_cast<double>(nnz) /
                    (static_cast<double>(rows) * max_nz_per_row) * 100.0)
                << "%\n";
        }
    };

    // CSR -> ELL
    template <typename T, typename Idx>
    ELLMatrix<T, Idx> csr_to_ell(const CSRMatrix<T, Idx>& A)
    {
        ELLMatrix<T, Idx> E;
        E.rows = A.rows;
        E.cols = A.cols;

        // Find Max Line
        Idx max_nz = 0;
        for (Idx i = 0; i < A.rows; ++i) {
            Idx r = A.row_nnz(i);
            if (r > max_nz) max_nz = r;
        }
        E.max_nz_per_row = max_nz;
        E.nnz = A.nnz;

        const size_t stride = A.rows;  // stride
        E.col_idx.resize(max_nz * stride);
        E.values.resize(max_nz * stride);

        // col_idx = 0, values = 0
        std::fill(E.col_idx.begin(), E.col_idx.end(), 0);
        std::fill(E.values.begin(), E.values.end(), T(0));

        const Idx* row_ptr = A.row_ptr.data();
        const Idx* col_idx = A.col_idx.data();
        const T* values = A.values.data();

        for (Idx i = 0; i < A.rows; ++i) {
            size_t start = row_ptr[i];
            size_t end = row_ptr[i + 1];
            Idx k = 0;
            for (size_t j = start; j < end; ++j, ++k) {
                //E.col_idx[k * stride] = col_idx[j];
                //E.values[k * stride] = values[j];

                E.col_idx[k * stride + i] = col_idx[j];
                E.values[k * stride + i] = values[j];
            }
        }
        return E;
    }
#if SFX_HAVE_AVX2_FMA

    inline void spmv_ell_avx2(const ELLMatrix<double, uint32_t>& A,
        const double* __restrict x,
        double* __restrict y)
    {
        const uint32_t n = A.rows;
        const uint32_t K = A.max_nz_per_row;
        const uint32_t stride = n;

        // clean y
        for (uint32_t i = 0; i < n; ++i) y[i] = 0.0;

        const uint32_t* __restrict col_idx = A.col_idx.data();
        const double* __restrict values = A.values.data();

        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t* col_k = col_idx + k * stride;
            const double* val_k = values + k * stride;

            uint32_t i = 0;
            for (; i + 4 <= n; i += 4) {
                __m128i idx = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(col_k + i));
                __m256d xv = _mm256_i32gather_pd(x, idx, 8);
                __m256d v = _mm256_loadu_pd(val_k + i);
                __m256d yv = _mm256_loadu_pd(y + i);
                yv = _mm256_fmadd_pd(v, xv, yv);
                _mm256_storeu_pd(y + i, yv);
            }
            for (; i < n; ++i) {
                y[i] += val_k[i] * x[col_k[i]];
            }
        }
    }

    inline void spmv_ell_avx2(const ELLMatrix<float, uint32_t>& A,
        const float* __restrict x,
        float* __restrict y)
    {
        const uint32_t n = A.rows;
        const uint32_t K = A.max_nz_per_row;
        const uint32_t stride = n;

        for (uint32_t i = 0; i < n; ++i) y[i] = 0.0f;

        const uint32_t* __restrict col_idx = A.col_idx.data();
        const float* __restrict values = A.values.data();

        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t* col_k = col_idx + k * stride;
            const float* val_k = values + k * stride;

            uint32_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m256i idx = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(col_k + i));
                __m256 xv = _mm256_i32gather_ps(x, idx, 4);
                __m256 v = _mm256_loadu_ps(val_k + i);
                __m256 yv = _mm256_loadu_ps(y + i);
                yv = _mm256_fmadd_ps(v, xv, yv);
                _mm256_storeu_ps(y + i, yv);
            }
            for (; i < n; ++i) {
                y[i] += val_k[i] * x[col_k[i]];
            }
        }
    }

#endif // SFX_HAVE_AVX2_FMA

    template <typename T, int C = 4, typename Idx = uint32_t>
    class SELLMatrix {
    public:
        static_assert(C > 0, "C must be positive");

        Idx rows{ 0 };
        Idx cols{ 0 };
        size_t nnz{ 0 };
        int sigma{ 256 };

        Idx num_chunks{ 0 };

        // chunk_ptr[num_chunks + 1]
        std::vector<Idx, AlignedAllocator<Idx, 64>> chunk_ptr;
        // chunk_width[num_chunks]
        std::vector<Idx, AlignedAllocator<Idx, 64>> chunk_width;

        // col_idx[total_cols x C], values[total_cols x C]
        std::vector<Idx, AlignedAllocator<Idx, 64>> col_idx;
        std::vector<T, AlignedAllocator<T, 64>> values;

        std::vector<Idx> perm;
        std::vector<Idx> inv_perm;

        SELLMatrix() noexcept = default;

        void print_info(const char* name = "SELL") const {
            size_t total_storage = 0;
            for (Idx c = 0; c < num_chunks; ++c)
                total_storage += chunk_width[c] * C;
            double fill_ratio = static_cast<double>(nnz) / total_storage * 100.0;
            std::cout << name << "(C=" << C << ", sigma=" << sigma << "): "
                << rows << " x " << cols
                << ", chunks = " << num_chunks
                << ", nnz = " << nnz
                << ", fill = " << fill_ratio << "%\n";
        }
    };

    // CSR -> SELL-C-sigma
    template <typename T, int C, typename Idx>
    SELLMatrix<T, C, Idx> csr_to_sell(const CSRMatrix<T, Idx>& A, int sigma = 256)
    {
        static_assert(std::is_same_v<Idx, uint32_t>,
            "SELL AVX2 requires uint32_t indices");

        SELLMatrix<T, C, Idx> S;
        S.rows = A.rows;
        S.cols = A.cols;
        S.nnz = A.nnz;
        S.sigma = sigma;

        const Idx n = A.rows;
        const Idx* row_ptr = A.row_ptr.data();
        const Idx* col_idx = A.col_idx.data();
        const T* values = A.values.data();

        S.perm.resize(n);
        S.inv_perm.resize(n);
        for (Idx i = 0; i < n; ++i) S.perm[i] = i;

        for (Idx g = 0; g < n; g += sigma) {
            Idx ge = std::min(g + (Idx)sigma, n);
            std::sort(S.perm.begin() + g, S.perm.begin() + ge,
                [&](Idx a, Idx b) {
                    return (row_ptr[a + 1] - row_ptr[a]) >
                        (row_ptr[b + 1] - row_ptr[b]);
                });
        }
        for (Idx i = 0; i < n; ++i)
            S.inv_perm[S.perm[i]] = i;

        Idx num_chunks = (n + C - 1) / C;
        S.num_chunks = num_chunks;
        S.chunk_width.resize(num_chunks, 0);

        for (Idx c = 0; c < num_chunks; ++c) {
            Idx max_w = 0;
            Idx cstart = c * C;
            Idx cend = std::min(cstart + (Idx)C, n);
            for (Idx ii = cstart; ii < cend; ++ii) {
                Idx orig_row = S.perm[ii];
                Idx r = static_cast<Idx>(row_ptr[orig_row + 1] - row_ptr[orig_row]);
                if (r > max_w) max_w = r;
            }
            S.chunk_width[c] = max_w;
        }

        S.chunk_ptr.resize(num_chunks + 1, 0);
        for (Idx c = 0; c < num_chunks; ++c)
            S.chunk_ptr[c + 1] = S.chunk_ptr[c] + S.chunk_width[c];

        Idx total_cols = S.chunk_ptr[num_chunks];
        S.col_idx.resize(total_cols * C);
        S.values.resize(total_cols * C);

        std::fill(S.col_idx.begin(), S.col_idx.end(), 0);
        std::fill(S.values.begin(), S.values.end(), T(0));

        for (Idx c = 0; c < num_chunks; ++c) {
            Idx cstart = c * C;
            Idx cend = std::min(cstart + (Idx)C, n);
            Idx width = S.chunk_width[c];
            Idx coff = S.chunk_ptr[c];

            for (Idx ii = cstart; ii < cend; ++ii) {
                Idx local_i = ii - cstart;
                Idx orig_row = S.perm[ii];
                size_t rstart = row_ptr[orig_row];
                size_t rend = row_ptr[orig_row + 1];

                Idx k = 0;
                for (size_t j = rstart; j < rend; ++j, ++k) {
                    size_t pos = (coff + k) * C + local_i;
                    S.col_idx[pos] = col_idx[j];
                    S.values[pos] = values[j];
                }
            }
        }
        return S;
    }

#if SFX_HAVE_AVX2_FMA

// double, C=4
    inline void spmv_sell_avx2(const SELLMatrix<double, 4, uint32_t>& A,
        const double* __restrict x,
        double* __restrict y)
    {
        const uint32_t n = A.rows;
        const uint32_t num_chunks = A.num_chunks;
        const uint32_t* chunk_ptr = A.chunk_ptr.data();
        const uint32_t* chunk_width = A.chunk_width.data();
        const uint32_t* col_idx = A.col_idx.data();
        const double* values = A.values.data();
        const uint32_t* perm = A.perm.data();

        std::vector<double, AlignedAllocator<double, 64>> y_sorted(n);
        double* __restrict ys = y_sorted.data();

        for (uint32_t c = 0; c < num_chunks; ++c) {
            uint32_t cstart = c * 4;
            uint32_t cend = std::min(cstart + 4u, n);
            uint32_t width = chunk_width[c];
            uint32_t coff = chunk_ptr[c];
            // ?
            __m256d sum = _mm256_setzero_pd();

            for (uint32_t k = 0; k < width; ++k) {
                const uint32_t* col_k = col_idx + (coff + k) * 4;
                const double* val_k = values + (coff + k) * 4;

                __m128i idx = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(col_k));
                __m256d xv = _mm256_i32gather_pd(x, idx, 8);
                __m256d v = _mm256_load_pd(val_k);
                sum = _mm256_fmadd_pd(v, xv, sum);
            }

            if (cend - cstart == 4) {
                _mm256_storeu_pd(ys + cstart, sum);
            }
            else {
                alignas(32) double tmp[4];
                _mm256_store_pd(tmp, sum);
                for (uint32_t i = 0; i < cend - cstart; ++i)
                    ys[cstart + i] = tmp[i];
            }
        }

        // y[perm[i]] = ys[i]
        for (uint32_t i = 0; i < n; ++i) {
            y[perm[i]] = ys[i];
        }
    }

    // float, C=8
    inline void spmv_sell_avx2(const SELLMatrix<float, 8, uint32_t>& A,
        const float* __restrict x,
        float* __restrict y)
    {
        const uint32_t n = A.rows;
        const uint32_t num_chunks = A.num_chunks;
        const uint32_t* chunk_ptr = A.chunk_ptr.data();
        const uint32_t* chunk_width = A.chunk_width.data();
        const uint32_t* col_idx = A.col_idx.data();
        const float* values = A.values.data();
        const uint32_t* perm = A.perm.data();

        std::vector<float, AlignedAllocator<float, 64>> y_sorted(n);
        float* __restrict ys = y_sorted.data();

        for (uint32_t c = 0; c < num_chunks; ++c) {
            uint32_t cstart = c * 8;
            uint32_t cend = std::min(cstart + 8u, n);
            uint32_t width = chunk_width[c];
            uint32_t coff = chunk_ptr[c];

            __m256 sum = _mm256_setzero_ps();

            for (uint32_t k = 0; k < width; ++k) {
                const uint32_t* col_k = col_idx + (coff + k) * 8;
                const float* val_k = values + (coff + k) * 8;

                __m256i idx = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(col_k));
                __m256 xv = _mm256_i32gather_ps(x, idx, 4);
                __m256 v = _mm256_load_ps(val_k);
                sum = _mm256_fmadd_ps(v, xv, sum);
            }

            if (cend - cstart == 8) {
                _mm256_storeu_ps(ys + cstart, sum);
            }
            else {
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, sum);
                for (uint32_t i = 0; i < cend - cstart; ++i)
                    ys[cstart + i] = tmp[i];
            }
        }

        for (uint32_t i = 0; i < n; ++i) {
            y[perm[i]] = ys[i];
        }
    }

#endif // SFX_HAVE_AVX2_FMA

    template <typename T, typename Idx>
    void spmv_transpose_csr(const CSRMatrix<T, Idx>& A,
        const T* __restrict x,
        T* __restrict y)
    {
        const Idx m = A.cols;
        // clean
        for (Idx j = 0; j < m; ++j) y[j] = T(0);

        const Idx* row_ptr = A.row_ptr.data();
        const Idx* col_idx = A.col_idx.data();
        const T* values = A.values.data();
        const Idx n = A.rows;

#if defined(_OPENMP)
#pragma omp parallel
        {
            std::vector<T> y_local(m, T(0));
            T* yl = y_local.data();

#pragma omp for schedule(dynamic, 64)
            for (int64_t i64 = 0; i64 < (int64_t)n; ++i64) {
                Idx i = static_cast<Idx>(i64);
                T xi = x[i];
                size_t start = row_ptr[i];
                size_t end = row_ptr[i + 1];
                for (size_t j = start; j < end; ++j) {
                    yl[col_idx[j]] += values[j] * xi;
                }
            }
#pragma omp critical
            {
                for (Idx j = 0; j < m; ++j)
                    y[j] += yl[j];
            }
        }
#else
        for (Idx i = 0; i < n; ++i) {
            T xi = x[i];
            size_t start = row_ptr[i];
            size_t end = row_ptr[i + 1];
            for (size_t j = start; j < end; ++j) {
                y[col_idx[j]] += values[j] * xi;
            }
        }
#endif
    }

    template <typename T, typename Idx = uint32_t>
    CSRMatrix<T, Idx> random_sparse(Idx rows, Idx cols, double density,
        uint64_t seed = 12345ULL) // random seed.nned to be improved
    {
        using COOEntry = typename CSRMatrix<T, Idx>::COOEntry;
        std::vector<COOEntry> coo;

        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        std::uniform_real_distribution<T> val_dist(T(-1), T(1));

        size_t expected_nnz = static_cast<size_t>(
            static_cast<double>(rows) * cols * density);
        coo.reserve(expected_nnz);

        for (Idx i = 0; i < rows; ++i) {
            for (Idx j = 0; j < cols; ++j) {
                if (prob(rng) < density) {
                    coo.push_back({ i, j, val_dist(rng) });
                }
            }
        }
        return CSRMatrix<T, Idx>::from_coo(rows, cols, coo, false);
    }

    template <typename T, typename Idx = uint32_t>
    CSRMatrix<T, Idx> diag_matrix(Idx n, T diag_val = T(1))
    {
        using COOEntry = typename CSRMatrix<T, Idx>::COOEntry;
        std::vector<COOEntry> coo(n);
        for (Idx i = 0; i < n; ++i) {
            coo[i] = { i, i, diag_val };
        }
        return CSRMatrix<T, Idx>::from_coo(n, n, coo, true);
    }

    template <typename T, typename Idx = uint32_t>
    CSRMatrix<T, Idx> tridiagonal(Idx n, T sub = T(-1), T diag = T(2), T super = T(-1))
    {
        using COOEntry = typename CSRMatrix<T, Idx>::COOEntry;
        std::vector<COOEntry> coo;
        coo.reserve(3 * n - 2);
        for (Idx i = 0; i < n; ++i) {
            if (i > 0) coo.push_back({ i, i - 1, sub });
            coo.push_back({ i, i, diag });
            if (i + 1 < n) coo.push_back({ i, i + 1, super });
        }
        return CSRMatrix<T, Idx>::from_coo(n, n, coo, true);
    }

    template <typename T, typename Idx = uint32_t>
    CSRMatrix<T, Idx> band_matrix(Idx n, Idx bandwidth, T diag_val = T(2), T off_val = T(-1))
    {
        using COOEntry = typename CSRMatrix<T, Idx>::COOEntry;
        std::vector<COOEntry> coo;
        coo.reserve(n * (2 * bandwidth + 1));
        for (Idx i = 0; i < n; ++i) {
            Idx lo = (i >= bandwidth) ? (i - bandwidth) : 0;
            Idx hi = std::min(i + bandwidth + 1, n);
            for (Idx j = lo; j < hi; ++j) {
                if (i == j) coo.push_back({ i, j, diag_val });
                else        coo.push_back({ i, j, off_val });
            }
        }
        return CSRMatrix<T, Idx>::from_coo(n, n, coo, true);
    }

    template <typename T, typename Idx = uint32_t>
    CSRMatrix<T, Idx> random_spd_sparse(Idx n, double density, uint64_t seed = 12345ULL)
    {
        auto A = random_sparse<T, Idx>(n, n, density, seed);
        // A = A^T x A + n x 0.1 x I  (SPD)
        using COOEntry = typename CSRMatrix<T, Idx>::COOEntry;
        std::vector<COOEntry> coo;
        coo.reserve(A.nnz * 2 + n);

        const Idx* row_ptr = A.row_ptr.data();
        const Idx* col_idx = A.col_idx.data();
        const T* values = A.values.data();

        for (Idx i = 0; i < n; ++i) {
            T diag_sum = T(0);
            for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                Idx c = col_idx[j];
                T v = values[j];
                if (c > i) {
                    coo.push_back({ i, c, v });
                    coo.push_back({ c, i, v });
                    diag_sum += std::abs(v);
                }
            }
            coo.push_back({ i, i, diag_sum + T(n) * T(0.01) });
        }
        std::sort(coo.begin(), coo.end(),
            [](const COOEntry& a, const COOEntry& b) {
                if (a.row != b.row) return a.row < b.row;
                return a.col < b.col;
            });
        return CSRMatrix<T, Idx>::from_coo(n, n, coo, true);
    }

    template <typename T, typename Idx>
    T spmv_max_error(const CSRMatrix<T, Idx>& A, const T* x, const T* y_sparse)
    {
        std::vector<T> y_dense(A.rows, T(0));
        const Idx* row_ptr = A.row_ptr.data();
        const Idx* col_idx = A.col_idx.data();
        const T* values = A.values.data();

        for (Idx i = 0; i < A.rows; ++i) {
            T sum = T(0);
            for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                sum += values[j] * x[col_idx[j]];
            }
            y_dense[i] = sum;
        }
        T err = T(0);
        for (Idx i = 0; i < A.rows; ++i) {
            err = std::max(err, std::abs(y_dense[i] - y_sparse[i]));
        }
        return err;
    }
    // tool
    template <typename T>
    T vec_max_abs_diff(const T* a, const T* b, size_t n) {
        T err = T(0);
        for (size_t i = 0; i < n; ++i)
            err = std::max(err, std::abs(a[i] - b[i]));
        return err;
    }

    template <typename T>
    T vec_norm2(const T* x, size_t n) {
        T s = T(0);
        for (size_t i = 0; i < n; ++i) s += x[i] * x[i];
        return std::sqrt(s);
    }
    // end
}
