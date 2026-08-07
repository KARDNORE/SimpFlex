#pragma once
#include <cstdint>
#include <type_traits>
#include <algorithm>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_pdep_u64)
#pragma intrinsic(_pext_u64)
#endif

namespace SimpFlex {

    struct uint128_t {
        uint64_t lo;
        uint64_t hi;

        constexpr uint128_t() noexcept : lo(0), hi(0) {}
        constexpr uint128_t(uint64_t l) noexcept : lo(l), hi(0) {}
        constexpr uint128_t(uint64_t h, uint64_t l) noexcept : lo(l), hi(h) {}

        constexpr explicit operator bool() const noexcept { return lo || hi; }
        constexpr explicit operator uint64_t() const noexcept { return lo; }
        constexpr explicit operator uint32_t() const noexcept { return static_cast<uint32_t>(lo); }

        constexpr bool operator==(const uint128_t& o) const noexcept { return lo == o.lo && hi == o.hi; }
        constexpr bool operator!=(const uint128_t& o) const noexcept { return !(*this == o); }
    };

    namespace detail {

        template<typename T>
        constexpr int bit_width_v = static_cast<int>(sizeof(T) * 8);

        inline uint64_t pdep64(uint64_t x, uint64_t mask) noexcept {
            return _pdep_u64(x, mask);
        }

        inline uint64_t pext64(uint64_t x, uint64_t mask) noexcept {
            return _pext_u64(x, mask);
        }

    }

    template<typename T>
    uint128_t bit_interleave(T a, T b) noexcept {
        static_assert(std::is_unsigned_v<T>);
        constexpr int W = detail::bit_width_v<T>;
        const uint64_t au = static_cast<uint64_t>(a);
        const uint64_t bu = static_cast<uint64_t>(b);
        constexpr uint64_t MASK_EVEN = 0x5555555555555555ULL;
        constexpr uint64_t MASK_ODD = 0xAAAAAAAAAAAAAAAAULL;

        if constexpr (W <= 32) {
            uint64_t lo = detail::pdep64(au, MASK_EVEN) | detail::pdep64(bu, MASK_ODD);
            return uint128_t(0, lo);
        }
        else {
            uint64_t lo = detail::pdep64(au & 0xFFFFFFFFULL, MASK_EVEN)
                | detail::pdep64(bu & 0xFFFFFFFFULL, MASK_ODD);
            uint64_t hi = detail::pdep64(au >> 32, MASK_EVEN)
                | detail::pdep64(bu >> 32, MASK_ODD);
            return uint128_t(hi, lo);
        }
    }

    template<typename T>
    void bit_deinterleave(uint128_t x, T& a, T& b) noexcept {
        static_assert(std::is_unsigned_v<T>);
        constexpr int W = detail::bit_width_v<T>;
        constexpr uint64_t MASK_EVEN = 0x5555555555555555ULL;
        constexpr uint64_t MASK_ODD = 0xAAAAAAAAAAAAAAAAULL;

        if constexpr (W <= 32) {
            a = static_cast<T>(detail::pext64(x.lo, MASK_EVEN));
            b = static_cast<T>(detail::pext64(x.lo, MASK_ODD));
        }
        else {
            uint64_t au = detail::pext64(x.lo, MASK_EVEN)
                | (detail::pext64(x.hi, MASK_EVEN) << 32);
            uint64_t bu = detail::pext64(x.lo, MASK_ODD)
                | (detail::pext64(x.hi, MASK_ODD) << 32);
            a = static_cast<T>(au);
            b = static_cast<T>(bu);
        }
    }

    template<typename T>
    T bit_compress(T x, T mask) noexcept {
        static_assert(std::is_unsigned_v<T>);
        return static_cast<T>(detail::pext64(static_cast<uint64_t>(x), static_cast<uint64_t>(mask)));
    }

    template<typename T>
    T bit_expand(T x, T mask) noexcept {
        static_assert(std::is_unsigned_v<T>);
        return static_cast<T>(detail::pdep64(static_cast<uint64_t>(x), static_cast<uint64_t>(mask)));
    }

    template<typename T>
    T bit_permute(T x, const int* table, int n) noexcept {
        static_assert(std::is_unsigned_v<T>);
        T result = 0;
        for (int i = 0; i < n; ++i) {
            T b = (x >> i) & 1;
            result |= b << table[i];
        }
        return result;
    }

    template<typename T>
    T bit_gather(T x, const int* indices, int n) noexcept {
        static_assert(std::is_unsigned_v<T>);
        T result = 0;
        for (int i = 0; i < n; ++i) {
            T b = (x >> indices[i]) & 1;
            result |= b << i;
        }
        return result;
    }

    template<typename T>
    T bit_scatter(T x, const int* indices, int n) noexcept {
        static_assert(std::is_unsigned_v<T>);
        T result = 0;
        for (int i = 0; i < n; ++i) {
            T b = (x >> i) & 1;
            result |= b << indices[i];
        }
        return result;
    }

    template<typename T>
    T bit_select(T a, T b, T mask) noexcept {
        static_assert(std::is_unsigned_v<T>);
        return (a & mask) | (b & ~mask);
    }

    template<typename T>
    uint128_t pack_bits(const T* src, int n, int bits_per_elem) noexcept {
        static_assert(std::is_unsigned_v<T>);
        uint64_t lo = 0, hi = 0;
        int pos = 0;
        // const uint64_t val_mask = (bits_per_elem >= 64) ? ~0ULL : (1ULL << bits_per_elem); KD-00009
        const uint64_t val_mask = (bits_per_elem >= 64) ? ~0ULL : ((1ULL << bits_per_elem) - 1);

        for (int i = 0; i < n; ++i) {
            uint64_t val = static_cast<uint64_t>(src[i]) & val_mask;
            if (pos < 64) {
                lo |= val << pos;
                const int end = pos + bits_per_elem;
                if (end > 64) {
                    hi |= val >> (64 - pos);
                }
            }
            else {
                hi |= val << (pos - 64);
            }
            pos += bits_per_elem;
        }
        return uint128_t(hi, lo);
    }

    template<typename T>
    void unpack_bits(uint128_t packed, T* dst, int n, int bits_per_elem) noexcept {
        static_assert(std::is_unsigned_v<T>);
        const uint64_t lo = packed.lo;
        const uint64_t hi = packed.hi;
        int pos = 0;
        // const uint64_t val_mask = (bits_per_elem >= 64) ? ~0ULL : (1ULL << bits_per_elem); KD-00009
        const uint64_t val_mask = (bits_per_elem >= 64) ? ~0ULL : ((1ULL << bits_per_elem) - 1);

        for (int i = 0; i < n; ++i) {
            uint64_t val;
            if (pos < 64) {
                val = lo >> pos;
                const int end = pos + bits_per_elem;
                if (end > 64) {
                    val |= hi << (64 - pos);
                }
            }
            else {
                val = hi >> (pos - 64);
            }
            val &= val_mask;
            dst[i] = static_cast<T>(val);
            pos += bits_per_elem;
        }
    }

    inline uint128_t bit_stream_extract(const uint8_t* data, int bit_offset, int n_bits) noexcept {
        uint64_t lo = 0, hi = 0;
        int byte_idx = bit_offset >> 3;
        int bit_idx = bit_offset & 7;
        int out_pos = 0;

        while (out_pos < n_bits) {
            const uint64_t byte = data[byte_idx];
            const int avail = 8 - bit_idx;
            const int remaining = n_bits - out_pos;
            const int take = (avail < remaining) ? avail : remaining;
            const uint64_t bits = (byte >> bit_idx) & ((1ULL << take) - 1);

            if (out_pos < 64) {
                lo |= bits << out_pos;
                const int end = out_pos + take;
                if (end > 64) {
                    hi |= bits >> (64 - out_pos);
                }
            }
            else {
                hi |= bits << (out_pos - 64);
            }

            out_pos += take;
            bit_idx += take;
            if (bit_idx >= 8) {
                bit_idx = 0;
                ++byte_idx;
            }
        }
        return uint128_t(hi, lo);
    }

    inline void bit_stream_insert(uint8_t* data, int bit_offset, uint128_t value, int n_bits) noexcept {
        const uint64_t lo = value.lo;
        const uint64_t hi = value.hi;
        int byte_idx = bit_offset >> 3;
        int bit_idx = bit_offset & 7;
        int in_pos = 0;

        while (in_pos < n_bits) {
            const int avail = 8 - bit_idx;
            const int remaining = n_bits - in_pos;
            const int put = (avail < remaining) ? avail : remaining;
            uint64_t bits;

            if (in_pos < 64) {
                bits = (lo >> in_pos) & ((1ULL << put) - 1);
                const int end = in_pos + put;
                // end >= 64
                if (end > 64) {
                    bits |= (hi << (64 - in_pos)) & ((1ULL << put) - 1);
                }
            }
            else {
                //bits = (hi >> (in_pos - 64)) & (1ULL << put);
                bits = (hi >> (in_pos - 64)) & ((1ULL << put) - 1);
            }

            const uint8_t mask = static_cast<uint8_t>(((1 << put) - 1) << bit_idx);
            data[byte_idx] = static_cast<uint8_t>(
                (data[byte_idx] & ~mask) | ((static_cast<uint8_t>(bits) << bit_idx) & mask));

            in_pos += put;
            bit_idx += put;
            if (bit_idx >= 8) {
                bit_idx = 0;
                ++byte_idx;
            }
        }
    }

    inline uint128_t bit_matrix_transpose(uint128_t x, int rows, int cols) noexcept {
        uint64_t lo = 0, hi = 0;

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                const int src = i * cols + j;
                const int dst = j * rows + i;

                uint64_t bit;
                if (src < 64) bit = (x.lo >> src) & 1;
                else          bit = (x.hi >> (src - 64)) & 1;

                if (dst < 64) lo |= bit << dst;
                else          hi |= bit << (dst - 64);
            }
        }
        return uint128_t(hi, lo);
    }

    namespace detail {

        inline uint128_t reverse_rows(uint128_t x, int rows, int cols) noexcept {
            uint64_t lo = 0, hi = 0;
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    const int src = i * cols + j;
                    //const int dst = i * cols + (cols - 1);
                    const int dst = i * cols + (cols - 1 - j);

                    uint64_t bit;
                    if (src < 64) bit = (x.lo >> src) & 1;
                    else          bit = (x.hi >> (src - 64)) & 1;

                    if (dst < 64) lo |= bit << dst;
                    else          hi |= bit << (dst - 64);
                }
            }
            return uint128_t(hi, lo);
        }

    }

    inline uint128_t bit_matrix_rotate(uint128_t x, int rows, int cols, int k) noexcept {
        //k = ((k + 4) % 4;
        k = ((k % 4) + 4) % 4;
        if (k == 0) return x;

        uint128_t result = x;
        int r = rows, c = cols;

        for (int i = 0; i < k; ++i) {
            result = bit_matrix_transpose(result, r, c);
            std::swap(r, c);
            result = detail::reverse_rows(result, r, c);
        }
        return result;
    }

}
