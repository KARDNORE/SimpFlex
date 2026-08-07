#pragma once
#include <cstdint>
#include <type_traits>
#include <initializer_list>
#include <vector>
#include <utility>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_byteswap_ushort)
#pragma intrinsic(_byteswap_ulong)
#pragma intrinsic(_byteswap_uint64)
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
    }

    template<typename T>
    constexpr T constexpr_bit_reverse(T x) noexcept {
        static_assert(std::is_unsigned_v<T>);
        using U = std::make_unsigned_t<T>;
        U v = static_cast<U>(x);

        if constexpr (sizeof(U) == 1) {
            v = static_cast<U>(((v >> 1) & 0x55U) | ((v & 0x55U) << 1));
            v = static_cast<U>(((v >> 2) & 0x33U) | ((v & 0x33U) << 2));
            v = static_cast<U>((v >> 4) | (v << 4));
        }
        else if constexpr (sizeof(U) == 2) {
            v = static_cast<U>(((v >> 1) & 0x5555U) | ((v & 0x5555U) << 1));
            v = static_cast<U>(((v >> 2) & 0x3333U) | ((v & 0x3333U) << 2));
            v = static_cast<U>(((v >> 4) & 0x0F0FU) | ((v & 0x0F0FU) << 4));
            v = static_cast<U>((v >> 8) | (v << 8));
        }
        else if constexpr (sizeof(U) == 4) {
            v = ((v >> 1) & 0x55555555U) | ((v & 0x55555555U) << 1);
            v = ((v >> 2) & 0x33333333U) | ((v & 0x33333333U) << 2);
            v = ((v >> 4) & 0x0F0F0F0FU) | ((v & 0x0F0F0F0FU) << 4);
            v = ((v >> 8) & 0x00FF00FFU) | ((v & 0x00FF00FFU) << 8);
            v = (v >> 16) | (v << 16);
        }
        else {
            v = ((v >> 1) & 0x5555555555555555ULL) | ((v & 0x5555555555555555ULL) << 1);
            v = ((v >> 2) & 0x3333333333333333ULL) | ((v & 0x3333333333333333ULL) << 2);
            v = ((v >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((v & 0x0F0F0F0F0F0F0F0FULL) << 4);
            v = ((v >> 8) & 0x00FF00FF00FF00FFULL) | ((v & 0x00FF00FF00FF00FFULL) << 8);
            v = ((v >> 16) & 0x0000FFFF0000FFFFULL) | ((v & 0x0000FFFF0000FFFFULL) << 16);
            v = (v >> 32) | (v << 32);
        }
        return static_cast<T>(v);
    }

    template<typename T>
    constexpr int constexpr_bit_count(T x) noexcept {
        static_assert(std::is_unsigned_v<T>);
        using U = std::make_unsigned_t<T>;
        U v = static_cast<U>(x);

        v = v - ((v >> 1) & (~U(0) / 3));
        v = (v & (~U(0) / 5)) + ((v >> 2) & (~U(0) / 5));
        v = (v + (v >> 4)) & (~U(0) / 17);

        if constexpr (sizeof(U) >= 2) {
            v += v >> 8;
        }
        if constexpr (sizeof(U) >= 4) {
            v += v >> 16;
        }
        if constexpr (sizeof(U) >= 8) {
            v += v >> 32;
        }
        return static_cast<int>(v & 0x7F);
    }

    template<typename T>
    constexpr int constexpr_clz(T x) noexcept {
        using U = std::make_unsigned_t<T>;
        constexpr int W = detail::bit_width_v<U>;
        U v = static_cast<U>(x);
        if (v == 0) return W;
        int n = 0;

        if constexpr (W >= 64) {
            if (v < (U(1) << 32)) { n += 32; v <<= 32; }
        }
        if constexpr (W >= 32) {
            if (v < (U(1) << (W - 16))) { n += 16; v <<= 16; }
        }
        if constexpr (W >= 16) {
            if (v < (U(1) << (W - 8))) { n += 8; v <<= 8; }
        }
        if (v < (U(1) << (W - 4))) { n += 4; v <<= 4; }
        if (v < (U(1) << (W - 2))) { n += 2; v <<= 2; }
        if (v < (U(1) << (W - 1))) { n += 1; }
        return n;
    }

    template<typename T>
    constexpr int constexpr_ctz(T x) noexcept {
        static_assert(std::is_unsigned_v<T>);
        using U = std::make_unsigned_t<T>;
        constexpr int W = detail::bit_width_v<U>;
        U v = static_cast<U>(x);
        if (v == 0) return W;
        int n = 0;

        if constexpr (W >= 64) {
            if ((v & 0xFFFFFFFFULL) == 0) { n += 32; v >>= 32; }
        }
        if constexpr (W >= 32) {
            if ((v & 0xFFFFULL) == 0) { n += 16; v >>= 16; }
        }
        if constexpr (W >= 16) {
            if ((v & 0xFFULL) == 0) { n += 8; v >>= 8; }
        }
        if ((v & 0xF) == 0) { n += 4; v >>= 4; }
        if ((v & 0x3) == 0) { n += 2; v >>= 2; }
        if ((v & 0x1) == 0) { n += 1; }
        return n;
    }

    template<typename T>
    constexpr T constexpr_bit_permute(T x, const int* table, int n) noexcept {
        static_assert(std::is_unsigned_v<T>);
        T result = 0;
        for (int i = 0; i < n; ++i) {
            T b = (x >> i) & 1;
            result |= b << table[i];
        }
        return result;
    }

    inline uint128_t pack_fields(std::initializer_list<std::pair<int, uint64_t>> fields) noexcept {
        uint64_t lo = 0, hi = 0;
        int pos = 0;
        for (const auto& f : fields) {
            int width = f.first;
            uint64_t val = f.second;
            const uint64_t val_mask = (width >= 64) ? ~0ULL : ((1ULL << width) - 1);
            val &= val_mask;
            if (pos < 64) {
                lo |= val << pos;
                const int end = pos + width;
                if (end > 64) {
                    hi |= val >> (64 - pos);
                }
            }
            else {
                hi |= val << (pos - 64);
            }
            pos += width;
        }
        return uint128_t(hi, lo);
    }

    inline std::vector<uint64_t> unpack_fields(uint128_t x, std::initializer_list<int> widths) {
        std::vector<uint64_t> result;
        result.reserve(widths.size());
        int pos = 0;
        for (int width : widths) {
            const uint64_t val_mask = (width >= 64) ? ~0ULL : ((1ULL << width) - 1);
            uint64_t val;
            if (pos < 64) {
                val = x.lo >> pos;
                const int end = pos + width;
                if (end > 64) {
                    val |= x.hi << (64 - pos);
                }
            }
            else {
                val = x.hi >> (pos - 64);
            }
            val &= val_mask;
            result.push_back(val);
            pos += width;
        }
        return result;
    }

    template<typename T>
    constexpr T byte_swap(T x) noexcept {
        static_assert(std::is_integral_v<T>);
        using U = std::make_unsigned_t<T>;
        U v = static_cast<U>(x);

        if constexpr (sizeof(U) == 1) {
            return static_cast<T>(v);
        }
        else if constexpr (sizeof(U) == 2) {
#ifdef _MSC_VER
            return static_cast<T>(static_cast<U>(_byteswap_ushort(static_cast<unsigned short>(v))));
#else
            return static_cast<T>((v >> 8) | (v << 8));
#endif
        }
        else if constexpr (sizeof(U) == 4) {
#ifdef _MSC_VER
            return static_cast<T>(static_cast<U>(_byteswap_ulong(static_cast<unsigned long>(v))));
#else
            v = ((v >> 8) & 0x00FF00FFU) | ((v & 0x00FF00FFU) << 8);
            return static_cast<T>((v >> 16) | (v << 16));
#endif
        }
        else {
#ifdef _MSC_VER
            return static_cast<T>(static_cast<U>(_byteswap_uint64(v)));
#else
            v = ((v >> 8) & 0x00FF00FF00FF00FFULL) | ((v & 0x00FF00FF00FF00FFULL) << 8);
            v = ((v >> 16) & 0x0000FFFF0000FFFFULL) | ((v & 0x0000FFFF0000FFFFULL) << 16);
            return static_cast<T>((v >> 32) | (v << 32));
#endif
        }
    }

    template<typename T>
    constexpr T bit_endian_swap(T x) noexcept {
        static_assert(std::is_unsigned_v<T>);
        using U = std::make_unsigned_t<T>;
        U v = static_cast<U>(x);

        if constexpr (sizeof(U) == 1) {
            v = static_cast<U>(((v >> 1) & 0x55U) | ((v & 0x55U) << 1));
            v = static_cast<U>(((v >> 2) & 0x33U) | ((v & 0x33U) << 2));
            v = static_cast<U>((v >> 4) | (v << 4));
        }
        else if constexpr (sizeof(U) == 2) {
            v = static_cast<U>(((v >> 1) & 0x5555U) | ((v & 0x5555U) << 1));
            v = static_cast<U>(((v >> 2) & 0x3333U) | ((v & 0x3333U) << 2));
            v = static_cast<U>(((v >> 4) & 0x0F0FU) | ((v & 0x0F0FU) << 4));
        }
        else if constexpr (sizeof(U) == 4) {
            v = ((v >> 1) & 0x55555555U) | ((v & 0x55555555U) << 1);
            v = ((v >> 2) & 0x33333333U) | ((v & 0x33333333U) << 2);
            v = ((v >> 4) & 0x0F0F0F0FU) | ((v & 0x0F0F0F0FU) << 4);
        }
        else {
            v = ((v >> 1) & 0x5555555555555555ULL) | ((v & 0x5555555555555555ULL) << 1);
            v = ((v >> 2) & 0x3333333333333333ULL) | ((v & 0x3333333333333333ULL) << 2);
            v = ((v >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((v & 0x0F0F0F0F0F0F0F0FULL) << 4);
        }
        return static_cast<T>(v);
    }

    template<typename T>
    constexpr T sign_extend(T x, int src_bits) noexcept {
        static_assert(std::is_integral_v<T>);
        using U = std::make_unsigned_t<T>;
        using S = std::make_signed_t<T>;
        constexpr int W = detail::bit_width_v<U>;
        U v = static_cast<U>(x);
        if (src_bits <= 0) return static_cast<T>(0);
        //if (src_bits > W) return x;
        if (src_bits >= W) return x;
        const int shift = W - src_bits;
        return static_cast<T>(static_cast<S>(v << shift) >> shift);
    }
}
