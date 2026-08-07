#pragma once
#include <cstdint>
#include <type_traits>
#include <cstddef>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanForward)
#pragma intrinsic(__popcnt)
#pragma intrinsic(__popcnt64)
#pragma intrinsic(_rotl)
#pragma intrinsic(_rotr)
#ifdef _WIN64
#pragma intrinsic(_rotl64)
#pragma intrinsic(_rotr64)
#pragma intrinsic(_BitScanReverse64)
#pragma intrinsic(_BitScanForward64)
#endif
#endif

namespace SimpFlex {

    template<typename T>
    struct make_unsigned { using type = std::make_unsigned_t<T>; };

    template<typename T>
    using make_unsigned_t = typename make_unsigned<T>::type;

    namespace detail {

        template<typename T>
        constexpr int bit_width() noexcept {
            return static_cast<int>(sizeof(T) * 8);
        }

        template<typename T>
        constexpr bool is_unsigned_v = std::is_unsigned_v<T>;
    }

    template<typename T>
    T bit_reverse(T x) noexcept {
        static_assert(std::is_integral_v<T>);
        using U = make_unsigned_t<T>;
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
    T bit_rotate_left(T x, int n) noexcept {
        static_assert(std::is_integral_v<T>);
        using U = make_unsigned_t<T>;
        constexpr int W = detail::bit_width<U>();
        n &= W - 1;
        if (n == 0) return x;
        U v = static_cast<U>(x);
        if constexpr (sizeof(U) == 4) {
            return static_cast<T>(static_cast<U>(_rotl(v, n)));
        }
        else if constexpr (sizeof(U) == 8) {
#ifdef _WIN64
            return static_cast<T>(static_cast<U>(_rotl64(v, n)));
#else
            return static_cast<T>((v << n) | (v >> (W - n)));
#endif
        }
        else {
            return static_cast<T>((v << n) | (v >> (W - n)));
        }
    }

    template<typename T>
    T bit_rotate_right(T x, int n) noexcept {
        static_assert(std::is_integral_v<T>);
        using U = make_unsigned_t<T>;
        constexpr int W = detail::bit_width<U>();
        n &= W - 1;
        if (n == 0) return x;
        U v = static_cast<U>(x);
        if constexpr (sizeof(U) == 4) {
            return static_cast<T>(static_cast<U>(_rotr(v, n)));
        }
        else if constexpr (sizeof(U) == 8) {
#ifdef _WIN64
            return static_cast<T>(static_cast<U>(_rotr64(v, n)));
#else
            return static_cast<T>((v >> n) | (v << (W - n)));
#endif
        }
        else {
            return static_cast<T>((v >> n) | (v << (W - n)));
        }
    }

    template<typename T>
    int bit_count(T x) noexcept {
        static_assert(std::is_integral_v<T>);
        using U = make_unsigned_t<T>;
        U v = static_cast<U>(x);
        if constexpr (sizeof(U) <= 4) {
            return __popcnt(static_cast<unsigned int>(v));
        }
        else {
#ifdef _WIN64
            return static_cast<int>(__popcnt64(v));
#else
            return __popcnt(static_cast<unsigned int>(v & 0xFFFFFFFFu))
                + __popcnt(static_cast<unsigned int>(v >> 32));
#endif
        }
    }

    template<typename T>
    int bit_count_zero(T x) noexcept {
        static_assert(std::is_integral_v<T>);
        using U = make_unsigned_t<T>;
        return detail::bit_width<U>() - bit_count(x);
    }

    template<typename T>
    int bit_length(T x) noexcept {
        static_assert(std::is_integral_v<T>);
        using U = make_unsigned_t<T>;
        U v = static_cast<U>(x);
        if (v == 0) return 0;
        unsigned long idx;
        if constexpr (sizeof(U) <= 4) {
            _BitScanReverse(&idx, static_cast<unsigned int>(v));
        }
        else {
#ifdef _WIN64
            _BitScanReverse64(&idx, v);
#else
            unsigned int hi = static_cast<unsigned int>(v >> 32);
            if (hi != 0) {
                _BitScanReverse(&idx, hi);
                return static_cast<int>(idx) + 33;
            }
            else {
                _BitScanReverse(&idx, static_cast<unsigned int>(v));
                return static_cast<int>(idx) + 1;
            }
#endif
        }
        return static_cast<int>(idx) + 1;
    }

    template<typename T>
    int clz(T x) noexcept {
        static_assert(std::is_integral_v<T>);
        using U = make_unsigned_t<T>;
        constexpr int W = detail::bit_width<U>();
        U v = static_cast<U>(x);
        if (v == 0) return W;
        return W - bit_length(v);
    }

    template<typename T>
    int ctz(T x) noexcept {
        static_assert(std::is_integral_v<T>);
        using U = make_unsigned_t<T>;
        U v = static_cast<U>(x);
        unsigned long idx;
        if constexpr (sizeof(U) <= 4) {
            _BitScanForward(&idx, static_cast<unsigned int>(v));
        }
        else {
#ifdef _WIN64
            _BitScanForward64(&idx, v);
#else
            unsigned int lo = static_cast<unsigned int>(v & 0xFFFFFFFFu);
            if (lo != 0) {
                _BitScanForward(&idx, lo);
                return static_cast<int>(idx);
            }
            else {
                _BitScanForward(&idx, static_cast<unsigned int>(v >> 32));
                return static_cast<int>(idx) + 32;
            }
#endif
        }
        return static_cast<int>(idx);
    }

    template<typename T>
    int bit_parity(T x) noexcept {
        static_assert(std::is_integral_v<T>);
        return bit_count(x) & 1;
    }

    template<typename T>
    bool is_power_of_two(T x) noexcept {
        static_assert(std::is_integral_v<T>);
        using U = make_unsigned_t<T>;
        U v = static_cast<U>(x);
        return v != 0 && (v & (v - 1)) == 0;
    }

    template<typename T>
    T next_power_of_two(T x) noexcept {
        static_assert(std::is_integral_v<T>);
        using U = make_unsigned_t<T>;
        U v = static_cast<U>(x);
        if (v <= 1) return static_cast<T>(1);
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        if constexpr (sizeof(U) >= 2) v |= v >> 8;
        if constexpr (sizeof(U) >= 4) v |= v >> 16;
        if constexpr (sizeof(U) >= 8) v |= v >> 32;
        ++v;
        return static_cast<T>(v);
    }

    template<typename T>
    T prev_power_of_two(T x) noexcept {
        static_assert(std::is_integral_v<T>);
        using U = make_unsigned_t<T>;
        U v = static_cast<U>(x);
        if (v == 0) return static_cast<T>(0);
        int len = bit_length(v);
        return static_cast<T>(static_cast<U>(1) << (len - 1));
    }
}
