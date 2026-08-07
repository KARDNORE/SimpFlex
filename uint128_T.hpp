#pragma once
#include <cstdint>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <utility>

namespace u128 {
    namespace detail {

        constexpr int clz64(uint64_t x) noexcept {
            if (x == 0) return 64;
            int n = 0;
            if ((x >> 32) == 0) { n += 32; x <<= 32; }
            if ((x >> 48) == 0) { n += 16; x <<= 16; }
            if ((x >> 56) == 0) { n += 8;  x <<= 8; }
            if ((x >> 60) == 0) { n += 4;  x <<= 4; }
            if ((x >> 62) == 0) { n += 2;  x <<= 2; }
            if ((x >> 63) == 0) { n += 1; }
            return n;
        }

        constexpr int ctz64(uint64_t x) noexcept {
            if (x == 0) return 64;
            int n = 0;
            if ((x & 0x00000000FFFFFFFFULL) == 0) { n += 32; x >>= 32; }
            if ((x & 0x000000000000FFFFULL) == 0) { n += 16; x >>= 16; }
            if ((x & 0x00000000000000FFULL) == 0) { n += 8;  x >>= 8; }
            if ((x & 0x000000000000000FULL) == 0) { n += 4;  x >>= 4; }
            if ((x & 0x0000000000000003ULL) == 0) { n += 2;  x >>= 2; }
            if ((x & 0x0000000000000001ULL) == 0) { n += 1; }
            return n;
        }

        constexpr int popcount64(uint64_t x) noexcept {
            x = x - ((x >> 1) & 0x5555555555555555ULL);
            x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
            x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
            return static_cast<int>((x * 0x0101010101010101ULL) >> 56);
        }

    }

    struct uint128_t {
        uint64_t hi;
        uint64_t lo;

        constexpr uint128_t() noexcept : hi(0), lo(0) {}

        template <typename T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) <= 8, int> = 0>
        constexpr uint128_t(T value) noexcept
            : hi(0), lo(static_cast<uint64_t>(value)) {
        }

        constexpr uint128_t(uint64_t high, uint64_t low) noexcept
            : hi(high), lo(low) {
        }

        constexpr explicit operator bool()     const noexcept { return hi || lo; }
        constexpr explicit operator uint8_t()  const noexcept { return static_cast<uint8_t>(lo); }
        constexpr explicit operator uint16_t() const noexcept { return static_cast<uint16_t>(lo); }
        constexpr explicit operator uint32_t() const noexcept { return static_cast<uint32_t>(lo); }
        constexpr explicit operator uint64_t() const noexcept { return lo; }

        constexpr uint128_t operator+() const noexcept { return *this; }
        constexpr uint128_t operator~() const noexcept { return uint128_t(~hi, ~lo); }

        constexpr uint128_t& operator++() noexcept {
            lo++;
            hi += (lo == 0);
            return *this;
        }
        constexpr uint128_t operator++(int) noexcept {
            uint128_t tmp = *this;
            ++(*this);
            return tmp;
        }
        constexpr uint128_t& operator--() noexcept {
            const uint64_t old_lo = lo;
            lo--;
            hi -= (old_lo == 0);
            return *this;
        }
        constexpr uint128_t operator--(int) noexcept {
            uint128_t tmp = *this;
            --(*this);
            return tmp;
        }

        constexpr uint128_t& operator+=(const uint128_t& rhs) noexcept {
            const uint64_t sum_lo = lo + rhs.lo;
            hi += rhs.hi + (sum_lo < lo);
            lo = sum_lo;
            return *this;
        }
        constexpr uint128_t operator+(const uint128_t& rhs) const noexcept {
            uint128_t tmp = *this;
            tmp += rhs;
            return tmp;
        }

        constexpr uint128_t& operator-=(const uint128_t& rhs) noexcept {
            const uint64_t diff_lo = lo - rhs.lo;
            hi -= rhs.hi + (diff_lo > lo);
            lo = diff_lo;
            return *this;
        }
        constexpr uint128_t operator-(const uint128_t& rhs) const noexcept {
            uint128_t tmp = *this;
            tmp -= rhs;
            return tmp;
        }

        static constexpr void mul64(uint64_t a, uint64_t b,
            uint64_t& out_hi, uint64_t& out_lo) noexcept {
            const uint64_t a_lo = a & 0xFFFFFFFFULL;
            const uint64_t a_hi = a >> 32;
            const uint64_t b_lo = b & 0xFFFFFFFFULL;
            const uint64_t b_hi = b >> 32;

            const uint64_t p0 = a_lo * b_lo;
            const uint64_t p1 = a_lo * b_hi;
            const uint64_t p2 = a_hi * b_lo;
            const uint64_t p3 = a_hi * b_hi;

            const uint64_t mid = p1 + (p0 >> 32);
            const uint64_t mid2 = p2 + (mid & 0xFFFFFFFFULL);

            out_lo = (mid2 << 32) | (p0 & 0xFFFFFFFFULL);
            out_hi = p3 + (mid >> 32) + (mid2 >> 32);
        }

        constexpr uint128_t& operator*=(const uint128_t& rhs) noexcept {
            uint64_t p_hi = 0, p_lo = 0;
            mul64(lo, rhs.lo, p_hi, p_lo);

            uint64_t q_hi = 0, q_lo = 0;
            mul64(lo, rhs.hi, q_hi, q_lo);
            p_hi += q_lo;

            uint64_t r_hi = 0, r_lo = 0;
            mul64(hi, rhs.lo, r_hi, r_lo);
            p_hi += r_lo;

            hi = p_hi;
            lo = p_lo;
            return *this;
        }
        constexpr uint128_t operator*(const uint128_t& rhs) const noexcept {
            uint128_t tmp = *this;
            tmp *= rhs;
            return tmp;
        }

        static constexpr std::pair<uint128_t, uint128_t>
            divmod(const uint128_t& num, const uint128_t& den) noexcept {
            if (den.hi == 0 && den.lo == 0) {
                return std::make_pair(uint128_t(UINT64_MAX, UINT64_MAX), uint128_t(0));
            }

            uint128_t quotient(0, 0);
            uint128_t remainder(0, 0);

            for (int i = 127; i >= 0; --i) {
                remainder.hi = (remainder.hi << 1) | (remainder.lo >> 63);
                remainder.lo <<= 1;

                const uint64_t bit = (i >= 64)
                    ? ((num.hi >> (i - 64)) & 1ULL)
                    : ((num.lo >> i) & 1ULL);
                remainder.lo |= bit;

                if (!(remainder < den)) {
                    remainder -= den;
                    if (i >= 64)
                        quotient.hi |= (1ULL << (i - 64));
                    else
                        quotient.lo |= (1ULL << i);
                }
            }
            return std::make_pair(quotient, remainder);
        }

        constexpr uint128_t& operator/=(const uint128_t& rhs) noexcept {
            *this = divmod(*this, rhs).first;
            return *this;
        }
        constexpr uint128_t operator/(const uint128_t& rhs) const noexcept {
            return divmod(*this, rhs).first;
        }
        constexpr uint128_t& operator%=(const uint128_t& rhs) noexcept {
            *this = divmod(*this, rhs).second;
            return *this;
        }
        constexpr uint128_t operator%(const uint128_t& rhs) const noexcept {
            return divmod(*this, rhs).second;
        }

        constexpr uint128_t& operator&=(const uint128_t& rhs) noexcept {
            hi &= rhs.hi;
            lo &= rhs.lo;
            return *this;
        }
        constexpr uint128_t operator&(const uint128_t& rhs) const noexcept {
            return uint128_t(hi & rhs.hi, lo & rhs.lo);
        }

        constexpr uint128_t& operator|=(const uint128_t& rhs) noexcept {
            hi |= rhs.hi;
            lo |= rhs.lo;
            return *this;
        }
        constexpr uint128_t operator|(const uint128_t& rhs) const noexcept {
            return uint128_t(hi | rhs.hi, lo | rhs.lo);
        }

        constexpr uint128_t& operator^=(const uint128_t& rhs) noexcept {
            hi ^= rhs.hi;
            lo ^= rhs.lo;
            return *this;
        }
        constexpr uint128_t operator^(const uint128_t& rhs) const noexcept {
            return uint128_t(hi ^ rhs.hi, lo ^ rhs.lo);
        }

        constexpr uint128_t& operator<<=(int shift) noexcept {
            if (shift == 0) return *this;
            if (shift >= 128) { hi = 0; lo = 0; return *this; }
            if (shift >= 64) {
                hi = lo << (shift - 64);
                lo = 0;
            }
            else {
                hi = (hi << shift) | (lo >> (64 - shift));
                lo <<= shift;
            }
            return *this;
        }
        constexpr uint128_t operator<<(int shift) const noexcept {
            uint128_t tmp = *this;
            tmp <<= shift;
            return tmp;
        }

        constexpr uint128_t& operator>>=(int shift) noexcept {
            if (shift == 0) return *this;
            if (shift >= 128) { hi = 0; lo = 0; return *this; }
            if (shift >= 64) {
                lo = hi >> (shift - 64);
                hi = 0;
            }
            else {
                lo = (lo >> shift) | (hi << (64 - shift));
                hi >>= shift;
            }
            return *this;
        }
        constexpr uint128_t operator>>(int shift) const noexcept {
            uint128_t tmp = *this;
            tmp >>= shift;
            return tmp;
        }

        constexpr bool operator==(const uint128_t& rhs) const noexcept {
            return hi == rhs.hi && lo == rhs.lo;
        }
        constexpr bool operator!=(const uint128_t& rhs) const noexcept {
            return !(*this == rhs);
        }
        constexpr bool operator<(const uint128_t& rhs) const noexcept {
            return (hi < rhs.hi) || (hi == rhs.hi && lo < rhs.lo);
        }
        constexpr bool operator<=(const uint128_t& rhs) const noexcept {
            return !(rhs < *this);
        }
        constexpr bool operator>(const uint128_t& rhs) const noexcept {
            return rhs < *this;
        }
        constexpr bool operator>=(const uint128_t& rhs) const noexcept {
            return !(*this < rhs);
        }

        constexpr bool is_zero() const noexcept { return hi == 0 && lo == 0; }

        constexpr int bits() const noexcept {
            if (hi != 0) return 128 - detail::clz64(hi);
            if (lo != 0) return 64 - detail::clz64(lo);
            return 0;
        }

        constexpr int clz() const noexcept {
            if (hi != 0) return detail::clz64(hi);
            if (lo != 0) return 64 + detail::clz64(lo);
            return 128;
        }

        constexpr int ctz() const noexcept {
            if (lo != 0) return detail::ctz64(lo);
            if (hi != 0) return 64 + detail::ctz64(hi);
            return 128;
        }

        constexpr int popcount() const noexcept {
            return detail::popcount64(hi) + detail::popcount64(lo);
        }

        std::string to_hex() const {
            std::ostringstream oss;
            if (hi != 0) {
                oss << std::hex << std::uppercase << hi
                    << std::setw(16) << std::setfill('0') << lo;
            }
            else {
                oss << std::hex << std::uppercase << lo;
            }
            return oss.str();
        }

        std::string to_dec() const {
            if (is_zero()) return "0";
            std::string result;
            uint128_t n(hi, lo);
            const uint128_t ten(10);
            while (n > 0) {
                std::pair<uint128_t, uint128_t> dm = divmod(n, ten);
                result = char('0' + static_cast<uint64_t>(dm.second.lo)) + result;
                n = dm.first;
            }
            return result;
        }

        static uint128_t from_dec(const std::string& s) {
            uint128_t result(0);
            const uint128_t ten(10);
            for (size_t i = 0; i < s.size(); ++i) {
                char c = s[i];
                if (c < '0' || c > '9')
                    throw std::invalid_argument("[-] invalid decimal digit");
                result = result * ten + uint128_t(c - '0');
            }
            return result;
        }

        static uint128_t from_hex(const std::string& s) {
            uint128_t result(0);
            for (size_t i = 0; i < s.size(); ++i) {
                char c = s[i];
                result <<= 4;
                if (c >= '0' && c <= '9')
                    result += uint128_t(c - '0');
                else if (c >= 'a' && c <= 'f')
                    result += uint128_t(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F')
                    result += uint128_t(c - 'A' + 10);
                else
                    throw std::invalid_argument("[-] invalid hex digit");
            }
            return result;
        }
    };

    constexpr uint128_t operator""_u128(unsigned long long value) noexcept {
        return uint128_t(value);
    }

    inline std::ostream& operator<<(std::ostream& os, const uint128_t& val) {
        if ((os.flags() & std::ios::basefield) == std::ios::hex) {
            os << val.to_hex();
        }
        else {
            os << val.to_dec();
        }
        return os;
    }

}

namespace std {
    template <>
    struct numeric_limits<u128::uint128_t> {
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = false;
        static constexpr bool is_integer = true;
        static constexpr bool is_exact = true;
        static constexpr int digits = 128;
        static constexpr int digits10 = 38;
        static constexpr u128::uint128_t min() noexcept { return u128::uint128_t(0); }
        static constexpr u128::uint128_t max() noexcept {
            return u128::uint128_t(UINT64_MAX, UINT64_MAX);
        }
        static constexpr u128::uint128_t lowest() noexcept { return min(); }
    };
}
