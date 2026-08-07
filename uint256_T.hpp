#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>

namespace u256 {
    class uint256_t {
    public:
        // Build
        constexpr uint256_t() noexcept : limbs_{ 0, 0, 0, 0 } {}

        constexpr uint256_t(uint64_t lo) noexcept : limbs_{ lo, 0, 0, 0 } {}

        constexpr uint256_t(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3) noexcept
            : limbs_{ l0, l1, l2, l3 } {
        }

        explicit uint256_t(const std::string& dec) : limbs_{ 0, 0, 0, 0 } {
            if (dec.empty()) return;
            uint256_t ten(10ULL);
            for (char c : dec) {
                if (c < '0' || c > '9')
                    throw std::invalid_argument("[-] uint256_t: invalid decimal digit");
                *this = *this * ten + uint256_t(static_cast<uint64_t>(c - '0'));
            }
        }

        // Factories
        static constexpr uint256_t zero() noexcept { return uint256_t(); }
        static constexpr uint256_t one() noexcept { return uint256_t(1); }
        static constexpr uint256_t max_val() noexcept {
            return uint256_t(~0ULL, ~0ULL, ~0ULL, ~0ULL);
        }

        // Visit
        constexpr uint64_t lo() const noexcept { return limbs_[0]; }
        constexpr uint64_t hi() const noexcept { return limbs_[3]; }
        constexpr uint64_t limb(int i) const noexcept { return limbs_[i]; }
        constexpr uint64_t to_uint64() const noexcept { return limbs_[0]; }

        constexpr bool is_zero() const noexcept {
            return limbs_[0] == 0 && limbs_[1] == 0 && limbs_[2] == 0 && limbs_[3] == 0;
        }

        constexpr int bits() const noexcept {
            for (int i = 3; i >= 0; --i) {
                if (limbs_[i] != 0)
                    return i * 64 + 64 - clz64(limbs_[i]);
            }
            return 0;
        }

        // Plus
        constexpr uint256_t operator+(const uint256_t& r) const noexcept {
            uint256_t res;
            uint64_t carry = 0;
            for (int i = 0; i < 4; ++i) {
                uint64_t a = limbs_[i], b = r.limbs_[i];
                uint64_t s = a + b;
                uint64_t c1 = (s < a) ? 1ULL : 0ULL;
                uint64_t s2 = s + carry;
                uint64_t c2 = (s2 < s) ? 1ULL : 0ULL;
                res.limbs_[i] = s2;
                carry = c1 + c2;
            }
            return res;
        }

        constexpr uint256_t& operator+=(const uint256_t& r) noexcept {
            *this = *this + r;
            return *this;
        }

        constexpr uint256_t operator++() noexcept {
            *this = *this + uint256_t(1);
            return *this;
        }

        constexpr uint256_t operator++(int) noexcept {
            uint256_t t = *this;
            ++(*this);
            return t;
        }

        // Minus
        constexpr uint256_t operator-(const uint256_t& r) const noexcept {
            uint256_t res;
            uint64_t borrow = 0;
            for (int i = 0; i < 4; ++i) {
                uint64_t a = limbs_[i], b = r.limbs_[i];
                uint64_t d = a - b;
                uint64_t b1 = (d > a) ? 1ULL : 0ULL;
                uint64_t d2 = d - borrow;
                uint64_t b2 = (d2 > d) ? 1ULL : 0ULL;
                res.limbs_[i] = d2;
                borrow = b1 + b2;
            }
            return res;
        }

        constexpr uint256_t& operator-=(const uint256_t& r) noexcept {
            *this = *this - r;
            return *this;
        }

        constexpr uint256_t operator--() noexcept {
            *this = *this - uint256_t(1);
            return *this;
        }

        constexpr uint256_t operator--(int) noexcept {
            uint256_t t = *this;
            --(*this);
            return t;
        }

        // Mul fuck this
        constexpr uint256_t operator*(const uint256_t& r) const noexcept {
            uint64_t t[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            for (int i = 0; i < 4; ++i) {
                uint64_t carry = 0;
                for (int j = 0; j < 4; ++j) {
                    uint64_t lo = 0, hi = 0;
                    mul64(limbs_[i], r.limbs_[j], lo, hi);

                    uint64_t old = t[i + j];
                    uint64_t sum = old + lo;
                    uint64_t c1 = (sum < old) ? 1ULL : 0ULL;
                    uint64_t sum2 = sum + carry;
                    uint64_t c2 = (sum2 < sum) ? 1ULL : 0ULL;
                    t[i + j] = sum2;
                    carry = hi + c1 + c2;
                }
                t[i + 4] = carry;
            }
            return uint256_t(t[0], t[1], t[2], t[3]);
        }

        constexpr uint256_t& operator*=(const uint256_t& r) noexcept {
            *this = *this * r;
            return *this;
        }

        // division
        uint256_t operator/(const uint256_t& r) const {
            if (r.is_zero()) throw std::domain_error("[-] uint256_t: division by zero");
            uint256_t q, rem;
            divmod(*this, r, q, rem);
            return q;
        }

        uint256_t& operator/=(const uint256_t& r) {
            *this = *this / r;
            return *this;
        }

        uint256_t operator%(const uint256_t& r) const {
            if (r.is_zero()) throw std::domain_error("[-] uint256_t: modulo by zero");
            uint256_t q, rem;
            divmod(*this, r, q, rem);
            return rem;
        }

        uint256_t& operator%=(const uint256_t& r) {
            *this = *this % r;
            return *this;
        }

        // bit
        constexpr uint256_t operator&(const uint256_t& r) const noexcept {
            return uint256_t(
                limbs_[0] & r.limbs_[0],
                limbs_[1] & r.limbs_[1],
                limbs_[2] & r.limbs_[2],
                limbs_[3] & r.limbs_[3]
            );
        }

        constexpr uint256_t operator|(const uint256_t& r) const noexcept {
            return uint256_t(
                limbs_[0] | r.limbs_[0],
                limbs_[1] | r.limbs_[1],
                limbs_[2] | r.limbs_[2],
                limbs_[3] | r.limbs_[3]
            );
        }

        constexpr uint256_t operator^(const uint256_t& r) const noexcept {
            return uint256_t(
                limbs_[0] ^ r.limbs_[0],
                limbs_[1] ^ r.limbs_[1],
                limbs_[2] ^ r.limbs_[2],
                limbs_[3] ^ r.limbs_[3]
            );
        }

        constexpr uint256_t operator~() const noexcept {
            return uint256_t(~limbs_[0], ~limbs_[1], ~limbs_[2], ~limbs_[3]);
        }

        constexpr uint256_t& operator&=(const uint256_t& r) noexcept {
            *this = *this & r; return *this;
        }
        constexpr uint256_t& operator|=(const uint256_t& r) noexcept {
            *this = *this | r; return *this;
        }
        constexpr uint256_t& operator^=(const uint256_t& r) noexcept {
            *this = *this ^ r; return *this;
        }

        // shift fuck
        constexpr uint256_t operator<<(int shift) const noexcept {
            if (shift <= 0) return *this;
            if (shift >= 256) return uint256_t();

            uint256_t res;
            int limb_s = shift / 64;
            int bit_s = shift % 64;

            for (int i = 0; i < limb_s; ++i) res.limbs_[i] = 0;

            if (bit_s == 0) {
                for (int i = 0; i < 4 - limb_s; ++i)
                    res.limbs_[i + limb_s] = limbs_[i];
            }
            else {
                uint64_t carry = 0;
                for (int i = 0; i < 4 - limb_s; ++i) {
                    uint64_t val = limbs_[i];
                    res.limbs_[i + limb_s] = (val << bit_s) | carry;
                    carry = val >> (64 - bit_s);
                }
            }
            return res;
        }

        constexpr uint256_t operator>>(int shift) const noexcept {
            if (shift <= 0) return *this;
            if (shift >= 256) return uint256_t();

            uint256_t res;
            int limb_s = shift / 64;
            //int bit_s = shift / 64;
            int bit_s = shift % 64;

            if (bit_s == 0) {
                for (int i = 0; i < 4 - limb_s; ++i)
                    res.limbs_[i] = limbs_[i + limb_s];
            }
            else {
                uint64_t carry = 0;
                for (int i = 3 - limb_s; i >= 0; --i) {
                    uint64_t val = limbs_[i + limb_s];
                    res.limbs_[i] = (val >> bit_s) | carry;
                    carry = val << (64 - bit_s);
                }
            }
            for (int i = 4 - limb_s; i < 4; ++i) res.limbs_[i] = 0;
            return res;
        }

        constexpr uint256_t& operator<<=(int shift) noexcept {
            *this = *this << shift; return *this;
        }
        constexpr uint256_t& operator>>=(int shift) noexcept {
            *this = *this >> shift; return *this;
        }

        // compare
        constexpr bool operator==(const uint256_t& r) const noexcept {
            return limbs_[0] == r.limbs_[0] && limbs_[1] == r.limbs_[1]
                && limbs_[2] == r.limbs_[2] && limbs_[3] == r.limbs_[3];
        }

        constexpr bool operator!=(const uint256_t& r) const noexcept {
            return !(*this == r);
        }

        constexpr bool operator<(const uint256_t& r) const noexcept {
            for (int i = 3; i >= 0; --i) {
                if (limbs_[i] < r.limbs_[i]) return true;
                if (limbs_[i] > r.limbs_[i]) return false;
            }
            return false;
        }

        constexpr bool operator<=(const uint256_t& r) const noexcept {
            return !(r < *this);
        }

        constexpr bool operator>(const uint256_t& r) const noexcept {
            return r < *this;
        }

        constexpr bool operator>=(const uint256_t& r) const noexcept {
            return !(*this < r);
        }

        // math
        uint256_t gcd(uint256_t r) const noexcept {
            uint256_t a = *this;
            while (!r.is_zero()) {
                uint256_t t = r;
                r = a % r;
                a = t;
            }
            return a;
        }

        uint256_t lcm(const uint256_t& r) const {
            if (is_zero() || r.is_zero()) return uint256_t();
            return (*this / gcd(r)) * r;
        }

        uint256_t pow(uint256_t exp) const noexcept {
            uint256_t base = *this;
            uint256_t result(1);
            while (!exp.is_zero()) {
                if ((exp.limbs_[0] & 1) != 0)
                    result = result * base;
                base = base * base;
                exp = exp >> 1;
            }
            return result;
        }

        uint256_t mod_pow(uint256_t exp, const uint256_t& mod) const noexcept {
            uint256_t base = *this % mod;
            uint256_t result(1);
            while (!exp.is_zero()) {
                if ((exp.limbs_[0] & 1) != 0)
                    result = (result * base) % mod;
                base = (base * base) % mod;
                exp = exp >> 1;
            }
            return result;
        }

        // convert
        std::string to_string() const {
            if (is_zero()) return "0";
            std::string s;
            uint256_t n = *this;
            uint256_t ten(10ULL);
            while (!n.is_zero()) {
                uint256_t q, r;
                divmod(n, ten, q, r);
                s = char('0' + static_cast<char>(r.limbs_[0])) + s;
                n = q;
            }
            return s;
        }

        std::string to_hex() const {
            if (is_zero()) return "0";
            std::ostringstream oss;
            bool leading = true;
            for (int i = 3; i >= 0; --i) {
                if (limbs_[i] != 0 || !leading) {
                    if (leading) {
                        oss << std::hex << std::uppercase << limbs_[i];
                    }
                    else {
                        oss << std::hex << std::uppercase
                            << std::setw(16) << std::setfill('0') << limbs_[i];
                    }
                    leading = false;
                }
            }
            return oss.str();
        }

        // 512 mul (son of the bitch)
        void mul_full(const uint256_t& r, uint256_t& out_lo, uint256_t& out_hi) const noexcept {
            uint64_t t[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            for (int i = 0; i < 4; ++i) {
                uint64_t carry = 0;
                for (int j = 0; j < 4; ++j) {
                    uint64_t lo = 0, hi = 0;
                    mul64(limbs_[i], r.limbs_[j], lo, hi);
                    uint64_t old = t[i + j];
                    uint64_t sum = old + lo;
                    uint64_t c1 = (sum < old) ? 1ULL : 0ULL;
                    uint64_t sum2 = sum + carry;
                    uint64_t c2 = (sum2 < sum) ? 1ULL : 0ULL;
                    t[i + j] = sum2;
                    carry = hi + c1 + c2;
                }
                t[i + 4] = carry;
            }

            //out_lo = uint256_t(t[0], t[1], t[2], t[4]);
            out_lo = uint256_t(t[0], t[1], t[2], t[3]);
            out_hi = uint256_t(t[4], t[5], t[6], t[7]);
        }

    private:
        uint64_t limbs_[4];

        // 128 mul
        static constexpr void mul64(uint64_t a, uint64_t b,
            uint64_t& lo, uint64_t& hi) noexcept {
            uint64_t a_lo = a & 0xFFFFFFFF;
            uint64_t a_hi = a >> 32;
            uint64_t b_lo = b & 0xFFFFFFFF;
            uint64_t b_hi = b >> 32;

            uint64_t p0 = a_lo * b_lo;
            uint64_t p1 = a_lo * b_hi;
            uint64_t p2 = a_hi * b_lo;
            uint64_t p3 = a_hi * b_hi;

            uint64_t mid = p1 + p2;
            uint64_t mid_carry = (mid < p1) ? (1ULL << 32) : 0ULL;

            lo = p0 + (mid << 32);
            hi = p3 + (mid >> 32) + mid_carry;
            if (lo < p0) hi++;
        }

        static constexpr int clz64(uint64_t x) noexcept {
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

        static void divmod(const uint256_t& a, const uint256_t& b,
            uint256_t& q, uint256_t& rem) noexcept {
            rem = a;
            q = uint256_t();

            int b_bits = b.bits();
            if (b_bits == 0) return;

            int a_bits = a.bits();

            for (int i = a_bits - b_bits; i >= 0; --i) {
                uint256_t shifted = b << i;
                if (rem >= shifted) {
                    rem = rem - shifted;
                    if (i < 256) {
                        q.limbs_[i / 64] |= (1ULL << (i % 64));
                    }
                }
            }
        }
    };

    constexpr uint256_t operator*(uint64_t s, const uint256_t& v) noexcept {
        return v * uint256_t(s);
    }

    inline std::ostream& operator<<(std::ostream& os, const uint256_t& v) {
        os << v.to_string();
        return os;
    }
    // end end end.....
}
