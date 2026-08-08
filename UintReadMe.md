# Uint API Development Documentation

## `u128::uint128_t` API

### Overview

`uint128_t` is a 128‑bit unsigned integer type designed specifically for MSVC environments, filling the gap in the compiler's lack of native 128‑bit integers. It provides an operation interface consistent with the built‑in integer type, supports all arithmetic, bitwise operations, comparisons, conversions, and formatting functions, and all operations are available as `constexpr` (except for some that involve I/O).

---

### Header

```cpp
#include "uint128_t.hpp"
```

### Namespace

```cpp
using namespace u128;
```

---

### Constructors

| Constructor | Description |
|-------------|-------------|
| `uint128_t() noexcept` | Default constructor, value `0`. |
| `uint128_t(T value) noexcept` | Construct from any integer type ≤64 bits (signed or unsigned), higher bits zero‑cleared. |
| `uint128_t(uint64_t high, uint64_t low) noexcept` | Construct from high and low 64‑bit words. |
| `uint128_t(const std::string& dec)` | Parse from decimal string. Use **from_dec** function for clarity. |

---

### Factories

| Function | Returns |
|----------|---------|
| `static constexpr uint128_t zero() noexcept` | `0` |
| `static constexpr uint128_t one() noexcept` | `1` |
| `static constexpr uint128_t max_val() noexcept` | All bits set to `1` |

---

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `is_zero` | `bool is_zero() const noexcept` | `true` if value is zero |
| `bits` | `int bits() const noexcept` | Position (1‑based) of highest set bit, or `0` if zero |
| `clz` | `int clz() const noexcept` | Count leading zeros (returns `128` for zero) |
| `ctz` | `int ctz() const noexcept` | Count trailing zeros (returns `128` for zero) |
| `popcount` | `int popcount() const noexcept` | Number of set bits |
| `to_dec` | `std::string to_dec() const` | Decimal string representation |
| `to_hex` | `std::string to_hex() const` | Hexadecimal string (uppercase, no prefix) |
| `from_dec` | `static uint128_t from_dec(const std::string& s)` | Parse decimal string (throws) |
| `from_hex` | `static uint128_t from_hex(const std::string& s)` | Parse hex string (case‑insensitive, ignores `0x`) |

---

### Operators

| Operator | Semantics |
|----------|-----------|
| `+` (unary) | Unary plus |
| `-` (unary) | Unary minus (kept for genericity) |
| `+` / `+=` | Addition, wraps modulo 2¹²⁸ |
| `-` / `-=` | Subtraction, wraps with borrow |
| `*` / `*=` | Multiplication (low 128 bits, overflow discarded) |
| `/` / `/=` | Integer division (throws `std::domain_error` on zero divisor) |
| `%` / `%=` | Modulo (throws on zero divisor) |
| `++` / `--` | Pre‑/post‑increment/decrement |
| `~` | Bitwise NOT |
| `&` / `&=` | Bitwise AND |
| `\|` / `\|=` | Bitwise OR |
| `^` / `^=` | Bitwise XOR |
| `<<` / `<<=` | Logical left shift (≥128 yields zero) |
| `>>` / `>>=` | Logical right shift (zero‑extended) |

---

### Conversions

Explicit conversions (use `static_cast`):

| To | Syntax |
|----|--------|
| `bool` | `explicit operator bool() const` |
| `uint64_t` | `explicit operator uint64_t() const` |
| `uint32_t` | `explicit operator uint32_t() const` |
| `uint16_t` | `explicit operator uint16_t() const` |
| `uint8_t` | `explicit operator uint8_t() const` |

---

### Output

```cpp
std::ostream& operator<<(std::ostream& os, const uint128_t& val)
```

- Default: decimal.
- If `std::hex` is set on the stream, outputs uppercase hexadecimal (no prefix).

---

### Example

```cpp
#include "uint128_T.hpp"
#include <iostream>

int main() {
    using namespace u128;

    uint128_t a;
    uint128_t b(42);
    uint128_t c(0x1234567890ABCDEFULL);
    uint128_t d(0xFEDCBA9876543210ULL, 0x1234567890ABCDEFULL);
    uint128_t e = uint128_t::from_dec("12345678901234567890");

    uint128_t sum = b + c;
    uint128_t diff = c - b;
    uint128_t prod = c * d;
    uint128_t quot = prod / c;
    uint128_t rem = prod % c;

    uint128_t and_ = a & b;
    uint128_t or_  = a | b;
    uint128_t xor_ = a ^ b;
    uint128_t not_ = ~a;
    uint128_t shl  = a << 4;
    uint128_t shr  = a >> 4;

    ++a;
    a--;
    a += 10;

    if (a < b) { }
    if (a == b) { }

    std::cout << "a = " << a.to_dec() << "\n";
    std::cout << "a = 0x" << a.to_hex() << "\n";

    uint128_t fromStr = uint128_t::from_dec("12345678901234567890");
    uint128_t fromHex = uint128_t::from_hex("DEADBEEFCAFEBABE");

    int bits = a.bits();
    int pop  = a.popcount();
    int clz  = a.clz();
    int ctz  = a.ctz();

    uint64_t lo = static_cast<uint64_t>(a);
    uint32_t low32 = static_cast<uint32_t>(a);
    bool isNonZero = static_cast<bool>(a);

    std::cout << std::dec << a << "\n";
    std::cout << std::hex << a << "\n";

    return 0;
}
```

---

## `u256::uint256_t` API

### Overview

`uint256_t` is a 256‑bit unsigned integer type designed for cryptographic, blockchain, and high‑precision numerical applications where 128 bits are insufficient. It provides a complete set of arithmetic, bitwise, comparison, conversion, and mathematical operations. Most functions are `constexpr`‑compatible (except I/O and parsing).

---

### Header

```cpp
#include "uint256_T.hpp"
```

### Namespace

```cpp
using namespace u256;
```

---

### Constructors

| Constructor | Description |
|-------------|-------------|
| `uint256_t() noexcept` | Default, value `0`. |
| `uint256_t(uint64_t lo) noexcept` | From a 64‑bit value (higher limbs zero). |
| `uint256_t(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3) noexcept` | From four 64‑bit words (least significant first). |
| `explicit uint256_t(const std::string& dec)` | Parse from a decimal string (throws `std::invalid_argument`). |

---

### Factories

| Function | Returns |
|----------|---------|
| `static constexpr uint256_t zero() noexcept` | `0` |
| `static constexpr uint256_t one() noexcept` | `1` |
| `static constexpr uint256_t max_val() noexcept` | All bits set to `1` |

---

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `lo` | `uint64_t lo() const noexcept` | Lowest 64‑bit word |
| `hi` | `uint64_t hi() const noexcept` | Highest 64‑bit word |
| `limb` | `uint64_t limb(int i) const noexcept` | `i`‑th word (0 = LSB) |
| `to_uint64` | `uint64_t to_uint64() const noexcept` | Low 64 bits as `uint64_t` |
| `is_zero` | `bool is_zero() const noexcept` | `true` if zero |
| `bits` | `int bits() const noexcept` | Position of highest set bit (1‑based), or 0 |
| `clz` | `int clz() const noexcept` | Count leading zeros (returns 256 for zero) |
| `ctz` | `int ctz() const noexcept` | Count trailing zeros (returns 256 for zero) |
| `popcount` | `int popcount() const noexcept` | Number of set bits |
| `to_string` | `std::string to_string() const` | Decimal string |
| `to_hex` | `std::string to_hex() const` | Hexadecimal string (uppercase, no prefix) |

---

### Operators

| Operator | Semantics |
|----------|-----------|
| `+` (unary) | Unary plus |
| `-` (unary) | Unary minus |
| `+` / `+=` | Addition (mod 2²⁵⁶) |
| `-` / `-=` | Subtraction (with borrow) |
| `*` / `*=` | Multiplication (low 256 bits, overflow discarded) |
| `/` / `/=` | Division (throws on zero divisor) |
| `%` / `%=` | Modulo (throws on zero divisor) |
| `++` / `--` | Pre‑/post‑increment/decrement |
| `~` | Bitwise NOT |
| `&` / `&=` | Bitwise AND |
| `\|` / `\|=` | Bitwise OR |
| `^` / `^=` | Bitwise XOR |
| `<<` / `<<=` | Logical left shift (≥256 yields zero) |
| `>>` / `>>=` | Logical right shift (zero‑extended) |

---


### Conversions

| To | Syntax |
|----|--------|
| `bool` | `explicit operator bool() const` |
| `uint64_t` | `explicit operator uint64_t() const` |

---

### Math Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `gcd` | `uint256_t gcd(uint256_t r) const noexcept` | Greatest common divisor |
| `lcm` | `uint256_t lcm(const uint256_t& r) const` | Least common multiple (0 if either zero) |
| `pow` | `uint256_t pow(uint256_t exp) const noexcept` | Fast exponentiation |
| `mod_pow` | `uint256_t mod_pow(uint256_t exp, const uint256_t& mod) const noexcept` | Modular exponentiation |
| `mul_full` | `void mul_full(const uint256_t& r, uint256_t& out_lo, uint256_t& out_hi) const noexcept` | Full 512‑bit product (low/high halves) |

---

### Output

```cpp
std::ostream& operator<<(std::ostream& os, const uint256_t& val)
```

Always outputs decimal (via `to_string()`). Use `to_hex()` for hexadecimal.

---

### Example

```cpp
#include "uint256_T.hpp"
#include <iostream>
#include <iomanip>

int main() {
    using namespace u256;

    uint256_t a;
    uint256_t b(42);
    uint256_t c(0x1234567890ABCDEFULL, 0x1111222233334444ULL,
        0x5555666677778888ULL, 0x9999AAAABBBBCCCCULL);
    uint256_t d("123456789012345678901234567890");

    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n";
    std::cout << "c = 0x" << c.to_hex() << "\n";
    std::cout << "d = " << d.to_string() << "\n";

    uint256_t sum = b + c;
    uint256_t diff = c - b;
    uint256_t prod = b * c;
    uint256_t quot = prod / b;
    uint256_t rem = prod % b;

    ++b;
    b--;

    uint256_t and_ = a & c;
    uint256_t or_ = a | c;
    uint256_t xor_ = a ^ c;
    uint256_t not_ = ~a;
    uint256_t shl = a << 4;
    uint256_t shr = a >> 4;

    if (b < c) { }
    if (b == c) { }

    std::cout << "is_zero(a) = " << a.is_zero() << "\n";
    std::cout << "bits(c) = " << c.bits() << "\n";

    uint256_t x(48), y(18);
    std::cout << "gcd(48,18) = " << x.gcd(y) << "\n";
    std::cout << "lcm(48,18) = " << x.lcm(y) << "\n";
    std::cout << "2^10 = " << uint256_t(2).pow(uint256_t(10)) << "\n";
    std::cout << "2^10 mod 1000 = " << uint256_t(2).mod_pow(uint256_t(10), uint256_t(1000)) << "\n";

    uint256_t z = uint256_t::zero();
    uint256_t o = uint256_t::one();
    uint256_t m = uint256_t::max_val();

    uint256_t sc = 3 * b;

    uint256_t hi, lo;
    c.mul_full(d, lo, hi);
    std::cout << "c * d (low)  = 0x" << lo.to_hex() << "\n";
    std::cout << "c * d (high) = 0x" << hi.to_hex() << "\n";

    std::cout << std::hex << "hex: " << b << "\n";
    std::cout << std::dec << "dec: " << b << "\n";

    return 0;
}
```
