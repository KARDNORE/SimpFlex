# Bit API Development Documentation

## MeskBit API Development Documentation

### Overview

`MeskBit` provides a comprehensive set of **compile‑time and runtime bitwise operations** for all fundamental integer types (`uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`, and their signed counterparts). It fills the gaps left by the C++ standard library, offering bit reversal, rotation, counting, leading/trailing zero detection, parity, and power‑of‑two utilities. All functions are **header‑only**, **zero‑dependency**, and optimized using compiler intrinsics on MSVC (e.g., `_rotl64`, `__popcnt64`, `_BitScanReverse`), with portable fallback implementations for other compilers.

---

### Header

```cpp
#include "MeskBit.hpp"
```

### Namespace

```cpp
using namespace SimpFlex;
```

---

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `bit_reverse` | `T bit_reverse(T x) noexcept` | Reverses all bits of `x` (MSB ↔ LSB). |
| `bit_rotate_left` | `T bit_rotate_left(T x, int n) noexcept` | Rotates bits left by `n` positions (cyclic shift). |
| `bit_rotate_right` | `T bit_rotate_right(T x, int n) noexcept` | Rotates bits right by `n` positions (cyclic shift). |
| `bit_count` | `int bit_count(T x) noexcept` | Returns the number of set bits (popcount). |
| `bit_count_zero` | `int bit_count_zero(T x) noexcept` | Returns the number of zero bits. |
| `bit_length` | `int bit_length(T x) noexcept` | Returns the position (1‑based) of the highest set bit; returns `0` if `x == 0`. |
| `clz` | `int clz(T x) noexcept` | Counts leading zeros (from MSB to the first `1`); returns the bit width for zero. |
| `ctz` | `int ctz(T x) noexcept` | Counts trailing zeros (from LSB to the first `1`); returns the bit width for zero. |
| `bit_parity` | `int bit_parity(T x) noexcept` | Returns `1` if the number of set bits is odd, otherwise `0`. |
| `is_power_of_two` | `bool is_power_of_two(T x) noexcept` | Returns `true` if `x` is a power of two (including `1`). |
| `next_power_of_two` | `T next_power_of_two(T x) noexcept` | Returns the smallest power of two not less than `x`; `0` yields `1`. |
| `prev_power_of_two` | `T prev_power_of_two(T x) noexcept` | Returns the largest power of two not greater than `x`; `0` yields `0`. |

---

### Example

```cpp
#include "MeskBit.hpp"
#include <iostream>
#include <cstdio>

int main() {
    using namespace SimpFlex;

    std::printf("bit_count(0x5A5A5A5Au) = %d\n", bit_count(0x5A5A5A5Au));

    uint32_t rev = bit_reverse(0x01234567u);
    std::printf("rev(0x01234567) = 0x%08X\n", rev);

    uint64_t x = 0x0000000000000001ULL;
    uint64_t rot = bit_rotate_left(x, 63);
    std::printf("rotl64(1,63) = 0x%016llX\n", rot);

    std::printf("clz(0x0000FFFF) = %d\n", clz(0x0000FFFFu));
    std::printf("ctz(0x00010000) = %d\n", ctz(0x00010000u));

    std::printf("bit_length(0x10000) = %d\n", bit_length(0x10000u));

    std::printf("parity(0x07) = %d\n", bit_parity(0x07u));
    std::printf("is_power_of_two(1024) = %d\n", is_power_of_two(1024u));

    std::printf("next_power_of_two(1023) = %u\n", next_power_of_two(1023u));
    std::printf("prev_power_of_two(1023) = %u\n", prev_power_of_two(1023u));

    return 0;
}
```

## MeskBit_R API Development Documentation

### Overview

`MeskBit_R` provides advanced bit‑manipulation operations that go beyond the standard library, including bit‑level interleaving, compression, expansion, permutation, gathering/scattering, bit‑stream packing/unpacking, and bit‑matrix transposition/rotation. These operations are essential for cryptography, data compression, protocol parsing, SIMD‑style bit packing, and graphics processing.All advanced functions operate on fundamental unsigned integer types (`uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`) and return `uint128_t` (a custom 128‑bit structure) where needed. On MSVC with BMI2 support, `bit_compress` and `bit_expand` use the hardware `pext` and `pdep` instructions for optimal performance; fallback implementations are **not** provided, so these functions require a CPU with BMI2 (Intel Haswell or later, AMD Excavator or later).

---

### Header

```cpp
#include "MeskBit_R.hpp"
```

### Namespace

```cpp
using namespace SimpFlex;
```

---

### Types

| Type | Description |
|------|-------------|
| `uint128_t` | Custom 128‑bit unsigned integer structure with `lo` and `hi` 64‑bit fields. |

---

### Interleaving / Deinterleaving

| Function | Signature | Description |
|----------|-----------|-------------|
| `bit_interleave` | `uint128_t bit_interleave(T a, T b) noexcept` | Interleaves the bits of `a` and `b`: `a0 b0 a1 b1 ...` (LSB first). |
| `bit_deinterleave` | `void bit_deinterleave(uint128_t x, T& a, T& b) noexcept` | Extracts the original `a` and `b` from an interleaved 128‑bit value. |

### Compression

| Function | Signature | Description |
|----------|-----------|-------------|
| `bit_compress` | `T bit_compress(T x, T mask) noexcept` | Compresses the bits of `x` selected by `mask` into the low bits of the result (using `pext`). |
| `bit_expand` | `T bit_expand(T x, T mask) noexcept` | Expands the low bits of `x` to positions set in `mask` (using `pdep`). |

### Permutation, Gather, Scatter

| Function | Signature | Description |
|----------|-----------|-------------|
| `bit_permute` | `T bit_permute(T x, const int* table, int n) noexcept` | Rearranges the lowest `n` bits of `x` according to `table` (source bit `i` → destination `table[i]`). |
| `bit_gather` | `T bit_gather(T x, const int* indices, int n) noexcept` | Collects bits from `x` at positions `indices[i]` and packs them into the low `n` bits of the result. |
| `bit_scatter` | `T bit_scatter(T x, const int* indices, int n) noexcept` | Scatters the low `n` bits of `x` to positions `indices[i]` in the result. |

### Other Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `bit_select` | `T bit_select(T a, T b, T mask) noexcept` | Selects bits from `a` where `mask` is 1, and from `b` where `mask` is 0. Equivalent to `(a & mask) \| (b & ~mask)`. |
| `pack_bits` | `uint128_t pack_bits(const T* src, int n, int bits_per_elem) noexcept` | Packs `n` elements of `bits_per_elem` bits each from `src` into a 128‑bit value. |
| `unpack_bits` | `void unpack_bits(uint128_t packed, T* dst, int n, int bits_per_elem) noexcept` | Unpacks `n` elements of `bits_per_elem` bits each from `packed` into `dst`. |
| `bit_stream_extract` | `uint128_t bit_stream_extract(const uint8_t* data, int bit_offset, int n_bits) noexcept` | Extracts `n_bits` bits from a byte stream starting at `bit_offset`. |
| `bit_stream_insert` | `void bit_stream_insert(uint8_t* data, int bit_offset, uint128_t value, int n_bits) noexcept` | Inserts the low `n_bits` of `value` into a byte stream at `bit_offset`. |
| `bit_matrix_transpose` | `uint128_t bit_matrix_transpose(uint128_t x, int rows, int cols) noexcept` | Transposes a `rows × cols` bit matrix stored in `x` (requires `rows * cols ≤ 128`). |
| `bit_matrix_rotate` | `uint128_t bit_matrix_rotate(uint128_t x, int rows, int cols, int k) noexcept` | Rotates the bit matrix clockwise by `k` quarters (1, 2, or 3). |

---

### Example

```cpp
#include "MeskBit_R.hpp"
#include <cstdio>

int main() {
    using namespace SimpFlex;

    uint32_t a = 0x0000FFFF, b = 0xFFFF0000;
    uint128_t inter = bit_interleave(a, b);
    printf("interleaved lo = 0x%016llX\n", inter.lo);

    uint32_t a_out, b_out;
    bit_deinterleave(inter, a_out, b_out);
    uint32_t x = 0b10110010, mask = 0b11010011;
    uint32_t comp = bit_compress(x, mask);
    uint32_t exp = bit_expand(comp, mask);

    int table[] = { 7, 6, 5, 4, 3, 2, 1, 0 };
    uint8_t perm = bit_permute((uint8_t)0b10110100, table, 8);

    int indices[] = { 7, 5, 3, 1 };
    uint8_t gathered = bit_gather((uint8_t)0b10110100, indices, 4);
    uint8_t scattered = bit_scatter((uint8_t)0b1010, indices, 4);

    uint8_t src[] = { 1,2,3,4,5,6,7,8 };
    uint128_t packed = pack_bits(src, 8, 4);

    uint128_t mat(0, 0b110010101);
    uint128_t trans = bit_matrix_transpose(mat, 3, 3);

    return 0;
}
```

## MeskBit_S API Development Documentation

### Overview

`MeskBit_S` provides **compile‑time (`constexpr`) versions** of core bit operations, along with additional utilities for bit‑field packing, byte/bit‑order swapping, and sign extension. These functions are designed for use in constant expressions, enabling compile‑time computation and zero‑runtime overhead in many scenarios.All `constexpr` functions are implemented using pure C++ bit‑twiddling (without compiler intrinsics), making them portable and safe for use in `static_assert` and constant initializers.

---

### Header

```cpp
#include "MeskBit_S.hpp"
```

### Namespace

```cpp
using namespace SimpFlex;
```

---

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `constexpr_bit_reverse` | `constexpr T bit_reverse(T x) noexcept` | Reverses all bits of `x` at compile‑time (MSB ↔ LSB). |
| `constexpr_bit_count` | `constexpr int bit_count(T x) noexcept` | Returns the number of set bits (popcount) at compile‑time. |
| `constexpr_clz` | `constexpr int clz(T x) noexcept` | Counts leading zeros at compile‑time (returns bit width for zero). |
| `constexpr_ctz` | `constexpr int ctz(T x) noexcept` | Counts trailing zeros at compile‑time (returns bit width for zero). |
| `constexpr_bit_permute` | `constexpr T bit_permute(T x, const int* table, int n) noexcept` | Applies a bit‑permutation specified by `table` at compile‑time. |
| `pack_fields` | `uint128_t pack_fields(std::initializer_list<std::pair<int, uint64_t>> fields) noexcept` | Packs multiple bit‑fields (width, value) into a `uint128_t`. |
| `unpack_fields` | `std::vector<uint64_t> unpack_fields(uint128_t x, std::initializer_list<int> widths)` | Unpacks a `uint128_t` into fields of given widths, returning the values. |
| `byte_swap` | `T byte_swap(T x) noexcept` | Swaps the byte order of `x` (little‑endian ↔ big‑endian). Uses compiler intrinsics on MSVC. |
| `bit_endian_swap` | `T bit_endian_swap(T x) noexcept` | Reverses the bit order **inside each byte** (not the full bit‑reversal). |
| `sign_extend` | `T sign_extend(T x, int src_bits) noexcept` | Extends the sign bit of a `src_bits`‑wide value to the full width of `T`. |

---

### Example

```cpp
#include "MeskBit_S.hpp"
#include <cstdio>

int main() {
    using namespace SimpFlex;

    constexpr uint32_t rev = constexpr_bit_reverse(0x01234567u);
    static_assert(rev == 0xE6A2C480u, "");

    static_assert(constexpr_bit_count(0x5A5A5A5Au) == 16, "");

    static_assert(constexpr_clz(0x0000FFFFu) == 16, "");
    static_assert(constexpr_ctz(0xFFFF0000u) == 16, "");

    constexpr int table[] = { 7,6,5,4,3,2,1,0 };
    constexpr uint8_t p = constexpr_bit_permute(uint8_t(0b10110100), table, 8);
    static_assert(p == 0b00101101, "");

    auto packed = pack_fields({ {4, 0xA}, {8, 0xBC}, {12, 0xDEF}, {3, 0x5} });
    auto fields = unpack_fields(packed, { 4, 8, 12, 3 });

    uint32_t swapped = byte_swap(0x12345678u);

    uint8_t bit_swapped = bit_endian_swap(uint8_t(0x12));

    int32_t neg = sign_extend(0x0F, 4);
    int32_t pos = sign_extend(0x07, 4);
    return 0;
}
```
