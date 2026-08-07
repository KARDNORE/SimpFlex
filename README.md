#                                                                                 SimpFlex
A powerful library created for MSVC

Under MSVC, this is a low-level computing toolbox that allows you to use it directly.

128 bit integers, 256 bit integers, bitwise operations, dense matrices, sparse matrices - all zero dependencies, pure header files.

# What's this?

SimpFlex is a pure header file C++low-level computing library designed specifically for MSVC environments

# What's inside

uint128_t / uint256_t, Bit Engine, Matrix / DMatrix, Sparse matrix CSR/ELL/SELL+SpMV

# Performance

just like uint128_t:

Environment:i5-1035G4 CPU @ 1.10GHz


----------------------------------------------------

Performance test for uint128_t (10000000 iterations each)

Addition                   27698 us         2.76985 ns/op       361030380 ops/sec

Subtraction                26760 us         2.67609 ns/op       373679510 ops/sec

Multiplication             80420 us         8.04209 ns/op       124345785 ops/sec

Division                 2672215 us         267.222 ns/op         3742213 ops/sec

Modulo                   2187476 us         218.748 ns/op         4571477 ops/sec

Left Shift                 66743 us         6.67434 ns/op       149827548 ops/sec

Right Shift                78392 us         7.83925 ns/op       127563223 ops/sec

XOR                        28700 us         2.87009 ns/op       348421129 ops/sec

Comparison (<)             51303 us         5.13037 ns/op       194917715 ops/sec

Global accumulator (checksum): ED9BF929CA671705DCF1675CE06CCF33

----------------------------------------------------

# Environment

Visual Studio 2022

ISO C++ 20


# License

Just use it casually, don't delete copyright information

