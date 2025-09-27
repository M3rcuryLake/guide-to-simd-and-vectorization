# SIMD and Vectorization.

According to wikipedia, **Single instruction, multiple data** (**SIMD**) is a type of [parallel computing](https://en.wikipedia.org/wiki/Parallel_computing) (processing) in [Flynn's taxonomy](https://en.wikipedia.org/wiki/Flynn%27s_taxonomy). SIMD describes computers with multiple processing elements that perform the same operation on multiple data points simultaneously. SIMD can be internal (part of the hardware design) and it can be directly accessible through an instruction set architecture (ISA), but it should not be confused with an ISA.

Such machines exploit [data level parallelism](https://en.wikipedia.org/wiki/Data_parallelism), but not [concurrency](https://en.wikipedia.org/wiki/Concurrent_computing): there are simultaneous (parallel) computations, but each unit performs exactly the same instruction at any given moment (just with different data). A simple example is to add many pairs of numbers together, all of the SIMD units are performing an addition, but each one has different pairs of values to add. SIMD is especially applicable to common tasks such as adjusting the contrast in a digital image or adjusting the volume of digital audio. Most modern CPU designs include SIMD instructions to improve the performance of multimedia use. In recent CPUs, SIMD units are tightly coupled with cache hierarchies and prefetch mechanisms, which minimize latency during large block operations

```c
				+---------------+--------------------------------------------+      
				|               |  CPU Core 1 -------------------->  Task 1  |      
				|  parallelism  |  CPU Core 2 -------------------->  Task 2  |      
				|               |  CPU Core 3 -------------------->  Task 3  |      
				|---------------|--------------------------------------------|      
				|               |             ---   ---   --->        Task 1 |      
				|  concurrency  |  CPU Core 1    ---   ---   --->     Task 2 |      
				|               |                    ---        --->  Task 3 |      
				+---------------+--------------------------------------------+      
```

There are multiple generations of ***SIMD instructions.*** 

- MMX : 64 Bit
- SSE : 128 Bit (both float and int, comprises of `4 $*$ float32` )
- AVX and AVX2 : 256 Bit  (`4 $*$ float32`  for float and `4 $*$ int32` for int)
- AVX-512 : 512 Bit (float and int)

## Standard C vs Intrinsics:

Lets take an example from a standard C code :

```c
for (__int 64 i = 0; i < N; i++){
		sumTotal += A[i];
}
```

with intrinsics, it has a speedup of ~4x with no optimisation flags. It may cause a speedup of ~20x after flags.

```c
for (__int 64 i = 0; i < N; i+=8){
		sumTotal256 = __mm256_add_ps(A256, sumtotal256)
}
```

# **SIMD Operations :**

## Assembly Instructions:

In this guide we will be focusing on **AVX (Advanced Vector Extensions).** These are special SIMD instructions added to the x86 architecture by Intel and later supported by AMD. They were first proposed in 2008, showed up in Intel’s *Sandy Bridge* CPUs in 2011, and in AMD’s *Bulldozer* CPUs later that same year. AVX introduced a bunch of new instructions, features, and a cleaner coding style for parallel processing. Refer to this link for list of [Intel CPU MicroArchs](https://en.wikipedia.org/wiki/List_of_Intel_CPU_microarchitectures).

**AVX2**, launched with Intel’s *Haswell* processors in 2013 and later supported by AMD’s [Excavator](https://en.wikipedia.org/wiki/Excavator_(microarchitecture)) processors (Q2 2015) and newer., made SIMD more powerful by adding full 256-bit integer support, gather operations, and better shuffle instructions — so it wasn’t just about floating-point math anymore.

**AVX-512**, announced in 2013, took things further by doubling the width to 512 bits using a new EVEX encoding. It first appeared in Intel’s *Knights Landing* co-processors in 2016, and later made its way into mainstream CPUs starting with *Skylake* server and HEDT chips in 2017.

- `zmm` = 512-bit AVX register (32 floats / 16 doubles / 64 ints / 32 longs depending on type)
- `ymm` = 256-bit AVX register (16 floats / 8 doubles / 32 ints / 16 longs depending on type)
- `xmm` = 128-bit SSE register (low half of ymm)
- `mem` = memory operand
- `{a/u}` = `a` requires memory address to be **aligned to 32 bytes** (for `ymm`) or **16 bytes** (for `xmm`) while `u` works on any memory address, no alignment restriction. Slower on older CPUs, but modern CPUs usually handle it with little or no penalty.
- `{ps}` = packed single-precision floats (32-bit). For example: in 128-bit `xmm` register `ps` (packed single) → 128 ÷ 32 = 8 **floats**
- `{pd}` = packed double-precision floats (64-bit). For example: in 128-bit `xmm` register `pd` (packed single) → 128 ÷ 64 = **4 floats**
- `{ph}` = packed half-precision floats.
- `{ss}` = scalar single floats (32-bit), i.e one **32-bit float**
- `{sd}` = scalar double floats (64-bit). i.e. one **64-bit float**
- `{epuX}` =  extended packed unsigned-integer. Here `x` is the ***element size*** which can be either 8, 16, 32 or 64.
- `{epiX}` =  extended packed signed-integer. Here `x` is the ***element size*** which can be either 8, 16, 32, 64 or 128 .
- `{siX}` =  32, 64, 128, 256 and 512-bit SIMD scalar integer type.
- `{d/q/w/b}` = packed integers (32/64/16/8-bit)

---

## **AVX1 (floating point only)**

| **Category** | **Instructions** | **Operands** |
| --- | --- | --- |
| **Data Transfer** | `VMOVAPS/UPD` | ymm, mem / mem, ymm |
|  | `VBROADCASTSS/SD/F128` | ymm, mem/xmm |
| **Arithmetic (FP)** | `VADDPS/PD`, `VSUBPS/PD`, `VMULPS/PD`, `VDIVPS/PD` | ymm, ymm/mem |
|  | `VMAXPS/PD`, `VMINPS/PD` | ymm, ymm/mem |
|  | `VSQRTPS/PD` | ymm, ymm/mem |
| **Comparison** | `VCMPPS/PD` | ymm, ymm/mem, imm8 |
| **Blend** | `VBLENDPS/PD`, `VBLENDVPS/PD` | ymm, ymm/mem, imm8 |
| **Shuffle / Permute** | `VSHUFPS`, `VSHUFPD` | ymm, ymm/mem, imm8 |
|  | `VPERMILPS/PD` | ymm, ymm/mem, imm8 |
| **Horizontal Ops** | `VHADDPS/PD`, `VHSUBPS/PD` | ymm, ymm/mem |

---

## **AVX2 (supports both floating point and integer)**

| **Category** | **Instructions** | **Operands** |
| --- | --- | --- |
| **Data Transfer** | `VMOVAPS/UPD` | ymm, mem / mem, ymm |
|  | `VPBROADCASTB/W/D/Q` | ymm, mem/xmm |
| **Arithmetic (FP)** | same as AVX1 (`VADDPS/PD`, `VMULPS/PD`, etc.) | ymm, ymm/mem |
| **Arithmetic (Int)** | `VPADDB/W/D/Q`, `VPSUBB/W/D/Q` | ymm, ymm/mem |
|  | `VPMULLW/D` | ymm, ymm/mem |
|  | `VPSLLW/D/Q`, `VPSRLW/D/Q`, `VPSRAW/D` | ymm, imm8 |
| **Comparison (Int)** | `VPCMPEQB/W/D/Q`, `VPCMPGTB/W/D/Q` | ymm, ymm/mem |
| **Logic** | `VPAND/VPOR/VPXOR/VPANDN` | ymm, ymm/mem |
| **Blend** | `VPBLENDD`, `VPBLENDVB` | ymm, ymm/mem, imm8 |
| **Shuffle / Permute** | `VPSHUFB/VPSHUFD`, `VPERMD/Q` | ymm, ymm/mem, imm8 |
|  | `VPERM2I128` | ymm, ymm/mem, imm8 |
| **Gather** | `VGATHERDPS/DPD/QPS/QPD` | ymm, [base+index*scale] |
| **FMA (Fused Multiply-Add)** | `VFMADD132PS/PD`, `VFMADD213PS/PD`, `VFMADD231PS/PD` | ymm, ymm/mem |

AVX1 introduced 256-bit SIMD registers but was limited to floating-point operations on single and double precision values. AVX2 extended the model by adding full integer SIMD support, gather instructions for more flexible memory access, and more advanced shuffle and permutation operations. Alongside AVX2, the FMA3 extension brought fused multiply-add instructions, which are especially valuable in high-performance computing and machine learning workloads.

# Intrinsics In C.

You'll need to include the header which gives you access to all the intrinsic functions. Just the `#include <immintrin.h>`  header should have you covered on all compilers for all instruction sets

```c
#include <mmintrin.h> //MMX 
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSE4.2
#include <ammintrin.h> //SSE4A
#include <wmmintrin.h> //AES
#include <immintrin.h> //AVX, AVX2, FMA, AVX-512
```

When using intrinsics, rather than declaring and allocating float, int, double types You'll be working with 'packed' float, int and double types consisting of blocks of data.

- SSE1, 2, 3, 4 (nearly obsolete) :
    - __m128 : (`4 $*$ float32`)
    - __m128i : (`2 $*$ int64, 4 $*$ int32, 8 $*$ int16, 16 $*$ int8`)
    - __m128d : (`2 $*$ float64`)
- AVX, AVX2 (2008 onwards), supports all x86 CPUs after ~2012
    - __m256 : 256bit block (`8 $*$ float32`)
    - __m256i : 256bit block (`4 $*$ int64, 8 $*$ int32, 16 $*$ int16, 32 $*$ int8`)
    - __m256d : 256bit block (`4 $*$ float64` )
- AVX-512, Support is mixed and is not a single set (i.e. it has subparts like AVX-512F, AVX-512BW etc). Processing big blocks of 512 bits make it pretty easy to run into memory bandwidth limitations.
    - __m512
    - __m512i
    - __m512d

## Intrinsic Function Naming Convention :

Intel intrinsics follow a **systematic naming scheme** that encodes the **operation, data type, and vector width** into the function name.

A typical intrinsic looks like this:

```c
__m256 _mm256_add_ps(__m256 a, __m256 b)

__type{i/d} _mm[width]_[operation]_[datatype]
```

Where `_mm[widh]` is the size of the SIMD, if we are using SSE it would be `_mm` , but as we are using AVX, it is `_mm256`. 

Variants:

- `_mm` → SSE (128-bit XMM registers).
- `_mm256` → AVX/AVX2 (256-bit YMM registers).
- `_mm512` → AVX-512 (512-bit ZMM registers).

`[instruction]` is the instruction or the function name, for eg: `add` , `cmpgt` , `mul` , `loadu` and `storeu`.

While the `[datatype]` are explained as follows:

- `{ps}` = packed single-precision floats (32-bit).
- `{pd}` = packed double-precision floats (64-bit).
- `{ph}` = packed half-precision floats.
- `{ss}` = scalar single floats (32-bit).
- `{sd}` = scalar double floats (64-bit).
- `{epuX}` =  extended packed unsigned-integer.
- `{epiX}` =  extended packed signed-integer.
- **`{**siX**}**` = 32, 64, 128, 256 and 512-bit SIMD scalar integer type.

# Instrinsic Functions in C :

Before we dive into this particular topic we need to understand what latency and throughput is. Latency is the delay or time it takes for data to travel from its source to its destination, while throughput is the actual rate at which data is successfully transmitted over a network or system in a given amount of time. We can also say that Latency is the number of cycles before the result is ready, and throughput is the number of cycles each operation takes. For the following, we will be using [***Intel Skylake Microarchitecture](https://en.wikipedia.org/wiki/Skylake_(microarchitecture))*** as as reference for performance in both latency and throughput.
For more information refer to the [Intel’s Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX_ALL) for help.

## *Arithmatic functions.*

### Packed float addition :

```c
__m256 _mm256_add_ps(__m256 a, __m256 b);
// vaddps ymm0, ymm1, ymm2
```

- **Latency:** ~4 cycles
- **Throughput:** 0.5 cycles (2 per cycle)

### Packed float subtraction :

```c
__m256 _mm256_sub_ps(__m256 a, __m256 b);
// vsubps ymm0, ymm1, ymm2
```

- **Latency:** ~4 cycles
- **Throughput:** 0.5 cycles (2 per cycle)

### Integer add :

Adds 8 packed 32-bit integers.

```c
__m256i _mm256_add_epi32(__m256i a, __m256i b);
// vpaddd ymm0, ymm1, ymm2   ; ymm0 = ymm1 + ymm2
```

- **Latency:** ~1 cycle
- **Throughput:** 0.33 cycles (3 per cycle)

### Packed double multiply :

Multiplies 4 double-precision floats in parallel.

```c
__m256d _mm256_mul_pd(__m256d a, __m256d b);
// vmulpd ymm0, ymm1, ymm2   ; ymm0 = ymm1 * ymm2
```

- **Latency:** ~4 cycles
- **Throughput:** 0.5 cycles

### Division (float, AVX)

Division is **expensive** compared to add/mul. Compilers often replace it with reciprocal + multiply (`vrcpps + vmulps`) for speed.

```c
__m256 _mm256_div_ps(__m256 a, __m256 b)
// vdivps ymm0, ymm0, ymm1
```

- **Latency:** ~14 cycles
- **Throughput:** ~4–6 cycles (one every few cycles, much slower)

### Fused Multiply-Add (FMA, AVX2+FMA3)

Computes `(a * b) + c` in one shot, with a single rounding, so more accurate. It also gives you multiply+add *for the cost of one instruction*, super powerful in HPC and ML. ~2 times lower latency in both latency and CPI than in a manually implemented multiply+add.

Its also a memory hungry operation. In therory, we need RAM capable of keeping up with the CPU as much as possible, though its not possble. So don’t be suprised if after changing the mul+add with an FMA, you don’t get a significant time delta.

```c
__m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c);
// vfmadd231ps ymm0, ymm1, ymm2   ; ymm0 = (ymm1 * ymm2) + ymm0
```

There are acutally more FMA instructions which can be used depending on our `ymmx` register:

- **`vfmadd132ps`:** `(operand1 * operand2) + operand3` ; The first operand is the source of the multiplication and the destination of the add.
- **`vfmadd231ps`:** `(operand2 * operand3) + operand1` ; The third operand (the addend) becomes the destination for the final result.
- **`vfmadd213ps`** **:** `(operand2 * operand1) + operand3` ; The second operand becomes the destination for the final result.
- this type of `132` , `231` & `213` notation exists for all FMA instructions.
- **Latency:** 4–5 cycles
- **Throughput:** 0.5 cycles

### **Fused Multiply-Subtract (FMSUB)**

its math is given by $fmsub(a,b,c)=(a×b)−c$

```c
#include <immintrin.h>

__m256 fmsub_example(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmsub_ps(a, b, c);
}

//vfmadd213ps ymm0, ymm1, ymm2   ; (ymm0*ymm1) - ymm2
```

- `ps` = packed single-precision floats (8 floats in ymm).
- There are also `pd` variants for double precision.

### **Fused Multiply-Add-Subtract (FMADDSUB)**

Here the math is given by : 

 $fmaddsub(a,b,c)={(a0×b0)+c0,(a1×b1)−c1,(a2×b2)+c2,…}$

Alternates between add and subtract across elements.

```c
__m256 fmaddsub_example(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmaddsub_ps(a, b, c);
}
// vfmaddsub213ps ymm0, ymm1, ymm2   ; (a*b)+c, (a*b)-c alternating
```

### **Fused Negative Multiply-Add (FNMADD)**

Here the math is given by : $fnmadd(a,b,c)=−(a×b)+c$

```c
__m256 fnmadd_example(__m256 a, __m256 b, __m256 c) {
    return _mm256_fnmadd_ps(a, b, c);
}
// vfnmadd213ps ymm0, ymm1, ymm2   ; -(a*b) + c
```

### **Fused Negative Multiply-Subtract (FNMSUB)**

Here the Math is given by : $fnmsub(a,b,c)=−(a×b)−c$

```c
__m256 fnmsub_example(__m256 a, __m256 b, __m256 c) {
    return _mm256_fnmsub_ps(a, b, c);
}
// vfnmsub213ps ymm0, ymm1, ymm2   ; -(a*b) - c
```

## *Logical Operations.*

In the following table, both `a` and `b` is a `__m256` 

| Intrinsic | Assembly | Operation | Latency | Throughput (cpi) |
| --- | --- | --- | --- | --- |
| `_mm256_and_ps(a, b)` | `vandps` | a & b | ~1 | 0.33–0.5 |
| `_mm256_or_ps(a, b)` | `vorps` | a | b | ~1 | 0.33–0.5 |
| `_mm256_xor_ps(a, b)` | `vxorps` | a ^ b | ~1 | 0.33–0.5 |
| `_mm256_andnot_ps(a, b)` | `vandnps` | (~a) & b | ~1 | 0.33–0.5 |
- Same intrinsics exist for **doubles and singles** (`_mm256_and_pd`,`_mm256_and_ps` .) → assembly uses `vandpd`, `vorpd`, etc.
- On **AVX2**, integer versions exist (`_mm256_and_si256`, `vanddqa` / `vpand`).
- These are **super cheap** instructions, often limited only by instruction issue width (ALU ports).

## *Math Functions.*

### Maximum

```c
__m256 _mm256_max_ps(__m256 a, __m256 b);
// vmaxps ymm0, ymm1, ymm2
```

- **Operation:** per-element max of packed floats.
- **Latency:** ~4 cycles
- **Throughput:** 0.5 cycles

---

### Minimum

```c
__m256 _mm256_min_ps(__m256 a, __m256 b);
// vminps ymm0, ymm1, ymm2
```

- **Operation:** per-element min of packed floats.
- **Latency:** ~4 cycles
- **Throughput:** 0.5 cycles

---

### Square Root

```c
__m256 _mm256_sqrt_ps(__m256 a);
// vsqrtps ymm0, ymm1
```

- **Operation:** per-element square root (float).
- **Latency:** ~7–13 cycles (depending on microarchitecture).
- **Throughput:** 0.5–1 cycle.

---

### Reciprocal

```c
__m256 _mm256_rcp_ps (__m256 a)
// vrcpps ymm, ymm
```

- **Operation:** Compute the approximate reciprocal of packed single-precision (32-bit) floating-point elements in a, and store the results in dst. The maximum relative error for this approximation is less than `1.5*2^-12`.
- **Latency:** 4 cycles
- **Throughput:** 1 cycles

---

### Reciprocal Square Root (approx)

```c
__m256 _mm256_rsqrt_ps(__m256 a);
// vrsqrtps ymm0, ymm1
```

- **Operation:** approximate `1/sqrt(x)` for each float (fast, low precision).
- **Latency:** ~3 cycles
- **Throughput:** 0.5 cycles

### Intel SVML (*a quick note*)

SVML is Intel’s vectorized math library. It provides high-performance SIMD implementations of common math functions (like sin, cos, exp, log, sqrt, etc.), so they can be used inside loops that have been auto-vectorized by compilers such as ICC, ICX, or ifort.

SVML is a proprietary Intel library that works with the Intel compiler (ICC). If you're not using ICC or Visual C++ 2019 then you'll need to find an equivalent library, e.g. [sse_mathfun](https://github.com/RJVB/sse_mathfun) or [avx_mathfun](https://github.com/reyoung/avx_mathfun).

## *Load and Store functions.*

### Load Aligned

```c
__m256 _mm256_load_ps(float const *mem_addr);
// vmovaps ymm0, YMMWORD PTR [rdi]
```

*vector load example with array :*

```c
double *arr = {1,2,3,4,5,6,7,8,9,0}
__m256d c = _mm256_load_pd(arr);
```

- **Operation:** load 8 floats (32-bit each) from an address aligned to 32 bytes.
- **Latency:** 7 cycles
- **Throughput:** 0.5 cycles

---

### Load Unaligned

```c
__m256 _mm256_loadu_ps(float const *mem_addr);
//vmovups ymm0, YMMWORD PTR [rdi]
```

- **Operation:** load 8 floats from memory, no alignment requirement.
- **Latency:** same as aligned (microarch hides it now).
- **Throughput:** 0.5 cycles

---

### Store Aligned

```c
void _mm256_store_ps(float *mem_addr, __m256 a);
// vmovaps YMMWORD PTR [rdi], ymm0
```

- **Operation:** store 8 floats (aligned).
- **Latency:** store latency is not directly relevant (depends on load→use), but write completion ~4–5 cycles.
- **Throughput:** 0.5 cycles

---

### Store Unaligned

```c
void _mm256_storeu_ps(float *mem_addr, __m256 a);
// vmovups YMMWORD PTR [rdi], ymm0
```

- **Operation:** store 8 floats (unaligned ok).
- **Latency:** ~4–5 cycles (hidden unless crossing cache line).
- **Throughput:** 0.5 cycles

---

### Non-Temporal Store (streaming, bypass caches)

Store 256-bits (composed of 8 packed single-precision (32-bit) floating-point elements) from a into memory using a non-temporal memory hint. mem_addr must be aligned on a 32-byte boundary or a general-protection exception may be generated.

```c
void _mm256_stream_ps(float *mem_addr, __m256 a);
// vmovntps YMMWORD PTR [rdi], ymm0
```

- **Operation:** store directly to memory, avoid polluting cache. Useful for streaming large arrays.
- **Latency:** depends on memory hierarchy. ~5 on SkyLake, but avoids L1→L3 writes.
- **Throughput:** 0.5–1 cycles

## *Comparison :*

Compare packed single-precision (32-bit) floating-point elements in a and b based on the comparison operand specified by imm8, and store the results in dst register.

```c
__m256 _mm256_cmp_ps(__m256 a, __m256 b, const int imm8);
// vcmpps ymm0, ymm1, ymm2, imm8
```

*Its Immidiate (imm8) mappings are given in the following table:*

| imm8 | Mnemonic | Meaning (per element) | Ordered/Unordered | Notes |
| --- | --- | --- | --- | --- |
| 0 | `_CMP_EQ_OQ` | Equal | Ordered | True if a == b, false if NaN |
| 1 | `_CMP_LT_OS` | Less-than | Ordered | True if a < b, false if NaN |
| 2 | `_CMP_LE_OS` | Less-or-equal | Ordered | True if a ≤ b, false if NaN |
| 3 | `_CMP_UNORD_Q` | Unordered | — | True if either operand is NaN |
| 4 | `_CMP_NEQ_UQ` | Not equal | Unordered | True if a ≠ b, true if NaN |
| 5 | `_CMP_NLT_US` | Not less-than | Unordered | Equivalent to a ≥ b or NaN |
| 6 | `_CMP_NLE_US` | Not less-or-equal | Unordered | Equivalent to a > b or NaN |
| 7 | `_CMP_ORD_Q` | Ordered (no NaN) | — | True if neither operand is NaN |
| 8 | `_CMP_EQ_UQ` | Equal | Unordered | True if a == b or NaN |
| 9 | `_CMP_NGE_US` | Not greater-or-equal | Unordered | Equivalent to a < b or NaN |
| 10 | `_CMP_NGT_US` | Not greater-than | Unordered | Equivalent to a ≤ b or NaN |
| 11 | `_CMP_FALSE_OQ` | Always false | Ordered | Constant false |
| 12 | `_CMP_NEQ_OQ` | Not equal | Ordered | True if a ≠ b, false if NaN |
| 13 | `_CMP_GE_OS` | Greater-or-equal | Ordered | True if a ≥ b, false if NaN |
| 14 | `_CMP_GT_OS` | Greater-than | Ordered | True if a > b, false if NaN |
| 15 | `_CMP_TRUE_UQ` | Always true | Unordered | Constant true |
| 16 | `_CMP_EQ_OS` | Equal | Ordered (signaling) | Like EQ_OQ but with signaling |
| 17 | `_CMP_LT_OQ` | Less-than | Ordered | Same as #1 but quiet NaN handling |
| 18 | `_CMP_LE_OQ` | Less-or-equal | Ordered | Same as #2 but quiet NaN handling |
| 19 | `_CMP_UNORD_S` | Unordered (signaling) | — | True if either NaN, exception if signaling NaN |
| 20 | `_CMP_NEQ_US` | Not equal | Unordered (signaling) | True if ≠ or NaN |
| 21 | `_CMP_NLT_UQ` | Not less-than | Unordered (quiet) | Equivalent to a ≥ b or NaN |
| 22 | `_CMP_NLE_UQ` | Not less-or-equal | Unordered (quiet) | Equivalent to a > b or NaN |
| 23 | `_CMP_ORD_S` | Ordered (signaling) | — | True if no NaNs, exception if signaling NaN |
| 24 | `_CMP_EQ_US` | Equal | Unordered (signaling) | True if a == b, true if NaN |
| 25 | `_CMP_NGE_UQ` | Not greater-or-equal | Unordered | Equivalent to a < b or NaN |
| 26 | `_CMP_NGT_UQ` | Not greater-than | Unordered | Equivalent to a ≤ b or NaN |
| 27 | `_CMP_FALSE_OS` | Always false | Ordered (signaling) | Constant false |
| 28 | `_CMP_NEQ_OS` | Not equal | Ordered (signaling) | True if ≠, false if NaN |
| 29 | `_CMP_GE_OQ` | Greater-or-equal | Ordered | True if ≥, false if NaN |
| 30 | `_CMP_GT_OQ` | Greater-than | Ordered | True if >, false if NaN |
| 31 | `_CMP_TRUE_US` | Always true | Unordered (signaling) | Constant true |

***Key to the Suffixes:***

- **`OQ / OS**` → Ordered (Quiet / Signaling)
    - *Quiet*: ignores signaling NaNs.
    - *Signaling*: raises exceptions on signaling NaNs.
- **`UQ / US**` → Unordered (Quiet / Signaling)
    - If NaN is present, condition evaluates to true.
- **`ORD** vs **UNORD**` → explicitly checks for NaN ordering.
- **`TRUE/FALSE**` → constant masks, useful for mask initialization.

---

In practice:

- HPC/ML code almost always uses the **ordered quiet** forms (`_CMP_EQ_OQ`, `_CMP_LT_OQ`, etc.), since they’re predictable and don’t trap on NaNs.
- The *signaling* variants exist mostly for IEEE 754 compliance testing.

the returned data `__m256` is binary, either all 32 bits are 1 or all are 0, the output is float32 like 1.0.

- **Latency:** ~4 cycles
- **Throughput:** 0.5 cycles

## *Bit Manipulation:*

### **Population Count (popcnt)**

```c
__m256i _mm256_popcnt_epi8 (__m256i a)
__m256i _mm256_popcnt_epi16 (__m256i a)
__m256i _mm256_popcnt_epi32 (__m256i a)
__m256i _mm256_popcnt_epi64 (__m256i a)

// vpopcnt{b/w/d/q} ymm, ymm
```

- **Operation:** Count the number of logical 1 bits in packed 8/16/32/64-bit integers in a, and store the results in destination.
- **Latency:** -
- **Throughput:** 1 cycle

### **Count Leading Zeros (lzcnt)**

```c
__m256i _mm256_lzcnt_epi8 (__m256i a)
__m256i _mm256_lzcnt_epi16 (__m256i a)
__m256i _mm256_lzcnt_epi32 (__m256i a)
__m256i _mm256_lzcnt_epi64 (__m256i a)

// vplzcnt{b/w/d/q} ymm, ymm
```

- **Operation:** Counts the number of leading zero bits in each packed 8/16/32/64-bit-bit integer in a, and store the results in dst.
- **Latency:** 4 Cycles
- **Throughput:** 0.5 cycles

# Branchless Programming with SIMD

Modern CPUs employ branch prediction to decide which path to execute when encountering conditional statements. While prediction is usually accurate, a **mispredicted branch** can incur significant penalties (often 15–20 cycles). To mitigate this, programmers often employ **branchless programming techniques** that use arithmetic or logical operations instead of conditional jumps.

When combined with **SIMD (Single Instruction, Multiple Data)**, branchless programming allows the same conditional logic to be applied to entire vectors of data in parallel, significantly improving performance and predictability.

---

## Branchless Conditionals

### *Scalar Example*

A conditional statement such as:

```c
if (x > 0)
    y = x;
else
    y = 0;
```

can be rewritten without a branch:

```c
y = (x > 0) ? x : 0;    // high-level
y = (x > 0) * x;        // arithmetic form
y = (x & -(x > 0));     // bitwise form
```

Here, `(x > 0)` evaluates to `1` or `0`, and multiplication or masking selects the appropriate value.

---

## SIMD Branchless Programming

### Example: ReLU (Rectified Linear Unit)

The ReLU function, common in machine learning, is defined as: 

$$
ReLU(x) =\begin{cases}
x, & \text{if } x > 0 \\
0, & \text{otherwise}
\end{cases}

$$

### Intrinsic Implementation (AVX):

```c
#include <immintrin.h>

__m256 relu(__m256 x) {
    __m256 zero = _mm256_setzero_ps();
    // Compare: mask = (x > 0) ? 0xFFFFFFFF : 0x00000000
    __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OS);
    // Select: (mask & x) | (~mask & 0)
    return _mm256_and_ps(mask, x);
}

```

### Assembly Mapping:

```
vxorps   ymm1, ymm1, ymm1        ; ymm1 = 0.0
vcmpps   ymm2, ymm0, ymm1, 0x0E  ; compare x > 0, result mask
vandps   ymm0, ymm2, ymm0        ; apply mask
```

### Performance:

- `vxorps` : latency ~1 cycle, throughput ~0.5 cpi
- `vcmpps` : latency ~3–4 cycles, throughput ~1 cpi
- `vandps` : latency ~1 cycle, throughput ~0.5 cpi

This implementation avoids branching entirely while operating on **eight 32-bit floats in parallel**.

---

### Example: Maximum of Two Vectors

To compute $max⁡(a,b)$ branchlessly, we formulate an operation :`(a > b) & a | ~(a > b) & b`    

### Intrinsic Implementation:

```c
__m256 max_branchless(__m256 a, __m256 b) {
    __m256 mask = _mm256_cmp_ps(a, b, _CMP_GT_OS);
    return _mm256_or_ps(
        _mm256_and_ps(mask, a),
        _mm256_andnot_ps(mask, b)
    );
}
```

### Assembly Mapping:

```
vcmpps   ymm2, ymm0, ymm1, 0x0E  ; mask = (a > b)
vandps   ymm3, ymm2, ymm0        ; mask & a
vandnps  ymm4, ymm2, ymm1        ; ~mask & b
vorps    ymm0, ymm3, ymm4        ; combine results
```

This evaluates both paths (`a` and `b`), then selects the correct value using bitwise masks.

# Some More Functions…

## *Permute*

```c
__m256 _mm256_permutevar8x32_ps (__m256 a, __m256i idx)
// vpermps ymm, ymm, ymm

__m256i _mm256_permutevar8x32_epi32 (__m256i a, __m256i idx)
// vpermd ymm, ymm
```

Here `idx` is basically mapping specified per element, which is `8*int32` , same as `a`

- **Operation:** Shuffle 32-bit integers or single-precision (32-bit) floating-point elements in a across lanes using the corresponding index in idx, and store the results in dst.
- **Latency:** 3 cycles
- **Throughput:** 1 cycle

```c
__m256i _mm256_permute4x64_epi64 (__m256i a, const int imm8)
// vpermpd ymm, ymm, ymm

__m256i _mm256_permute4x64_ps (__m256i a, const int imm8)
// vpermq ymm, ymm
```

Here `imm8` is a constant int type, which has mapping specified per two bits (`1*int32`), so it uses the first eight bits (`4*2 bits`) with individual two bit mapping each 64-bit chunks to required position , imagine the following :

```c
								  64-Bit Blocks
|-------------|-------------------|
| Argument1		| a3 | a2 | a1 | a0 |
| Argument2		| 00 | 01 | 10 | 11 | = 0b00011011 (27)
| Output   		| a0 | a1 | a2 | a3 | = remmapped output
|-------------|-------------------|
```

- **Operation:** Shuffle double-precision (64-bit) floating-point elements or 64-bit integers in a across lanes using the control in imm8, and store the results in dst.
- **Latency:** 3 cycles
- **Throughput:** 1 cycle

## *Lanes*

### **Lane count table :**

We would need this for studying lanes and for other complicated instructions.

| Intrinsic type | Element size | Elements per 256-bit reg |
| --- | --- | --- |
| `epi8` | 8 bits | 32 lanes |
| `epi16` | 16 bits | 16 lanes |
| `epi32` | 32 bits | 8 lanes |
| `epi64` | 64 bits | 4 lanes |
| `ps` (float) | 32 bits | 8 lanes |
| `pd` (double) | 64 bits | 4 lanes |

Before AVX2 started supporting proper permute,  AVX instructions often **treat the 256-bit register as two 128-bit halves**:

```c
				|------------------|-------------------|
	YMM = | Lane 1 (low 128) ∣ Lane 2 (high 128) |
				|------------------|-------------------|
```

so the following operations,

```c
__m256 _mm256_permute_ps (__m256 a, int imm8)
// vpermilps ymm, ymm, ymm
```

where `imm8` is a constant int type, which has mapping specified per two bits (`1*int32`), similar to the `_mm256_permute4x64_ps (__m256i a, const int imm8)` , but reorders the whole `8*float32` block within its `128 bit` lane with respect to the `imm8`, so they cannot write to other lanes from other lane . The final remapping looks like :

```c
|							|	  32-Bit Blocks   |  32-Bit Blocks    |
|-------------|-------------------|-------------------|
|Argument1		| a7 | a6 | a5 | a4 | a3 | a2 | a1 | a0 |
|Argument2		| 00 | 01 | 10 | 11 | 00 | 01 | 10 | 11 | = 0b00011011 (27)
|Output   		| a4 | a5 | a6 | a7 | a0 | a1 | a2 | a3 | = remmapped output
|-------------|-------------------|-------------------|
| 256-bit ymm |    128-bit lane   |    128-bit lane   |

// here a7 cannot be written to a0
```

- **A Lane is a 128-bit chunk of a wider register like `ymmX` or `zmmX`.**
- Many permutes (AVX1) are **lane-restricted**, shuffling only inside each lane.
- Cross-lane permutes require special instructions (`vperm2f128`, `vpermd`, AVX-512 `vpermt2*`).
- **Operation:** Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in imm8, and store the results in dst.
- **Latency:** 1 cycles (bit faster than others)
- **Throughput:** 1 cycle

## *Insert*

```c
__m256d _mm256_insertf128_pd (__m256d a, __m128d b, int imm8)
// vinsertf128 ymm, ymm, xmm, imm8
```

Copy a to dst, then insert 128 bits (composed of 2 packed double-precision (64-bit) floating-point elements) from b into dst at the location specified by imm8.

```c
__m256 _mm256_insertf128_ps (__m256 a, __m128 b, int imm8)
// vinsertf128 ymm, ymm, xmm, imm8
```

Copy a to dst, then insert 128 bits from b into dst at the location specified by imm8.

`imm8` can be either 0 or 1, depending on the index, example :

```c
|							|	 4*32-Bit Blocks  | 4*32-Bit Blocks   |
|-------------|-------------------|-------------------|
|Argument1		| a7 | a6 | a5 | a4 | a3 | a2 | a1 | a0 |
|Argument2		|                   | b3 | b2 | b1 | b0 | = lets say imm8 = 1
|Output   		| b3 | b2 | b1 | b0 | a0 | a1 | a2 | a3 |
|-------------|-------------------|-------------------|
| 256-bit ymm |    128-bit lane   |    128-bit lane   |
```

- **Latency:** 3 cycles
- **Throughput:** 1 cycle

## *Extract*

Extract 128 bits composed of 2 packed double-precision (64-bit) floating-point elements or 128 bits composed of 4 packed single-precision (32-bit) floating-point elements from a, selected with imm8, and store the result in dst. `imm8` can be either 0 or 1, depending on the index of the part we want to extract, example :

```c
__m128 _mm256_extractf128_ps (__m256 a, const int imm8)
__m128d _mm256_extractf128_pd (__m256d a, const int imm8)
// vextractf128 xmm, ymm, imm8
```

- **Latency:** 3 cycles
- **Throughput:** 1 cycle

## *Blend*

```c
__m256i _mm256_blend_epi16 (__m256i a, __m256i b, const int imm8)
__m256i _mm256_blend_epi32 (__m256i a, __m256i b, const int imm8)
__m256d _mm256_blend_pd (__m256d a, __m256d b, const int imm8)
__m256 _mm256_blend_ps (__m256 a, __m256 b, const int imm8)

// vblend{w/d} ymm0, ymm1, ymm2, 0xAA
// vblend{pd/ps} ymm0, ymm1, ymm2, 0xAA
```

In `_mm256_blend_ps`, `0b10101010` (0xAA) is the **mask**:

- Each bit corresponds to one of the 8 packed floats.
- `0 → take from a`, `1 → take from b`.

So result = `[b0, a1, b2, a3, b4, a5, b6, a7]`.

- **Latency**: 1 cycle
- **Throughput (CPI)**: 0.5 (i.e., 2 instructions per cycle, if enough ports free)

## *Convert*

The **`cvt` intrinsics** are the SIMD “bridge operators.” They convert between datatypes (float ↔ int, float32 ↔ float64, etc.), lane by lane, no branching.

- **Packed element conversion:** Each element in the input vector is independently converted to the new format.
- **Widening vs. narrowing:** Some conversions increase the precision or element size (e.g., `vcvtps2pd`), while others reduce it (e.g., `vcvtpd2ps`).
- **Rounding modes:** Floating-point to integer conversions come in two forms: with truncation (`vcvttps2dq`) or with rounding according to the MXCSR control word (`vcvtps2dq`).

```c

*function naming convention :*

		 __m{size}{i/d} _mm{size}_cvt{***datatype***}_{***datatype***} (__m{size}{i/d} a) 
		 vcvt*opcode* ymm, xmm

		|	 assembly		|           C Language Mapping                |
		|-------------|---------------------------------------------|
		|  vpmovsxwd  |  __m256i _mm256_cvtepi16_epi32 (__m128i a)  |
		|  vpmovsxwq  |  __m256i _mm256_cvtepi16_epi64 (__m128i a)  |
		|  vpmovsxdq  |  __m256i _mm256_cvtepi32_epi64 (__m128i a)  |
		|  vcvtdq2pd  |  __m256d _mm256_cvtepi32_pd (__m128i a)     |
		|  vcvtdq2ps  |  __m256 _mm256_cvtepi32_ps (__m256i a)      |
		|  vpmovsxbw  |  __m256i _mm256_cvtepi8_epi16 (__m128i a)   |
		|  vpmovsxbd  |  __m256i _mm256_cvtepi8_epi32 (__m128i a)   |
		|  vpmovsxbq  |  __m256i _mm256_cvtepi8_epi64 (__m128i a)   |
		|  vpmovzxwd  |  __m256i _mm256_cvtepu16_epi32 (__m128i a)  |
		|  vpmovzxwq  |  __m256i _mm256_cvtepu16_epi64 (__m128i a)  |
		|  vpmovzxdq  |  __m256i _mm256_cvtepu32_epi64 (__m128i a)  |
		|  vpmovzxbw  |  __m256i _mm256_cvtepu8_epi16 (__m128i a)   |
		|  vpmovzxbd  |  __m256i _mm256_cvtepu8_epi32 (__m128i a)   |
		|  vpmovzxbq  |  __m256i _mm256_cvtepu8_epi64 (__m128i a)   |
		|  vcvtpd2ps  |  __m128 _mm256_cvtpd_ps (__m256d a)         |
		|  vcvtpd2dq  |  __m128i _mm256_cvtpd_epi32 (__m256d a)     |
		|  vcvtpd2ps  |  __m128 _mm256_cvtpd_ps (__m256d a)         |
		|  vcvtps2dq  |  __m256i _mm256_cvtps_epi32 (__m256 a)      |
		|  vcvtps2pd  |  __m256d _mm256_cvtps_pd (__m128 a)         |
		|-------------|---------------------------------------------|
```

## *Set*

### Set with values

Manually set all elements (in reverse order — last argument is the lowest lane).

```c
__m256i _mm256_set_epi16 (16 * short ei)
__m256i _mm256_set_epi32 (8 * int ei)
__m256i _mm256_set_epi64x (__int64 e3, __int64 e2, __int64 e1, __int64 e0)
__m256i _mm256_set_epi8 (32 * char ei)

__m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
__m256 _mm256_set_ps (8 * float ei)

// these above functions do not map one to one with assembly instructions but are a sequence

__m256 _mm256_set_m128 (__m128 hi, __m128 lo)
__m256d _mm256_set_m128d (__m128d hi, __m128d lo)
__m256i _mm256_set_m128i (__m128i hi, __m128i lo)

// vinsertf128 ymm, ymm, xmm, imm8
```

Usually compiled into multiple `mov` + `vinsertf128` sequence.

- **Latency:** depends, but often free if the compiler optimizes to a memory constant.

### **Broadcast a single value**

Broadcast 8, 16, 32 or 64-bit integer a to all elements of dst. This intrinsic may generate the `vpbroadcast{b/w/d/q}`. Or you can say replicate one scalar value into all lanes.

```c
__m256 a = _mm256_set1_ps(3.14f);    __m256i b = _mm256_set1_epi32(42);   
__m256d c = _mm256_set1_pd(2.718);   

								---x---

__m256i _mm256_set1_epi16 (short a)
__m256i _mm256_set1_epi32 (int a)
__m256i _mm256_set1_epi64x (long long a)
__m256i _mm256_set1_epi8 (char a)         // all 8 lanes = 42
__m256d _mm256_set1_pd (double a)         // all 4 lanes = a
__m256 _mm256_set1_ps (float a)           // all 8 lanes = a
```

assembly generated does not map one to one but is a sequence but may contain the `vpbroadcast{b/w/d/q}` instruction.

Latency: 1-3 cycles, Throughput: ~0.5-1 CPI.

### **Zero vector**

The most common and fastest initializer. Maps to `vxorps`.

```c
__m256 a = _mm256_setzero_ps();
// vxorps ymm0, ymm0, ymm0

__m256i b = _mm256_setzero_si256();
// vxor ymm0, ymm0, ymm0

__m256d c = _mm256_setzero_pd();
// vxorpd ymm0, ymm0, ymm0
```

Latency: 1, Throughput: 0.25 CPI (super fast, no dependency).

# **Memory Alignment**

Memory alignment means that data is stored in memory at an address that matches its "natural boundary."

- A **4-byte (32-bit) int** is aligned if its address is divisible by 4.
- An **8-byte (64-bit) double** is aligned if its address is divisible by 8.
- For SIMD loads/stores:
    - **128-bit SSE register** wants 16-byte alignment.
    - **256-bit AVX register** wants 32-byte alignment.
    - **512-bit AVX-512 register** wants 64-byte alignment.

If the data isn’t aligned, the CPU might need multiple memory accesses to grab it, which kills performance — and in older ISAs (like SSE before `movdqu`), misaligned loads could even crash.

## aligned_alloc

Using `_mm256_load_si256` or `_mm256_store_si256` with pointers returned by the following code :

```c
__m256 *data = (__m256*) malloc(10 * sizeof(__m256));
```

Can cause performance penalties (if the CPU handles misaligned loads), or segmentation faults (on older CPUs or when crossing cache boundaries). This is where `aligned_alloc` or `posix_memalign` comes in. `aligned_alloc(align, size)` (introduced in C11) allocates memory **explicitly aligned** to a power-of-two boundary.

```c
#include <immintrin.h>
#include <stdlib.h>

// __m256 *A = (__m256*) aligned_alloc(alignement, size * sizeof(__m256));
__m256 *data = (__m256*) aligned_alloc(32, 10 * sizeof(__m256));
```

- `align` must be a power of two and a multiple of `sizeof(void*)`. In the above example it is given 32 bytes, but it lets you request a **size that is a multiple of that alignment** (AVX `__m256` requires 32-byte alignment, AVX-512 requires 64).
- `size` must be a multiple of `align`.
- Thus result is always a 32-byte aligned, perfectly safe for `_mm256_load_ps` and `_mm256_store_ps`.

# `inline static` keywords in C functions

In C, the `inline` and `static` keywords serve different purposes but are often used together for efficiency and modularity. The `inline` specifier is a hint to the compiler, suggesting that the function’s body should be expanded directly at each call site instead of being invoked through the usual function call mechanism. This can reduce the overhead of function calls and improve performance for small, frequently used functions, although the compiler is free to ignore the suggestion if the function is large, complex, or recursive. The `static` specifier, when applied to a function, gives it internal linkage, meaning the function’s name is visible only within the source file where it is defined. This promotes encapsulation, prevents external access to helper functions, and avoids naming conflicts between files. When combined as `static inline`, the function inherits both properties: each source file that includes the function definition treats it as a local function, and the compiler can choose to inline it wherever it is called. If the compiler decides not to inline it, it generates a separate local copy of the function for that file, ensuring there are no linker conflicts. This approach is commonly used for small helper functions defined in header files, offering speed benefits through inlining while maintaining internal linkage. The trade-off is that multiple local copies of the function may increase the final code size if inlining is not applied consistently.

take the following for an examle :

```c
#include <stdio.h>

static inline int square(int x) {
    return x * x;
}

int main(void) {
    printf("%d\n", square(5));
    return 0;
}
```

compiling to assembly with `gcc -g -o main.s -masm=intel -fno-verbose-asm -S -fdiagnostics-color=always -fstack-usage -O2 main.c` we get the output as :

```nasm
.LC0:
        .string "%d\n"
main:
        sub     rsp, 8
        mov     esi, 25
        mov     edi, OFFSET FLAT:.LC0
        xor     eax, eax
        call    printf
        xor     eax, eax
        add     rsp, 8
        ret
```

The `-O2` flag asks the compiler to optimize, so we’ll likely see the inlining in `main.s`.

## *Random*

These are specialized CPU instructions that provide access to hardware-based random number generators (RNGs). The most well-known example is Intel's RDRAND instruction, which generates high-quality, cryptographically secure random numbers directly from the processor's built-in hardware.

It has a CPUID Flag of `RDRAND` and when compiling using gcc, we require the `-mrdrnd` option.

```c
int _rdrand16_step (unsigned short* val)
//rdrand r16

int _rdrand32_step (unsigned int* val)
//rdrand r32

int _rdrand64_step (unsigned __int64* val)
//rdrand r64
```

*following C example would help us to understand better :*

```c
#include <immintrin.h>
unsigned int random_value;
if (_rdrand32_step(&random_value)) {
// random_value contains a 32-bit random number
} else {
// RDRAND failed
}
```

`rdrand` is exceptionally resource hungry, it takes 463 clock cycles, regardless of the operand size (16/32/64 bits). This number of clock cycles applies to all processors with Skylake or Kaby Lake microarchitecture. And on Ivy Bridge processors `RDRAND` takes up to 117 clock cycles.

# *Code Examples:*

$Example\ I$  : The following code generates two arrays with randomly generated data and adds them up. 

```c
#include <stdint.h>
#include <immintrin.h>

#define N (1<<10)   // 1024 elements

int main(void) {
    unsigned long long a[N];
    unsigned long long b[N];
    unsigned long long c[N];

    // Fill arrays with random numbers
    for (int16_t i = 0; i < N; i++) {
        unsigned long long val;
        if (_rdrand64_step(&val)) a[i] = val;
        if (_rdrand64_step(&val)) b[i] = val;
    }

    // Vectorized addition: process 8 elements (512 bits) per iteration
    for (int16_t i = 0; i < N; i += 8) {
        __m512i va = _mm512_loadu_si512(&a[i]);
        __m512i vb = _mm512_loadu_si512(&b[i]);
        __m512i vc = _mm512_add_epi64(va, vb);
        _mm512_storeu_si512(&c[i], vc);
    }

    return 0;
}
```

$Example\ II$ : The following code finds the answer to $∫x^2+x$ where limits are $(0, 100)$ using Trapezoidal integration.

```c
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>

// Function f(x) = x^2 + x using AVX2
static inline __m256d f(__m256d x) {
    return _mm256_fmadd_pd(x, x, x); // x^2 + x
}

// Trapezoidal integration using AVX2
double integral(const double a, const double b) {
    __m256d integral_val;
    double n = 1 << 26;  // number of intervals

    __m256d va = _mm256_set1_pd(a);
    __m256d vb = _mm256_set1_pd(b);

    // Step size (h)
    __m256d vn = _mm256_set1_pd(1.4901161193847656e-08);
    __m256d vh = _mm256_fmsub_pd(vb, vn, _mm256_mul_pd(va, vn));

    // Initialize integral with (f(a) + f(b)) / 2
    __m256d vr = _mm256_set1_pd(0.5);
    integral_val = _mm256_fmadd_pd(f(va), vr, _mm256_mul_pd(f(vb), vr));

    // Sum interior points in chunks of 4
    for (int64_t i = 1; i + 3 < n; i += 4) {
        __m256d idx = _mm256_set_pd(i + 3, i + 2, i + 1, i);
        __m256d k = _mm256_fmadd_pd(idx, vh, va);
        integral_val = _mm256_add_pd(integral_val, f(k));
    }

    // Multiply by h to get final approximation
    integral_val = _mm256_mul_pd(integral_val, vh);

    // Horizontal sum of the vector
    double arr[4];
    _mm256_storeu_pd(arr, integral_val);
    double sum = arr[0] + arr[1] + arr[2] + arr[3];

    return sum;
}

int main(int argc, char *argv[]) {
    int a = 0, b = 100;
    double _integral = integral(a, b);
    printf("%lf\n", _integral);

    return 0;
}
```

$Example\ III$  : The following code finds the number of occurances of an element in a list.

```c
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>

int count_occurrences(const int *arr, size_t n, int target) {
    __m256i vtarget = _mm256_set1_epi32(target); // broadcast target
    __m256i vcount  = _mm256_setzero_si256();    // vector counter

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // Load 8 ints from array (something like malloc)
        __m256i vdata = _mm256_loadu_si256((__m256i const*)&arr[i]);

        // Compare with target → 0xFFFFFFFF if equal, 0 if not
        __m256i vcmp = _mm256_cmpeq_epi32(vdata, vtarget);

        // The bitwise AND operation ensures that:
        // If vcmp[i] == 0xFFFFFFFF, then (0xFFFFFFFF & 0x00000001) = 0x00000001.
        // If vcmp[i] == 0x00000000, then (0 & 0x00000001) = 0x00000000.
        __m256i vone  = _mm256_set1_epi32(1);
        __m256i vadd  = _mm256_and_si256(vcmp, vone);
        vcount = _mm256_add_epi32(vcount, vadd);
    }

    // Horizontal sum of vector counts
    int buf[8];
    _mm256_storeu_si256((__m256i*)buf, vcount);
    int count = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

    return count;
}

int main() {
    int arr[20] = {1,2,3,2,2,5,2,6,7,2,8,9,2,10,11,12,13,2,14,2};
    int n = 20;
    int target = 2;

    int result = count_occurrences(arr, n, target);
    printf("Occurrences of %d = %d\n", target, result);

    return 0;
}
```

# *Conclusion.*

This very crude and unreliable guide to intrinsics was written by me while learning assembly intrinsics in C. This markdown file covers the basics of it and a bit more. If there is a mistake, whether it be a major error or just a disagreement on a particular topic, feel free to correct me. I am just a newb and I only know so much. Thank you for reading this.
