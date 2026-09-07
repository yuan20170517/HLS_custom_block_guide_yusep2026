#ifndef HG_PIPE_TYPES_H
#define HG_PIPE_TYPES_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <cmath>

// Standard fixed-point data types for FPGA acceleration
typedef ap_fixed<16, 6>  data_t;       // 16-bit fixed point: 1 sign, 5 integer, 10 fraction
typedef ap_fixed<24, 8>  accum_t;      // 24-bit accumulator
typedef ap_uint<32>      dim_t;        // Dimension and loop counters

// Floating point definitions for high-precision operators (LayerNorm, RMSNorm)
typedef float fp32_t;

// Architectural constants
#ifndef NACC
#define NACC 16 // 16-way partial accumulators to break multi-cycle pipeline feedback
#endif

#ifndef MAX_DEPTH
#define MAX_DEPTH 8192
#endif

#endif // HG_PIPE_TYPES_H
