#pragma once

/**
 * @file hls_common.hpp
 * @brief Common Fixed-Point and Arbitrary-Precision Integer Type Definitions for Vitis HLS.
 *
 * Defines uniform integer width types and numerical saturation utilities utilized
 * across quantized neural network operators (GELU LUT, Multi-Head Attention, Tiled MatMul).
 */

#if defined(__SYNTHESIS__) || defined(HLS_NO_XIL_FPO_LIB) || defined(__VITIS_HLS__) || defined(__XILINX_HLS__)
#include <ap_int.h>
#include <ap_fixed.h>
#else
// Fallback definitions for host/C++ standard unit testing environments
#include <cstdint>
#include <ap_int.h>
#include <ap_fixed.h>
#endif

// ==============================================================================
// Arbitrary-Precision Integer Typedefs
// ==============================================================================
typedef ap_int<8>   int8_t_hls;   ///< Signed 8-bit quantized weights and activation tensors
typedef ap_int<16>  int16_t_hls;  ///< Signed 16-bit intermediate accumulator / product
typedef ap_int<32>  int32_t_hls;  ///< Signed 32-bit dot-product reduction accumulator
typedef ap_uint<8>  uint8_t_hls;  ///< Unsigned 8-bit index for 256-entry Look-Up Tables
typedef ap_uint<16> uint16_t_hls; ///< Unsigned 16-bit normalisation denominator

/**
 * @brief Numerical saturation and bounding utility.
 *
 * Inlines a branchless hardware clamp to prevent numerical rollover or overflow
 * during fixed-point quantization.
 *
 * @tparam T   Primitive integer or fixed-point type.
 * @param  v   Input value to be clamped.
 * @param  lo  Lower saturation boundary.
 * @param  hi  Upper saturation boundary.
 * @return Saturation-bounded output within [lo, hi].
 */
template <typename T>
static T clamp_signed(T v, T lo, T hi) {
#pragma HLS INLINE
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

