/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

/**
 * @file dct.h
 * @brief Configuration Definitions, Arithmetic Macros, and Function Prototypes for 2D-DCT Kernel.
 *
 * This header defines architectural constants, fixed-point bit widths, rounding macros,
 * and top-level function declarations for the 2-Dimensional 8x8 Discrete Cosine Transform
 * High-Level Synthesis (HLS) design.
 */

#ifndef __DCT_H__
#define __DCT_H__

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>

// ============================================================================
// Architectural Constants & Dimensional Parameters
// ============================================================================

/**
 * @def DW
 * @brief Data Width in bits for each input/output word (16-bit signed integers).
 */
#define DW 16

/**
 * @def N
 * @brief Total number of sample elements per 2D block: 1024 bits / 16 bits = 64 elements.
 *        Corresponds to an 8x8 matrix (DCT_SIZE x DCT_SIZE).
 */
#define N (1024 / DW)

/**
 * @def NUM_TRANS
 * @brief Total number of 1D-DCT transforms executed per 2D block:
 *        8 row transforms + 8 column transforms = 16 1D transforms.
 */
#define NUM_TRANS 16

/**
 * @typedef dct_data_t
 * @brief Primary data type representation for audio/video/RF matrix elements (16-bit signed integer).
 */
typedef short dct_data_t;

/**
 * @def DCT_SIZE
 * @brief 1D transform length and dimension of the square 8x8 matrix block.
 */
#define DCT_SIZE 8

// ============================================================================
// Fixed-Point Arithmetic & Rounding Parameters
// ============================================================================

/**
 * @def CONST_BITS
 * @brief Number of fractional fractional scaling bits for the pre-computed DCT cosine coefficients.
 *        Scale factor = 2^13 = 8192 (representing unity gain 1.0 in fixed-point).
 */
#define CONST_BITS 13

/**
 * @def DESCALE(x, n)
 * @brief Fixed-point symmetrical descaling macro with convergent rounding.
 *
 * Adds half the divisor (1 << (n - 1)) prior to right-shifting by n bits to implement
 * round-to-nearest integer arithmetic:
 *   round(x / 2^n) = ((x + 2^(n-1)) >> n)
 *
 * @param x The 32-bit accumulated sum to be descaled.
 * @param n Number of fractional bits to discard (CONST_BITS = 13).
 */
#define DESCALE(x, n) (((x) + (1 << ((n) - 1))) >> (n))

// ============================================================================
// Top-Level Hardware Kernel Function Prototype
// ============================================================================

extern "C" {
  /**
   * @brief Top-level hardware entry function for the 2D 8x8 Discrete Cosine Transform.
   *
   * @param[in]  input  Linear input buffer containing N=64 samples (16-bit signed).
   * @param[out] output Linear output buffer containing N=64 transformed frequency coefficients.
   *
   * @note 'extern "C"' linkage is mandated to prevent C++ compiler symbol name
   *       mangling, ensuring clean mapping to the synthesised RTL entity/module name.
   */
  void dct(short input[N], short output[N]);
}

#endif // __DCT_H__

