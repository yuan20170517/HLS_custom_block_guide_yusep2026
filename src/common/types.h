/**
 * @file types.h
 * @brief Common Floating-Point and System-Level Type Definitions for Custom HLS Kernels.
 *
 * Provides standardized typedefs, interface depth parameters, and accumulation constants
 * utilized across Transformer sub-layer kernels (LayerNorm, Softmax, GELU).
 */

#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <cmath>
#include "hls_common.hpp"

// ==============================================================================
// Floating-Point & Fixed-Point Type Definitions
// ==============================================================================
typedef float fp32_t;
typedef double fp64_t;

// ==============================================================================
// Memory & Interface Sizing Parameters
// ==============================================================================
/**
 * @def MAX_DEPTH
 * @brief Maximum burst transaction depth for AXI master interfaces (m_axi).
 *        Allocates storage for up to 4096 elements per transaction stream.
 */
#ifndef MAX_DEPTH
#define MAX_DEPTH 4096
#endif

/**
 * @def NACC
 * @brief Number of interleaved partial accumulators for floating-point pipelining.
 *        Breaks cyclic latency feedback (II=1) on DSP48/DSP58 primitives.
 */
#ifndef NACC
#define NACC 16
#endif

#endif // TYPES_H
