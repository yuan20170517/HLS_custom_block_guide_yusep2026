#ifndef LAYERNORM_KERNEL_H
#define LAYERNORM_KERNEL_H

#include "../common/types.h"

#define LN_DIM 768

extern "C" {
void layernorm_kernel(
    const fp32_t in[MAX_DEPTH],
    const fp32_t gamma[LN_DIM],
    const fp32_t beta[LN_DIM],
    fp32_t out[MAX_DEPTH],
    int total_tokens
);
}

#endif // LAYERNORM_KERNEL_H
