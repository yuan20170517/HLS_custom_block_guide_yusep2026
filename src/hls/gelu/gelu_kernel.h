#ifndef GELU_KERNEL_H
#define GELU_KERNEL_H

#include "../common/types.h"

extern "C" {
void gelu_kernel(
    const fp32_t in[MAX_DEPTH],
    fp32_t out[MAX_DEPTH],
    int total_elements
);
}

#endif // GELU_KERNEL_H
