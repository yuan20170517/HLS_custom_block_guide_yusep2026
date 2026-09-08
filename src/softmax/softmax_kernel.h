#ifndef SOFTMAX_KERNEL_H
#define SOFTMAX_KERNEL_H

#include "../common/types.h"

#define SOFTMAX_MAX_LEN 1024

extern "C" {
void softmax_kernel(
    const fp32_t in[MAX_DEPTH],
    fp32_t out[MAX_DEPTH],
    int total_rows,
    int row_length
);
}

#endif // SOFTMAX_KERNEL_H
