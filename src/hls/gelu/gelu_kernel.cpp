#include "gelu_kernel.h"

// GELU Approximation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
extern "C" {
void gelu_kernel(
    const fp32_t in[MAX_DEPTH],
    fp32_t out[MAX_DEPTH],
    int total_elements
) {
    #pragma HLS INTERFACE m_axi port=in  bundle=gmem0 depth=MAX_DEPTH offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=out bundle=gmem1 depth=MAX_DEPTH offset=slave max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=total_elements bundle=control
    #pragma HLS INTERFACE s_axilite port=return         bundle=control

    const fp32_t sqrt_2_over_pi = 0.7978845608f;
    const fp32_t coeff = 0.044715f;

    Gelu_Loop: for (int i = 0; i < total_elements; i++) {
        #pragma HLS PIPELINE II=1
        fp32_t x = in[i];
        fp32_t x_cube = x * x * x;
        fp32_t inner = sqrt_2_over_pi * (x + coeff * x_cube);
        fp32_t tanh_val = tanhf(inner);
        out[i] = 0.5f * x * (1.0f + tanh_val);
    }
}
}
