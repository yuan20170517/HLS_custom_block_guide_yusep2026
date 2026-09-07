#include "gelu_kernel.h"

// Parametrically generated GELU Kernel from HG-PIPE Template
// Model: Vision_Transformer_Base_16 | MaxDepth: 8192

extern "C" {
void gelu_kernel(
    const float in[8192],
    float out[8192],
    int total_elements
) {
    #pragma HLS INTERFACE m_axi port=in  bundle=gmem0 depth=8192 offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=out bundle=gmem1 depth=8192 offset=slave max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=total_elements bundle=control
    #pragma HLS INTERFACE s_axilite port=return         bundle=control

    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    Gelu_Loop: for (int i = 0; i < total_elements; i++) {
        #pragma HLS PIPELINE II=1
        float x = in[i];
        float x_cube = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cube);
        float tanh_val = tanhf(inner);
        out[i] = 0.5f * (float)x * (1.0f + tanh_val);
    }
}
}
