#include "layernorm_kernel.h"

// Parametrically generated LayerNorm Kernel from HG-PIPE Template
// Model: Vision_Transformer_Base_16 | Dim: 768 | NACC: 16

void layernorm_kernel(
    const float in[8192],
    const float gamma[768],
    const float beta[768],
    float out[8192],
    int total_tokens
) {
    #pragma HLS INTERFACE m_axi port=in    bundle=gmem0 depth=8192 offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=gamma bundle=gmem1 depth=768 offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=beta  bundle=gmem2 depth=768 offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=out   bundle=gmem3 depth=8192 offset=slave max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=total_tokens bundle=control
    #pragma HLS INTERFACE s_axilite port=return       bundle=control

    float acc_mu[16];
    float acc_var[16];
    #pragma HLS ARRAY_PARTITION variable=acc_mu complete
    #pragma HLS ARRAY_PARTITION variable=acc_var complete

    float local_buf[768];
    #pragma HLS ARRAY_PARTITION variable=local_buf cyclic factor=16

    Token_Loop: for (int t = 0; t < total_tokens; t++) {
        int base = t * 768;

        // Reset partial accumulators
        Init_Acc: for (int a = 0; a < 16; a++) {
            #pragma HLS UNROLL
            acc_mu[a] = 0.0f;
            acc_var[a] = 0.0f;
        }

        // 1. Mean accumulation with II=1
        Mean_Loop: for (int i = 0; i < 768; i++) {
            #pragma HLS PIPELINE II=1
            float val = in[base + i];
            local_buf[i] = val;
            acc_mu[i % 16] += val;
        }

        float sum_mu = 0.0f;
        Reduce_Mu: for (int a = 0; a < 16; a++) {
            #pragma HLS UNROLL
            sum_mu += acc_mu[a];
        }
        float mean = sum_mu / (float)768;

        // 2. Variance accumulation with II=1
        Var_Loop: for (int i = 0; i < 768; i++) {
            #pragma HLS PIPELINE II=1
            float diff = local_buf[i] - mean;
            acc_var[i % 16] += diff * diff;
        }

        float sum_var = 0.0f;
        Reduce_Var: for (int a = 0; a < 16; a++) {
            #pragma HLS UNROLL
            sum_var += acc_var[a];
        }
        float inv_std = 1.0f / sqrtf((sum_var / (float)768) + 1e-5f);

        // 3. Normalise, scale & affine transform
        Norm_Loop: for (int i = 0; i < 768; i++) {
            #pragma HLS PIPELINE II=1
            float norm = (local_buf[i] - mean) * inv_std;
            out[base + i] = norm * gamma[i] + beta[i];
        }
    }
}
