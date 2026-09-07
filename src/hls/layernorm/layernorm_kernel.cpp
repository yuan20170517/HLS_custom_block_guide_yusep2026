#include "layernorm_kernel.h"

extern "C" {
void layernorm_kernel(
    const fp32_t in[MAX_DEPTH],
    const fp32_t gamma[LN_DIM],
    const fp32_t beta[LN_DIM],
    fp32_t out[MAX_DEPTH],
    int total_tokens
) {
    #pragma HLS INTERFACE m_axi port=in    bundle=gmem0 depth=MAX_DEPTH offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=gamma bundle=gmem1 depth=LN_DIM    offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=beta  bundle=gmem2 depth=LN_DIM    offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=out   bundle=gmem3 depth=MAX_DEPTH offset=slave max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=total_tokens bundle=control
    #pragma HLS INTERFACE s_axilite port=return       bundle=control

    fp32_t acc_mu[NACC];
    fp32_t acc_var[NACC];
    #pragma HLS ARRAY_PARTITION variable=acc_mu complete
    #pragma HLS ARRAY_PARTITION variable=acc_var complete

    fp32_t local_buf[LN_DIM];
    #pragma HLS ARRAY_PARTITION variable=local_buf cyclic factor=16

    Token_Loop: for (int t = 0; t < total_tokens; t++) {
        int base = t * LN_DIM;

        // Reset 16-way partial accumulators
        Init_Acc: for (int a = 0; a < NACC; a++) {
            #pragma HLS UNROLL
            acc_mu[a] = 0.0f;
            acc_var[a] = 0.0f;
        }

        // 1. Mean accumulation with II=1
        Mean_Loop: for (int i = 0; i < LN_DIM; i++) {
            #pragma HLS PIPELINE II=1
            fp32_t val = in[base + i];
            local_buf[i] = val;
            acc_mu[i % NACC] += val;
        }

        fp32_t sum_mu = 0.0f;
        Reduce_Mu: for (int a = 0; a < NACC; a++) {
            #pragma HLS UNROLL
            sum_mu += acc_mu[a];
        }
        fp32_t mean = sum_mu / (fp32_t)LN_DIM;

        // 2. Variance accumulation with II=1
        Var_Loop: for (int i = 0; i < LN_DIM; i++) {
            #pragma HLS PIPELINE II=1
            fp32_t diff = local_buf[i] - mean;
            acc_var[i % NACC] += diff * diff;
        }

        fp32_t sum_var = 0.0f;
        Reduce_Var: for (int a = 0; a < NACC; a++) {
            #pragma HLS UNROLL
            sum_var += acc_var[a];
        }
        fp32_t inv_std = 1.0f / sqrtf((sum_var / (fp32_t)LN_DIM) + 1e-5f);

        // 3. Normalise, scale & affine transform
        Norm_Loop: for (int i = 0; i < LN_DIM; i++) {
            #pragma HLS PIPELINE II=1
            fp32_t norm = (local_buf[i] - mean) * inv_std;
            out[base + i] = norm * gamma[i] + beta[i];
        }
    }
}
}
