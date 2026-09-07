#include "softmax_kernel.h"

// Parametrically generated Softmax Kernel from HG-PIPE Template
// Model: Vision_Transformer_Base_16 | SeqLen: 197 | NACC: 16

void softmax_kernel(
    const float in[8192],
    float out[8192],
    int total_rows,
    int row_length
) {
    #pragma HLS INTERFACE m_axi port=in  bundle=gmem0 depth=8192 offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=out bundle=gmem1 depth=8192 offset=slave max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=total_rows bundle=control
    #pragma HLS INTERFACE s_axilite port=row_length bundle=control
    #pragma HLS INTERFACE s_axilite port=return     bundle=control

    float local_row[197];
    #pragma HLS ARRAY_PARTITION variable=local_row cyclic factor=16

    float acc_sum[16];
    #pragma HLS ARRAY_PARTITION variable=acc_sum complete

    Row_Loop: for (int r = 0; r < total_rows; r++) {
        int base = r * row_length;

        // 1. Find max for numerical stability
        float max_val = -1e30f;
        Find_Max: for (int c = 0; c < row_length; c++) {
            #pragma HLS PIPELINE II=1
            float v = in[base + c];
            local_row[c] = v;
            if (v > max_val) {
                max_val = v;
            }
        }

        // 2. Compute exp and accumulate sum
        Init_Acc: for (int a = 0; a < 16; a++) {
            #pragma HLS UNROLL
            acc_sum[a] = 0.0f;
        }

        Exp_Sum: for (int c = 0; c < row_length; c++) {
            #pragma HLS PIPELINE II=1
            float e = expf(local_row[c] - max_val);
            local_row[c] = e;
            acc_sum[c % 16] += e;
        }

        float total_sum = 0.0f;
        Reduce_Sum: for (int a = 0; a < 16; a++) {
            #pragma HLS UNROLL
            total_sum += acc_sum[a];
        }
        float inv_sum = 1.0f / total_sum;

        // 3. Normalise output probabilities
        Normalise: for (int c = 0; c < row_length; c++) {
            #pragma HLS PIPELINE II=1
            out[base + c] = local_row[c] * inv_sum;
        }
    }
}
