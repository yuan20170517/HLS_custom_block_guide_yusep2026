#include "softmax_kernel.h"

extern "C" {
void softmax_kernel(
    const fp32_t in[MAX_DEPTH],
    fp32_t out[MAX_DEPTH],
    int total_rows,
    int row_length
) {
    #pragma HLS INTERFACE m_axi port=in  bundle=gmem0 depth=MAX_DEPTH offset=slave max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=out bundle=gmem1 depth=MAX_DEPTH offset=slave max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=total_rows bundle=control
    #pragma HLS INTERFACE s_axilite port=row_length bundle=control
    #pragma HLS INTERFACE s_axilite port=return     bundle=control

    fp32_t local_row[SOFTMAX_MAX_LEN];
    #pragma HLS ARRAY_PARTITION variable=local_row cyclic factor=16

    fp32_t acc_sum[NACC];
    #pragma HLS ARRAY_PARTITION variable=acc_sum complete

    Row_Loop: for (int r = 0; r < total_rows; r++) {
        int base = r * row_length;

        // 1. Online max search for numerical stability
        fp32_t max_val = -1e30f;
        Find_Max: for (int c = 0; c < row_length; c++) {
            #pragma HLS PIPELINE II=1
            fp32_t v = in[base + c];
            local_row[c] = v;
            if (v > max_val) {
                max_val = v;
            }
        }

        // 2. Exponential accumulation with 16-way partial accumulators
        Init_Acc: for (int a = 0; a < NACC; a++) {
            #pragma HLS UNROLL
            acc_sum[a] = 0.0f;
        }

        Exp_Sum: for (int c = 0; c < row_length; c++) {
            #pragma HLS PIPELINE II=1
            fp32_t e = expf(local_row[c] - max_val);
            local_row[c] = e;
            acc_sum[c % NACC] += e;
        }

        fp32_t total_sum = 0.0f;
        Reduce_Sum: for (int a = 0; a < NACC; a++) {
            #pragma HLS UNROLL
            total_sum += acc_sum[a];
        }
        fp32_t inv_sum = 1.0f / total_sum;

        // 3. Normalise output distribution
        Normalise: for (int c = 0; c < row_length; c++) {
            #pragma HLS PIPELINE II=1
            out[base + c] = local_row[c] * inv_sum;
        }
    }
}
}
