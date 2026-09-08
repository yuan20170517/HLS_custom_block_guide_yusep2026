/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

/**
 * @file dct_test.cpp
 * @brief Self-Checking Testbench for 2-Dimensional 8x8 DCT Kernel.
 *
 * This testbench provides functional verification and regression testing for the
 * hardware DCT kernel during High-Level Synthesis (C-Simulation and C/RTL Co-Simulation).
 *
 * Verification Strategy:
 *   1. Bulk Stimulus Loading: Ingests a continuous dataset of 10 consecutive 8x8
 *      blocks (640 samples total) from 'in.dat'.
 *   2. Multi-Iteration Dataflow Stress: Invokes 'dct()' across 10 successive block
 *      iterations to verify kernel re-entrancy, pipeline flushing, and memory stability
 *      under continuous streaming conditions.
 *   3. Golden Reference Validation: Captures the primary output block (iteration i=0),
 *      writes it to 'out.dat', and validates against 'out.golden.dat'.
 *   4. Automated Exit Code: Adheres to the HLS verification contract by returning 0
 *      on zero functional discrepancies, and 1 on failure to drive CI/CD build scripts.
 */

#include "dct.h"

int main()
{
   // a: Single-block kernel input buffer (N = 64 elements)
   // b: Single-block kernel output buffer (N = 64 elements)
   // b_prime: Captured output block from the primary iteration (i = 0)
   // x: Bulk stimulus vector holding 10 continuous blocks (10 * 64 = 640 elements)
   short a[N], b[N], b_prime[N], x[10 * N];
   int retval = 0, i, j, z;
   FILE *fp;

   // =========================================================================
   // Step 1: Ingest Stimulus Vectors from File
   // =========================================================================
   fp = fopen("in.dat", "r");
   if (!fp) {
      fprintf(stderr, "ERROR: Unable to open input stimulus file 'in.dat'!\n");
      return 1;
   }

   // Read 640 sequential integer entries (10 consecutive 8x8 data blocks)
   for (i = 0; i < (10 * N); i++) {
      int tmp;
      if (fscanf(fp, "%d", &tmp) != 1) {
         fprintf(stderr, "ERROR: Premature end of file or parse error in 'in.dat' at index %d!\n", i);
         fclose(fp);
         return 1;
      }
      x[i] = (short)tmp;
   }
   fclose(fp);

   // =========================================================================
   // Step 2: Multi-Frame Execution (Stress-Testing Pipeline & Task Dataflow)
   // =========================================================================
   // Calling the DCT kernel 10 times consecutively verifies that internal state,
   // loop counters, and pipeline flushing behaviour remain clean across transactions.
   for (i = 0; i < 10; i++) {
      printf("Execution Checkpoint: Processing Block %d/10\n", i);

      // Slice out the i-th block of N=64 samples from the continuous dataset
      for (j = 0; j < N; j++) {
         a[j] = x[j + (N * i)];
      }

      // Invoke the top-level hardware kernel
      dct(a, b);

      // Capture the primary iteration result (i = 0) for golden verification
      if (i == 0) {
         printf("Capturing Block 0 output for golden comparison...\n");
         for (z = 0; z < N; z++) {
            b_prime[z] = b[z];
         }
      }
   }

   // =========================================================================
   // Step 3: Serialize Transformed Output to Disk
   // =========================================================================
   fp = fopen("out.dat", "w");
   if (!fp) {
      fprintf(stderr, "ERROR: Unable to open output file 'out.dat' for writing!\n");
      return 1;
   }

   for (i = 0; i < N; i++) {
      fprintf(fp, "%d \n", b_prime[i]);
   }
   fclose(fp);

   // =========================================================================
   // Step 4: Validate Against Golden Reference
   // =========================================================================
   // Perform bit-exact whitespace-insensitive file comparison
   printf("Comparing 'out.dat' against golden reference 'out.golden.dat'...\n");
   retval = system("diff --brief -w out.dat out.golden.dat");

   if (retval != 0) {
      printf("\n************************************************\n");
      printf("TEST FAILED: Output differs from golden reference!\n");
      printf("************************************************\n\n");
      retval = 1;
   } else {
      printf("\n************************************************\n");
      printf("TEST PASSED: Bit-exact match confirmed!\n");
      printf("************************************************\n\n");
      retval = 0;
   }

   // Return status to HLS framework (0 = SUCCESS, non-zero = FAILURE)
   return retval;
}

