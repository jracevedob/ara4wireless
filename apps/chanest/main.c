// Copyright 2026 TUD Technical University of Dresden
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Javier Acevedo, TU Dresden Telekom Chair of Communication Networks

// 5G NR DMRS channel estimation benchmark: LSE and MMSE.

#include <stdint.h>
#include <string.h>

#include "kernel/chanest.h"
#include "runtime.h"
#include "util.h"

#ifndef SPIKE
#include "printf.h"
#else
#include <stdio.h>
#endif

extern uint64_t N;
extern double sigma_h2;
extern double sigma_n2;

extern double xp_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double xp_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double yp_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double yp_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern double h_lse_re[]
    __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double h_lse_im[]
    __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double h_mmse_re[]
    __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double h_mmse_im[]
    __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern double g_lse_re[]
    __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_lse_im[]
    __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_mmse_re[]
    __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_mmse_im[]
    __attribute__((aligned(32 * NR_LANES), section(".l2")));

#define THRESHOLD 0.0000000001

// LSE: 6 mul + 3 add/sub + 2 div per RE. MMSE adds 3 mul + 1 add.
#define FLOP_LSE 11
#define FLOP_MMSE 15

static int verify_cplx(const double *re, const double *im, const double *g_re,
                       const double *g_im, size_t n, double threshold,
                       const char *name) {
  for (size_t k = 0; k < n; ++k) {
    if (!similarity_check(re[k], g_re[k], threshold) ||
        !similarity_check(im[k], g_im[k], threshold)) {
#ifndef SPIKE
      printf("Error: %s[%d] = (%f, %f) != (%f, %f)\n", name, (int)k, re[k],
             im[k], g_re[k], g_im[k]);
#else
      printf("Error: %s[%d] mismatch\n", name, (int)k);
#endif
      return k ? (int)k : -1;
    }
  }
  return 0;
}

int main() {
  printf("\n");
  printf("==============\n");
  printf("=  CHANEST   =\n");
  printf("==============\n");
  printf("\n");
  printf("5G NR DMRS channel estimation, %d subcarriers.\n", (int)N);
#ifndef SPIKE
  printf("sigma_h2 = %f, sigma_n2 = %f\n", sigma_h2, sigma_n2);
#else
  printf("sigma_h2 (x1000) = %d, sigma_n2 (x1000) = %d\n",
         (int)(sigma_h2 * 1000.0), (int)(sigma_n2 * 1000.0));
#endif
  printf("\n");

  // ---- LSE / least-squares ----
  start_timer();
  lse_vec(N, xp_re, xp_im, yp_re, yp_im, h_lse_re, h_lse_im);
  stop_timer();
  int64_t runtime_lse = get_timer();
  printf("LSE kernel took %d cycles.\n", (int)runtime_lse);
#ifndef SPIKE
  if (runtime_lse > 0) {
    printf("LSE performance: %f DP-FLOP/cycle.\n",
           (double)N * FLOP_LSE / runtime_lse);
  }
#endif

  lse_gold(N, xp_re, xp_im, yp_re, yp_im, g_lse_re, g_lse_im);
  int err = verify_cplx(h_lse_re, h_lse_im, g_lse_re, g_lse_im, N, THRESHOLD,
                        "h_lse");
  if (err)
    return err;
  printf("LSE passed.\n");
  printf("\n");

  // ---- MMSE / Wiener ----
  start_timer();
  mmse_vec(N, xp_re, xp_im, yp_re, yp_im, sigma_h2, sigma_n2, h_mmse_re,
           h_mmse_im);
  stop_timer();
  int64_t runtime_mmse = get_timer();
  printf("MMSE kernel took %d cycles.\n", (int)runtime_mmse);
#ifndef SPIKE
  if (runtime_mmse > 0) {
    printf("MMSE performance: %f DP-FLOP/cycle.\n",
           (double)N * FLOP_MMSE / runtime_mmse);
  }
#endif

  mmse_gold(N, xp_re, xp_im, yp_re, yp_im, sigma_h2, sigma_n2, g_mmse_re,
            g_mmse_im);
  err = verify_cplx(h_mmse_re, h_mmse_im, g_mmse_re, g_mmse_im, N, THRESHOLD,
                    "h_mmse");
  if (err)
    return err;
  printf("MMSE passed.\n");
  printf("\n");
  printf("Passed.\n");

  return 0;
}
