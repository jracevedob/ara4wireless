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

// Zero-forcing dirty-paper coding with Tomlinson-Harashima modulo.

#include <stdint.h>

#include "kernel/dpc.h"
#include "runtime.h"
#include "util.h"

#ifndef SPIKE
#include "printf.h"
#else
#include <stdio.h>
#endif

#define THRESHOLD 1e-5

extern uint64_t M;
extern uint64_t K;
extern uint64_t N;
extern double P;
extern double h_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double h_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double s_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double s_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern double q_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double q_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double rdiag[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double u_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double u_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double x_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double x_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern double gq_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double gq_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double grdiag[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double gu_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double gu_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double gx_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double gx_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

static int verify(const double *a, const double *b, size_t n, const char *name) {
  for (size_t i = 0; i < n; ++i) {
    if (!similarity_check(a[i], b[i], THRESHOLD)) {
#ifndef SPIKE
      printf("Error: %s[%d] = %f != %f\n", name, (int)i, a[i], b[i]);
#else
      printf("Error: %s[%d] mismatch\n", name, (int)i);
#endif
      return i ? (int)i : -1;
    }
  }
  return 0;
}

static int verify_cplx(const double *re, const double *im, const double *g_re,
                       const double *g_im, size_t n, const char *name) {
  for (size_t i = 0; i < n; ++i) {
    if (!similarity_check(re[i], g_re[i], THRESHOLD) ||
        !similarity_check(im[i], g_im[i], THRESHOLD)) {
      printf("Error: %s[%d] mismatch\n", name, (int)i);
      return i ? (int)i : -1;
    }
  }
  return 0;
}

static int check_dpc(size_t k_users, size_t m_ant, double power, const double *q_r,
                     const double *q_i, const double *x_r, const double *x_i) {
  for (size_t k = 0; k < k_users; ++k) {
    double nrm = 0.0;
    for (size_t m = 0; m < m_ant; ++m) {
      double qr = q_r[k * m_ant + m];
      double qi = q_i[k * m_ant + m];
      nrm += qr * qr + qi * qi;
    }
    if (!similarity_check(nrm, 1.0, THRESHOLD))
      return -1;
  }
  double px = 0.0;
  for (size_t m = 0; m < m_ant; ++m)
    px += x_r[m] * x_r[m] + x_i[m] * x_i[m];
  if (!similarity_check(px, power, THRESHOLD))
    return -1;
  return 0;
}

int main(void) {
  printf("\n");
  printf("====================\n");
  printf("=  DIRTY PAPER     =\n");
  printf("====================\n");
  printf("\n");
  size_t S = K * N;
  printf("M = %d transmit antennas, K = %d users, N = %d antennas each.\n", (int)M,
         (int)K, (int)N);

  start_timer();
  dpc_gold(K, N, M, P, h_re, h_im, s_re, s_im, gq_re, gq_im, grdiag, gu_re, gu_im,
           gx_re, gx_im);
  stop_timer();
  printf("DPC Ariane took %d cycles.\n", (int)get_timer());
  start_timer();
  dpc_vec(K, N, M, P, h_re, h_im, s_re, s_im, q_re, q_im, rdiag, u_re, u_im, x_re,
          x_im);
  stop_timer();
  printf("DPC Ariane+Ara took %d cycles.\n", (int)get_timer());
  if (verify_cplx(q_re, q_im, gq_re, gq_im, S * M, "Q") ||
      verify(rdiag, grdiag, S, "Rdiag") ||
      verify_cplx(u_re, u_im, gu_re, gu_im, S, "u") ||
      verify_cplx(x_re, x_im, gx_re, gx_im, M, "x"))
    return 1;
  start_timer();
  dpc_ll_gold(K, N, M, P, h_re, h_im, s_re, s_im, rdiag, u_re, u_im, x_re, x_im);
  stop_timer();
  printf("DPC Ariane low-latency took %d cycles.\n", (int)get_timer());
  if (verify(rdiag, grdiag, S, "Rdiag_ll") ||
      verify_cplx(u_re, u_im, gu_re, gu_im, S, "u_ll") ||
      verify_cplx(x_re, x_im, gx_re, gx_im, M, "x_ll"))
    return 1;
  start_timer();
  dpc_ll_vec(K, N, M, P, h_re, h_im, s_re, s_im, rdiag, u_re, u_im, x_re, x_im);
  stop_timer();
  printf("DPC Ariane+Ara low-latency took %d cycles.\n", (int)get_timer());
  if (verify(rdiag, grdiag, S, "Rdiag_ll_vec") ||
      verify_cplx(u_re, u_im, gu_re, gu_im, S, "u_ll_vec") ||
      verify_cplx(x_re, x_im, gx_re, gx_im, M, "x_ll_vec"))
    return 1;
  if (check_dpc(S, M, P, gq_re, gq_im, gx_re, gx_im)) {
    printf("DPC orthogonality or power check failed.\n");
    return 1;
  }
  printf("DPC passed.\n");
  printf("Passed.\n");
  return 0;
}
