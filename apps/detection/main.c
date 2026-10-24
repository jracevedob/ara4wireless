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

// Uplink detectors: zero-forcing, parallel interference cancellation, and
// sphere decoding.

#include <stdint.h>

#include "kernel/detection.h"
#include "runtime.h"
#include "util.h"

#ifndef SPIKE
#include "printf.h"
#else
#include <stdio.h>
#endif

#define THRESHOLD 1e-5
#define INV_SQRT2 0.7071067811865476

extern uint64_t M;
extern uint64_t K;
extern uint64_t N;
extern uint64_t niter;
extern double sigma_n2;
extern double h_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double h_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double y_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double y_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double x_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double x_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern double zf_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double zf_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_zf_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_zf_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern double pic_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double pic_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_pic_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_pic_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern double sd_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double sd_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_sd_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_sd_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

static int verify_cplx(const double *re, const double *im, const double *g_re,
                       const double *g_im, size_t n, const char *name) {
  for (size_t i = 0; i < n; ++i) {
    if (!similarity_check(re[i], g_re[i], THRESHOLD) ||
        !similarity_check(im[i], g_im[i], THRESHOLD)) {
#ifndef SPIKE
      printf("Error: %s[%d] = (%f, %f) != (%f, %f)\n", name, (int)i, re[i], im[i],
             g_re[i], g_im[i]);
#else
      printf("Error: %s[%d] mismatch\n", name, (int)i);
#endif
      return i ? (int)i : -1;
    }
  }
  return 0;
}

static void slice_qpsk(const double *re, const double *im, double *out_re, double *out_im,
                       size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out_re[i] = (re[i] >= 0.0) ? INV_SQRT2 : -INV_SQRT2;
    out_im[i] = (im[i] >= 0.0) ? INV_SQRT2 : -INV_SQRT2;
  }
}

static int symbol_errors(const double *hat_re, const double *hat_im, const double *true_re,
                         const double *true_im, size_t n, int already_sliced) {
  double sl_re[n];
  double sl_im[n];
  const double *re = hat_re;
  const double *im = hat_im;
  if (!already_sliced) {
    slice_qpsk(hat_re, hat_im, sl_re, sl_im, n);
    re = sl_re;
    im = sl_im;
  }
  int err = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!similarity_check(re[i], true_re[i], THRESHOLD) ||
        !similarity_check(im[i], true_im[i], THRESHOLD))
      ++err;
  }
  return err;
}

// Normal equations: Hᴴ (H hat{x} - y) should be ~ 0 for ZF.
static int check_zf_normal(size_t S, size_t m_ant, const double *h_r, const double *h_i,
                           const double *y_r, const double *y_i, const double *xhat_r,
                           const double *xhat_i) {
  for (size_t s = 0; s < S; ++s) {
    double rr = 0.0;
    double ri = 0.0;
    for (size_t m = 0; m < m_ant; ++m) {
      double hx_r = -y_r[m];
      double hx_i = -y_i[m];
      for (size_t t = 0; t < S; ++t) {
        double hr = h_r[t * m_ant + m];
        double hi = h_i[t * m_ant + m];
        hx_r += hr * xhat_r[t] - hi * xhat_i[t];
        hx_i += hr * xhat_i[t] + hi * xhat_r[t];
      }
      double hr = h_r[s * m_ant + m];
      double hi = h_i[s * m_ant + m];
      rr += hr * hx_r + hi * hx_i;
      ri += hr * hx_i - hi * hx_r;
    }
    if (!similarity_check(rr, 0.0, 1e-8) || !similarity_check(ri, 0.0, 1e-8))
      return -1;
  }
  return 0;
}

int main(void) {
  printf("\n");
  printf("=====================\n");
  printf("=  UPLINK DETECTION =\n");
  printf("=====================\n");
  printf("\n");
  size_t S = K * N;
  printf("M = %d receive antennas, K = %d users, N = %d antennas each, PIC iters = %d.\n",
         (int)M, (int)K, (int)N, (int)niter);
#ifndef SPIKE
  printf("sigma_n2 = %f\n", sigma_n2);
#else
  printf("sigma_n2 (x1000) = %d\n", (int)(sigma_n2 * 1000.0));
#endif

  start_timer();
  zf_gold(K, N, M, h_re, h_im, y_re, y_im, g_zf_re, g_zf_im);
  stop_timer();
  printf("ZF Ariane took %d cycles.\n", (int)get_timer());
  start_timer();
  zf_vec(K, N, M, h_re, h_im, y_re, y_im, zf_re, zf_im);
  stop_timer();
  printf("ZF Ariane+Ara took %d cycles.\n", (int)get_timer());
  if (verify_cplx(zf_re, zf_im, g_zf_re, g_zf_im, S, "x_zf"))
    return 1;
  start_timer();
  zf_ll_gold(K, N, M, h_re, h_im, y_re, y_im, zf_re, zf_im);
  stop_timer();
  printf("ZF Ariane low-latency took %d cycles.\n", (int)get_timer());
  if (verify_cplx(zf_re, zf_im, g_zf_re, g_zf_im, S, "x_zf_ll"))
    return 1;
  start_timer();
  zf_ll_vec(K, N, M, h_re, h_im, y_re, y_im, zf_re, zf_im);
  stop_timer();
  printf("ZF Ariane+Ara low-latency took %d cycles.\n", (int)get_timer());
  if (verify_cplx(zf_re, zf_im, g_zf_re, g_zf_im, S, "x_zf_ll_vec"))
    return 1;
  if (check_zf_normal(S, M, h_re, h_im, y_re, y_im, g_zf_re, g_zf_im)) {
    printf("ZF normal-equation check failed.\n");
    return 1;
  }
  printf("ZF symbol errors: %d / %d\n",
         symbol_errors(g_zf_re, g_zf_im, x_re, x_im, S, 0), (int)S);
  printf("ZF passed.\n\n");

  start_timer();
  pic_gold(K, N, M, niter, h_re, h_im, y_re, y_im, g_pic_re, g_pic_im);
  stop_timer();
  printf("PIC Ariane took %d cycles.\n", (int)get_timer());
  start_timer();
  pic_vec(K, N, M, niter, h_re, h_im, y_re, y_im, pic_re, pic_im);
  stop_timer();
  printf("PIC Ariane+Ara took %d cycles.\n", (int)get_timer());
  if (verify_cplx(pic_re, pic_im, g_pic_re, g_pic_im, S, "x_pic"))
    return 1;
  start_timer();
  pic_ll_gold(K, N, M, niter, h_re, h_im, y_re, y_im, pic_re, pic_im);
  stop_timer();
  printf("PIC Ariane low-latency took %d cycles.\n", (int)get_timer());
  if (verify_cplx(pic_re, pic_im, g_pic_re, g_pic_im, S, "x_pic_ll"))
    return 1;
  start_timer();
  pic_ll_vec(K, N, M, niter, h_re, h_im, y_re, y_im, pic_re, pic_im);
  stop_timer();
  printf("PIC Ariane+Ara low-latency took %d cycles.\n", (int)get_timer());
  if (verify_cplx(pic_re, pic_im, g_pic_re, g_pic_im, S, "x_pic_ll_vec"))
    return 1;
  printf("PIC symbol errors: %d / %d\n",
         symbol_errors(g_pic_re, g_pic_im, x_re, x_im, S, 0), (int)S);
  printf("PIC passed.\n\n");

  start_timer();
  sd_gold(K, N, M, h_re, h_im, y_re, y_im, g_sd_re, g_sd_im);
  stop_timer();
  printf("SD Ariane took %d cycles.\n", (int)get_timer());
  start_timer();
  sd_vec(K, N, M, h_re, h_im, y_re, y_im, sd_re, sd_im);
  stop_timer();
  printf("SD Ariane+Ara took %d cycles.\n", (int)get_timer());
  if (verify_cplx(sd_re, sd_im, g_sd_re, g_sd_im, S, "x_sd"))
    return 1;
  start_timer();
  sd_ll_gold(K, N, M, h_re, h_im, y_re, y_im, sd_re, sd_im);
  stop_timer();
  printf("SD Ariane low-latency took %d cycles.\n", (int)get_timer());
  if (verify_cplx(sd_re, sd_im, g_sd_re, g_sd_im, S, "x_sd_ll"))
    return 1;
  start_timer();
  sd_ll_vec(K, N, M, h_re, h_im, y_re, y_im, sd_re, sd_im);
  stop_timer();
  printf("SD Ariane+Ara low-latency took %d cycles.\n", (int)get_timer());
  if (verify_cplx(sd_re, sd_im, g_sd_re, g_sd_im, S, "x_sd_ll_vec"))
    return 1;
  printf("SD symbol errors: %d / %d\n",
         symbol_errors(g_sd_re, g_sd_im, x_re, x_im, S, 1), (int)S);
  printf("SD passed.\n\n");

  printf("Passed.\n");
  return 0;
}
