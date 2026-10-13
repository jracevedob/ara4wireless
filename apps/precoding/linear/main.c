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

// Linear downlink precoders: zero-forcing, MMSE, and large-scale fading.

#include <math.h>
#include <stdint.h>

#include "kernel/precoding.h"
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
extern double sigma_n2;
extern double h_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double h_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double s_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double s_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern double w_zf_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double w_zf_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double x_zf_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double x_zf_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_zf_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_zf_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double gx_zf_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double gx_zf_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern double w_mmse_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double w_mmse_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double x_mmse_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double x_mmse_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_mmse_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_mmse_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double gx_mmse_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double gx_mmse_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern double w_lsf_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double w_lsf_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double x_lsf_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double x_lsf_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_lsf_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double g_lsf_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double gx_lsf_re[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double gx_lsf_im[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

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

static double frobenius(const double *re, const double *im, size_t n) {
  double p = 0.0;
  for (size_t i = 0; i < n; ++i)
    p += re[i] * re[i] + im[i] * im[i];
  return p;
}

// (H W)_ij. W is stored by columns, w[j * M + m].
static void hw_entry(size_t i, size_t j, size_t m_ant, const double *h_r,
                     const double *h_i, const double *w_r, const double *w_i,
                     double *re, double *im) {
  double sr = 0.0;
  double si = 0.0;
  for (size_t m = 0; m < m_ant; ++m) {
    double hr = h_r[i * m_ant + m];
    double hi = h_i[i * m_ant + m];
    double wr = w_r[j * m_ant + m];
    double wi = w_i[j * m_ant + m];
    sr += hr * wr - hi * wi;
    si += hr * wi + hi * wr;
  }
  *re = sr;
  *im = si;
}

static int check_zf(size_t k_users, size_t m_ant, double power, const double *h_r,
                    const double *h_i, const double *w_r, const double *w_i) {
  double diag = 0.0;
  for (size_t i = 0; i < k_users; ++i) {
    for (size_t j = 0; j < k_users; ++j) {
      double re;
      double im;
      hw_entry(i, j, m_ant, h_r, h_i, w_r, w_i, &re, &im);
      if (i == j) {
        if (!similarity_check(im, 0.0, THRESHOLD))
          return -1;
        if (i == 0)
          diag = re;
        else if (!similarity_check(re, diag, THRESHOLD))
          return -1;
      } else if (!similarity_check(re, 0.0, THRESHOLD) ||
                 !similarity_check(im, 0.0, THRESHOLD)) {
        return -1;
      }
    }
  }
  if (!similarity_check(frobenius(w_r, w_i, k_users * m_ant), power, THRESHOLD))
    return -1;
  return 0;
}

static int check_power(size_t n, double power, const double *re, const double *im) {
  return similarity_check(frobenius(re, im, n), power, THRESHOLD) ? 0 : -1;
}

static void report_sinr(const char *name, size_t n_streams, size_t n_rx, size_t m_ant,
                        double noise, const double *h_r, const double *h_i,
                        const double *w_r, const double *w_i) {
  for (size_t k = 0; k < n_streams; ++k) {
    double sig_re, sig_im, sig, inter = 0.0;
    hw_entry(k, k, m_ant, h_r, h_i, w_r, w_i, &sig_re, &sig_im);
    sig = sig_re * sig_re + sig_im * sig_im;
    for (size_t j = 0; j < n_streams; ++j) {
      double re, im;
      if (j == k)
        continue;
      hw_entry(k, j, m_ant, h_r, h_i, w_r, w_i, &re, &im);
      inter += re * re + im * im;
    }
    double sinr = sig / (inter + noise);
#ifndef SPIKE
    printf("%s SINR user %d antenna %d: %f dB\n", name, (int)(k / n_rx),
           (int)(k % n_rx), 10.0 * log10(sinr));
#else
    printf("%s SINR user %d antenna %d (x100): %d\n", name, (int)(k / n_rx),
           (int)(k % n_rx), (int)(sinr * 100.0));
#endif
  }
}

int main(void) {
  printf("\n");
  printf("=====================\n");
  printf("=  LINEAR PRECODING =\n");
  printf("=====================\n");
  printf("\n");
  size_t S = K * N;
  printf("M = %d transmit antennas, K = %d users, N = %d antennas each.\n", (int)M,
         (int)K, (int)N);

  start_timer();
  zf_gold(K, N, M, P, h_re, h_im, s_re, s_im, g_zf_re, g_zf_im, gx_zf_re, gx_zf_im);
  stop_timer();
  printf("ZF Ariane took %d cycles.\n", (int)get_timer());
  start_timer();
  zf_vec(K, N, M, P, h_re, h_im, s_re, s_im, w_zf_re, w_zf_im, x_zf_re, x_zf_im);
  stop_timer();
  printf("ZF Ariane+Ara took %d cycles.\n", (int)get_timer());
  if (verify_cplx(w_zf_re, w_zf_im, g_zf_re, g_zf_im, S * M, "W_zf") ||
      verify_cplx(x_zf_re, x_zf_im, gx_zf_re, gx_zf_im, M, "x_zf"))
    return 1;
  start_timer();
  zf_ll_gold(K, N, M, P, h_re, h_im, s_re, s_im, w_zf_re, w_zf_im, x_zf_re, x_zf_im);
  stop_timer();
  printf("ZF Ariane low-latency took %d cycles.\n", (int)get_timer());
  if (verify_cplx(w_zf_re, w_zf_im, g_zf_re, g_zf_im, S * M, "W_zf_ll") ||
      verify_cplx(x_zf_re, x_zf_im, gx_zf_re, gx_zf_im, M, "x_zf_ll"))
    return 1;
  start_timer();
  zf_ll_vec(K, N, M, P, h_re, h_im, s_re, s_im, w_zf_re, w_zf_im, x_zf_re, x_zf_im);
  stop_timer();
  printf("ZF Ariane+Ara low-latency took %d cycles.\n", (int)get_timer());
  if (verify_cplx(w_zf_re, w_zf_im, g_zf_re, g_zf_im, S * M, "W_zf_ll_vec") ||
      verify_cplx(x_zf_re, x_zf_im, gx_zf_re, gx_zf_im, M, "x_zf_ll_vec"))
    return 1;
  if (check_zf(S, M, P, h_re, h_im, g_zf_re, g_zf_im)) {
    printf("ZF nulling check failed.\n");
    return 1;
  }
  report_sinr("ZF", S, N, M, sigma_n2, h_re, h_im, g_zf_re, g_zf_im);
  printf("ZF passed.\n\n");

  start_timer();
  mmse_gold(K, N, M, P, sigma_n2, h_re, h_im, s_re, s_im, g_mmse_re, g_mmse_im,
            gx_mmse_re, gx_mmse_im);
  stop_timer();
  printf("MMSE Ariane took %d cycles.\n", (int)get_timer());
  start_timer();
  mmse_vec(K, N, M, P, sigma_n2, h_re, h_im, s_re, s_im, w_mmse_re, w_mmse_im,
           x_mmse_re, x_mmse_im);
  stop_timer();
  printf("MMSE Ariane+Ara took %d cycles.\n", (int)get_timer());
  if (verify_cplx(w_mmse_re, w_mmse_im, g_mmse_re, g_mmse_im, S * M, "W_mmse") ||
      verify_cplx(x_mmse_re, x_mmse_im, gx_mmse_re, gx_mmse_im, M, "x_mmse"))
    return 1;
  start_timer();
  mmse_ll_gold(K, N, M, P, sigma_n2, h_re, h_im, s_re, s_im, w_mmse_re, w_mmse_im,
               x_mmse_re, x_mmse_im);
  stop_timer();
  printf("MMSE Ariane low-latency took %d cycles.\n", (int)get_timer());
  if (verify_cplx(w_mmse_re, w_mmse_im, g_mmse_re, g_mmse_im, S * M, "W_mmse_ll") ||
      verify_cplx(x_mmse_re, x_mmse_im, gx_mmse_re, gx_mmse_im, M, "x_mmse_ll"))
    return 1;
  start_timer();
  mmse_ll_vec(K, N, M, P, sigma_n2, h_re, h_im, s_re, s_im, w_mmse_re, w_mmse_im,
              x_mmse_re, x_mmse_im);
  stop_timer();
  printf("MMSE Ariane+Ara low-latency took %d cycles.\n", (int)get_timer());
  if (verify_cplx(w_mmse_re, w_mmse_im, g_mmse_re, g_mmse_im, S * M, "W_mmse_ll_vec") ||
      verify_cplx(x_mmse_re, x_mmse_im, gx_mmse_re, gx_mmse_im, M, "x_mmse_ll_vec"))
    return 1;
  if (check_power(S * M, P, g_mmse_re, g_mmse_im)) {
    printf("MMSE power check failed.\n");
    return 1;
  }
  report_sinr("MMSE", S, N, M, sigma_n2, h_re, h_im, g_mmse_re, g_mmse_im);
  printf("MMSE passed.\n\n");

  start_timer();
  lsf_gold(K, N, M, P, h_re, h_im, s_re, s_im, g_lsf_re, g_lsf_im, gx_lsf_re, gx_lsf_im);
  stop_timer();
  printf("LSF Ariane took %d cycles.\n", (int)get_timer());
  start_timer();
  lsf_vec(K, N, M, P, h_re, h_im, s_re, s_im, w_lsf_re, w_lsf_im, x_lsf_re, x_lsf_im);
  stop_timer();
  printf("LSF Ariane+Ara took %d cycles.\n", (int)get_timer());
  if (verify_cplx(w_lsf_re, w_lsf_im, g_lsf_re, g_lsf_im, S * M, "W_lsf") ||
      verify_cplx(x_lsf_re, x_lsf_im, gx_lsf_re, gx_lsf_im, M, "x_lsf"))
    return 1;
  double beta[K];
  lsf_beta_gold(K, N, M, h_re, h_im, beta);
  start_timer();
  lsf_ll_gold(K, N, M, P, h_re, h_im, beta, s_re, s_im, w_lsf_re, w_lsf_im, x_lsf_re,
              x_lsf_im);
  stop_timer();
  printf("LSF Ariane low-latency took %d cycles.\n", (int)get_timer());
  if (verify_cplx(w_lsf_re, w_lsf_im, g_lsf_re, g_lsf_im, S * M, "W_lsf_ll") ||
      verify_cplx(x_lsf_re, x_lsf_im, gx_lsf_re, gx_lsf_im, M, "x_lsf_ll"))
    return 1;
  start_timer();
  lsf_ll_vec(K, N, M, P, h_re, h_im, beta, s_re, s_im, w_lsf_re, w_lsf_im, x_lsf_re,
             x_lsf_im);
  stop_timer();
  printf("LSF Ariane+Ara low-latency took %d cycles.\n", (int)get_timer());
  if (verify_cplx(w_lsf_re, w_lsf_im, g_lsf_re, g_lsf_im, S * M, "W_lsf_ll_vec") ||
      verify_cplx(x_lsf_re, x_lsf_im, gx_lsf_re, gx_lsf_im, M, "x_lsf_ll_vec"))
    return 1;
  if (check_power(S * M, P, g_lsf_re, g_lsf_im)) {
    printf("LSF power check failed.\n");
    return 1;
  }
  report_sinr("LSF", S, N, M, sigma_n2, h_re, h_im, g_lsf_re, g_lsf_im);
  printf("LSF passed.\n\n");

  printf("Passed.\n");
  return 0;
}
