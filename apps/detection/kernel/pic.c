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

// Parallel interference cancellation. A joint ZF detector initializes all
// streams. Each iteration slices the previous estimates onto QPSK, cancels
// every other user from y, and re-detects user k with a local ZF filter.
// The low-latency kernels factor each H_kᴴ H_k once and form a single
// residual r = y - H x̃ per iteration.

#include "detection.h"
#include "linalg.h"

static void pic_run(size_t K, size_t N, size_t M, size_t niter, int use_vec,
                    const double *h_re, const double *h_im, const double *y_re,
                    const double *y_im, double *x_re, double *x_im) {
  size_t S = K * N;
  double prev_re[S];
  double prev_im[S];
  double sl_re[S];
  double sl_im[S];
  double yk_re[M];
  double yk_im[M];
  double xk_re[N];
  double xk_im[N];

  zf_combine(S, M, use_vec, h_re, h_im, y_re, y_im, x_re, x_im);
  if (niter == 0)
    return;

  for (size_t t = 0; t < niter; ++t) {
    vcopy(prev_re, x_re, S, 0);
    vcopy(prev_im, x_im, S, 0);
    slice_qpsk(prev_re, prev_im, sl_re, sl_im, S);

    for (size_t k = 0; k < K; ++k) {
      vcopy(yk_re, y_re, M, use_vec);
      vcopy(yk_im, y_im, M, use_vec);
      for (size_t j = 0; j < K; ++j) {
        if (j == k)
          continue;
        for (size_t n = 0; n < N; ++n) {
          size_t s = j * N + n;
          caxpy(yk_re, yk_im, &h_re[s * M], &h_im[s * M], -sl_re[s], -sl_im[s], M,
                use_vec);
        }
      }
      zf_combine(N, M, use_vec, &h_re[k * N * M], &h_im[k * N * M], yk_re, yk_im, xk_re,
                 xk_im);
      for (size_t n = 0; n < N; ++n) {
        x_re[k * N + n] = xk_re[n];
        x_im[k * N + n] = xk_im[n];
      }
    }
  }
}

static void pic_run_ll(size_t K, size_t N, size_t M, size_t niter, int use_vec,
                       const double *h_re, const double *h_im, const double *y_re,
                       const double *y_im, double *x_re, double *x_im) {
  size_t S = K * N;
  double sl_re[S];
  double sl_im[S];
  double r_re[M];
  double r_im[M];
  double yk_re[M];
  double yk_im[M];
  double xk_re[N];
  double xk_im[N];
  double lk_re[K * N * N];
  double lk_im[K * N * N];

  zf_combine_ll(S, M, use_vec, h_re, h_im, y_re, y_im, x_re, x_im);
  if (niter == 0)
    return;

  for (size_t k = 0; k < K; ++k)
    zf_factor(N, M, use_vec, &h_re[k * N * M], &h_im[k * N * M], &lk_re[k * N * N],
              &lk_im[k * N * N]);

  for (size_t t = 0; t < niter; ++t) {
    slice_qpsk(x_re, x_im, sl_re, sl_im, S);

    vcopy(r_re, y_re, M, use_vec);
    vcopy(r_im, y_im, M, use_vec);
    for (size_t s = 0; s < S; ++s)
      caxpy(r_re, r_im, &h_re[s * M], &h_im[s * M], -sl_re[s], -sl_im[s], M, use_vec);

    for (size_t k = 0; k < K; ++k) {
      vcopy(yk_re, r_re, M, use_vec);
      vcopy(yk_im, r_im, M, use_vec);
      for (size_t n = 0; n < N; ++n) {
        size_t s = k * N + n;
        caxpy(yk_re, yk_im, &h_re[s * M], &h_im[s * M], sl_re[s], sl_im[s], M, use_vec);
      }
      zf_apply(N, M, use_vec, &h_re[k * N * M], &h_im[k * N * M], &lk_re[k * N * N],
               &lk_im[k * N * N], yk_re, yk_im, xk_re, xk_im);
      for (size_t n = 0; n < N; ++n) {
        x_re[k * N + n] = xk_re[n];
        x_im[k * N + n] = xk_im[n];
      }
    }
  }
}

void pic_gold(size_t K, size_t N, size_t M, size_t niter, const double *h_re,
              const double *h_im, const double *y_re, const double *y_im, double *x_re,
              double *x_im) {
  pic_run(K, N, M, niter, 0, h_re, h_im, y_re, y_im, x_re, x_im);
}

void pic_vec(size_t K, size_t N, size_t M, size_t niter, const double *h_re,
             const double *h_im, const double *y_re, const double *y_im, double *x_re,
             double *x_im) {
  pic_run(K, N, M, niter, 1, h_re, h_im, y_re, y_im, x_re, x_im);
}

void pic_ll_gold(size_t K, size_t N, size_t M, size_t niter, const double *h_re,
                 const double *h_im, const double *y_re, const double *y_im, double *x_re,
                 double *x_im) {
  pic_run_ll(K, N, M, niter, 0, h_re, h_im, y_re, y_im, x_re, x_im);
}

void pic_ll_vec(size_t K, size_t N, size_t M, size_t niter, const double *h_re,
                const double *h_im, const double *y_re, const double *y_im, double *x_re,
                double *x_im) {
  pic_run_ll(K, N, M, niter, 1, h_re, h_im, y_re, y_im, x_re, x_im);
}
