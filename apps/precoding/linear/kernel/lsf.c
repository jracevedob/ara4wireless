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

// Large-scale fading precoding. User k has channel H_k (N x M). All N
// antennas share one large-scale coefficient
//   β_k = ||H_k||_F² / (M N)
// Power is split across users, then the user's precoder is the matched
// filter of the whole block, with ||W_k||_F² = p_k:
//   p_k = P / (β_k Σ_j 1/β_j),  W_k = sqrt(p_k) H_kᴴ / ||H_k||_F

#include "linalg.h"
#include "precoding.h"

#include <math.h>

static void scale_row(const double *h_re, const double *h_im, double *w_re, double *w_im,
                      size_t M, double gain, int use_vec) {
  if (use_vec) {
#if !defined(__riscv)
    for (size_t m = 0; m < M; ++m) {
      w_re[m] = gain * h_re[m];
      w_im[m] = -gain * h_im[m];
    }
#else
    size_t vl = 0;
    const double *hr = h_re;
    const double *hi = h_im;
    double *wr = w_re;
    double *wi = w_im;
    double neg = -gain;
    for (size_t avl = M; avl > 0; avl -= vl) {
      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vle64.v v0, (%0)" ::"r"(hr));
      asm volatile("vle64.v v4, (%0)" ::"r"(hi));
      asm volatile("vfmul.vf v0, v0, %0" ::"f"(gain));
      asm volatile("vfmul.vf v4, v4, %0" ::"f"(neg));
      asm volatile("vse64.v v0, (%0)" ::"r"(wr));
      asm volatile("vse64.v v4, (%0)" ::"r"(wi));
      hr += vl;
      hi += vl;
      wr += vl;
      wi += vl;
    }
#endif
  } else {
    for (size_t m = 0; m < M; ++m) {
      w_re[m] = gain * h_re[m];
      w_im[m] = -gain * h_im[m];
    }
  }
}

static size_t tile_step(size_t n, size_t tile) { return (tile == 0 || tile > n) ? n : tile; }

static void row_energy(const double *hr, const double *hi, size_t M, size_t tile, int use_vec,
                       double *re) {
  double im;
  if (tile == 0 || tile >= M) {
    if (use_vec)
      hdot_vec(hr, hi, hr, hi, M, re, &im);
    else
      hdot_gold(hr, hi, hr, hi, M, re, &im);
    return;
  }
  double sr = 0.0;
  size_t step = tile_step(M, tile);
  for (size_t m = 0; m < M; m += step) {
    size_t len = M - m < step ? M - m : step;
    double pr;
    hdot_vec(hr + m, hi + m, hr + m, hi + m, len, &pr, &im);
    sr += pr;
  }
  *re = sr;
}

static void scale_panels(const double *hr, const double *hi, double *wr, double *wi, size_t M,
                         size_t tile, double gain, int use_vec) {
  size_t step = tile_step(M, tile);
  for (size_t m = 0; m < M; m += step) {
    size_t len = M - m < step ? M - m : step;
    scale_row(hr + m, hi + m, wr + m, wi + m, len, gain, use_vec);
  }
}

static void lsf_beta(size_t K, size_t N, size_t M, int use_vec, size_t tile, const double *h_re,
                     const double *h_im, double *beta) {
  size_t users = (tile == 0 || tile < N) ? K : tile / N;
  for (size_t k0 = 0; k0 < K; k0 += users) {
    size_t nk = K - k0 < users ? K - k0 : users;
    for (size_t kk = 0; kk < nk; ++kk) {
      size_t k = k0 + kk;
      double pwr = 0.0;
      for (size_t n = 0; n < N; ++n) {
        size_t row = (k * N + n) * M;
        double re;
        row_energy(&h_re[row], &h_im[row], M, tile, use_vec, &re);
        pwr += re;
      }
      beta[k] = pwr / (double)(M * N);
    }
  }
}

static void lsf_from_beta(size_t K, size_t N, size_t M, double P, int use_vec, size_t tile,
                          const double *h_re, const double *h_im, const double *beta,
                          const double *s_re, const double *s_im, double *w_re,
                          double *w_im, double *x_re, double *x_im) {
  double inv_beta_sum = 0.0;
  size_t users = (tile == 0 || tile < N) ? K : tile / N;
  for (size_t k = 0; k < K; ++k)
    inv_beta_sum += 1.0 / beta[k];

  for (size_t k0 = 0; k0 < K; k0 += users) {
    size_t nk = K - k0 < users ? K - k0 : users;
    for (size_t kk = 0; kk < nk; ++kk) {
      size_t k = k0 + kk;
      double pwr = 0.0;
      for (size_t n = 0; n < N; ++n) {
        size_t row = (k * N + n) * M;
        double re;
        row_energy(&h_re[row], &h_im[row], M, tile, use_vec, &re);
        pwr += re;
      }
      double pk = P / (beta[k] * inv_beta_sum);
      double gain = sqrt(pk) / sqrt(pwr);
      for (size_t n = 0; n < N; ++n) {
        size_t row = (k * N + n) * M;
        scale_panels(&h_re[row], &h_im[row], &w_re[row], &w_im[row], M, tile, gain, use_vec);
      }
    }
  }
  apply_precoder(K * N, M, w_re, w_im, s_re, s_im, x_re, x_im);
}

void lsf_beta_gold(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
                   double *beta) {
  lsf_beta(K, N, M, 0, 0, h_re, h_im, beta);
}

void lsf_beta_vec(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
                  double *beta) {
  lsf_beta(K, N, M, 1, 0, h_re, h_im, beta);
}

void lsf_beta_vec_tile(size_t K, size_t N, size_t M, size_t tile, const double *h_re,
                       const double *h_im, double *beta) {
  lsf_beta(K, N, M, 1, tile, h_re, h_im, beta);
}

void lsf_ll_gold(size_t K, size_t N, size_t M, double P, const double *h_re,
                 const double *h_im, const double *beta, const double *s_re,
                 const double *s_im, double *w_re, double *w_im, double *x_re,
                 double *x_im) {
  lsf_from_beta(K, N, M, P, 0, 0, h_re, h_im, beta, s_re, s_im, w_re, w_im, x_re, x_im);
}

void lsf_ll_vec(size_t K, size_t N, size_t M, double P, const double *h_re,
                const double *h_im, const double *beta, const double *s_re,
                const double *s_im, double *w_re, double *w_im, double *x_re,
                double *x_im) {
  lsf_from_beta(K, N, M, P, 1, 0, h_re, h_im, beta, s_re, s_im, w_re, w_im, x_re, x_im);
}

void lsf_ll_vec_tile(size_t K, size_t N, size_t M, double P, size_t tile, const double *h_re,
                     const double *h_im, const double *beta, const double *s_re,
                     const double *s_im, double *w_re, double *w_im, double *x_re,
                     double *x_im) {
  lsf_from_beta(K, N, M, P, 1, tile, h_re, h_im, beta, s_re, s_im, w_re, w_im, x_re, x_im);
}

static void lsf_run(size_t K, size_t N, size_t M, double P, int use_vec, size_t tile,
                    const double *h_re, const double *h_im, const double *s_re,
                    const double *s_im, double *w_re, double *w_im, double *x_re,
                    double *x_im) {
  double beta[K];
  double nrm[K];
  double inv_beta_sum = 0.0;
  size_t users = (tile == 0 || tile < N) ? K : tile / N;

  for (size_t k0 = 0; k0 < K; k0 += users) {
    size_t nk = K - k0 < users ? K - k0 : users;
    for (size_t kk = 0; kk < nk; ++kk) {
      size_t k = k0 + kk;
      double pwr = 0.0;
      for (size_t n = 0; n < N; ++n) {
        size_t row = (k * N + n) * M;
        double re;
        row_energy(&h_re[row], &h_im[row], M, tile, use_vec, &re);
        pwr += re;
      }
      beta[k] = pwr / (double)(M * N);
      nrm[k] = sqrt(pwr);
      inv_beta_sum += 1.0 / beta[k];
    }
  }
  for (size_t k0 = 0; k0 < K; k0 += users) {
    size_t nk = K - k0 < users ? K - k0 : users;
    for (size_t kk = 0; kk < nk; ++kk) {
      size_t k = k0 + kk;
      double pk = P / (beta[k] * inv_beta_sum);
      double gain = sqrt(pk) / nrm[k];
      for (size_t n = 0; n < N; ++n) {
        size_t row = (k * N + n) * M;
        if (tile == 0)
          scale_row(&h_re[row], &h_im[row], &w_re[row], &w_im[row], M, gain, use_vec);
        else
          scale_panels(&h_re[row], &h_im[row], &w_re[row], &w_im[row], M, tile, gain, use_vec);
      }
    }
  }
  apply_precoder(K * N, M, w_re, w_im, s_re, s_im, x_re, x_im);
}

void lsf_gold(size_t K, size_t N, size_t M, double P, const double *h_re, const double *h_im,
              const double *s_re, const double *s_im, double *w_re, double *w_im,
              double *x_re, double *x_im) {
  lsf_run(K, N, M, P, 0, 0, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
}

void lsf_vec(size_t K, size_t N, size_t M, double P, const double *h_re, const double *h_im,
             const double *s_re, const double *s_im, double *w_re, double *w_im,
             double *x_re, double *x_im) {
  lsf_run(K, N, M, P, 1, 0, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
}

void lsf_vec_tile(size_t K, size_t N, size_t M, double P, size_t tile, const double *h_re,
                  const double *h_im, const double *s_re, const double *s_im, double *w_re,
                  double *w_im, double *x_re, double *x_im) {
  lsf_run(K, N, M, P, 1, tile, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
}
