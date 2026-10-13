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

// Downlink precoders. The base station has M transmit antennas. Each of the
// K users has N receive antennas. The stacked channel H is (KN) x M, stored
// row-major: receive antenna n of user k is row s = k*N+n, and h[s*M+m] is
// the coefficient to transmit antenna m. One stream is sent to each receive
// antenna, so W is M x (KN), stored by columns (w[s*M+m]). ||W||_F^2 = P.
//
//   ZF:   W ∝ Hᴴ (H Hᴴ)⁻¹
//   MMSE: W ∝ Hᴴ (H Hᴴ + α I)⁻¹,  α = (KN) σ² / P
//   LSF:  one large-scale coefficient per user, shared by its N antennas,
//           β_k = ||H_k||_F² / (MN)
//           p_k = P / (β_k Σ_j 1/β_j)
//           W_k = sqrt(p_k) H_kᴴ / ||H_k||_F
//
// The *_vec kernels form the length-M inner products on Ara. The (KN) x (KN)
// factorization stays on Ariane. Zero-forcing and dirty-paper coding need
// M >= KN and a full-rank stacked channel.

#ifndef _PRECODING_H_
#define _PRECODING_H_

#include <stddef.h>

void zf_gold(size_t K, size_t N, size_t M, double P, const double *h_re,
             const double *h_im, const double *s_re, const double *s_im, double *w_re,
             double *w_im, double *x_re, double *x_im);
void zf_vec(size_t K, size_t N, size_t M, double P, const double *h_re,
            const double *h_im, const double *s_re, const double *s_im, double *w_re,
            double *w_im, double *x_re, double *x_im);

// Same filter as zf_gold. Forms only the upper triangle of H Hᴴ and solves
// the Gram with its Cholesky factor instead of building the explicit inverse.
void zf_ll_gold(size_t K, size_t N, size_t M, double P, const double *h_re,
                const double *h_im, const double *s_re, const double *s_im, double *w_re,
                double *w_im, double *x_re, double *x_im);
void zf_ll_vec(size_t K, size_t N, size_t M, double P, const double *h_re,
               const double *h_im, const double *s_re, const double *s_im, double *w_re,
               double *w_im, double *x_re, double *x_im);
void zf_vec_tile(size_t K, size_t N, size_t M, double P, size_t tile, const double *h_re,
                 const double *h_im, const double *s_re, const double *s_im, double *w_re,
                 double *w_im, double *x_re, double *x_im);
void zf_ll_vec_tile(size_t K, size_t N, size_t M, double P, size_t tile, const double *h_re,
                    const double *h_im, const double *s_re, const double *s_im, double *w_re,
                    double *w_im, double *x_re, double *x_im);

void mmse_gold(size_t K, size_t N, size_t M, double P, double sigma_n2,
               const double *h_re, const double *h_im, const double *s_re,
               const double *s_im, double *w_re, double *w_im, double *x_re,
               double *x_im);
void mmse_vec(size_t K, size_t N, size_t M, double P, double sigma_n2,
              const double *h_re, const double *h_im, const double *s_re,
              const double *s_im, double *w_re, double *w_im, double *x_re,
              double *x_im);

// Same filter as mmse_gold, with the low-latency Gram and Cholesky solve.
void mmse_ll_gold(size_t K, size_t N, size_t M, double P, double sigma_n2,
                  const double *h_re, const double *h_im, const double *s_re,
                  const double *s_im, double *w_re, double *w_im, double *x_re,
                  double *x_im);
void mmse_ll_vec(size_t K, size_t N, size_t M, double P, double sigma_n2,
                 const double *h_re, const double *h_im, const double *s_re,
                 const double *s_im, double *w_re, double *w_im, double *x_re,
                 double *x_im);
void mmse_vec_tile(size_t K, size_t N, size_t M, double P, double sigma_n2, size_t tile,
                   const double *h_re, const double *h_im, const double *s_re,
                   const double *s_im, double *w_re, double *w_im, double *x_re,
                   double *x_im);
void mmse_ll_vec_tile(size_t K, size_t N, size_t M, double P, double sigma_n2, size_t tile,
                      const double *h_re, const double *h_im, const double *s_re,
                      const double *s_im, double *w_re, double *w_im, double *x_re,
                      double *x_im);

void lsf_gold(size_t K, size_t N, size_t M, double P, const double *h_re,
              const double *h_im, const double *s_re, const double *s_im, double *w_re,
              double *w_im, double *x_re, double *x_im);
void lsf_vec(size_t K, size_t N, size_t M, double P, const double *h_re,
             const double *h_im, const double *s_re, const double *s_im, double *w_re,
             double *w_im, double *x_re, double *x_im);

// Large-scale coefficients, one per user. This is the slow update: β changes
// much more slowly than H. beta has length K.
void lsf_beta_gold(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
                   double *beta);
void lsf_beta_vec(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
                  double *beta);

// Matched-filter refresh from the current H, using a previously computed beta.
// The N beams of user k are still normalized to power p_k(beta).
void lsf_ll_gold(size_t K, size_t N, size_t M, double P, const double *h_re,
                 const double *h_im, const double *beta, const double *s_re,
                 const double *s_im, double *w_re, double *w_im, double *x_re,
                 double *x_im);
void lsf_ll_vec(size_t K, size_t N, size_t M, double P, const double *h_re,
                const double *h_im, const double *beta, const double *s_re,
                const double *s_im, double *w_re, double *w_im, double *x_re,
                double *x_im);
void lsf_vec_tile(size_t K, size_t N, size_t M, double P, size_t tile, const double *h_re,
                  const double *h_im, const double *s_re, const double *s_im, double *w_re,
                  double *w_im, double *x_re, double *x_im);
void lsf_beta_vec_tile(size_t K, size_t N, size_t M, size_t tile, const double *h_re,
                       const double *h_im, double *beta);
void lsf_ll_vec_tile(size_t K, size_t N, size_t M, double P, size_t tile, const double *h_re,
                     const double *h_im, const double *beta, const double *s_re,
                     const double *s_im, double *w_re, double *w_im, double *x_re,
                     double *x_im);

#endif
