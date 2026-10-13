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

// Zero-forcing dirty-paper coding for the MIMO broadcast channel
// (Caire, Shamai), with Tomlinson-Harashima modulo on QPSK.
//
//   Hᴴ = Q R     (thin QR, modified Gram-Schmidt)
//   y  = H Q u + n = Rᴴ u + n
//
// KN streams, one per receive antenna, in order (user, antenna). Rᴴ is lower
// triangular in that order, so stream i sees interference only from streams
// already encoded (j < i). The transmitter cancels it:
//   u_i = s_i - Σ_{j<i} (Rᴴ)_ij / R_ii · u_j
// and folds u_i back into the QPSK fundamental region with modulo τ = 2√2.
// The transmit vector is x = sqrt(P / ||Q u||²) Q u, so ||x||² = P.
//
// q holds the orthonormal columns of Q (q[s * M + m], s = k*N+n). rdiag
// holds the real R_ii gains, one per receive antenna.

#ifndef _DPC_H_
#define _DPC_H_

#include <stddef.h>

void dpc_gold(size_t K, size_t N, size_t M, double P, const double *h_re,
              const double *h_im, const double *s_re, const double *s_im, double *q_re,
              double *q_im, double *rdiag, double *u_re, double *u_im, double *x_re,
              double *x_im);
void dpc_vec(size_t K, size_t N, size_t M, double P, const double *h_re,
             const double *h_im, const double *s_re, const double *s_im, double *q_re,
             double *q_im, double *rdiag, double *u_re, double *u_im, double *x_re,
             double *x_im);

// Same dirty-paper coding without the QR. The Cholesky factor of H Hᴴ supplies
// the Tomlinson-Harashima coefficients and the spatial filter. Q is not formed.
// rdiag holds the real Cholesky diagonal, which equals the R_ii gains.
void dpc_ll_gold(size_t K, size_t N, size_t M, double P, const double *h_re,
                 const double *h_im, const double *s_re, const double *s_im,
                 double *rdiag, double *u_re, double *u_im, double *x_re, double *x_im);
void dpc_ll_vec(size_t K, size_t N, size_t M, double P, const double *h_re,
                const double *h_im, const double *s_re, const double *s_im, double *rdiag,
                double *u_re, double *u_im, double *x_re, double *x_im);
void dpc_vec_tile(size_t K, size_t N, size_t M, double P, size_t tile, const double *h_re,
                  const double *h_im, const double *s_re, const double *s_im, double *q_re,
                  double *q_im, double *rdiag, double *u_re, double *u_im, double *x_re,
                  double *x_im);
void dpc_ll_vec_tile(size_t K, size_t N, size_t M, double P, size_t tile, const double *h_re,
                     const double *h_im, const double *s_re, const double *s_im,
                     double *rdiag, double *u_re, double *u_im, double *x_re, double *x_im);

#endif
