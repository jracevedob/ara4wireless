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

// Uplink multi-user MIMO detection. The base station has M receive antennas.
// Each of the K users has N transmit antennas. The stacked channel H is
// M x (KN), stored by columns: transmit antenna n of user k is column
// s = k*N+n, and h[s*M+m] is the coefficient to receive antenna m. The
// received vector is y = H x + n, with x in C^{KN}. Requires M >= KN.
//
//   ZF:  hat{x} = (Hᴴ H)⁻¹ Hᴴ y. Cholesky; the inverse is not formed.
//        The *_ll kernels batch Hᴴ H and Hᴴ y.
//   PIC: ZF init, QPSK slice, per-user local ZF. *_ll caches H_kᴴ H_k and
//        forms r = y - H x̃ once per iteration.
//   SD:  thin QR and Schnorr–Euchner search of ||ỹ - R x||². *_ll fuses
//        ỹ = Qᴴ y and updates the residual incrementally.

#ifndef _DETECTION_H_
#define _DETECTION_H_

#include <stddef.h>

void zf_gold(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
             const double *y_re, const double *y_im, double *x_re, double *x_im);
void zf_vec(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
            const double *y_re, const double *y_im, double *x_re, double *x_im);
void zf_ll_gold(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
                const double *y_re, const double *y_im, double *x_re, double *x_im);
void zf_ll_vec(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
               const double *y_re, const double *y_im, double *x_re, double *x_im);

void pic_gold(size_t K, size_t N, size_t M, size_t niter, const double *h_re,
              const double *h_im, const double *y_re, const double *y_im, double *x_re,
              double *x_im);
void pic_vec(size_t K, size_t N, size_t M, size_t niter, const double *h_re,
             const double *h_im, const double *y_re, const double *y_im, double *x_re,
             double *x_im);
void pic_ll_gold(size_t K, size_t N, size_t M, size_t niter, const double *h_re,
                 const double *h_im, const double *y_re, const double *y_im, double *x_re,
                 double *x_im);
void pic_ll_vec(size_t K, size_t N, size_t M, size_t niter, const double *h_re,
                const double *h_im, const double *y_re, const double *y_im, double *x_re,
                double *x_im);

void sd_gold(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
             const double *y_re, const double *y_im, double *x_re, double *x_im);
void sd_vec(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
            const double *y_re, const double *y_im, double *x_re, double *x_im);
void sd_ll_gold(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
                const double *y_re, const double *y_im, double *x_re, double *x_im);
void sd_ll_vec(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
               const double *y_re, const double *y_im, double *x_re, double *x_im);

#endif
