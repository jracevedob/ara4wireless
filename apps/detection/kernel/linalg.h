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

#ifndef _DETECTION_LINALG_H_
#define _DETECTION_LINALG_H_

#include <stddef.h>

void hdot_gold(const double *ar, const double *ai, const double *br, const double *bi,
               size_t n, double *re, double *im);
void hdot_vec(const double *ar, const double *ai, const double *br, const double *bi,
              size_t n, double *re, double *im);

void vcopy(double *dst, const double *src, size_t n, int use_vec);
void vscale(double *re, double *im, double gain, size_t n, int use_vec);
// y += (h_re + j h_im) (xr + j xi)
void caxpy(double *y_re, double *y_im, const double *h_re, const double *h_im, double xr,
           double xi, size_t n, int use_vec);

// Cholesky factor of Hᴴ H. Reused across PIC iterations for a fixed H.
void zf_factor(size_t S, size_t M, int use_vec, const double *h_re, const double *h_im,
               double *l_re, double *l_im);
// hat{x} from a cached factor: b = Hᴴ y, then L Lᴴ hat{x} = b.
void zf_apply(size_t S, size_t M, int use_vec, const double *h_re, const double *h_im,
              const double *l_re, const double *l_im, const double *y_re,
              const double *y_im, double *x_re, double *x_im);

// ZF combining on S contiguous columns of H: hat{x} = (Hᴴ H)⁻¹ Hᴴ y.
void zf_combine(size_t S, size_t M, int use_vec, const double *h_re, const double *h_im,
                const double *y_re, const double *y_im, double *x_re, double *x_im);
// Same filter. Batched inner products, one reduction per batch, 1/L_ii in the
// triangular solve.
void zf_combine_ll(size_t S, size_t M, int use_vec, const double *h_re, const double *h_im,
                   const double *y_re, const double *y_im, double *x_re, double *x_im);

// Thin QR, H = Q R. Baseline copies H into a workspace; the low-latency form
// orthogonalizes in place and fuses ỹ = Qᴴ y.
void thin_qr(size_t S, size_t M, int use_vec, const double *h_re, const double *h_im,
             double *q_re, double *q_im, double *r_re, double *r_im);
void thin_qr_ll(size_t S, size_t M, int use_vec, const double *h_re, const double *h_im,
                const double *y_re, const double *y_im, double *q_re, double *q_im,
                double *r_re, double *r_im, double *yt_re, double *yt_im);

void slice_qpsk(const double *re, const double *im, double *out_re, double *out_im,
                size_t n);

#endif
