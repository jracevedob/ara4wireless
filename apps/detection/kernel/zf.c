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

// Zero-forcing uplink combining:
//   hat{x} = (Hᴴ H)⁻¹ Hᴴ y
// Baseline forms each Gram entry with a separate inner product. The
// low-latency kernels batch four products against a live column and reduce
// once per batch.

#include "detection.h"
#include "linalg.h"

void zf_gold(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
             const double *y_re, const double *y_im, double *x_re, double *x_im) {
  zf_combine(K * N, M, 0, h_re, h_im, y_re, y_im, x_re, x_im);
}

void zf_vec(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
            const double *y_re, const double *y_im, double *x_re, double *x_im) {
  zf_combine(K * N, M, 1, h_re, h_im, y_re, y_im, x_re, x_im);
}

void zf_ll_gold(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
                const double *y_re, const double *y_im, double *x_re, double *x_im) {
  zf_combine_ll(K * N, M, 0, h_re, h_im, y_re, y_im, x_re, x_im);
}

void zf_ll_vec(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
               const double *y_re, const double *y_im, double *x_re, double *x_im) {
  zf_combine_ll(K * N, M, 1, h_re, h_im, y_re, y_im, x_re, x_im);
}
