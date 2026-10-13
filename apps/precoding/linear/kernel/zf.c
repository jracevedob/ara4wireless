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

// Zero-forcing precoder, sum-power normalized:
//   W = sqrt(P / ||V||_F²) V,  V = Hᴴ (H Hᴴ)⁻¹
// H is (KN) x M. H W is a real scaled identity, so interference among all
// KN receive antennas is nulled.

#include "linalg.h"
#include "precoding.h"

void zf_gold(size_t K, size_t N, size_t M, double P, const double *h_re,
             const double *h_im, const double *s_re, const double *s_im, double *w_re,
             double *w_im, double *x_re, double *x_im) {
  rzf_precoder(K * N, M, P, 0.0, 0, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
}

void zf_vec(size_t K, size_t N, size_t M, double P, const double *h_re,
            const double *h_im, const double *s_re, const double *s_im, double *w_re,
            double *w_im, double *x_re, double *x_im) {
  rzf_precoder(K * N, M, P, 0.0, 1, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
}

void zf_ll_gold(size_t K, size_t N, size_t M, double P, const double *h_re,
                const double *h_im, const double *s_re, const double *s_im, double *w_re,
                double *w_im, double *x_re, double *x_im) {
  rzf_lowlat(K * N, M, P, 0.0, 0, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
}

void zf_ll_vec(size_t K, size_t N, size_t M, double P, const double *h_re,
               const double *h_im, const double *s_re, const double *s_im, double *w_re,
               double *w_im, double *x_re, double *x_im) {
  rzf_lowlat(K * N, M, P, 0.0, 1, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
}

void zf_vec_tile(size_t K, size_t N, size_t M, double P, size_t tile, const double *h_re,
                 const double *h_im, const double *s_re, const double *s_im, double *w_re,
                 double *w_im, double *x_re, double *x_im) {
  rzf_tiled(K * N, M, P, 0.0, tile, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
}

void zf_ll_vec_tile(size_t K, size_t N, size_t M, double P, size_t tile, const double *h_re,
                    const double *h_im, const double *s_re, const double *s_im, double *w_re,
                    double *w_im, double *x_re, double *x_im) {
  rzf_lowlat_tiled(K * N, M, P, 0.0, tile, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
}
