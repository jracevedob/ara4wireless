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

#ifndef _PRECODING_LINALG_H_
#define _PRECODING_LINALG_H_

#include <stddef.h>

// aᴴ b = Σ conj(a) b. Real and imaginary parts are accumulated separately.
void hdot_gold(const double *ar, const double *ai, const double *br, const double *bi,
               size_t n, double *re, double *im);
void hdot_vec(const double *ar, const double *ai, const double *br, const double *bi,
              size_t n, double *re, double *im);

// Regularized zero-forcing. alpha = 0 is ZF. alpha = (stream count) σ² / P is MMSE.
// use_vec selects the Ara inner product. K is the number of streams, KN.
void rzf_precoder(size_t K, size_t M, double P, double alpha, int use_vec,
                  const double *h_re, const double *h_im, const double *s_re,
                  const double *s_im, double *w_re, double *w_im, double *x_re,
                  double *x_im);

// Same precoder as rzf_precoder. Only the upper triangle of the Gram is formed.
// The explicit inverse is replaced by a Cholesky factor and one triangular
// solve per transmit antenna.
void rzf_lowlat(size_t K, size_t M, double P, double alpha, int use_vec,
                const double *h_re, const double *h_im, const double *s_re,
                const double *s_im, double *w_re, double *w_im, double *x_re,
                double *x_im);

void apply_precoder(size_t K, size_t M, const double *w_re, const double *w_im,
                    const double *s_re, const double *s_im, double *x_re,
                    double *x_im);

// Same filters as rzf_precoder and rzf_lowlat. The Gram and the product
// Hᴴ A⁻¹ are formed in tile×tile blocks. Inner products run on Ara over
// antenna panels of width tile. The low-latency form solves the Cholesky
// system for a whole antenna panel in one vector pass.
void rzf_tiled(size_t K, size_t M, double P, double alpha, size_t tile,
               const double *h_re, const double *h_im, const double *s_re,
               const double *s_im, double *w_re, double *w_im, double *x_re,
               double *x_im);
void rzf_lowlat_tiled(size_t K, size_t M, double P, double alpha, size_t tile,
                      const double *h_re, const double *h_im, const double *s_re,
                      const double *s_im, double *w_re, double *w_im, double *x_re,
                      double *x_im);

#endif
