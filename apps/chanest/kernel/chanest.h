// Copyright 2026 ETH Zurich and University of Bologna.
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

// Per-subcarrier 5G NR DMRS channel estimation.
// Complex I/Q samples are stored as separate real/imag double arrays.

#ifndef _CHANEST_H_
#define _CHANEST_H_

#include <stddef.h>
#include <stdint.h>

#include "runtime.h"

// Least-squares estimator (LSE): ĥ = y * conj(x) / |x|²
void lse_gold(size_t n, const double *xp_re, const double *xp_im,
              const double *yp_re, const double *yp_im, double *h_re,
              double *h_im);
void lse_vec(size_t n, const double *xp_re, const double *xp_im,
             const double *yp_re, const double *yp_im, double *h_re,
             double *h_im);

// Per-RE MMSE (Wiener): ĥ = y * conj(x) * σ_h² / (|x|² σ_h² + σ_n²)
void mmse_gold(size_t n, const double *xp_re, const double *xp_im,
               const double *yp_re, const double *yp_im, double sigma_h2,
               double sigma_n2, double *h_re, double *h_im);
void mmse_vec(size_t n, const double *xp_re, const double *xp_im,
              const double *yp_re, const double *yp_im, double sigma_h2,
              double sigma_n2, double *h_re, double *h_im);

#endif
