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

// Per-RE MMSE (Wiener) channel estimate for 5G NR DMRS:
//   ĥ[k] = y[k] * conj(x[k]) * σ_h² / (|x[k]|² σ_h² + σ_n²)
// Equivalent to shrinking the LS estimate by
//   w[k] = σ_h² / (σ_h² + σ_n² / |x[k]|²)
// when h ~ CN(0, σ_h²) is uncorrelated across REs (diagonal R_hh).

#include "chanest.h"

void mmse_gold(size_t n, const double *xp_re, const double *xp_im,
               const double *yp_re, const double *yp_im, double sigma_h2,
               double sigma_n2, double *h_re, double *h_im) {
  for (size_t k = 0; k < n; ++k) {
    double den = xp_re[k] * xp_re[k] + xp_im[k] * xp_im[k];
    double num_re = yp_re[k] * xp_re[k] + yp_im[k] * xp_im[k];
    double num_im = yp_im[k] * xp_re[k] - yp_re[k] * xp_im[k];
    double den_mmse = den * sigma_h2 + sigma_n2;
    h_re[k] = num_re * sigma_h2 / den_mmse;
    h_im[k] = num_im * sigma_h2 / den_mmse;
  }
}

void mmse_vec(size_t n, const double *xp_re, const double *xp_im,
              const double *yp_re, const double *yp_im, double sigma_h2,
              double sigma_n2, double *h_re, double *h_im) {
  unsigned int vl;
  double *xr = (double *)xp_re;
  double *xi = (double *)xp_im;
  double *yr = (double *)yp_re;
  double *yi = (double *)yp_im;
  double *hr = h_re;
  double *hi = h_im;

  asm volatile("vsetvli %[vl], %[n], e64, m4, ta, ma"
               : [vl] "=r"(vl)
               : [n] "r"(n));

#ifdef VCD_DUMP
  event_trigger = +1;
#endif

  for (unsigned int avl = n; avl > 0; avl -= vl) {
    asm volatile("vsetvli %[vl], %[avl], e64, m4, ta, ma"
                 : [vl] "=r"(vl)
                 : [avl] "r"(avl));
    asm volatile("vle64.v v0,  (%0)" ::"r"(xr));
    asm volatile("vle64.v v4,  (%0)" ::"r"(xi));
    asm volatile("vle64.v v8,  (%0)" ::"r"(yr));
    asm volatile("vle64.v v12, (%0)" ::"r"(yi));
    // den = |x|²
    asm volatile("vfmul.vv v16, v0, v0");
    asm volatile("vfmacc.vv v16, v4, v4");
    // Re(y conj(x)), Im(y conj(x))
    asm volatile("vfmul.vv v20, v8, v0");
    asm volatile("vfmacc.vv v20, v12, v4");
    asm volatile("vfmul.vv v24, v12, v0");
    asm volatile("vfnmsac.vv v24, v8, v4");
    // den_mmse = |x|² σ_h² + σ_n²
    asm volatile("vfmul.vf v16, v16, %0" ::"f"(sigma_h2));
    asm volatile("vfadd.vf v16, v16, %0" ::"f"(sigma_n2));
    // num * σ_h² / den_mmse
    asm volatile("vfmul.vf v20, v20, %0" ::"f"(sigma_h2));
    asm volatile("vfmul.vf v24, v24, %0" ::"f"(sigma_h2));
    asm volatile("vfdiv.vv v20, v20, v16");
    asm volatile("vfdiv.vv v24, v24, v16");
    asm volatile("vse64.v v20, (%0)" ::"r"(hr));
    asm volatile("vse64.v v24, (%0)" ::"r"(hi));
    xr += vl;
    xi += vl;
    yr += vl;
    yi += vl;
    hr += vl;
    hi += vl;
  }

#ifdef VCD_DUMP
  event_trigger = -1;
#endif
}
