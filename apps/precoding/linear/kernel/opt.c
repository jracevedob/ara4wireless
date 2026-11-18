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

// Optimization of the Ara precoders. The arithmetic matches zf_vec, mmse_vec,
// and lsf_vec. Three round-trips are gone:
//   * each Gram entry is reduced once, and only the lower triangle is formed
//   * an antenna strip of W is accumulated in registers and stored once
//   * the Frobenius norm and the power scale are vector instructions

#include "precoding.h"

#include <math.h>

#if defined(__riscv)

// Up to eight streams live in v0..v15 as (real, imag) pairs.
#define NMAC_ASM(DR, DI, SR, SI)                                      \
  asm volatile("vfnmsac.vf v" #DR ", %0, v" #SR ::"f"(lr));           \
  asm volatile("vfmacc.vf v" #DR ", %0, v" #SI ::"f"(li));            \
  asm volatile("vfnmsac.vf v" #DI ", %0, v" #SI ::"f"(lr));           \
  asm volatile("vfnmsac.vf v" #DI ", %0, v" #SR ::"f"(li))

#define NMAC_FN(d, dr, di)                                            \
  static void nmac_##d(int src, double lr, double li) {              \
    switch (src) {                                                    \
      case 0: NMAC_ASM(dr, di, 0, 1); break;                          \
      case 1: NMAC_ASM(dr, di, 2, 3); break;                          \
      case 2: NMAC_ASM(dr, di, 4, 5); break;                          \
      case 3: NMAC_ASM(dr, di, 6, 7); break;                          \
      case 4: NMAC_ASM(dr, di, 8, 9); break;                          \
      case 5: NMAC_ASM(dr, di, 10, 11); break;                        \
      case 6: NMAC_ASM(dr, di, 12, 13); break;                        \
      case 7: NMAC_ASM(dr, di, 14, 15); break;                        \
      default: break;                                                 \
    }                                                                 \
  }

NMAC_FN(0, 0, 1)
NMAC_FN(1, 2, 3)
NMAC_FN(2, 4, 5)
NMAC_FN(3, 6, 7)
NMAC_FN(4, 8, 9)
NMAC_FN(5, 10, 11)
NMAC_FN(6, 12, 13)
NMAC_FN(7, 14, 15)

static void nmac(int dst, int src, double lr, double li) {
  switch (dst) {
    case 0: nmac_0(src, lr, li); break;
    case 1: nmac_1(src, lr, li); break;
    case 2: nmac_2(src, lr, li); break;
    case 3: nmac_3(src, lr, li); break;
    case 4: nmac_4(src, lr, li); break;
    case 5: nmac_5(src, lr, li); break;
    case 6: nmac_6(src, lr, li); break;
    default: nmac_7(src, lr, li); break;
  }
}

static void load_row(int row, const double *hr, const double *hi) {
  switch (row) {
    case 0:
      asm volatile("vle64.v v0, (%0)" ::"r"(hr));
      asm volatile("vle64.v v1, (%0)" ::"r"(hi));
      break;
    case 1:
      asm volatile("vle64.v v2, (%0)" ::"r"(hr));
      asm volatile("vle64.v v3, (%0)" ::"r"(hi));
      break;
    case 2:
      asm volatile("vle64.v v4, (%0)" ::"r"(hr));
      asm volatile("vle64.v v5, (%0)" ::"r"(hi));
      break;
    case 3:
      asm volatile("vle64.v v6, (%0)" ::"r"(hr));
      asm volatile("vle64.v v7, (%0)" ::"r"(hi));
      break;
    case 4:
      asm volatile("vle64.v v8, (%0)" ::"r"(hr));
      asm volatile("vle64.v v9, (%0)" ::"r"(hi));
      break;
    case 5:
      asm volatile("vle64.v v10, (%0)" ::"r"(hr));
      asm volatile("vle64.v v11, (%0)" ::"r"(hi));
      break;
    case 6:
      asm volatile("vle64.v v12, (%0)" ::"r"(hr));
      asm volatile("vle64.v v13, (%0)" ::"r"(hi));
      break;
    default:
      asm volatile("vle64.v v14, (%0)" ::"r"(hr));
      asm volatile("vle64.v v15, (%0)" ::"r"(hi));
      break;
  }
}

static void scale_row(int row, double gain) {
  switch (row) {
    case 0:
      asm volatile("vfmul.vf v0, v0, %0" ::"f"(gain));
      asm volatile("vfmul.vf v1, v1, %0" ::"f"(gain));
      break;
    case 1:
      asm volatile("vfmul.vf v2, v2, %0" ::"f"(gain));
      asm volatile("vfmul.vf v3, v3, %0" ::"f"(gain));
      break;
    case 2:
      asm volatile("vfmul.vf v4, v4, %0" ::"f"(gain));
      asm volatile("vfmul.vf v5, v5, %0" ::"f"(gain));
      break;
    case 3:
      asm volatile("vfmul.vf v6, v6, %0" ::"f"(gain));
      asm volatile("vfmul.vf v7, v7, %0" ::"f"(gain));
      break;
    case 4:
      asm volatile("vfmul.vf v8, v8, %0" ::"f"(gain));
      asm volatile("vfmul.vf v9, v9, %0" ::"f"(gain));
      break;
    case 5:
      asm volatile("vfmul.vf v10, v10, %0" ::"f"(gain));
      asm volatile("vfmul.vf v11, v11, %0" ::"f"(gain));
      break;
    case 6:
      asm volatile("vfmul.vf v12, v12, %0" ::"f"(gain));
      asm volatile("vfmul.vf v13, v13, %0" ::"f"(gain));
      break;
    default:
      asm volatile("vfmul.vf v14, v14, %0" ::"f"(gain));
      asm volatile("vfmul.vf v15, v15, %0" ::"f"(gain));
      break;
  }
}

static void store_conj(int row, double *wr, double *wi) {
  double neg = -1.0;
  switch (row) {
    case 0:
      asm volatile("vfmul.vf v16, v1, %0" ::"f"(neg));
      asm volatile("vse64.v v0, (%0)" ::"r"(wr) : "memory");
      asm volatile("vse64.v v16, (%0)" ::"r"(wi) : "memory");
      break;
    case 1:
      asm volatile("vfmul.vf v16, v3, %0" ::"f"(neg));
      asm volatile("vse64.v v2, (%0)" ::"r"(wr) : "memory");
      asm volatile("vse64.v v16, (%0)" ::"r"(wi) : "memory");
      break;
    case 2:
      asm volatile("vfmul.vf v16, v5, %0" ::"f"(neg));
      asm volatile("vse64.v v4, (%0)" ::"r"(wr) : "memory");
      asm volatile("vse64.v v16, (%0)" ::"r"(wi) : "memory");
      break;
    case 3:
      asm volatile("vfmul.vf v16, v7, %0" ::"f"(neg));
      asm volatile("vse64.v v6, (%0)" ::"r"(wr) : "memory");
      asm volatile("vse64.v v16, (%0)" ::"r"(wi) : "memory");
      break;
    case 4:
      asm volatile("vfmul.vf v16, v9, %0" ::"f"(neg));
      asm volatile("vse64.v v8, (%0)" ::"r"(wr) : "memory");
      asm volatile("vse64.v v16, (%0)" ::"r"(wi) : "memory");
      break;
    case 5:
      asm volatile("vfmul.vf v16, v11, %0" ::"f"(neg));
      asm volatile("vse64.v v10, (%0)" ::"r"(wr) : "memory");
      asm volatile("vse64.v v16, (%0)" ::"r"(wi) : "memory");
      break;
    case 6:
      asm volatile("vfmul.vf v16, v13, %0" ::"f"(neg));
      asm volatile("vse64.v v12, (%0)" ::"r"(wr) : "memory");
      asm volatile("vse64.v v16, (%0)" ::"r"(wi) : "memory");
      break;
    default:
      asm volatile("vfmul.vf v16, v15, %0" ::"f"(neg));
      asm volatile("vse64.v v14, (%0)" ::"r"(wr) : "memory");
      asm volatile("vse64.v v16, (%0)" ::"r"(wi) : "memory");
      break;
  }
}

static void accum_energy(int row, int first) {
  if (first) {
    switch (row) {
      case 0:
        asm volatile("vfmul.vv v17, v0, v0");
        asm volatile("vfmacc.vv v17, v1, v1");
        break;
      case 1:
        asm volatile("vfmul.vv v17, v2, v2");
        asm volatile("vfmacc.vv v17, v3, v3");
        break;
      case 2:
        asm volatile("vfmul.vv v17, v4, v4");
        asm volatile("vfmacc.vv v17, v5, v5");
        break;
      case 3:
        asm volatile("vfmul.vv v17, v6, v6");
        asm volatile("vfmacc.vv v17, v7, v7");
        break;
      case 4:
        asm volatile("vfmul.vv v17, v8, v8");
        asm volatile("vfmacc.vv v17, v9, v9");
        break;
      case 5:
        asm volatile("vfmul.vv v17, v10, v10");
        asm volatile("vfmacc.vv v17, v11, v11");
        break;
      case 6:
        asm volatile("vfmul.vv v17, v12, v12");
        asm volatile("vfmacc.vv v17, v13, v13");
        break;
      default:
        asm volatile("vfmul.vv v17, v14, v14");
        asm volatile("vfmacc.vv v17, v15, v15");
        break;
    }
    return;
  }
  switch (row) {
    case 0:
      asm volatile("vfmacc.vv v17, v0, v0");
      asm volatile("vfmacc.vv v17, v1, v1");
      break;
    case 1:
      asm volatile("vfmacc.vv v17, v2, v2");
      asm volatile("vfmacc.vv v17, v3, v3");
      break;
    case 2:
      asm volatile("vfmacc.vv v17, v4, v4");
      asm volatile("vfmacc.vv v17, v5, v5");
      break;
    case 3:
      asm volatile("vfmacc.vv v17, v6, v6");
      asm volatile("vfmacc.vv v17, v7, v7");
      break;
    case 4:
      asm volatile("vfmacc.vv v17, v8, v8");
      asm volatile("vfmacc.vv v17, v9, v9");
      break;
    case 5:
      asm volatile("vfmacc.vv v17, v10, v10");
      asm volatile("vfmacc.vv v17, v11, v11");
      break;
    case 6:
      asm volatile("vfmacc.vv v17, v12, v12");
      asm volatile("vfmacc.vv v17, v13, v13");
      break;
    default:
      asm volatile("vfmacc.vv v17, v14, v14");
      asm volatile("vfmacc.vv v17, v15, v15");
      break;
  }
}

static void herm_chol(size_t K, const double *a_re, const double *a_im, double *l_re,
                      double *l_im) {
  for (size_t n = 0; n < K * K; ++n) {
    l_re[n] = 0.0;
    l_im[n] = 0.0;
  }
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < i; ++j) {
      double sr = a_re[i * K + j];
      double si = a_im[i * K + j];
      for (size_t p = 0; p < j; ++p) {
        double lr = l_re[i * K + p];
        double li = l_im[i * K + p];
        double rr = l_re[j * K + p];
        double ri = -l_im[j * K + p];
        sr -= lr * rr - li * ri;
        si -= lr * ri + li * rr;
      }
      double diag = l_re[j * K + j];
      l_re[i * K + j] = sr / diag;
      l_im[i * K + j] = si / diag;
    }
    double diag = a_re[i * K + i];
    for (size_t p = 0; p < i; ++p)
      diag -= l_re[i * K + p] * l_re[i * K + p] + l_im[i * K + p] * l_im[i * K + p];
    l_re[i * K + i] = sqrt(diag);
    l_im[i * K + i] = 0.0;
  }
}

static void herm_inv(size_t K, const double *a_re, const double *a_im, double *ainv_re,
                     double *ainv_im) {
  double l_re[K * K];
  double l_im[K * K];
  herm_chol(K, a_re, a_im, l_re, l_im);

  for (size_t c = 0; c < K; ++c) {
    double y_re[K];
    double y_im[K];
    double x_re[K];
    double x_im[K];
    for (size_t i = 0; i < K; ++i) {
      double sr = (i == c) ? 1.0 : 0.0;
      double si = 0.0;
      for (size_t j = 0; j < i; ++j) {
        double lr = l_re[i * K + j];
        double li = l_im[i * K + j];
        sr -= lr * y_re[j] - li * y_im[j];
        si -= lr * y_im[j] + li * y_re[j];
      }
      double diag = l_re[i * K + i];
      y_re[i] = sr / diag;
      y_im[i] = si / diag;
    }
    for (size_t i = K; i-- > 0;) {
      double sr = y_re[i];
      double si = y_im[i];
      for (size_t j = i + 1; j < K; ++j) {
        double lr = l_re[j * K + i];
        double li = -l_im[j * K + i];
        sr -= lr * x_re[j] - li * x_im[j];
        si -= lr * x_im[j] + li * x_re[j];
      }
      double diag = l_re[i * K + i];
      x_re[i] = sr / diag;
      x_im[i] = si / diag;
    }
    for (size_t i = 0; i < K; ++i) {
      ainv_re[i * K + c] = x_re[i];
      ainv_im[i * K + c] = x_im[i];
    }
  }
}

// G_ij = h_jᴴ h_i. Only i >= j is computed. Each dot reduces once.
static void gram_opt(size_t K, size_t M, const double *h_re, const double *h_im, double *a_re,
                     double *a_im) {
  for (size_t n = 0; n < K * K; ++n) {
    a_re[n] = 0.0;
    a_im[n] = 0.0;
  }
  for (size_t j = 0; j < K; ++j) {
    const double *jr = &h_re[j * M];
    const double *ji = &h_im[j * M];
    for (size_t i0 = j; i0 < K; i0 += 4) {
      size_t nb = K - i0 < 4 ? K - i0 : 4;
      const double *ir[4];
      const double *ii[4];
      double pr[4];
      double pi[4];
      double zero = 0.0;
      size_t vl = 0;
      for (size_t t = 0; t < nb; ++t) {
        ir[t] = &h_re[(i0 + t) * M];
        ii[t] = &h_im[(i0 + t) * M];
      }
      asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(M));
      asm volatile("vfmv.s.f v24, %0" ::"f"(zero));
      asm volatile("vfmv.s.f v25, %0" ::"f"(zero));
      asm volatile("vfmv.s.f v26, %0" ::"f"(zero));
      asm volatile("vfmv.s.f v27, %0" ::"f"(zero));
      asm volatile("vfmv.s.f v28, %0" ::"f"(zero));
      asm volatile("vfmv.s.f v29, %0" ::"f"(zero));
      asm volatile("vfmv.s.f v30, %0" ::"f"(zero));
      asm volatile("vfmv.s.f v31, %0" ::"f"(zero));
      for (size_t m0 = 0; m0 < M; m0 += vl) {
        asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(M - m0));
        asm volatile("vle64.v v0, (%0)" ::"r"(jr + m0));
        asm volatile("vle64.v v1, (%0)" ::"r"(ji + m0));
        if (nb > 0) {
          asm volatile("vle64.v v2, (%0)" ::"r"(ir[0] + m0));
          asm volatile("vle64.v v3, (%0)" ::"r"(ii[0] + m0));
          asm volatile("vfmul.vv v4, v0, v2");
          asm volatile("vfmacc.vv v4, v1, v3");
          asm volatile("vfmul.vv v8, v0, v3");
          asm volatile("vfnmsac.vv v8, v1, v2");
          asm volatile("vfredusum.vs v24, v4, v24");
          asm volatile("vfredusum.vs v28, v8, v28");
        }
        if (nb > 1) {
          asm volatile("vle64.v v2, (%0)" ::"r"(ir[1] + m0));
          asm volatile("vle64.v v3, (%0)" ::"r"(ii[1] + m0));
          asm volatile("vfmul.vv v5, v0, v2");
          asm volatile("vfmacc.vv v5, v1, v3");
          asm volatile("vfmul.vv v9, v0, v3");
          asm volatile("vfnmsac.vv v9, v1, v2");
          asm volatile("vfredusum.vs v25, v5, v25");
          asm volatile("vfredusum.vs v29, v9, v29");
        }
        if (nb > 2) {
          asm volatile("vle64.v v2, (%0)" ::"r"(ir[2] + m0));
          asm volatile("vle64.v v3, (%0)" ::"r"(ii[2] + m0));
          asm volatile("vfmul.vv v6, v0, v2");
          asm volatile("vfmacc.vv v6, v1, v3");
          asm volatile("vfmul.vv v10, v0, v3");
          asm volatile("vfnmsac.vv v10, v1, v2");
          asm volatile("vfredusum.vs v26, v6, v26");
          asm volatile("vfredusum.vs v30, v10, v30");
        }
        if (nb > 3) {
          asm volatile("vle64.v v2, (%0)" ::"r"(ir[3] + m0));
          asm volatile("vle64.v v3, (%0)" ::"r"(ii[3] + m0));
          asm volatile("vfmul.vv v7, v0, v2");
          asm volatile("vfmacc.vv v7, v1, v3");
          asm volatile("vfmul.vv v11, v0, v3");
          asm volatile("vfnmsac.vv v11, v1, v2");
          asm volatile("vfredusum.vs v27, v7, v27");
          asm volatile("vfredusum.vs v31, v11, v31");
        }
      }
      asm volatile("vfmv.f.s %0, v24" : "=f"(pr[0]));
      asm volatile("vfmv.f.s %0, v28" : "=f"(pi[0]));
      if (nb > 1) {
        asm volatile("vfmv.f.s %0, v25" : "=f"(pr[1]));
        asm volatile("vfmv.f.s %0, v29" : "=f"(pi[1]));
      }
      if (nb > 2) {
        asm volatile("vfmv.f.s %0, v26" : "=f"(pr[2]));
        asm volatile("vfmv.f.s %0, v30" : "=f"(pi[2]));
      }
      if (nb > 3) {
        asm volatile("vfmv.f.s %0, v27" : "=f"(pr[3]));
        asm volatile("vfmv.f.s %0, v31" : "=f"(pi[3]));
      }
      for (size_t t = 0; t < nb; ++t) {
        a_re[(i0 + t) * K + j] = pr[t];
        a_im[(i0 + t) * K + j] = pi[t];
      }
    }
  }
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = i + 1; j < K; ++j) {
      a_re[i * K + j] = a_re[j * K + i];
      a_im[i * K + j] = -a_im[j * K + i];
    }
  }
}

static void scale_vec(double *re, double *im, size_t n, double gain) {
  size_t vl = 0;
  for (size_t m0 = 0; m0 < n; m0 += vl) {
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(n - m0));
    asm volatile("vle64.v v0, (%0)" ::"r"(re + m0));
    asm volatile("vle64.v v4, (%0)" ::"r"(im + m0));
    asm volatile("vfmul.vf v0, v0, %0" ::"f"(gain));
    asm volatile("vfmul.vf v4, v4, %0" ::"f"(gain));
    asm volatile("vse64.v v0, (%0)" ::"r"(re + m0) : "memory");
    asm volatile("vse64.v v4, (%0)" ::"r"(im + m0) : "memory");
  }
}

static void apply_vec(size_t K, size_t M, const double *w_re, const double *w_im,
                      const double *s_re, const double *s_im, double *x_re, double *x_im) {
  double zero = 0.0;
  size_t vl = 0;
  for (size_t m0 = 0; m0 < M; m0 += vl) {
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(M - m0));
    asm volatile("vfmv.v.f v8, %0" ::"f"(zero));
    asm volatile("vfmv.v.f v12, %0" ::"f"(zero));
    for (size_t k = 0; k < K; ++k) {
      double sr = s_re[k];
      double si = s_im[k];
      asm volatile("vle64.v v0, (%0)" ::"r"(w_re + k * M + m0));
      asm volatile("vle64.v v4, (%0)" ::"r"(w_im + k * M + m0));
      asm volatile("vfmacc.vf v8, %0, v0" ::"f"(sr));
      asm volatile("vfnmsac.vf v8, %0, v4" ::"f"(si));
      asm volatile("vfmacc.vf v12, %0, v0" ::"f"(si));
      asm volatile("vfmacc.vf v12, %0, v4" ::"f"(sr));
    }
    asm volatile("vse64.v v8, (%0)" ::"r"(x_re + m0) : "memory");
    asm volatile("vse64.v v12, (%0)" ::"r"(x_im + m0) : "memory");
  }
}

// W = Hᴴ A⁻¹. Four output rows stay in v4..v11 for the whole strip.
static void apply_inverse_opt(size_t K, size_t M, const double *h_re, const double *h_im,
                              const double *ainv_re, const double *ainv_im, double *w_re,
                              double *w_im, double *fro) {
  double zero = 0.0;
  size_t vl = 0;
  asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(M));
  asm volatile("vfmv.s.f v31, %0" ::"f"(zero));
  for (size_t k0 = 0; k0 < K; k0 += 4) {
    size_t nb = K - k0 < 4 ? K - k0 : 4;
    for (size_t m0 = 0; m0 < M; m0 += vl) {
      asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(M - m0));
      for (size_t j = 0; j < K; ++j) {
        double ar0 = ainv_re[j * K + k0];
        double ai0 = ainv_im[j * K + k0];
        asm volatile("vle64.v v0, (%0)" ::"r"(h_re + j * M + m0));
        asm volatile("vle64.v v1, (%0)" ::"r"(h_im + j * M + m0));
        if (j == 0) {
          asm volatile("vfmul.vf v4, v0, %0" ::"f"(ar0));
          asm volatile("vfmacc.vf v4, %0, v1" ::"f"(ai0));
          asm volatile("vfmul.vf v8, v0, %0" ::"f"(ai0));
          asm volatile("vfnmsac.vf v8, %0, v1" ::"f"(ar0));
          if (nb > 1) {
            double ar = ainv_re[j * K + k0 + 1];
            double ai = ainv_im[j * K + k0 + 1];
            asm volatile("vfmul.vf v5, v0, %0" ::"f"(ar));
            asm volatile("vfmacc.vf v5, %0, v1" ::"f"(ai));
            asm volatile("vfmul.vf v9, v0, %0" ::"f"(ai));
            asm volatile("vfnmsac.vf v9, %0, v1" ::"f"(ar));
          }
          if (nb > 2) {
            double ar = ainv_re[j * K + k0 + 2];
            double ai = ainv_im[j * K + k0 + 2];
            asm volatile("vfmul.vf v6, v0, %0" ::"f"(ar));
            asm volatile("vfmacc.vf v6, %0, v1" ::"f"(ai));
            asm volatile("vfmul.vf v10, v0, %0" ::"f"(ai));
            asm volatile("vfnmsac.vf v10, %0, v1" ::"f"(ar));
          }
          if (nb > 3) {
            double ar = ainv_re[j * K + k0 + 3];
            double ai = ainv_im[j * K + k0 + 3];
            asm volatile("vfmul.vf v7, v0, %0" ::"f"(ar));
            asm volatile("vfmacc.vf v7, %0, v1" ::"f"(ai));
            asm volatile("vfmul.vf v11, v0, %0" ::"f"(ai));
            asm volatile("vfnmsac.vf v11, %0, v1" ::"f"(ar));
          }
        } else {
          asm volatile("vfmacc.vf v4, %0, v0" ::"f"(ar0));
          asm volatile("vfmacc.vf v4, %0, v1" ::"f"(ai0));
          asm volatile("vfmacc.vf v8, %0, v0" ::"f"(ai0));
          asm volatile("vfnmsac.vf v8, %0, v1" ::"f"(ar0));
          if (nb > 1) {
            double ar = ainv_re[j * K + k0 + 1];
            double ai = ainv_im[j * K + k0 + 1];
            asm volatile("vfmacc.vf v5, %0, v0" ::"f"(ar));
            asm volatile("vfmacc.vf v5, %0, v1" ::"f"(ai));
            asm volatile("vfmacc.vf v9, %0, v0" ::"f"(ai));
            asm volatile("vfnmsac.vf v9, %0, v1" ::"f"(ar));
          }
          if (nb > 2) {
            double ar = ainv_re[j * K + k0 + 2];
            double ai = ainv_im[j * K + k0 + 2];
            asm volatile("vfmacc.vf v6, %0, v0" ::"f"(ar));
            asm volatile("vfmacc.vf v6, %0, v1" ::"f"(ai));
            asm volatile("vfmacc.vf v10, %0, v0" ::"f"(ai));
            asm volatile("vfnmsac.vf v10, %0, v1" ::"f"(ar));
          }
          if (nb > 3) {
            double ar = ainv_re[j * K + k0 + 3];
            double ai = ainv_im[j * K + k0 + 3];
            asm volatile("vfmacc.vf v7, %0, v0" ::"f"(ar));
            asm volatile("vfmacc.vf v7, %0, v1" ::"f"(ai));
            asm volatile("vfmacc.vf v11, %0, v0" ::"f"(ai));
            asm volatile("vfnmsac.vf v11, %0, v1" ::"f"(ar));
          }
        }
      }
      asm volatile("vfmul.vv v2, v4, v4");
      asm volatile("vfmacc.vv v2, v8, v8");
      if (nb > 1) {
        asm volatile("vfmacc.vv v2, v5, v5");
        asm volatile("vfmacc.vv v2, v9, v9");
      }
      if (nb > 2) {
        asm volatile("vfmacc.vv v2, v6, v6");
        asm volatile("vfmacc.vv v2, v10, v10");
      }
      if (nb > 3) {
        asm volatile("vfmacc.vv v2, v7, v7");
        asm volatile("vfmacc.vv v2, v11, v11");
      }
      asm volatile("vfredusum.vs v31, v2, v31");
      asm volatile("vse64.v v4, (%0)" ::"r"(w_re + k0 * M + m0) : "memory");
      asm volatile("vse64.v v8, (%0)" ::"r"(w_im + k0 * M + m0) : "memory");
      if (nb > 1) {
        asm volatile("vse64.v v5, (%0)" ::"r"(w_re + (k0 + 1) * M + m0) : "memory");
        asm volatile("vse64.v v9, (%0)" ::"r"(w_im + (k0 + 1) * M + m0) : "memory");
      }
      if (nb > 2) {
        asm volatile("vse64.v v6, (%0)" ::"r"(w_re + (k0 + 2) * M + m0) : "memory");
        asm volatile("vse64.v v10, (%0)" ::"r"(w_im + (k0 + 2) * M + m0) : "memory");
      }
      if (nb > 3) {
        asm volatile("vse64.v v7, (%0)" ::"r"(w_re + (k0 + 3) * M + m0) : "memory");
        asm volatile("vse64.v v11, (%0)" ::"r"(w_im + (k0 + 3) * M + m0) : "memory");
      }
    }
  }
  asm volatile("vfmv.f.s %0, v31" : "=f"(*fro));
}

// L Lᴴ Z = H over one antenna strip. All stream rows stay in v0..v15.
static void solve_regs(size_t K, size_t M, const double *l_re, const double *l_im,
                       const double *h_re, const double *h_im, double *w_re, double *w_im,
                       double *fro) {
  double zero = 0.0;
  size_t vl = 0;
  asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(M));
  asm volatile("vfmv.s.f v31, %0" ::"f"(zero));
  for (size_t m0 = 0; m0 < M; m0 += vl) {
    asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(M - m0));
    for (size_t i = 0; i < K; ++i)
      load_row((int)i, h_re + i * M + m0, h_im + i * M + m0);
    for (size_t i = 0; i < K; ++i) {
      for (size_t j = 0; j < i; ++j)
        nmac((int)i, (int)j, l_re[i * K + j], l_im[i * K + j]);
      scale_row((int)i, 1.0 / l_re[i * K + i]);
    }
    for (size_t i = K; i-- > 0;) {
      for (size_t j = i + 1; j < K; ++j)
        nmac((int)i, (int)j, l_re[j * K + i], -l_im[j * K + i]);
      scale_row((int)i, 1.0 / l_re[i * K + i]);
    }
    for (size_t i = 0; i < K; ++i)
      accum_energy((int)i, i == 0);
    asm volatile("vfredusum.vs v31, v17, v31");
    for (size_t i = 0; i < K; ++i)
      store_conj((int)i, w_re + i * M + m0, w_im + i * M + m0);
  }
  asm volatile("vfmv.f.s %0, v31" : "=f"(*fro));
}

static void rzf_opt(size_t K, size_t M, double P, double alpha, const double *h_re,
                    const double *h_im, const double *s_re, const double *s_im, double *w_re,
                    double *w_im, double *x_re, double *x_im) {
  double a_re[K * K];
  double a_im[K * K];
  double ainv_re[K * K];
  double ainv_im[K * K];
  double fro = 0.0;

  gram_opt(K, M, h_re, h_im, a_re, a_im);
  for (size_t i = 0; i < K; ++i) {
    a_re[i * K + i] += alpha;
    a_im[i * K + i] = 0.0;
  }
  herm_inv(K, a_re, a_im, ainv_re, ainv_im);
  apply_inverse_opt(K, M, h_re, h_im, ainv_re, ainv_im, w_re, w_im, &fro);
  scale_vec(w_re, w_im, K * M, sqrt(P / fro));
  apply_vec(K, M, w_re, w_im, s_re, s_im, x_re, x_im);
}

static void rzf_ll_opt(size_t K, size_t M, double P, double alpha, const double *h_re,
                       const double *h_im, const double *s_re, const double *s_im,
                       double *w_re, double *w_im, double *x_re, double *x_im) {
  double a_re[K * K];
  double a_im[K * K];
  double l_re[K * K];
  double l_im[K * K];
  double fro = 0.0;

  gram_opt(K, M, h_re, h_im, a_re, a_im);
  for (size_t i = 0; i < K; ++i) {
    a_re[i * K + i] += alpha;
    a_im[i * K + i] = 0.0;
  }
  herm_chol(K, a_re, a_im, l_re, l_im);
  solve_regs(K, M, l_re, l_im, h_re, h_im, w_re, w_im, &fro);
  scale_vec(w_re, w_im, K * M, sqrt(P / fro));
  apply_vec(K, M, w_re, w_im, s_re, s_im, x_re, x_im);
}

static void row_energy_opt(const double *hr, const double *hi, size_t M, double *re) {
  double zero = 0.0;
  size_t vl = 0;
  asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(M));
  asm volatile("vfmv.s.f v24, %0" ::"f"(zero));
  for (size_t m0 = 0; m0 < M; m0 += vl) {
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(M - m0));
    asm volatile("vle64.v v0, (%0)" ::"r"(hr + m0));
    asm volatile("vle64.v v4, (%0)" ::"r"(hi + m0));
    asm volatile("vfmul.vv v16, v0, v0");
    asm volatile("vfmacc.vv v16, v4, v4");
    asm volatile("vfredusum.vs v24, v16, v24");
  }
  asm volatile("vfmv.f.s %0, v24" : "=f"(*re));
}

static void scale_conj_row(const double *hr, const double *hi, double *wr, double *wi,
                           size_t M, double gain) {
  double neg = -gain;
  size_t vl = 0;
  for (size_t m0 = 0; m0 < M; m0 += vl) {
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(M - m0));
    asm volatile("vle64.v v0, (%0)" ::"r"(hr + m0));
    asm volatile("vle64.v v4, (%0)" ::"r"(hi + m0));
    asm volatile("vfmul.vf v0, v0, %0" ::"f"(gain));
    asm volatile("vfmul.vf v4, v4, %0" ::"f"(neg));
    asm volatile("vse64.v v0, (%0)" ::"r"(wr + m0) : "memory");
    asm volatile("vse64.v v4, (%0)" ::"r"(wi + m0) : "memory");
  }
}

static int m4_covers(size_t n) {
  size_t vl = 0;
  asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(n));
  return vl >= n;
}

static void lsf_users(size_t K, size_t N, size_t M, double P, const double *beta_in,
                      const double *h_re, const double *h_im, const double *s_re,
                      const double *s_im, double *w_re, double *w_im, double *x_re,
                      double *x_im) {
  double beta_local[K];
  double inv_beta_sum = 0.0;
  int have_beta = beta_in != 0;
  int fused = N <= 2 && m4_covers(M);
  const double *beta = have_beta ? beta_in : beta_local;

  if (!have_beta && !fused) {
    for (size_t k = 0; k < K; ++k) {
      double pwr = 0.0;
      for (size_t n = 0; n < N; ++n) {
        size_t row = (k * N + n) * M;
        double re;
        row_energy_opt(&h_re[row], &h_im[row], M, &re);
        pwr += re;
      }
      beta_local[k] = pwr / (double)(M * N);
    }
  }

  if (!fused || have_beta) {
    for (size_t k = 0; k < K; ++k)
      inv_beta_sum += 1.0 / beta[k];
  }

  for (size_t k = 0; k < K; ++k) {
    if (fused && have_beta) {
      double zero = 0.0;
      double pwr;
      size_t row = (k * N) * M;
      asm volatile("vsetvli x0, %0, e64, m4, ta, ma" ::"r"(M));
      asm volatile("vle64.v v0, (%0)" ::"r"(h_re + row));
      asm volatile("vle64.v v4, (%0)" ::"r"(h_im + row));
      asm volatile("vfmul.vv v16, v0, v0");
      asm volatile("vfmacc.vv v16, v4, v4");
      if (N > 1) {
        asm volatile("vle64.v v8, (%0)" ::"r"(h_re + row + M));
        asm volatile("vle64.v v12, (%0)" ::"r"(h_im + row + M));
        asm volatile("vfmacc.vv v16, v8, v8");
        asm volatile("vfmacc.vv v16, v12, v12");
      }
      asm volatile("vfmv.s.f v24, %0" ::"f"(zero));
      asm volatile("vfredusum.vs v24, v16, v24");
      asm volatile("vfmv.f.s %0, v24" : "=f"(pwr));
      double pk = P / (beta[k] * inv_beta_sum);
      double gain = sqrt(pk) / sqrt(pwr);
      double neg = -gain;
      asm volatile("vfmul.vf v0, v0, %0" ::"f"(gain));
      asm volatile("vfmul.vf v4, v4, %0" ::"f"(neg));
      asm volatile("vse64.v v0, (%0)" ::"r"(w_re + row) : "memory");
      asm volatile("vse64.v v4, (%0)" ::"r"(w_im + row) : "memory");
      if (N > 1) {
        asm volatile("vfmul.vf v8, v8, %0" ::"f"(gain));
        asm volatile("vfmul.vf v12, v12, %0" ::"f"(neg));
        asm volatile("vse64.v v8, (%0)" ::"r"(w_re + row + M) : "memory");
        asm volatile("vse64.v v12, (%0)" ::"r"(w_im + row + M) : "memory");
      }
    } else if (fused) {
      double zero = 0.0;
      double pwr;
      size_t row = (k * N) * M;
      asm volatile("vsetvli x0, %0, e64, m4, ta, ma" ::"r"(M));
      asm volatile("vle64.v v0, (%0)" ::"r"(h_re + row));
      asm volatile("vle64.v v4, (%0)" ::"r"(h_im + row));
      asm volatile("vfmul.vv v16, v0, v0");
      asm volatile("vfmacc.vv v16, v4, v4");
      if (N > 1) {
        asm volatile("vle64.v v8, (%0)" ::"r"(h_re + row + M));
        asm volatile("vle64.v v12, (%0)" ::"r"(h_im + row + M));
        asm volatile("vfmacc.vv v16, v8, v8");
        asm volatile("vfmacc.vv v16, v12, v12");
      }
      asm volatile("vfmv.s.f v24, %0" ::"f"(zero));
      asm volatile("vfredusum.vs v24, v16, v24");
      asm volatile("vfmv.f.s %0, v24" : "=f"(pwr));
      beta_local[k] = pwr / (double)(M * N);
    } else {
      double pwr = 0.0;
      for (size_t n = 0; n < N; ++n) {
        size_t row = (k * N + n) * M;
        double re;
        row_energy_opt(&h_re[row], &h_im[row], M, &re);
        pwr += re;
      }
      double pk = P / (beta[k] * inv_beta_sum);
      double gain = sqrt(pk) / sqrt(pwr);
      for (size_t n = 0; n < N; ++n) {
        size_t row = (k * N + n) * M;
        scale_conj_row(&h_re[row], &h_im[row], &w_re[row], &w_im[row], M, gain);
      }
    }
  }

  if (fused && !have_beta) {
    for (size_t k = 0; k < K; ++k)
      inv_beta_sum += 1.0 / beta_local[k];
    for (size_t k = 0; k < K; ++k) {
      double pwr = beta_local[k] * (double)(M * N);
      double pk = P / (beta_local[k] * inv_beta_sum);
      double gain = sqrt(pk) / sqrt(pwr);
      size_t row = (k * N) * M;
      double neg = -gain;
      asm volatile("vsetvli x0, %0, e64, m4, ta, ma" ::"r"(M));
      asm volatile("vle64.v v0, (%0)" ::"r"(h_re + row));
      asm volatile("vle64.v v4, (%0)" ::"r"(h_im + row));
      asm volatile("vfmul.vf v0, v0, %0" ::"f"(gain));
      asm volatile("vfmul.vf v4, v4, %0" ::"f"(neg));
      asm volatile("vse64.v v0, (%0)" ::"r"(w_re + row) : "memory");
      asm volatile("vse64.v v4, (%0)" ::"r"(w_im + row) : "memory");
      if (N > 1) {
        asm volatile("vle64.v v8, (%0)" ::"r"(h_re + row + M));
        asm volatile("vle64.v v12, (%0)" ::"r"(h_im + row + M));
        asm volatile("vfmul.vf v8, v8, %0" ::"f"(gain));
        asm volatile("vfmul.vf v12, v12, %0" ::"f"(neg));
        asm volatile("vse64.v v8, (%0)" ::"r"(w_re + row + M) : "memory");
        asm volatile("vse64.v v12, (%0)" ::"r"(w_im + row + M) : "memory");
      }
    }
  }
  apply_vec(K * N, M, w_re, w_im, s_re, s_im, x_re, x_im);
}

#endif

void zf_opt(size_t K, size_t N, size_t M, double P, const double *h_re, const double *h_im,
            const double *s_re, const double *s_im, double *w_re, double *w_im,
            double *x_re, double *x_im) {
#if !defined(__riscv)
  zf_gold(K, N, M, P, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
#else
  rzf_opt(K * N, M, P, 0.0, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
#endif
}

void zf_ll_opt(size_t K, size_t N, size_t M, double P, const double *h_re,
               const double *h_im, const double *s_re, const double *s_im, double *w_re,
               double *w_im, double *x_re, double *x_im) {
#if !defined(__riscv)
  zf_ll_gold(K, N, M, P, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
#else
  if (K * N <= 8)
    rzf_ll_opt(K * N, M, P, 0.0, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
  else
    zf_ll_vec(K, N, M, P, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
#endif
}

void mmse_opt(size_t K, size_t N, size_t M, double P, double sigma_n2, const double *h_re,
              const double *h_im, const double *s_re, const double *s_im, double *w_re,
              double *w_im, double *x_re, double *x_im) {
#if !defined(__riscv)
  mmse_gold(K, N, M, P, sigma_n2, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
#else
  double alpha = (double)(K * N) * sigma_n2 / P;
  rzf_opt(K * N, M, P, alpha, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
#endif
}

void mmse_ll_opt(size_t K, size_t N, size_t M, double P, double sigma_n2, const double *h_re,
                 const double *h_im, const double *s_re, const double *s_im, double *w_re,
                 double *w_im, double *x_re, double *x_im) {
#if !defined(__riscv)
  mmse_ll_gold(K, N, M, P, sigma_n2, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
#else
  double alpha = (double)(K * N) * sigma_n2 / P;
  if (K * N <= 8)
    rzf_ll_opt(K * N, M, P, alpha, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
  else
    mmse_ll_vec(K, N, M, P, sigma_n2, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
#endif
}

void lsf_opt(size_t K, size_t N, size_t M, double P, const double *h_re, const double *h_im,
             const double *s_re, const double *s_im, double *w_re, double *w_im,
             double *x_re, double *x_im) {
#if !defined(__riscv)
  lsf_gold(K, N, M, P, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
#else
  lsf_users(K, N, M, P, 0, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
#endif
}

void lsf_ll_opt(size_t K, size_t N, size_t M, double P, const double *h_re,
                const double *h_im, const double *beta, const double *s_re,
                const double *s_im, double *w_re, double *w_im, double *x_re,
                double *x_im) {
#if !defined(__riscv)
  lsf_ll_gold(K, N, M, P, h_re, h_im, beta, s_re, s_im, w_re, w_im, x_re, x_im);
#else
  lsf_users(K, N, M, P, beta, h_re, h_im, s_re, s_im, w_re, w_im, x_re, x_im);
#endif
}
