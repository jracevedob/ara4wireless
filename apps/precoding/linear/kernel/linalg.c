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

#include "linalg.h"

#include <math.h>

void hdot_gold(const double *ar, const double *ai, const double *br, const double *bi,
               size_t n, double *re, double *im) {
  double sr = 0.0;
  double si = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sr += ar[i] * br[i] + ai[i] * bi[i];
    si += ar[i] * bi[i] - ai[i] * br[i];
  }
  *re = sr;
  *im = si;
}

void hdot_vec(const double *ar, const double *ai, const double *br, const double *bi,
              size_t n, double *re, double *im) {
#if !defined(__riscv)
  hdot_gold(ar, ai, br, bi, n, re, im);
#else
  double sr = 0.0;
  double si = 0.0;
  size_t vl = 0;
  const double *ar_ = ar;
  const double *ai_ = ai;
  const double *br_ = br;
  const double *bi_ = bi;

  for (size_t avl = n; avl > 0; avl -= vl) {
    double partial_re;
    double partial_im;
    double zero = 0.0;
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle64.v v0,  (%0)" ::"r"(ar_));
    asm volatile("vle64.v v4,  (%0)" ::"r"(ai_));
    asm volatile("vle64.v v8,  (%0)" ::"r"(br_));
    asm volatile("vle64.v v12, (%0)" ::"r"(bi_));
    // re = ar br + ai bi
    asm volatile("vfmul.vv v16, v0, v8");
    asm volatile("vfmacc.vv v16, v4, v12");
    // im = ar bi - ai br
    asm volatile("vfmul.vv v20, v0, v12");
    asm volatile("vfnmsac.vv v20, v4, v8");
    asm volatile("vfmv.s.f v24, %0" ::"f"(zero));
    asm volatile("vfredusum.vs v24, v16, v24");
    asm volatile("vfmv.f.s %0, v24" : "=f"(partial_re));
    asm volatile("vfmv.s.f v24, %0" ::"f"(zero));
    asm volatile("vfredusum.vs v24, v20, v24");
    asm volatile("vfmv.f.s %0, v24" : "=f"(partial_im));
    sr += partial_re;
    si += partial_im;
    ar_ += vl;
    ai_ += vl;
    br_ += vl;
    bi_ += vl;
  }
  *re = sr;
  *im = si;
#endif
}

static void hdot(const double *ar, const double *ai, const double *br, const double *bi,
                 size_t n, int use_vec, double *re, double *im) {
  if (use_vec)
    hdot_vec(ar, ai, br, bi, n, re, im);
  else
    hdot_gold(ar, ai, br, bi, n, re, im);
}

// Invert a Hermitian positive-definite matrix with Cholesky: A = L Lᴴ.
// L is lower triangular with a real diagonal. Each column of A⁻¹ is one
// forward substitution through L and one back substitution through Lᴴ.
static void herm_inv(size_t K, const double *a_re, const double *a_im, double *ainv_re,
                     double *ainv_im) {
  double l_re[K * K];
  double l_im[K * K];

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

void apply_precoder(size_t K, size_t M, const double *w_re, const double *w_im,
                    const double *s_re, const double *s_im, double *x_re, double *x_im) {
  for (size_t m = 0; m < M; ++m) {
    double xr = 0.0;
    double xi = 0.0;
    for (size_t k = 0; k < K; ++k) {
      double wr = w_re[k * M + m];
      double wi = w_im[k * M + m];
      xr += wr * s_re[k] - wi * s_im[k];
      xi += wr * s_im[k] + wi * s_re[k];
    }
    x_re[m] = xr;
    x_im[m] = xi;
  }
}

// G_ij = h_jᴴ h_i. Row j stays in v0/v1 while up to four rows i are accumulated
// with LMUL=1, and the reduction runs once per antenna chunk.
static void gram_vec(size_t K, size_t M, const double *h_re, const double *h_im, double *a_re,
                     double *a_im) {
  for (size_t n = 0; n < K * K; ++n) {
    a_re[n] = 0.0;
    a_im[n] = 0.0;
  }
#if !defined(__riscv)
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < K; ++j)
      hdot_gold(&h_re[j * M], &h_im[j * M], &h_re[i * M], &h_im[i * M], M, &a_re[i * K + j],
                &a_im[i * K + j]);
  }
#else
  for (size_t j = 0; j < K; ++j) {
    for (size_t i0 = 0; i0 < K; i0 += 4) {
      size_t nb = K - i0 < 4 ? K - i0 : 4;
      const double *jr = &h_re[j * M];
      const double *ji = &h_im[j * M];
      const double *ir[4];
      const double *ii[4];
      double pr[4] = {0.0, 0.0, 0.0, 0.0};
      double pi[4] = {0.0, 0.0, 0.0, 0.0};
      for (size_t t = 0; t < nb; ++t) {
        ir[t] = &h_re[(i0 + t) * M];
        ii[t] = &h_im[(i0 + t) * M];
      }
      size_t vl = 0;
      for (size_t m0 = 0; m0 < M; m0 += vl) {
        size_t avl = M - m0;
        double zero = 0.0;
        double part;
        asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle64.v v0, (%0)" ::"r"(jr + m0));
        asm volatile("vle64.v v1, (%0)" ::"r"(ji + m0));
        asm volatile("vfmv.v.f v4, %0" ::"f"(zero));
        asm volatile("vfmv.v.f v5, %0" ::"f"(zero));
        asm volatile("vfmv.v.f v6, %0" ::"f"(zero));
        asm volatile("vfmv.v.f v7, %0" ::"f"(zero));
        asm volatile("vfmv.v.f v8, %0" ::"f"(zero));
        asm volatile("vfmv.v.f v9, %0" ::"f"(zero));
        asm volatile("vfmv.v.f v10, %0" ::"f"(zero));
        asm volatile("vfmv.v.f v11, %0" ::"f"(zero));
        if (nb > 0) {
          asm volatile("vle64.v v2, (%0)" ::"r"(ir[0] + m0));
          asm volatile("vle64.v v3, (%0)" ::"r"(ii[0] + m0));
          asm volatile("vfmacc.vv v4, v0, v2");
          asm volatile("vfmacc.vv v4, v1, v3");
          asm volatile("vfmacc.vv v8, v0, v3");
          asm volatile("vfnmsac.vv v8, v1, v2");
        }
        if (nb > 1) {
          asm volatile("vle64.v v2, (%0)" ::"r"(ir[1] + m0));
          asm volatile("vle64.v v3, (%0)" ::"r"(ii[1] + m0));
          asm volatile("vfmacc.vv v5, v0, v2");
          asm volatile("vfmacc.vv v5, v1, v3");
          asm volatile("vfmacc.vv v9, v0, v3");
          asm volatile("vfnmsac.vv v9, v1, v2");
        }
        if (nb > 2) {
          asm volatile("vle64.v v2, (%0)" ::"r"(ir[2] + m0));
          asm volatile("vle64.v v3, (%0)" ::"r"(ii[2] + m0));
          asm volatile("vfmacc.vv v6, v0, v2");
          asm volatile("vfmacc.vv v6, v1, v3");
          asm volatile("vfmacc.vv v10, v0, v3");
          asm volatile("vfnmsac.vv v10, v1, v2");
        }
        if (nb > 3) {
          asm volatile("vle64.v v2, (%0)" ::"r"(ir[3] + m0));
          asm volatile("vle64.v v3, (%0)" ::"r"(ii[3] + m0));
          asm volatile("vfmacc.vv v7, v0, v2");
          asm volatile("vfmacc.vv v7, v1, v3");
          asm volatile("vfmacc.vv v11, v0, v3");
          asm volatile("vfnmsac.vv v11, v1, v2");
        }
        asm volatile("vfmv.s.f v31, %0" ::"f"(zero));
        asm volatile("vfredusum.vs v31, v4, v31");
        asm volatile("vfmv.f.s %0, v31" : "=f"(part));
        pr[0] += part;
        asm volatile("vfmv.s.f v31, %0" ::"f"(zero));
        asm volatile("vfredusum.vs v31, v8, v31");
        asm volatile("vfmv.f.s %0, v31" : "=f"(part));
        pi[0] += part;
        if (nb > 1) {
          asm volatile("vfmv.s.f v31, %0" ::"f"(zero));
          asm volatile("vfredusum.vs v31, v5, v31");
          asm volatile("vfmv.f.s %0, v31" : "=f"(part));
          pr[1] += part;
          asm volatile("vfmv.s.f v31, %0" ::"f"(zero));
          asm volatile("vfredusum.vs v31, v9, v31");
          asm volatile("vfmv.f.s %0, v31" : "=f"(part));
          pi[1] += part;
        }
        if (nb > 2) {
          asm volatile("vfmv.s.f v31, %0" ::"f"(zero));
          asm volatile("vfredusum.vs v31, v6, v31");
          asm volatile("vfmv.f.s %0, v31" : "=f"(part));
          pr[2] += part;
          asm volatile("vfmv.s.f v31, %0" ::"f"(zero));
          asm volatile("vfredusum.vs v31, v10, v31");
          asm volatile("vfmv.f.s %0, v31" : "=f"(part));
          pi[2] += part;
        }
        if (nb > 3) {
          asm volatile("vfmv.s.f v31, %0" ::"f"(zero));
          asm volatile("vfredusum.vs v31, v7, v31");
          asm volatile("vfmv.f.s %0, v31" : "=f"(part));
          pr[3] += part;
          asm volatile("vfmv.s.f v31, %0" ::"f"(zero));
          asm volatile("vfredusum.vs v31, v11, v31");
          asm volatile("vfmv.f.s %0, v31" : "=f"(part));
          pi[3] += part;
        }
      }
      for (size_t t = 0; t < nb; ++t) {
        a_re[(i0 + t) * K + j] = pr[t];
        a_im[(i0 + t) * K + j] = pi[t];
      }
    }
  }
#endif
}

// W = Hᴴ A⁻¹. Each channel row is loaded once and applied to every column.
static void apply_inverse_full(size_t K, size_t M, const double *h_re, const double *h_im,
                               const double *ainv_re, const double *ainv_im, double *w_re,
                               double *w_im, double *fro) {
  for (size_t n = 0; n < K * M; ++n) {
    w_re[n] = 0.0;
    w_im[n] = 0.0;
  }
  for (size_t j = 0; j < K; ++j) {
#if !defined(__riscv)
    for (size_t k = 0; k < K; ++k) {
      double ar = ainv_re[j * K + k];
      double ai = ainv_im[j * K + k];
      for (size_t m = 0; m < M; ++m) {
        double hr = h_re[j * M + m];
        double hi = h_im[j * M + m];
        w_re[k * M + m] += hr * ar + hi * ai;
        w_im[k * M + m] += hr * ai - hi * ar;
      }
    }
#else
    size_t vl = 0;
    for (size_t m0 = 0; m0 < M; m0 += vl) {
      size_t avl = M - m0;
      asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(avl));
      asm volatile("vle64.v v0, (%0)" ::"r"(&h_re[j * M + m0]));
      asm volatile("vle64.v v1, (%0)" ::"r"(&h_im[j * M + m0]));
      for (size_t k = 0; k < K; ++k) {
        double ar = ainv_re[j * K + k];
        double ai = ainv_im[j * K + k];
        double *wr = &w_re[k * M + m0];
        double *wi = &w_im[k * M + m0];
        asm volatile("vle64.v v2, (%0)" ::"r"(wr));
        asm volatile("vle64.v v3, (%0)" ::"r"(wi));
        asm volatile("vfmacc.vf v2, %0, v0" ::"f"(ar));
        asm volatile("vfmacc.vf v2, %0, v1" ::"f"(ai));
        asm volatile("vfmacc.vf v3, %0, v0" ::"f"(ai));
        asm volatile("vfnmsac.vf v3, %0, v1" ::"f"(ar));
        asm volatile("vse64.v v2, (%0)" ::"r"(wr));
        asm volatile("vse64.v v3, (%0)" ::"r"(wi));
      }
    }
#endif
  }
  *fro = 0.0;
  for (size_t n = 0; n < K * M; ++n)
    *fro += w_re[n] * w_re[n] + w_im[n] * w_im[n];
}

void rzf_precoder(size_t K, size_t M, double P, double alpha, int use_vec,
                  const double *h_re, const double *h_im, const double *s_re,
                  const double *s_im, double *w_re, double *w_im, double *x_re,
                  double *x_im) {
  double a_re[K * K];
  double a_im[K * K];
  double ainv_re[K * K];
  double ainv_im[K * K];
  double fro = 0.0;

  if (use_vec) {
    gram_vec(K, M, h_re, h_im, a_re, a_im);
  } else {
    for (size_t i = 0; i < K; ++i) {
      for (size_t j = 0; j < K; ++j)
        hdot_gold(&h_re[j * M], &h_im[j * M], &h_re[i * M], &h_im[i * M], M,
                  &a_re[i * K + j], &a_im[i * K + j]);
    }
  }
  for (size_t i = 0; i < K; ++i) {
    a_re[i * K + i] += alpha;
    a_im[i * K + i] = 0.0;
  }

  herm_inv(K, a_re, a_im, ainv_re, ainv_im);

  if (use_vec) {
    apply_inverse_full(K, M, h_re, h_im, ainv_re, ainv_im, w_re, w_im, &fro);
  } else {
    for (size_t k = 0; k < K; ++k) {
      for (size_t m = 0; m < M; ++m) {
        double wr = 0.0;
        double wi = 0.0;
        for (size_t j = 0; j < K; ++j) {
          double hr = h_re[j * M + m];
          double hi = h_im[j * M + m];
          double ar = ainv_re[j * K + k];
          double ai = ainv_im[j * K + k];
          wr += hr * ar + hi * ai;
          wi += hr * ai - hi * ar;
        }
        w_re[k * M + m] = wr;
        w_im[k * M + m] = wi;
        fro += wr * wr + wi * wi;
      }
    }
  }

  double scale = sqrt(P / fro);
  for (size_t n = 0; n < K * M; ++n) {
    w_re[n] *= scale;
    w_im[n] *= scale;
  }
  apply_precoder(K, M, w_re, w_im, s_re, s_im, x_re, x_im);
}

// Lower Cholesky factor of a Hermitian positive-definite matrix, A = L Lᴴ.
// The diagonal of L is real and positive.
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

// Solve L Lᴴ x = b. L is the factor from herm_chol.
static void chol_solve(size_t K, const double *l_re, const double *l_im, const double *b_re,
                       const double *b_im, double *x_re, double *x_im) {
  double y_re[K];
  double y_im[K];

  for (size_t i = 0; i < K; ++i) {
    double sr = b_re[i];
    double si = b_im[i];
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
}

static void chol_solve_panel(size_t K, size_t M, size_t m0, size_t tm, const double *l_re,
                             const double *l_im, const double *h_re, const double *h_im,
                             double *w_re, double *w_im, double *y_re, double *y_im);

void rzf_lowlat(size_t K, size_t M, double P, double alpha, int use_vec,
                const double *h_re, const double *h_im, const double *s_re,
                const double *s_im, double *w_re, double *w_im, double *x_re,
                double *x_im) {
  double a_re[K * K];
  double a_im[K * K];
  double l_re[K * K];
  double l_im[K * K];
  double b_re[K];
  double b_im[K];
  double z_re[K];
  double z_im[K];

  if (use_vec) {
    gram_vec(K, M, h_re, h_im, a_re, a_im);
    for (size_t i = 0; i < K; ++i) {
      a_re[i * K + i] += alpha;
      a_im[i * K + i] = 0.0;
    }
  } else {
    for (size_t i = 0; i < K; ++i) {
      for (size_t j = i; j < K; ++j)
        hdot_gold(&h_re[j * M], &h_im[j * M], &h_re[i * M], &h_im[i * M], M,
                  &a_re[i * K + j], &a_im[i * K + j]);
      a_re[i * K + i] += alpha;
      a_im[i * K + i] = 0.0;
    }
    for (size_t i = 0; i < K; ++i) {
      for (size_t j = 0; j < i; ++j) {
        a_re[i * K + j] = a_re[j * K + i];
        a_im[i * K + j] = -a_im[j * K + i];
      }
    }
  }

  herm_chol(K, a_re, a_im, l_re, l_im);

  // G Z = H, then the precoder column is conj(Z). The vector path solves every
  // antenna in one panel. The factor itself stays scalar.
  double fro = 0.0;
  if (use_vec) {
    double y_re[K * M];
    double y_im[K * M];
    chol_solve_panel(K, M, 0, M, l_re, l_im, h_re, h_im, w_re, w_im, y_re, y_im);
    for (size_t n = 0; n < K * M; ++n)
      fro += w_re[n] * w_re[n] + w_im[n] * w_im[n];
  } else {
    for (size_t m = 0; m < M; ++m) {
      for (size_t i = 0; i < K; ++i) {
        b_re[i] = h_re[i * M + m];
        b_im[i] = h_im[i * M + m];
      }
      chol_solve(K, l_re, l_im, b_re, b_im, z_re, z_im);
      for (size_t i = 0; i < K; ++i) {
        double wr = z_re[i];
        double wi = -z_im[i];
        w_re[i * M + m] = wr;
        w_im[i * M + m] = wi;
        fro += wr * wr + wi * wi;
      }
    }
  }

  double scale = sqrt(P / fro);
  for (size_t n = 0; n < K * M; ++n) {
    w_re[n] *= scale;
    w_im[n] *= scale;
  }
  apply_precoder(K, M, w_re, w_im, s_re, s_im, x_re, x_im);
}

static size_t tile_step(size_t n, size_t tile) { return (tile == 0 || tile > n) ? n : tile; }

// Length-n Hermitian product, accumulated over panels of width tile.
static void hdot_panels(const double *ar, const double *ai, const double *br, const double *bi,
                        size_t n, size_t tile, double *re, double *im) {
  double sr = 0.0;
  double si = 0.0;
  size_t step = tile_step(n, tile);
  for (size_t m = 0; m < n; m += step) {
    size_t len = n - m < step ? n - m : step;
    double pr;
    double pi;
    hdot_vec(ar + m, ai + m, br + m, bi + m, len, &pr, &pi);
    sr += pr;
    si += pi;
  }
  *re = sr;
  *im = si;
}

// w += conj(h) * a over a contiguous antenna panel.
static void caxpy_conj_panel(const double *hr, const double *hi, double ar, double ai,
                             double *wr, double *wi, size_t n) {
#if !defined(__riscv)
  for (size_t m = 0; m < n; ++m) {
    wr[m] += hr[m] * ar + hi[m] * ai;
    wi[m] += hr[m] * ai - hi[m] * ar;
  }
#else
  size_t vl = 0;
  for (size_t avl = n; avl > 0; avl -= vl) {
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle64.v v0, (%0)" ::"r"(hr));
    asm volatile("vle64.v v4, (%0)" ::"r"(hi));
    asm volatile("vle64.v v8, (%0)" ::"r"(wr));
    asm volatile("vle64.v v12, (%0)" ::"r"(wi));
    asm volatile("vfmacc.vf v8, %0, v0" ::"f"(ar));
    asm volatile("vfmacc.vf v8, %0, v4" ::"f"(ai));
    asm volatile("vfmacc.vf v12, %0, v0" ::"f"(ai));
    asm volatile("vfnmsac.vf v12, %0, v4" ::"f"(ar));
    asm volatile("vse64.v v8, (%0)" ::"r"(wr));
    asm volatile("vse64.v v12, (%0)" ::"r"(wi));
    hr += vl;
    hi += vl;
    wr += vl;
    wi += vl;
  }
#endif
}

static void gram_tiled(size_t K, size_t M, size_t tile, double alpha, const double *h_re,
                       const double *h_im, double *a_re, double *a_im, int upper_only) {
  size_t step = tile_step(K, tile);
  for (size_t n = 0; n < K * K; ++n) {
    a_re[n] = 0.0;
    a_im[n] = 0.0;
  }
  for (size_t i0 = 0; i0 < K; i0 += step) {
    size_t ni = K - i0 < step ? K - i0 : step;
    for (size_t j0 = 0; j0 < K; j0 += step) {
      size_t nj = K - j0 < step ? K - j0 : step;
      for (size_t ii = 0; ii < ni; ++ii) {
        for (size_t jj = 0; jj < nj; ++jj) {
          size_t i = i0 + ii;
          size_t j = j0 + jj;
          if (upper_only && j < i)
            continue;
          hdot_panels(&h_re[j * M], &h_im[j * M], &h_re[i * M], &h_im[i * M], M, tile,
                      &a_re[i * K + j], &a_im[i * K + j]);
        }
      }
    }
  }
  for (size_t i = 0; i < K; ++i) {
    a_re[i * K + i] += alpha;
    a_im[i * K + i] = 0.0;
  }
  if (upper_only) {
    for (size_t i = 0; i < K; ++i) {
      for (size_t j = 0; j < i; ++j) {
        a_re[i * K + j] = a_re[j * K + i];
        a_im[i * K + j] = -a_im[j * K + i];
      }
    }
  }
}

static void apply_inverse_tiled(size_t K, size_t M, size_t tile, const double *h_re,
                                const double *h_im, const double *ainv_re,
                                const double *ainv_im, double *w_re, double *w_im,
                                double *fro) {
  size_t step_k = tile_step(K, tile);
  size_t step_m = tile_step(M, tile);
  for (size_t n = 0; n < K * M; ++n) {
    w_re[n] = 0.0;
    w_im[n] = 0.0;
  }
  for (size_t m0 = 0; m0 < M; m0 += step_m) {
    size_t tm = M - m0 < step_m ? M - m0 : step_m;
    for (size_t k0 = 0; k0 < K; k0 += step_k) {
      size_t nk = K - k0 < step_k ? K - k0 : step_k;
      for (size_t kk = 0; kk < nk; ++kk) {
        size_t k = k0 + kk;
        for (size_t j0 = 0; j0 < K; j0 += step_k) {
          size_t nj = K - j0 < step_k ? K - j0 : step_k;
          for (size_t jj = 0; jj < nj; ++jj) {
            size_t j = j0 + jj;
            caxpy_conj_panel(&h_re[j * M + m0], &h_im[j * M + m0], ainv_re[j * K + k],
                             ainv_im[j * K + k], &w_re[k * M + m0], &w_im[k * M + m0], tm);
          }
        }
      }
    }
  }
  *fro = 0.0;
  for (size_t n = 0; n < K * M; ++n)
    *fro += w_re[n] * w_re[n] + w_im[n] * w_im[n];
}

static void scale_and_apply(size_t K, size_t M, double P, double fro, const double *s_re,
                            const double *s_im, double *w_re, double *w_im, double *x_re,
                            double *x_im) {
  double scale = sqrt(P / fro);
  for (size_t n = 0; n < K * M; ++n) {
    w_re[n] *= scale;
    w_im[n] *= scale;
  }
  apply_precoder(K, M, w_re, w_im, s_re, s_im, x_re, x_im);
}

void rzf_tiled(size_t K, size_t M, double P, double alpha, size_t tile, const double *h_re,
               const double *h_im, const double *s_re, const double *s_im, double *w_re,
               double *w_im, double *x_re, double *x_im) {
  double a_re[K * K];
  double a_im[K * K];
  double ainv_re[K * K];
  double ainv_im[K * K];
  double fro;

  gram_tiled(K, M, tile, alpha, h_re, h_im, a_re, a_im, 0);
  herm_inv(K, a_re, a_im, ainv_re, ainv_im);
  apply_inverse_tiled(K, M, tile, h_re, h_im, ainv_re, ainv_im, w_re, w_im, &fro);
  scale_and_apply(K, M, P, fro, s_re, s_im, w_re, w_im, x_re, x_im);
}

// vr -= lr * ar - li * ai,  vi -= lr * ai + li * ar, over one antenna panel.
static void panel_nmac(double *vr, double *vi, const double *ar, const double *ai, double lr,
                       double li, size_t n) {
#if !defined(__riscv)
  for (size_t m = 0; m < n; ++m) {
    vr[m] -= lr * ar[m] - li * ai[m];
    vi[m] -= lr * ai[m] + li * ar[m];
  }
#else
  size_t vl = 0;
  for (size_t avl = n; avl > 0; avl -= vl) {
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle64.v v0, (%0)" ::"r"(ar));
    asm volatile("vle64.v v4, (%0)" ::"r"(ai));
    asm volatile("vle64.v v8, (%0)" ::"r"(vr));
    asm volatile("vle64.v v12, (%0)" ::"r"(vi));
    asm volatile("vfnmsac.vf v8, %0, v0" ::"f"(lr));
    asm volatile("vfmacc.vf v8, %0, v4" ::"f"(li));
    asm volatile("vfnmsac.vf v12, %0, v4" ::"f"(lr));
    asm volatile("vfnmsac.vf v12, %0, v0" ::"f"(li));
    asm volatile("vse64.v v8, (%0)" ::"r"(vr));
    asm volatile("vse64.v v12, (%0)" ::"r"(vi));
    ar += vl;
    ai += vl;
    vr += vl;
    vi += vl;
  }
#endif
}

static void panel_scale(double *vr, double *vi, double gain, size_t n) {
#if !defined(__riscv)
  for (size_t m = 0; m < n; ++m) {
    vr[m] *= gain;
    vi[m] *= gain;
  }
#else
  size_t vl = 0;
  for (size_t avl = n; avl > 0; avl -= vl) {
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle64.v v0, (%0)" ::"r"(vr));
    asm volatile("vle64.v v4, (%0)" ::"r"(vi));
    asm volatile("vfmul.vf v0, v0, %0" ::"f"(gain));
    asm volatile("vfmul.vf v4, v4, %0" ::"f"(gain));
    asm volatile("vse64.v v0, (%0)" ::"r"(vr));
    asm volatile("vse64.v v4, (%0)" ::"r"(vi));
    vr += vl;
    vi += vl;
  }
#endif
}

static void panel_copy(double *dst, const double *src, size_t n) {
#if !defined(__riscv)
  for (size_t m = 0; m < n; ++m)
    dst[m] = src[m];
#else
  size_t vl = 0;
  for (size_t avl = n; avl > 0; avl -= vl) {
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle64.v v0, (%0)" ::"r"(src));
    asm volatile("vse64.v v0, (%0)" ::"r"(dst));
    src += vl;
    dst += vl;
  }
#endif
}

// Solve L Lᴴ Z = H for one contiguous antenna panel. Row i of the panel is
// H[i, m0 : m0+tm). The precoder panel is conj(Z).
static void chol_solve_panel(size_t K, size_t M, size_t m0, size_t tm, const double *l_re,
                             const double *l_im, const double *h_re, const double *h_im,
                             double *w_re, double *w_im, double *y_re, double *y_im) {
  for (size_t i = 0; i < K; ++i) {
    panel_copy(y_re + i * tm, h_re + i * M + m0, tm);
    panel_copy(y_im + i * tm, h_im + i * M + m0, tm);
    for (size_t j = 0; j < i; ++j)
      panel_nmac(y_re + i * tm, y_im + i * tm, y_re + j * tm, y_im + j * tm,
                 l_re[i * K + j], l_im[i * K + j], tm);
    panel_scale(y_re + i * tm, y_im + i * tm, 1.0 / l_re[i * K + i], tm);
  }

  for (size_t i = K; i-- > 0;) {
    for (size_t j = i + 1; j < K; ++j)
      panel_nmac(y_re + i * tm, y_im + i * tm, y_re + j * tm, y_im + j * tm,
                 l_re[j * K + i], -l_im[j * K + i], tm);
    panel_scale(y_re + i * tm, y_im + i * tm, 1.0 / l_re[i * K + i], tm);
    for (size_t m = 0; m < tm; ++m) {
      w_re[i * M + m0 + m] = y_re[i * tm + m];
      w_im[i * M + m0 + m] = -y_im[i * tm + m];
    }
  }
}

void rzf_lowlat_tiled(size_t K, size_t M, double P, double alpha, size_t tile,
                      const double *h_re, const double *h_im, const double *s_re,
                      const double *s_im, double *w_re, double *w_im, double *x_re,
                      double *x_im) {
  double a_re[K * K];
  double a_im[K * K];
  double l_re[K * K];
  double l_im[K * K];
  double fro = 0.0;
  size_t step_m = tile_step(M, tile);
  double y_re[K * step_m];
  double y_im[K * step_m];

  gram_tiled(K, M, tile, alpha, h_re, h_im, a_re, a_im, 1);
  herm_chol(K, a_re, a_im, l_re, l_im);

  for (size_t m0 = 0; m0 < M; m0 += step_m) {
    size_t tm = M - m0 < step_m ? M - m0 : step_m;
    chol_solve_panel(K, M, m0, tm, l_re, l_im, h_re, h_im, w_re, w_im, y_re, y_im);
  }
  for (size_t n = 0; n < K * M; ++n)
    fro += w_re[n] * w_re[n] + w_im[n] * w_im[n];
  scale_and_apply(K, M, P, fro, s_re, s_im, w_re, w_im, x_re, x_im);
}
