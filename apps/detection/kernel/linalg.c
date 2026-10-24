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

static const double kInvSqrt2 = 0.7071067811865476;

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
    asm volatile("vfmul.vv v16, v0, v8");
    asm volatile("vfmacc.vv v16, v4, v12");
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

static void hdot_vec_acc(const double *ar, const double *ai, const double *br,
                         const double *bi, size_t n, double *re, double *im) {
#if !defined(__riscv)
  hdot_gold(ar, ai, br, bi, n, re, im);
#else
  size_t vl0;
  size_t vl = 0;
  const double *ar_ = ar;
  const double *ai_ = ai;
  const double *br_ = br;
  const double *bi_ = bi;
  int first = 1;

  asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl0) : "r"(n));
  asm volatile("vmv.v.i v16, 0");
  asm volatile("vmv.v.i v20, 0");
  for (size_t avl = n; avl > 0; avl -= vl) {
    asm volatile("vsetvli %0, %1, e64, m4, tu, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle64.v v0,  (%0)" ::"r"(ar_));
    asm volatile("vle64.v v4,  (%0)" ::"r"(ai_));
    asm volatile("vle64.v v8,  (%0)" ::"r"(br_));
    asm volatile("vle64.v v12, (%0)" ::"r"(bi_));
    if (first) {
      asm volatile("vfmul.vv v16, v0, v8");
      asm volatile("vfmacc.vv v16, v4, v12");
      asm volatile("vfmul.vv v20, v0, v12");
      asm volatile("vfnmsac.vv v20, v4, v8");
      first = 0;
    } else {
      asm volatile("vfmacc.vv v16, v0, v8");
      asm volatile("vfmacc.vv v16, v4, v12");
      asm volatile("vfmacc.vv v20, v0, v12");
      asm volatile("vfnmsac.vv v20, v4, v8");
    }
    ar_ += vl;
    ai_ += vl;
    br_ += vl;
    bi_ += vl;
  }
  asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(vl0));
  asm volatile("vmv.s.x v24, zero");
  asm volatile("vfredusum.vs v24, v16, v24");
  asm volatile("vfmv.f.s %0, v24" : "=f"(*re));
  asm volatile("vmv.s.x v24, zero");
  asm volatile("vfredusum.vs v24, v20, v24");
  asm volatile("vfmv.f.s %0, v24" : "=f"(*im));
#endif
}

static void hdot(const double *ar, const double *ai, const double *br, const double *bi,
                 size_t n, int use_vec, double *re, double *im) {
  if (use_vec)
    hdot_vec(ar, ai, br, bi, n, re, im);
  else
    hdot_gold(ar, ai, br, bi, n, re, im);
}

static void hdot_ll(const double *ar, const double *ai, const double *br, const double *bi,
                    size_t n, int use_vec, double *re, double *im) {
  if (use_vec)
    hdot_vec_acc(ar, ai, br, bi, n, re, im);
  else
    hdot_gold(ar, ai, br, bi, n, re, im);
}

void vcopy(double *dst, const double *src, size_t n, int use_vec) {
#if !defined(__riscv)
  (void)use_vec;
  for (size_t i = 0; i < n; ++i)
    dst[i] = src[i];
#else
  if (!use_vec) {
    for (size_t i = 0; i < n; ++i)
      dst[i] = src[i];
    return;
  }
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

void vscale(double *re, double *im, double gain, size_t n, int use_vec) {
#if !defined(__riscv)
  (void)use_vec;
  for (size_t i = 0; i < n; ++i) {
    re[i] *= gain;
    im[i] *= gain;
  }
#else
  if (!use_vec) {
    for (size_t i = 0; i < n; ++i) {
      re[i] *= gain;
      im[i] *= gain;
    }
    return;
  }
  size_t vl = 0;
  for (size_t avl = n; avl > 0; avl -= vl) {
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle64.v v0, (%0)" ::"r"(re));
    asm volatile("vle64.v v4, (%0)" ::"r"(im));
    asm volatile("vfmul.vf v0, v0, %0" ::"f"(gain));
    asm volatile("vfmul.vf v4, v4, %0" ::"f"(gain));
    asm volatile("vse64.v v0, (%0)" ::"r"(re));
    asm volatile("vse64.v v4, (%0)" ::"r"(im));
    re += vl;
    im += vl;
  }
#endif
}

void caxpy(double *y_re, double *y_im, const double *h_re, const double *h_im, double xr,
           double xi, size_t n, int use_vec) {
#if !defined(__riscv)
  (void)use_vec;
  for (size_t m = 0; m < n; ++m) {
    y_re[m] += h_re[m] * xr - h_im[m] * xi;
    y_im[m] += h_re[m] * xi + h_im[m] * xr;
  }
#else
  if (!use_vec) {
    for (size_t m = 0; m < n; ++m) {
      y_re[m] += h_re[m] * xr - h_im[m] * xi;
      y_im[m] += h_re[m] * xi + h_im[m] * xr;
    }
    return;
  }
  size_t vl = 0;
  for (size_t avl = n; avl > 0; avl -= vl) {
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle64.v v0, (%0)" ::"r"(h_re));
    asm volatile("vle64.v v4, (%0)" ::"r"(h_im));
    asm volatile("vle64.v v8, (%0)" ::"r"(y_re));
    asm volatile("vle64.v v12, (%0)" ::"r"(y_im));
    asm volatile("vfmacc.vf v8, %0, v0" ::"f"(xr));
    asm volatile("vfnmsac.vf v8, %0, v4" ::"f"(xi));
    asm volatile("vfmacc.vf v12, %0, v0" ::"f"(xi));
    asm volatile("vfmacc.vf v12, %0, v4" ::"f"(xr));
    asm volatile("vse64.v v8, (%0)" ::"r"(y_re));
    asm volatile("vse64.v v12, (%0)" ::"r"(y_im));
    h_re += vl;
    h_im += vl;
    y_re += vl;
    y_im += vl;
  }
#endif
}

// Four Hermitian products against a live vector, one reduction after the last
// strip. G_ij = liveᴴ col_j for the live = row index i.
#if defined(__riscv)
static void batch_hdot4(const double *live_r, const double *live_i, const double *cr[4],
                        const double *ci[4], size_t n, size_t nb, double *pr, double *pi) {
  size_t vl0;
  size_t vl = 0;
  asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl0) : "r"(n));
  asm volatile("vmv.v.i v4, 0");
  asm volatile("vmv.v.i v5, 0");
  asm volatile("vmv.v.i v6, 0");
  asm volatile("vmv.v.i v7, 0");
  asm volatile("vmv.v.i v8, 0");
  asm volatile("vmv.v.i v9, 0");
  asm volatile("vmv.v.i v10, 0");
  asm volatile("vmv.v.i v11, 0");
  for (size_t m0 = 0; m0 < n; m0 += vl) {
    size_t avl = n - m0;
    asm volatile("vsetvli %0, %1, e64, m1, tu, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle64.v v0, (%0)" ::"r"(live_r + m0));
    asm volatile("vle64.v v1, (%0)" ::"r"(live_i + m0));
    if (nb > 0) {
      asm volatile("vle64.v v2, (%0)" ::"r"(cr[0] + m0));
      asm volatile("vle64.v v3, (%0)" ::"r"(ci[0] + m0));
      asm volatile("vfmacc.vv v4, v0, v2");
      asm volatile("vfmacc.vv v4, v1, v3");
      asm volatile("vfmacc.vv v8, v0, v3");
      asm volatile("vfnmsac.vv v8, v1, v2");
    }
    if (nb > 1) {
      asm volatile("vle64.v v2, (%0)" ::"r"(cr[1] + m0));
      asm volatile("vle64.v v3, (%0)" ::"r"(ci[1] + m0));
      asm volatile("vfmacc.vv v5, v0, v2");
      asm volatile("vfmacc.vv v5, v1, v3");
      asm volatile("vfmacc.vv v9, v0, v3");
      asm volatile("vfnmsac.vv v9, v1, v2");
    }
    if (nb > 2) {
      asm volatile("vle64.v v2, (%0)" ::"r"(cr[2] + m0));
      asm volatile("vle64.v v3, (%0)" ::"r"(ci[2] + m0));
      asm volatile("vfmacc.vv v6, v0, v2");
      asm volatile("vfmacc.vv v6, v1, v3");
      asm volatile("vfmacc.vv v10, v0, v3");
      asm volatile("vfnmsac.vv v10, v1, v2");
    }
    if (nb > 3) {
      asm volatile("vle64.v v2, (%0)" ::"r"(cr[3] + m0));
      asm volatile("vle64.v v3, (%0)" ::"r"(ci[3] + m0));
      asm volatile("vfmacc.vv v7, v0, v2");
      asm volatile("vfmacc.vv v7, v1, v3");
      asm volatile("vfmacc.vv v11, v0, v3");
      asm volatile("vfnmsac.vv v11, v1, v2");
    }
  }
  asm volatile("vsetvli zero, %0, e64, m1, ta, ma" ::"r"(vl0));
  asm volatile("vmv.s.x v31, zero");
  asm volatile("vfredusum.vs v31, v4, v31");
  asm volatile("vfmv.f.s %0, v31" : "=f"(pr[0]));
  asm volatile("vmv.s.x v31, zero");
  asm volatile("vfredusum.vs v31, v8, v31");
  asm volatile("vfmv.f.s %0, v31" : "=f"(pi[0]));
  if (nb > 1) {
    asm volatile("vmv.s.x v31, zero");
    asm volatile("vfredusum.vs v31, v5, v31");
    asm volatile("vfmv.f.s %0, v31" : "=f"(pr[1]));
    asm volatile("vmv.s.x v31, zero");
    asm volatile("vfredusum.vs v31, v9, v31");
    asm volatile("vfmv.f.s %0, v31" : "=f"(pi[1]));
  }
  if (nb > 2) {
    asm volatile("vmv.s.x v31, zero");
    asm volatile("vfredusum.vs v31, v6, v31");
    asm volatile("vfmv.f.s %0, v31" : "=f"(pr[2]));
    asm volatile("vmv.s.x v31, zero");
    asm volatile("vfredusum.vs v31, v10, v31");
    asm volatile("vfmv.f.s %0, v31" : "=f"(pi[2]));
  }
  if (nb > 3) {
    asm volatile("vmv.s.x v31, zero");
    asm volatile("vfredusum.vs v31, v7, v31");
    asm volatile("vfmv.f.s %0, v31" : "=f"(pr[3]));
    asm volatile("vmv.s.x v31, zero");
    asm volatile("vfredusum.vs v31, v11, v31");
    asm volatile("vfmv.f.s %0, v31" : "=f"(pi[3]));
  }
}
#endif

static void gram_lower(size_t S, size_t M, int use_vec, const double *h_re,
                       const double *h_im, double *g_re, double *g_im) {
#if defined(__riscv)
  if (use_vec) {
    for (size_t i = 0; i < S; ++i) {
      for (size_t j0 = 0; j0 <= i; j0 += 4) {
        size_t nb = i + 1 - j0;
        if (nb > 4)
          nb = 4;
        const double *cr[4];
        const double *ci[4];
        double pr[4];
        double pi[4];
        for (size_t t = 0; t < nb; ++t) {
          cr[t] = &h_re[(j0 + t) * M];
          ci[t] = &h_im[(j0 + t) * M];
        }
        batch_hdot4(&h_re[i * M], &h_im[i * M], cr, ci, M, nb, pr, pi);
        for (size_t t = 0; t < nb; ++t) {
          g_re[i * S + (j0 + t)] = pr[t];
          g_im[i * S + (j0 + t)] = pi[t];
        }
      }
    }
    for (size_t i = 0; i < S; ++i) {
      g_im[i * S + i] = 0.0;
      for (size_t j = 0; j < i; ++j) {
        g_re[j * S + i] = g_re[i * S + j];
        g_im[j * S + i] = -g_im[i * S + j];
      }
    }
    return;
  }
#else
  (void)use_vec;
#endif
  for (size_t i = 0; i < S; ++i) {
    for (size_t j = 0; j <= i; ++j)
      hdot_gold(&h_re[i * M], &h_im[i * M], &h_re[j * M], &h_im[j * M], M,
                &g_re[i * S + j], &g_im[i * S + j]);
    g_im[i * S + i] = 0.0;
    for (size_t j = 0; j < i; ++j) {
      g_re[j * S + i] = g_re[i * S + j];
      g_im[j * S + i] = -g_im[i * S + j];
    }
  }
}

static void matched_filter(size_t S, size_t M, int use_vec, const double *h_re,
                           const double *h_im, const double *y_re, const double *y_im,
                           double *b_re, double *b_im) {
#if defined(__riscv)
  if (use_vec) {
    for (size_t i0 = 0; i0 < S; i0 += 4) {
      size_t nb = S - i0;
      if (nb > 4)
        nb = 4;
      const double *cr[4];
      const double *ci[4];
      double pr[4];
      double pi[4];
      for (size_t t = 0; t < nb; ++t) {
        cr[t] = &h_re[(i0 + t) * M];
        ci[t] = &h_im[(i0 + t) * M];
      }
      // b_i = h_iᴴ y. Live vector is y; the four columns are the H side.
      batch_hdot4(y_re, y_im, cr, ci, M, nb, pr, pi);
      // batch_hdot4 computes liveᴴ col = yᴴ h_i = (h_iᴴ y)̅. Conjugate back.
      for (size_t t = 0; t < nb; ++t) {
        b_re[i0 + t] = pr[t];
        b_im[i0 + t] = -pi[t];
      }
    }
    return;
  }
#else
  (void)use_vec;
#endif
  for (size_t i = 0; i < S; ++i)
    hdot_gold(&h_re[i * M], &h_im[i * M], y_re, y_im, M, &b_re[i], &b_im[i]);
}

static void herm_chol(size_t S, const double *a_re, const double *a_im, double *l_re,
                      double *l_im) {
  for (size_t i = 0; i < S; ++i) {
    for (size_t j = 0; j < i; ++j) {
      double sr = a_re[i * S + j];
      double si = a_im[i * S + j];
      for (size_t p = 0; p < j; ++p) {
        double lr = l_re[i * S + p];
        double li = l_im[i * S + p];
        double rr = l_re[j * S + p];
        double ri = -l_im[j * S + p];
        sr -= lr * rr - li * ri;
        si -= lr * ri + li * rr;
      }
      double inv = 1.0 / l_re[j * S + j];
      l_re[i * S + j] = sr * inv;
      l_im[i * S + j] = si * inv;
    }
    double diag = a_re[i * S + i];
    for (size_t p = 0; p < i; ++p)
      diag -= l_re[i * S + p] * l_re[i * S + p] + l_im[i * S + p] * l_im[i * S + p];
    l_re[i * S + i] = sqrt(diag);
    l_im[i * S + i] = 0.0;
  }
}

static void chol_solve(size_t S, const double *l_re, const double *l_im, const double *b_re,
                       const double *b_im, double *x_re, double *x_im) {
  double y_re[S];
  double y_im[S];
  double inv[S];

  for (size_t i = 0; i < S; ++i)
    inv[i] = 1.0 / l_re[i * S + i];

  for (size_t i = 0; i < S; ++i) {
    double sr = b_re[i];
    double si = b_im[i];
    for (size_t j = 0; j < i; ++j) {
      double lr = l_re[i * S + j];
      double li = l_im[i * S + j];
      sr -= lr * y_re[j] - li * y_im[j];
      si -= lr * y_im[j] + li * y_re[j];
    }
    y_re[i] = sr * inv[i];
    y_im[i] = si * inv[i];
  }

  for (size_t i = S; i-- > 0;) {
    double sr = y_re[i];
    double si = y_im[i];
    for (size_t j = i + 1; j < S; ++j) {
      double lr = l_re[j * S + i];
      double li = -l_im[j * S + i];
      sr -= lr * x_re[j] - li * x_im[j];
      si -= lr * x_im[j] + li * x_re[j];
    }
    x_re[i] = sr * inv[i];
    x_im[i] = si * inv[i];
  }
}

void zf_factor(size_t S, size_t M, int use_vec, const double *h_re, const double *h_im,
               double *l_re, double *l_im) {
  double g_re[S * S];
  double g_im[S * S];
  gram_lower(S, M, use_vec, h_re, h_im, g_re, g_im);
  herm_chol(S, g_re, g_im, l_re, l_im);
}

void zf_apply(size_t S, size_t M, int use_vec, const double *h_re, const double *h_im,
              const double *l_re, const double *l_im, const double *y_re,
              const double *y_im, double *x_re, double *x_im) {
  double b_re[S];
  double b_im[S];
  matched_filter(S, M, use_vec, h_re, h_im, y_re, y_im, b_re, b_im);
  chol_solve(S, l_re, l_im, b_re, b_im, x_re, x_im);
}

void zf_combine(size_t S, size_t M, int use_vec, const double *h_re, const double *h_im,
                const double *y_re, const double *y_im, double *x_re, double *x_im) {
  double g_re[S * S];
  double g_im[S * S];
  double l_re[S * S];
  double l_im[S * S];
  double b_re[S];
  double b_im[S];

  for (size_t i = 0; i < S; ++i) {
    for (size_t j = i; j < S; ++j)
      hdot(&h_re[i * M], &h_im[i * M], &h_re[j * M], &h_im[j * M], M, use_vec,
           &g_re[i * S + j], &g_im[i * S + j]);
    hdot(&h_re[i * M], &h_im[i * M], y_re, y_im, M, use_vec, &b_re[i], &b_im[i]);
  }
  for (size_t i = 0; i < S; ++i) {
    g_im[i * S + i] = 0.0;
    for (size_t j = 0; j < i; ++j) {
      g_re[i * S + j] = g_re[j * S + i];
      g_im[i * S + j] = -g_im[j * S + i];
    }
  }
  herm_chol(S, g_re, g_im, l_re, l_im);
  chol_solve(S, l_re, l_im, b_re, b_im, x_re, x_im);
}

void zf_combine_ll(size_t S, size_t M, int use_vec, const double *h_re, const double *h_im,
                   const double *y_re, const double *y_im, double *x_re, double *x_im) {
  double l_re[S * S];
  double l_im[S * S];
  zf_factor(S, M, use_vec, h_re, h_im, l_re, l_im);
  zf_apply(S, M, use_vec, h_re, h_im, l_re, l_im, y_re, y_im, x_re, x_im);
}

void thin_qr(size_t S, size_t M, int use_vec, const double *h_re, const double *h_im,
             double *q_re, double *q_im, double *r_re, double *r_im) {
  double v_re[S * M];
  double v_im[S * M];

  for (size_t n = 0; n < S * S; ++n) {
    r_re[n] = 0.0;
    r_im[n] = 0.0;
  }
  for (size_t s = 0; s < S; ++s) {
    for (size_t m = 0; m < M; ++m) {
      v_re[s * M + m] = h_re[s * M + m];
      v_im[s * M + m] = h_im[s * M + m];
    }
  }

  for (size_t k = 0; k < S; ++k) {
    for (size_t j = 0; j < k; ++j) {
      double rr;
      double ri;
      hdot(&q_re[j * M], &q_im[j * M], &v_re[k * M], &v_im[k * M], M, use_vec, &rr, &ri);
      r_re[j * S + k] = rr;
      r_im[j * S + k] = ri;
      caxpy(&v_re[k * M], &v_im[k * M], &q_re[j * M], &q_im[j * M], -rr, -ri, M, use_vec);
    }
    double nr;
    double ni;
    hdot(&v_re[k * M], &v_im[k * M], &v_re[k * M], &v_im[k * M], M, use_vec, &nr, &ni);
    double rkk = sqrt(nr);
    r_re[k * S + k] = rkk;
    r_im[k * S + k] = 0.0;
    double inv = 1.0 / rkk;
    for (size_t m = 0; m < M; ++m) {
      q_re[k * M + m] = v_re[k * M + m] * inv;
      q_im[k * M + m] = v_im[k * M + m] * inv;
    }
  }
}

void thin_qr_ll(size_t S, size_t M, int use_vec, const double *h_re, const double *h_im,
                const double *y_re, const double *y_im, double *q_re, double *q_im,
                double *r_re, double *r_im, double *yt_re, double *yt_im) {
  vcopy(q_re, h_re, S * M, use_vec);
  vcopy(q_im, h_im, S * M, use_vec);

  for (size_t k = 0; k < S; ++k) {
    for (size_t j = 0; j < k; ++j) {
      double rr;
      double ri;
      hdot_ll(&q_re[j * M], &q_im[j * M], &q_re[k * M], &q_im[k * M], M, use_vec, &rr,
              &ri);
      r_re[j * S + k] = rr;
      r_im[j * S + k] = ri;
      caxpy(&q_re[k * M], &q_im[k * M], &q_re[j * M], &q_im[j * M], -rr, -ri, M, use_vec);
    }
    double nr;
    double ni;
    hdot_ll(&q_re[k * M], &q_im[k * M], &q_re[k * M], &q_im[k * M], M, use_vec, &nr, &ni);
    double rkk = sqrt(nr);
    r_re[k * S + k] = rkk;
    r_im[k * S + k] = 0.0;
    vscale(&q_re[k * M], &q_im[k * M], 1.0 / rkk, M, use_vec);
    hdot_ll(&q_re[k * M], &q_im[k * M], y_re, y_im, M, use_vec, &yt_re[k], &yt_im[k]);
  }
}

void slice_qpsk(const double *re, const double *im, double *out_re, double *out_im,
                size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out_re[i] = (re[i] >= 0.0) ? kInvSqrt2 : -kInvSqrt2;
    out_im[i] = (im[i] >= 0.0) ? kInvSqrt2 : -kInvSqrt2;
  }
}
