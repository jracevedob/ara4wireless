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

// Optimization of dirty-paper coding. The QR columns and the working column
// stay in the vector registers, so each Gram-Schmidt update is a dot and a
// saxpy with no reload. The low-latency form reduces each Gram entry once and
// builds x in a single pass over H.

#include "dpc.h"

#include <math.h>

static const double kTau = 2.0 * 1.4142135623730951;

static double nearest(double x) {
  return (x >= 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
}

static double modulo_tau(double x) { return x - kTau * nearest(x / kTau); }

#if defined(__riscv)

#define DOT_COL(QR, QI)                                               \
  asm volatile("vfmul.vv v18, v" #QR ", v16");                        \
  asm volatile("vfmacc.vv v18, v" #QI ", v17");                       \
  asm volatile("vfmul.vv v19, v" #QR ", v17");                        \
  asm volatile("vfnmsac.vv v19, v" #QI ", v16")

#define SAX_COL(QR, QI)                                               \
  asm volatile("vfnmsac.vf v16, %0, v" #QR ::"f"(rr));                \
  asm volatile("vfmacc.vf v16, %0, v" #QI ::"f"(ri));                 \
  asm volatile("vfnmsac.vf v17, %0, v" #QR ::"f"(ri));                \
  asm volatile("vfnmsac.vf v17, %0, v" #QI ::"f"(rr))

#define NORM_COL(QR, QI)                                              \
  asm volatile("vfmul.vf v" #QR ", v16, %0" ::"f"(inv));              \
  asm volatile("vfmul.vf v" #QI ", v17, %0" ::"f"(inv));              \
  asm volatile("vse64.v v" #QR ", (%0)" ::"r"(qr) : "memory");        \
  asm volatile("vse64.v v" #QI ", (%0)" ::"r"(qi) : "memory")

#define X_FIRST(QR, QI)                                               \
  asm volatile("vfmul.vf v22, v" #QR ", %0" ::"f"(ur));               \
  asm volatile("vfnmsac.vf v22, %0, v" #QI ::"f"(ui));                \
  asm volatile("vfmul.vf v23, v" #QR ", %0" ::"f"(ui));               \
  asm volatile("vfmacc.vf v23, %0, v" #QI ::"f"(ur))

#define X_NEXT(QR, QI)                                                \
  asm volatile("vfmacc.vf v22, %0, v" #QR ::"f"(ur));                 \
  asm volatile("vfnmsac.vf v22, %0, v" #QI ::"f"(ui));                \
  asm volatile("vfmacc.vf v23, %0, v" #QR ::"f"(ui));                 \
  asm volatile("vfmacc.vf v23, %0, v" #QI ::"f"(ur))

static void dot_col(int col, double *re, double *im) {
  switch (col) {
    case 0: DOT_COL(0, 1); break;
    case 1: DOT_COL(2, 3); break;
    case 2: DOT_COL(4, 5); break;
    case 3: DOT_COL(6, 7); break;
    case 4: DOT_COL(8, 9); break;
    case 5: DOT_COL(10, 11); break;
    case 6: DOT_COL(12, 13); break;
    default: DOT_COL(14, 15); break;
  }
  double zero = 0.0;
  asm volatile("vfmv.s.f v20, %0" ::"f"(zero));
  asm volatile("vfredusum.vs v20, v18, v20");
  asm volatile("vfmv.f.s %0, v20" : "=f"(*re));
  asm volatile("vfmv.s.f v21, %0" ::"f"(zero));
  asm volatile("vfredusum.vs v21, v19, v21");
  asm volatile("vfmv.f.s %0, v21" : "=f"(*im));
}

static void saxpy_col(int col, double rr, double ri) {
  switch (col) {
    case 0: SAX_COL(0, 1); break;
    case 1: SAX_COL(2, 3); break;
    case 2: SAX_COL(4, 5); break;
    case 3: SAX_COL(6, 7); break;
    case 4: SAX_COL(8, 9); break;
    case 5: SAX_COL(10, 11); break;
    case 6: SAX_COL(12, 13); break;
    default: SAX_COL(14, 15); break;
  }
}

static void norm_into_col(int col, double *qr, double *qi, double *rdiag) {
  double zero = 0.0;
  double nr;
  asm volatile("vfmul.vv v18, v16, v16");
  asm volatile("vfmacc.vv v18, v17, v17");
  asm volatile("vfmv.s.f v20, %0" ::"f"(zero));
  asm volatile("vfredusum.vs v20, v18, v20");
  asm volatile("vfmv.f.s %0, v20" : "=f"(nr));
  double rkk = sqrt(nr);
  *rdiag = rkk;
  double inv = 1.0 / rkk;
  switch (col) {
    case 0: NORM_COL(0, 1); break;
    case 1: NORM_COL(2, 3); break;
    case 2: NORM_COL(4, 5); break;
    case 3: NORM_COL(6, 7); break;
    case 4: NORM_COL(8, 9); break;
    case 5: NORM_COL(10, 11); break;
    case 6: NORM_COL(12, 13); break;
    default: NORM_COL(14, 15); break;
  }
}

static void x_from_col(int col, double ur, double ui, int first) {
  if (first) {
    switch (col) {
      case 0: X_FIRST(0, 1); break;
      case 1: X_FIRST(2, 3); break;
      case 2: X_FIRST(4, 5); break;
      case 3: X_FIRST(6, 7); break;
      case 4: X_FIRST(8, 9); break;
      case 5: X_FIRST(10, 11); break;
      case 6: X_FIRST(12, 13); break;
      default: X_FIRST(14, 15); break;
    }
    return;
  }
  switch (col) {
    case 0: X_NEXT(0, 1); break;
    case 1: X_NEXT(2, 3); break;
    case 2: X_NEXT(4, 5); break;
    case 3: X_NEXT(6, 7); break;
    case 4: X_NEXT(8, 9); break;
    case 5: X_NEXT(10, 11); break;
    case 6: X_NEXT(12, 13); break;
    default: X_NEXT(14, 15); break;
  }
}

static void load_conj_v(const double *hr, const double *hi) {
  double neg = -1.0;
  asm volatile("vle64.v v16, (%0)" ::"r"(hr));
  asm volatile("vle64.v v17, (%0)" ::"r"(hi));
  asm volatile("vfmul.vf v17, v17, %0" ::"f"(neg));
}

static int m1_covers(size_t n) {
  size_t vl = 0;
  asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(n));
  return vl >= n;
}

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
      double pr[4] = {0.0, 0.0, 0.0, 0.0};
      double pi[4] = {0.0, 0.0, 0.0, 0.0};
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

static void chol_back(size_t K, const double *l_re, const double *l_im, const double *u_re,
                      const double *u_im, double *z_re, double *z_im) {
  for (size_t i = K; i-- > 0;) {
    double sr = u_re[i];
    double si = u_im[i];
    for (size_t j = i + 1; j < K; ++j) {
      double lr = l_re[j * K + i];
      double li = -l_im[j * K + i];
      sr -= lr * z_re[j] - li * z_im[j];
      si -= lr * z_im[j] + li * z_re[j];
    }
    double diag = l_re[i * K + i];
    z_re[i] = sr / diag;
    z_im[i] = si / diag;
  }
}

static void cancel(size_t K, const double *coeff_re, const double *coeff_im,
                   const double *diag, const double *s_re, const double *s_im, double *u_re,
                   double *u_im) {
  for (size_t i = 0; i < K; ++i) {
    double ur = s_re[i];
    double ui = s_im[i];
    double inv = 1.0 / diag[i];
    for (size_t j = 0; j < i; ++j) {
      double cr = coeff_re[j * K + i] * inv;
      double ci = -coeff_im[j * K + i] * inv;
      ur -= cr * u_re[j] - ci * u_im[j];
      ui -= cr * u_im[j] + ci * u_re[j];
    }
    u_re[i] = modulo_tau(ur);
    u_im[i] = modulo_tau(ui);
  }
}

static void dpc_regs(size_t K, size_t M, double P, const double *h_re, const double *h_im,
                     const double *s_re, const double *s_im, double *q_re, double *q_im,
                     double *rdiag, double *u_re, double *u_im, double *x_re, double *x_im) {
  double r_re[K * K];
  double r_im[K * K];
  size_t vl = 0;
  asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(M));

  for (size_t k = 0; k < K; ++k) {
    load_conj_v(&h_re[k * M], &h_im[k * M]);
    for (size_t j = 0; j < k; ++j) {
      double rr;
      double ri;
      dot_col((int)j, &rr, &ri);
      r_re[j * K + k] = rr;
      r_im[j * K + k] = ri;
      saxpy_col((int)j, rr, ri);
    }
    norm_into_col((int)k, &q_re[k * M], &q_im[k * M], &rdiag[k]);
    r_re[k * K + k] = rdiag[k];
    r_im[k * K + k] = 0.0;
  }

  // The QR coefficient stored above is R_jk for j < k. Cancellation divides
  // by R_ii and conjugates, which is what cancel() does with this layout.
  cancel(K, r_re, r_im, rdiag, s_re, s_im, u_re, u_im);

  for (size_t k = 0; k < K; ++k)
    x_from_col((int)k, u_re[k], u_im[k], k == 0);

  double zero = 0.0;
  double power;
  asm volatile("vfmul.vv v18, v22, v22");
  asm volatile("vfmacc.vv v18, v23, v23");
  asm volatile("vfmv.s.f v20, %0" ::"f"(zero));
  asm volatile("vfredusum.vs v20, v18, v20");
  asm volatile("vfmv.f.s %0, v20" : "=f"(power));
  double scale = sqrt(P / power);
  asm volatile("vfmul.vf v22, v22, %0" ::"f"(scale));
  asm volatile("vfmul.vf v23, v23, %0" ::"f"(scale));
  asm volatile("vse64.v v22, (%0)" ::"r"(x_re) : "memory");
  asm volatile("vse64.v v23, (%0)" ::"r"(x_im) : "memory");
  (void)vl;
}

static void apply_from_h(size_t K, size_t M, double P, const double *h_re, const double *h_im,
                         const double *z_re, const double *z_im, double *x_re, double *x_im) {
  double zero = 0.0;
  double power;
  size_t vl = 0;
  asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(M));
  asm volatile("vfmv.s.f v31, %0" ::"f"(zero));
  for (size_t m0 = 0; m0 < M; m0 += vl) {
    asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(M - m0));
    asm volatile("vle64.v v0, (%0)" ::"r"(h_re + m0));
    asm volatile("vle64.v v1, (%0)" ::"r"(h_im + m0));
    asm volatile("vfmul.vf v2, v0, %0" ::"f"(z_re[0]));
    asm volatile("vfmacc.vf v2, %0, v1" ::"f"(z_im[0]));
    asm volatile("vfmul.vf v3, v0, %0" ::"f"(z_im[0]));
    asm volatile("vfnmsac.vf v3, %0, v1" ::"f"(z_re[0]));
    for (size_t k = 1; k < K; ++k) {
      asm volatile("vle64.v v0, (%0)" ::"r"(h_re + k * M + m0));
      asm volatile("vle64.v v1, (%0)" ::"r"(h_im + k * M + m0));
      asm volatile("vfmacc.vf v2, %0, v0" ::"f"(z_re[k]));
      asm volatile("vfmacc.vf v2, %0, v1" ::"f"(z_im[k]));
      asm volatile("vfmacc.vf v3, %0, v0" ::"f"(z_im[k]));
      asm volatile("vfnmsac.vf v3, %0, v1" ::"f"(z_re[k]));
    }
    asm volatile("vfmul.vv v4, v2, v2");
    asm volatile("vfmacc.vv v4, v3, v3");
    asm volatile("vfredusum.vs v31, v4, v31");
    asm volatile("vse64.v v2, (%0)" ::"r"(x_re + m0) : "memory");
    asm volatile("vse64.v v3, (%0)" ::"r"(x_im + m0) : "memory");
  }
  asm volatile("vfmv.f.s %0, v31" : "=f"(power));
  double scale = sqrt(P / power);
  for (size_t m0 = 0; m0 < M; m0 += vl) {
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(M - m0));
    asm volatile("vle64.v v0, (%0)" ::"r"(x_re + m0));
    asm volatile("vle64.v v4, (%0)" ::"r"(x_im + m0));
    asm volatile("vfmul.vf v0, v0, %0" ::"f"(scale));
    asm volatile("vfmul.vf v4, v4, %0" ::"f"(scale));
    asm volatile("vse64.v v0, (%0)" ::"r"(x_re + m0) : "memory");
    asm volatile("vse64.v v4, (%0)" ::"r"(x_im + m0) : "memory");
  }
}

static void dpc_ll_fast(size_t K, size_t M, double P, const double *h_re, const double *h_im,
                        const double *s_re, const double *s_im, double *rdiag, double *u_re,
                        double *u_im, double *x_re, double *x_im) {
  double a_re[K * K];
  double a_im[K * K];
  double l_re[K * K];
  double l_im[K * K];
  double z_re[K];
  double z_im[K];

  gram_opt(K, M, h_re, h_im, a_re, a_im);
  herm_chol(K, a_re, a_im, l_re, l_im);
  for (size_t i = 0; i < K; ++i)
    rdiag[i] = l_re[i * K + i];

  // L is lower. The TH coefficient of stream j into stream i is L_ij / L_ii.
  for (size_t i = 0; i < K; ++i) {
    double ur = s_re[i];
    double ui = s_im[i];
    double inv = 1.0 / rdiag[i];
    for (size_t j = 0; j < i; ++j) {
      double cr = l_re[i * K + j] * inv;
      double ci = l_im[i * K + j] * inv;
      ur -= cr * u_re[j] - ci * u_im[j];
      ui -= cr * u_im[j] + ci * u_re[j];
    }
    u_re[i] = modulo_tau(ur);
    u_im[i] = modulo_tau(ui);
  }
  chol_back(K, l_re, l_im, u_re, u_im, z_re, z_im);
  apply_from_h(K, M, P, h_re, h_im, z_re, z_im, x_re, x_im);
}

#endif

void dpc_opt(size_t K, size_t N, size_t M, double P, const double *h_re, const double *h_im,
             const double *s_re, const double *s_im, double *q_re, double *q_im,
             double *rdiag, double *u_re, double *u_im, double *x_re, double *x_im) {
#if !defined(__riscv)
  dpc_gold(K, N, M, P, h_re, h_im, s_re, s_im, q_re, q_im, rdiag, u_re, u_im, x_re, x_im);
#else
  size_t S = K * N;
  if (S <= 8 && m1_covers(M))
    dpc_regs(S, M, P, h_re, h_im, s_re, s_im, q_re, q_im, rdiag, u_re, u_im, x_re, x_im);
  else
    dpc_vec(K, N, M, P, h_re, h_im, s_re, s_im, q_re, q_im, rdiag, u_re, u_im, x_re, x_im);
#endif
}

void dpc_ll_opt(size_t K, size_t N, size_t M, double P, const double *h_re,
                const double *h_im, const double *s_re, const double *s_im, double *rdiag,
                double *u_re, double *u_im, double *x_re, double *x_im) {
#if !defined(__riscv)
  dpc_ll_gold(K, N, M, P, h_re, h_im, s_re, s_im, rdiag, u_re, u_im, x_re, x_im);
#else
  dpc_ll_fast(K * N, M, P, h_re, h_im, s_re, s_im, rdiag, u_re, u_im, x_re, x_im);
#endif
}
