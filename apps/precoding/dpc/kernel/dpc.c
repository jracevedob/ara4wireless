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

#include "dpc.h"

#include <math.h>

// QPSK fundamental interval. Symbols (±1 ± j)/√2 lie inside (−τ/2, τ/2).
static const double kTau = 2.0 * 1.4142135623730951;

static double nearest(double x) {
  return (x >= 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
}

static double modulo_tau(double x) { return x - kTau * nearest(x / kTau); }

static void hdot_gold(const double *ar, const double *ai, const double *br,
                      const double *bi, size_t n, double *re, double *im) {
  double sr = 0.0;
  double si = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sr += ar[i] * br[i] + ai[i] * bi[i];
    si += ar[i] * bi[i] - ai[i] * br[i];
  }
  *re = sr;
  *im = si;
}

static void hdot_vec(const double *ar, const double *ai, const double *br,
                     const double *bi, size_t n, double *re, double *im) {
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

static void dpc_run(size_t K, size_t M, double P, int use_vec, const double *h_re,
                    const double *h_im, const double *s_re, const double *s_im,
                    double *q_re, double *q_im, double *rdiag, double *u_re,
                    double *u_im, double *x_re, double *x_im) {
  double v_re[K * M];
  double v_im[K * M];
  double r_re[K * K];
  double r_im[K * K];

  for (size_t n = 0; n < K * K; ++n) {
    r_re[n] = 0.0;
    r_im[n] = 0.0;
  }

  // Columns of Hᴴ. Column k is conj(h_k).
  for (size_t k = 0; k < K; ++k) {
    for (size_t m = 0; m < M; ++m) {
      v_re[k * M + m] = h_re[k * M + m];
      v_im[k * M + m] = -h_im[k * M + m];
    }
  }

  for (size_t k = 0; k < K; ++k) {
    for (size_t j = 0; j < k; ++j) {
      double rr;
      double ri;
      if (use_vec)
        hdot_vec(&q_re[j * M], &q_im[j * M], &v_re[k * M], &v_im[k * M], M, &rr, &ri);
      else
        hdot_gold(&q_re[j * M], &q_im[j * M], &v_re[k * M], &v_im[k * M], M, &rr, &ri);
      r_re[j * K + k] = rr;
      r_im[j * K + k] = ri;
      for (size_t m = 0; m < M; ++m) {
        double qr = q_re[j * M + m];
        double qi = q_im[j * M + m];
        v_re[k * M + m] -= qr * rr - qi * ri;
        v_im[k * M + m] -= qr * ri + qi * rr;
      }
    }
    double nr;
    double ni;
    if (use_vec)
      hdot_vec(&v_re[k * M], &v_im[k * M], &v_re[k * M], &v_im[k * M], M, &nr, &ni);
    else
      hdot_gold(&v_re[k * M], &v_im[k * M], &v_re[k * M], &v_im[k * M], M, &nr, &ni);
    double rkk = sqrt(nr);
    rdiag[k] = rkk;
    r_re[k * K + k] = rkk;
    r_im[k * K + k] = 0.0;
    double inv = 1.0 / rkk;
    for (size_t m = 0; m < M; ++m) {
      q_re[k * M + m] = v_re[k * M + m] * inv;
      q_im[k * M + m] = v_im[k * M + m] * inv;
    }
  }

  // Successive cancellation. User 0 is encoded first and has nothing to cancel.
  for (size_t i = 0; i < K; ++i) {
    double ur = s_re[i];
    double ui = s_im[i];
    for (size_t j = 0; j < i; ++j) {
      double cr = r_re[j * K + i] / rdiag[i];
      double ci = -r_im[j * K + i] / rdiag[i];
      double urj = u_re[j];
      double uij = u_im[j];
      ur -= cr * urj - ci * uij;
      ui -= cr * uij + ci * urj;
    }
    u_re[i] = modulo_tau(ur);
    u_im[i] = modulo_tau(ui);
  }

  double power = 0.0;
  for (size_t m = 0; m < M; ++m) {
    double xr = 0.0;
    double xi = 0.0;
    for (size_t k = 0; k < K; ++k) {
      double qr = q_re[k * M + m];
      double qi = q_im[k * M + m];
      xr += qr * u_re[k] - qi * u_im[k];
      xi += qr * u_im[k] + qi * u_re[k];
    }
    x_re[m] = xr;
    x_im[m] = xi;
    power += xr * xr + xi * xi;
  }
  double scale = sqrt(P / power);
  for (size_t m = 0; m < M; ++m) {
    x_re[m] *= scale;
    x_im[m] *= scale;
  }
}

// Lower Cholesky factor of a Hermitian positive-definite matrix, A = L Lᴴ.
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

// Solve Lᴴ z = u. L is lower triangular with a real diagonal.
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

static void dpc_ll_run(size_t K, size_t M, double P, int use_vec, const double *h_re,
                       const double *h_im, const double *s_re, const double *s_im,
                       double *rdiag, double *u_re, double *u_im, double *x_re,
                       double *x_im) {
  double a_re[K * K];
  double a_im[K * K];
  double l_re[K * K];
  double l_im[K * K];
  double z_re[K];
  double z_im[K];

  // Upper triangle of G = H Hᴴ. G = L Lᴴ and, for the thin QR Hᴴ = Q R, L = Rᴴ.
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = i; j < K; ++j) {
      double re;
      double im;
      if (use_vec)
        hdot_vec(&h_re[j * M], &h_im[j * M], &h_re[i * M], &h_im[i * M], M, &re, &im);
      else
        hdot_gold(&h_re[j * M], &h_im[j * M], &h_re[i * M], &h_im[i * M], M, &re, &im);
      a_re[i * K + j] = re;
      a_im[i * K + j] = im;
    }
    a_im[i * K + i] = 0.0;
  }
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < i; ++j) {
      a_re[i * K + j] = a_re[j * K + i];
      a_im[i * K + j] = -a_im[j * K + i];
    }
  }

  herm_chol(K, a_re, a_im, l_re, l_im);
  for (size_t i = 0; i < K; ++i)
    rdiag[i] = l_re[i * K + i];

  for (size_t i = 0; i < K; ++i) {
    double ur = s_re[i];
    double ui = s_im[i];
    double inv = 1.0 / rdiag[i];
    for (size_t j = 0; j < i; ++j) {
      double cr = l_re[i * K + j] * inv;
      double ci = l_im[i * K + j] * inv;
      double urj = u_re[j];
      double uij = u_im[j];
      ur -= cr * urj - ci * uij;
      ui -= cr * uij + ci * urj;
    }
    u_re[i] = modulo_tau(ur);
    u_im[i] = modulo_tau(ui);
  }

  // x = Hᴴ (Lᴴ)⁻¹ u, which equals Q u from the QR form.
  chol_back(K, l_re, l_im, u_re, u_im, z_re, z_im);

  double power = 0.0;
  for (size_t m = 0; m < M; ++m) {
    double xr = 0.0;
    double xi = 0.0;
    for (size_t k = 0; k < K; ++k) {
      double hr = h_re[k * M + m];
      double hi = h_im[k * M + m];
      xr += hr * z_re[k] + hi * z_im[k];
      xi += hr * z_im[k] - hi * z_re[k];
    }
    x_re[m] = xr;
    x_im[m] = xi;
    power += xr * xr + xi * xi;
  }
  double scale = sqrt(P / power);
  for (size_t m = 0; m < M; ++m) {
    x_re[m] *= scale;
    x_im[m] *= scale;
  }
}

static size_t tile_step(size_t n, size_t tile) { return (tile == 0 || tile > n) ? n : tile; }

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

// w += conj(h) * a over one antenna panel.
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

// v -= q * r on one antenna panel. r is a complex scalar.
static void saxpy_panel(double *vr, double *vi, const double *qr, const double *qi, double rr,
                        double ri, size_t n) {
#if !defined(__riscv)
  for (size_t m = 0; m < n; ++m) {
    vr[m] -= qr[m] * rr - qi[m] * ri;
    vi[m] -= qr[m] * ri + qi[m] * rr;
  }
#else
  size_t vl = 0;
  for (size_t avl = n; avl > 0; avl -= vl) {
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle64.v v0, (%0)" ::"r"(qr));
    asm volatile("vle64.v v4, (%0)" ::"r"(qi));
    asm volatile("vle64.v v8, (%0)" ::"r"(vr));
    asm volatile("vle64.v v12, (%0)" ::"r"(vi));
    asm volatile("vfnmsac.vf v8, %0, v0" ::"f"(rr));
    asm volatile("vfmacc.vf v8, %0, v4" ::"f"(ri));
    asm volatile("vfnmsac.vf v12, %0, v0" ::"f"(ri));
    asm volatile("vfnmsac.vf v12, %0, v4" ::"f"(rr));
    asm volatile("vse64.v v8, (%0)" ::"r"(vr));
    asm volatile("vse64.v v12, (%0)" ::"r"(vi));
    qr += vl;
    qi += vl;
    vr += vl;
    vi += vl;
  }
#endif
}

static void dpc_tiled_run(size_t K, size_t M, double P, size_t tile, const double *h_re,
                          const double *h_im, const double *s_re, const double *s_im,
                          double *q_re, double *q_im, double *rdiag, double *u_re,
                          double *u_im, double *x_re, double *x_im) {
  double v_re[K * M];
  double v_im[K * M];
  double r_re[K * K];
  double r_im[K * K];
  size_t step_m = tile_step(M, tile);
  size_t step_k = tile_step(K, tile);

  for (size_t n = 0; n < K * K; ++n) {
    r_re[n] = 0.0;
    r_im[n] = 0.0;
  }
  for (size_t k = 0; k < K; ++k) {
    for (size_t m = 0; m < M; ++m) {
      v_re[k * M + m] = h_re[k * M + m];
      v_im[k * M + m] = -h_im[k * M + m];
    }
  }

  for (size_t k0 = 0; k0 < K; k0 += step_k) {
    size_t nk = K - k0 < step_k ? K - k0 : step_k;
    for (size_t kk = 0; kk < nk; ++kk) {
      size_t k = k0 + kk;
      for (size_t j = 0; j < k; ++j) {
        hdot_panels(&q_re[j * M], &q_im[j * M], &v_re[k * M], &v_im[k * M], M, tile,
                    &r_re[j * K + k], &r_im[j * K + k]);
        for (size_t m0 = 0; m0 < M; m0 += step_m) {
          size_t tm = M - m0 < step_m ? M - m0 : step_m;
          saxpy_panel(&v_re[k * M + m0], &v_im[k * M + m0], &q_re[j * M + m0],
                      &q_im[j * M + m0], r_re[j * K + k], r_im[j * K + k], tm);
        }
      }
      double nr;
      double ni;
      hdot_panels(&v_re[k * M], &v_im[k * M], &v_re[k * M], &v_im[k * M], M, tile, &nr, &ni);
      double rkk = sqrt(nr);
      rdiag[k] = rkk;
      r_re[k * K + k] = rkk;
      r_im[k * K + k] = 0.0;
      double inv = 1.0 / rkk;
      for (size_t m0 = 0; m0 < M; m0 += step_m) {
        size_t tm = M - m0 < step_m ? M - m0 : step_m;
        for (size_t m = 0; m < tm; ++m) {
          q_re[k * M + m0 + m] = v_re[k * M + m0 + m] * inv;
          q_im[k * M + m0 + m] = v_im[k * M + m0 + m] * inv;
        }
      }
    }
  }

  for (size_t i0 = 0; i0 < K; i0 += step_k) {
    size_t ni = K - i0 < step_k ? K - i0 : step_k;
    for (size_t ii = 0; ii < ni; ++ii) {
      size_t i = i0 + ii;
      double ur = s_re[i];
      double ui = s_im[i];
      for (size_t j = 0; j < i; ++j) {
        double cr = r_re[j * K + i] / rdiag[i];
        double ci = -r_im[j * K + i] / rdiag[i];
        ur -= cr * u_re[j] - ci * u_im[j];
        ui -= cr * u_im[j] + ci * u_re[j];
      }
      u_re[i] = modulo_tau(ur);
      u_im[i] = modulo_tau(ui);
    }
  }

  double power = 0.0;
  for (size_t m0 = 0; m0 < M; m0 += step_m) {
    size_t tm = M - m0 < step_m ? M - m0 : step_m;
    for (size_t m = 0; m < tm; ++m) {
      double xr = 0.0;
      double xi = 0.0;
      for (size_t k0 = 0; k0 < K; k0 += step_k) {
        size_t nk = K - k0 < step_k ? K - k0 : step_k;
        for (size_t kk = 0; kk < nk; ++kk) {
          size_t k = k0 + kk;
          double qr = q_re[k * M + m0 + m];
          double qi = q_im[k * M + m0 + m];
          xr += qr * u_re[k] - qi * u_im[k];
          xi += qr * u_im[k] + qi * u_re[k];
        }
      }
      x_re[m0 + m] = xr;
      x_im[m0 + m] = xi;
      power += xr * xr + xi * xi;
    }
  }
  double scale = sqrt(P / power);
  for (size_t m = 0; m < M; ++m) {
    x_re[m] *= scale;
    x_im[m] *= scale;
  }
}

static void dpc_ll_tiled_run(size_t K, size_t M, double P, size_t tile, const double *h_re,
                             const double *h_im, const double *s_re, const double *s_im,
                             double *rdiag, double *u_re, double *u_im, double *x_re,
                             double *x_im) {
  double a_re[K * K];
  double a_im[K * K];
  double l_re[K * K];
  double l_im[K * K];
  double z_re[K];
  double z_im[K];
  size_t step_k = tile_step(K, tile);
  size_t step_m = tile_step(M, tile);

  for (size_t n = 0; n < K * K; ++n) {
    a_re[n] = 0.0;
    a_im[n] = 0.0;
  }
  for (size_t i0 = 0; i0 < K; i0 += step_k) {
    size_t ni = K - i0 < step_k ? K - i0 : step_k;
    for (size_t j0 = 0; j0 < K; j0 += step_k) {
      size_t nj = K - j0 < step_k ? K - j0 : step_k;
      for (size_t ii = 0; ii < ni; ++ii) {
        for (size_t jj = 0; jj < nj; ++jj) {
          size_t i = i0 + ii;
          size_t j = j0 + jj;
          if (j < i)
            continue;
          hdot_panels(&h_re[j * M], &h_im[j * M], &h_re[i * M], &h_im[i * M], M, tile,
                      &a_re[i * K + j], &a_im[i * K + j]);
        }
      }
    }
  }
  for (size_t i = 0; i < K; ++i)
    a_im[i * K + i] = 0.0;
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < i; ++j) {
      a_re[i * K + j] = a_re[j * K + i];
      a_im[i * K + j] = -a_im[j * K + i];
    }
  }

  herm_chol(K, a_re, a_im, l_re, l_im);
  for (size_t i = 0; i < K; ++i)
    rdiag[i] = l_re[i * K + i];

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

  double power = 0.0;
  for (size_t m0 = 0; m0 < M; m0 += step_m) {
    size_t tm = M - m0 < step_m ? M - m0 : step_m;
    for (size_t m = 0; m < tm; ++m) {
      x_re[m0 + m] = 0.0;
      x_im[m0 + m] = 0.0;
    }
    for (size_t k0 = 0; k0 < K; k0 += step_k) {
      size_t nk = K - k0 < step_k ? K - k0 : step_k;
      for (size_t kk = 0; kk < nk; ++kk) {
        size_t k = k0 + kk;
        caxpy_conj_panel(&h_re[k * M + m0], &h_im[k * M + m0], z_re[k], z_im[k],
                         &x_re[m0], &x_im[m0], tm);
      }
    }
    for (size_t m = 0; m < tm; ++m)
      power += x_re[m0 + m] * x_re[m0 + m] + x_im[m0 + m] * x_im[m0 + m];
  }
  double scale = sqrt(P / power);
  for (size_t m = 0; m < M; ++m) {
    x_re[m] *= scale;
    x_im[m] *= scale;
  }
}

void dpc_vec_tile(size_t K, size_t N, size_t M, double P, size_t tile, const double *h_re,
                  const double *h_im, const double *s_re, const double *s_im, double *q_re,
                  double *q_im, double *rdiag, double *u_re, double *u_im, double *x_re,
                  double *x_im) {
  dpc_tiled_run(K * N, M, P, tile, h_re, h_im, s_re, s_im, q_re, q_im, rdiag, u_re, u_im,
                x_re, x_im);
}

void dpc_ll_vec_tile(size_t K, size_t N, size_t M, double P, size_t tile, const double *h_re,
                     const double *h_im, const double *s_re, const double *s_im,
                     double *rdiag, double *u_re, double *u_im, double *x_re, double *x_im) {
  dpc_ll_tiled_run(K * N, M, P, tile, h_re, h_im, s_re, s_im, rdiag, u_re, u_im, x_re, x_im);
}

void dpc_ll_gold(size_t K, size_t N, size_t M, double P, const double *h_re,
                 const double *h_im, const double *s_re, const double *s_im,
                 double *rdiag, double *u_re, double *u_im, double *x_re, double *x_im) {
  dpc_ll_run(K * N, M, P, 0, h_re, h_im, s_re, s_im, rdiag, u_re, u_im, x_re, x_im);
}

void dpc_ll_vec(size_t K, size_t N, size_t M, double P, const double *h_re,
                const double *h_im, const double *s_re, const double *s_im, double *rdiag,
                double *u_re, double *u_im, double *x_re, double *x_im) {
  dpc_ll_run(K * N, M, P, 1, h_re, h_im, s_re, s_im, rdiag, u_re, u_im, x_re, x_im);
}

void dpc_gold(size_t K, size_t N, size_t M, double P, const double *h_re,
              const double *h_im, const double *s_re, const double *s_im, double *q_re,
              double *q_im, double *rdiag, double *u_re, double *u_im, double *x_re,
              double *x_im) {
  dpc_run(K * N, M, P, 0, h_re, h_im, s_re, s_im, q_re, q_im, rdiag, u_re, u_im, x_re,
          x_im);
}

void dpc_vec(size_t K, size_t N, size_t M, double P, const double *h_re,
             const double *h_im, const double *s_re, const double *s_im, double *q_re,
             double *q_im, double *rdiag, double *u_re, double *u_im, double *x_re,
             double *x_im) {
  dpc_run(K * N, M, P, 1, h_re, h_im, s_re, s_im, q_re, q_im, rdiag, u_re, u_im, x_re,
          x_im);
}
