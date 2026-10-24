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

// Sphere decoder for QPSK. Thin QR reduces
//   argmin ||y - H x||²
// to
//   argmin ||Qᴴ y - R x||²,  x in S^{KN}.
// A Babai point sets the first radius. Schnorr–Euchner enumeration then
// searches the tree from layer KN-1 down to 0. The projected residual is
// updated incrementally when a symbol is written, so each node is O(layer)
// rather than a fresh sum over the already-decided symbols.

#include "detection.h"
#include "linalg.h"

static const double kQpskRe[4] = {0.7071067811865476, -0.7071067811865476,
                                  0.7071067811865476, -0.7071067811865476};
static const double kQpskIm[4] = {0.7071067811865476, 0.7071067811865476,
                                  -0.7071067811865476, -0.7071067811865476};

static double metric_inc(double pr, double pi, double rii, double xr, double xi) {
  double dr = pr - rii * xr;
  double di = pi - rii * xi;
  return dr * dr + di * di;
}

// SE order for real-positive R_ii: closest QPSK, then flip the I/Q component
// with the smaller |partial|, then the other, then both.
static void se_order(double pr, double pi, int *ord) {
  int c0 = (pr < 0.0) + 2 * (pi < 0.0);
  int flip_re = c0 ^ 1;
  int flip_im = c0 ^ 2;
  ord[0] = c0;
  if ((pr < 0.0 ? -pr : pr) <= (pi < 0.0 ? -pi : pi)) {
    ord[1] = flip_re;
    ord[2] = flip_im;
  } else {
    ord[1] = flip_im;
    ord[2] = flip_re;
  }
  ord[3] = c0 ^ 3;
}

static void cancel_layer(size_t i, size_t S, const double *r_re, const double *r_im,
                         double xr, double xi, double sign, double *res_re,
                         double *res_im) {
  for (size_t k = 0; k < i; ++k) {
    double rr = r_re[k * S + i];
    double ri = r_im[k * S + i];
    res_re[k] += sign * (rr * xr - ri * xi);
    res_im[k] += sign * (rr * xi + ri * xr);
  }
}

static void sd_search(size_t i, size_t S, double dist, const double *r_re, const double *r_im,
                      double *res_re, double *res_im, double *x_re, double *x_im,
                      double *best_re, double *best_im, double *r2) {
  double pr = res_re[i];
  double pi = res_im[i];
  double rii = r_re[i * S + i];
  int ord[4];
  se_order(pr, pi, ord);
  for (int t = 0; t < 4; ++t) {
    int c = ord[t];
    double xr = kQpskRe[c];
    double xi = kQpskIm[c];
    double ndist = dist + metric_inc(pr, pi, rii, xr, xi);
    if (ndist > *r2)
      continue;
    x_re[i] = xr;
    x_im[i] = xi;
    if (i == 0) {
      *r2 = ndist;
      for (size_t s = 0; s < S; ++s) {
        best_re[s] = x_re[s];
        best_im[s] = x_im[s];
      }
    } else {
      cancel_layer(i, S, r_re, r_im, xr, xi, -1.0, res_re, res_im);
      sd_search(i - 1, S, ndist, r_re, r_im, res_re, res_im, x_re, x_im, best_re, best_im,
                r2);
      cancel_layer(i, S, r_re, r_im, xr, xi, 1.0, res_re, res_im);
    }
  }
}

static double babai(size_t S, const double *r_re, const double *r_im, const double *yt_re,
                    const double *yt_im, double *x_re, double *x_im, double *res_re,
                    double *res_im) {
  for (size_t s = 0; s < S; ++s) {
    res_re[s] = yt_re[s];
    res_im[s] = yt_im[s];
  }
  double dist = 0.0;
  for (size_t i = S; i-- > 0;) {
    double pr = res_re[i];
    double pi = res_im[i];
    double rii = r_re[i * S + i];
    double zr = pr / rii;
    double zi = pi / rii;
    slice_qpsk(&zr, &zi, &x_re[i], &x_im[i], 1);
    dist += metric_inc(pr, pi, rii, x_re[i], x_im[i]);
    cancel_layer(i, S, r_re, r_im, x_re[i], x_im[i], -1.0, res_re, res_im);
  }
  return dist;
}

static void layer_partial(size_t i, size_t S, const double *r_re, const double *r_im,
                          const double *yt_re, const double *yt_im, const double *x_re,
                          const double *x_im, double *pr, double *pi) {
  double sr = yt_re[i];
  double si = yt_im[i];
  for (size_t j = i + 1; j < S; ++j) {
    double rr = r_re[i * S + j];
    double ri = r_im[i * S + j];
    sr -= rr * x_re[j] - ri * x_im[j];
    si -= rr * x_im[j] + ri * x_re[j];
  }
  *pr = sr;
  *pi = si;
}

static void se_order_sort(double pr, double pi, double rii, int *ord) {
  double d[4];
  for (int c = 0; c < 4; ++c)
    d[c] = metric_inc(pr, pi, rii, kQpskRe[c], kQpskIm[c]);
  ord[0] = 0;
  ord[1] = 1;
  ord[2] = 2;
  ord[3] = 3;
  for (int a = 0; a < 3; ++a) {
    for (int b = a + 1; b < 4; ++b) {
      if (d[ord[b]] < d[ord[a]]) {
        int tmp = ord[a];
        ord[a] = ord[b];
        ord[b] = tmp;
      }
    }
  }
}

static void sd_search_base(size_t i, size_t S, double dist, const double *r_re,
                           const double *r_im, const double *yt_re, const double *yt_im,
                           double *x_re, double *x_im, double *best_re, double *best_im,
                           double *r2) {
  double pr;
  double pi;
  int ord[4];
  layer_partial(i, S, r_re, r_im, yt_re, yt_im, x_re, x_im, &pr, &pi);
  double rii = r_re[i * S + i];
  se_order_sort(pr, pi, rii, ord);
  for (int t = 0; t < 4; ++t) {
    int c = ord[t];
    double ndist = dist + metric_inc(pr, pi, rii, kQpskRe[c], kQpskIm[c]);
    if (ndist > *r2)
      continue;
    x_re[i] = kQpskRe[c];
    x_im[i] = kQpskIm[c];
    if (i == 0) {
      *r2 = ndist;
      for (size_t s = 0; s < S; ++s) {
        best_re[s] = x_re[s];
        best_im[s] = x_im[s];
      }
    } else {
      sd_search_base(i - 1, S, ndist, r_re, r_im, yt_re, yt_im, x_re, x_im, best_re,
                     best_im, r2);
    }
  }
}

static void sd_run(size_t K, size_t N, size_t M, int use_vec, const double *h_re,
                   const double *h_im, const double *y_re, const double *y_im, double *x_re,
                   double *x_im) {
  size_t S = K * N;
  double q_re[S * M];
  double q_im[S * M];
  double r_re[S * S];
  double r_im[S * S];
  double yt_re[S];
  double yt_im[S];
  double cur_re[S];
  double cur_im[S];
  double best_re[S];
  double best_im[S];

  thin_qr(S, M, use_vec, h_re, h_im, q_re, q_im, r_re, r_im);
  for (size_t s = 0; s < S; ++s) {
    if (use_vec)
      hdot_vec(&q_re[s * M], &q_im[s * M], y_re, y_im, M, &yt_re[s], &yt_im[s]);
    else
      hdot_gold(&q_re[s * M], &q_im[s * M], y_re, y_im, M, &yt_re[s], &yt_im[s]);
  }

  double dist = 0.0;
  for (size_t i = S; i-- > 0;) {
    double pr;
    double pi;
    layer_partial(i, S, r_re, r_im, yt_re, yt_im, best_re, best_im, &pr, &pi);
    double rii = r_re[i * S + i];
    double zr = pr / rii;
    double zi = pi / rii;
    slice_qpsk(&zr, &zi, &best_re[i], &best_im[i], 1);
    dist += metric_inc(pr, pi, rii, best_re[i], best_im[i]);
  }
  for (size_t s = 0; s < S; ++s) {
    cur_re[s] = best_re[s];
    cur_im[s] = best_im[s];
  }
  sd_search_base(S - 1, S, 0.0, r_re, r_im, yt_re, yt_im, cur_re, cur_im, best_re, best_im,
                 &dist);
  for (size_t s = 0; s < S; ++s) {
    x_re[s] = best_re[s];
    x_im[s] = best_im[s];
  }
}

static void sd_run_ll(size_t K, size_t N, size_t M, int use_vec, const double *h_re,
                      const double *h_im, const double *y_re, const double *y_im,
                      double *x_re, double *x_im) {
  size_t S = K * N;
  double q_re[S * M];
  double q_im[S * M];
  double r_re[S * S];
  double r_im[S * S];
  double yt_re[S];
  double yt_im[S];
  double res_re[S];
  double res_im[S];
  double cur_re[S];
  double cur_im[S];
  double best_re[S];
  double best_im[S];

  thin_qr_ll(S, M, use_vec, h_re, h_im, y_re, y_im, q_re, q_im, r_re, r_im, yt_re, yt_im);

  double r2 = babai(S, r_re, r_im, yt_re, yt_im, best_re, best_im, res_re, res_im);
  for (size_t s = 0; s < S; ++s) {
    cur_re[s] = best_re[s];
    cur_im[s] = best_im[s];
    res_re[s] = yt_re[s];
    res_im[s] = yt_im[s];
  }
  sd_search(S - 1, S, 0.0, r_re, r_im, res_re, res_im, cur_re, cur_im, best_re, best_im,
            &r2);
  for (size_t s = 0; s < S; ++s) {
    x_re[s] = best_re[s];
    x_im[s] = best_im[s];
  }
}

void sd_gold(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
             const double *y_re, const double *y_im, double *x_re, double *x_im) {
  sd_run(K, N, M, 0, h_re, h_im, y_re, y_im, x_re, x_im);
}

void sd_vec(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
            const double *y_re, const double *y_im, double *x_re, double *x_im) {
  sd_run(K, N, M, 1, h_re, h_im, y_re, y_im, x_re, x_im);
}

void sd_ll_gold(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
                const double *y_re, const double *y_im, double *x_re, double *x_im) {
  sd_run_ll(K, N, M, 0, h_re, h_im, y_re, y_im, x_re, x_im);
}

void sd_ll_vec(size_t K, size_t N, size_t M, const double *h_re, const double *h_im,
               const double *y_re, const double *y_im, double *x_re, double *x_im) {
  sd_run_ll(K, N, M, 1, h_re, h_im, y_re, y_im, x_re, x_im);
}
