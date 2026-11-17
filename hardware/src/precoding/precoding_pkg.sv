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

// Precoding coprocessor register map. The unit is an AXI slave at PREC_BASE.
// It reads H from L2, forms the regularized Gram, factorizes it, and writes
// the sum-power precoder W. The vector unit is not on this path.
//
//   W = sqrt(P / ||V||_F^2) V,  V = H^H (H H^H + alpha I)^{-1}
//
// alpha = 0 is zero-forcing. A symbol apply x = W s runs when both symbol
// pointers and both transmit pointers are non-zero.

package precoding_pkg;
  localparam logic [63:0] PREC_BASE   = 64'hE000_0000;
  localparam logic [63:0] PREC_LENGTH = 64'h0000_1000;

  localparam int unsigned PREC_MAX_S = 16;
  localparam int unsigned PREC_MAX_M = 64;

  localparam logic [11:0] PREC_CTRL   = 12'h00;
  localparam logic [11:0] PREC_STATUS = 12'h08;
  localparam logic [11:0] PREC_H_RE   = 12'h10;
  localparam logic [11:0] PREC_H_IM   = 12'h18;
  localparam logic [11:0] PREC_W_RE   = 12'h20;
  localparam logic [11:0] PREC_W_IM   = 12'h28;
  localparam logic [11:0] PREC_S_RE   = 12'h30;
  localparam logic [11:0] PREC_S_IM   = 12'h38;
  localparam logic [11:0] PREC_X_RE   = 12'h40;
  localparam logic [11:0] PREC_X_IM   = 12'h48;
  localparam logic [11:0] PREC_DIMS   = 12'h50;
  localparam logic [11:0] PREC_P      = 12'h58;
  localparam logic [11:0] PREC_ALPHA  = 12'h60;
  localparam logic [11:0] PREC_CYCLES = 12'h68;

  typedef struct packed {
    logic [63:0] h_re;
    logic [63:0] h_im;
    logic [63:0] w_re;
    logic [63:0] w_im;
    logic [63:0] s_re;
    logic [63:0] s_im;
    logic [63:0] x_re;
    logic [63:0] x_im;
    logic [63:0] p;
    logic [63:0] alpha;
    logic [15:0] k;
    logic [15:0] n;
    logic [15:0] m;
  } prec_cfg_t;
endpackage
