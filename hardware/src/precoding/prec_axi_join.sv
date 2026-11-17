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

// Two AXI masters, one port. Ara already drives every ID to zero, and the
// coprocessor does the same, so this join does not widen IDs. A master keeps
// the bus until its outstanding reads and writes have drained.

module prec_axi_join #(
  parameter type req_t  = logic,
  parameter type resp_t = logic
) (
  input  logic  clk_i,
  input  logic  rst_ni,
  input  req_t  ara_req_i,
  output resp_t ara_resp_o,
  input  req_t  prec_req_i,
  output resp_t prec_resp_o,
  output req_t  mst_req_o,
  input  resp_t mst_resp_i
);

  localparam logic [1:0] OWN_NONE = 2'd0;
  localparam logic [1:0] OWN_ARA  = 2'd1;
  localparam logic [1:0] OWN_PREC = 2'd2;

  logic [1:0] rd_own_q, rd_own_d;
  logic [1:0] wr_own_q, wr_own_d;
  logic [8:0] rd_cnt_q, rd_cnt_d;
  logic [8:0] wr_cnt_q, wr_cnt_d;

  req_t  mst_req;
  resp_t ara_resp, prec_resp;

  logic mst_ar_hs, mst_r_hs, mst_aw_hs, mst_b_hs;

  assign mst_req_o   = mst_req;
  assign ara_resp_o  = ara_resp;
  assign prec_resp_o = prec_resp;

  always_comb begin
    rd_own_d = rd_own_q;
    wr_own_d = wr_own_q;
    rd_cnt_d = rd_cnt_q;
    wr_cnt_d = wr_cnt_q;

    mst_req         = '0;
    ara_resp        = '0;
    prec_resp       = '0;
    ara_resp.r      = mst_resp_i.r;
    prec_resp.r     = mst_resp_i.r;
    ara_resp.b      = mst_resp_i.b;
    prec_resp.b     = mst_resp_i.b;

    if (rd_own_q == OWN_NONE) begin
      if (ara_req_i.ar_valid) rd_own_d = OWN_ARA;
      else if (prec_req_i.ar_valid) rd_own_d = OWN_PREC;
    end
    if (wr_own_q == OWN_NONE) begin
      if (ara_req_i.aw_valid || ara_req_i.w_valid) wr_own_d = OWN_ARA;
      else if (prec_req_i.aw_valid || prec_req_i.w_valid) wr_own_d = OWN_PREC;
    end

    if (rd_own_d == OWN_ARA) begin
      mst_req.ar        = ara_req_i.ar;
      mst_req.ar_valid  = ara_req_i.ar_valid;
      mst_req.r_ready   = ara_req_i.r_ready;
      ara_resp.ar_ready = mst_resp_i.ar_ready;
      ara_resp.r_valid  = mst_resp_i.r_valid;
    end else if (rd_own_d == OWN_PREC) begin
      mst_req.ar         = prec_req_i.ar;
      mst_req.ar_valid   = prec_req_i.ar_valid;
      mst_req.r_ready    = prec_req_i.r_ready;
      prec_resp.ar_ready = mst_resp_i.ar_ready;
      prec_resp.r_valid  = mst_resp_i.r_valid;
    end

    if (wr_own_d == OWN_ARA) begin
      mst_req.aw        = ara_req_i.aw;
      mst_req.aw_valid  = ara_req_i.aw_valid;
      mst_req.w         = ara_req_i.w;
      mst_req.w_valid   = ara_req_i.w_valid;
      mst_req.b_ready   = ara_req_i.b_ready;
      ara_resp.aw_ready = mst_resp_i.aw_ready;
      ara_resp.w_ready  = mst_resp_i.w_ready;
      ara_resp.b_valid  = mst_resp_i.b_valid;
    end else if (wr_own_d == OWN_PREC) begin
      mst_req.aw         = prec_req_i.aw;
      mst_req.aw_valid   = prec_req_i.aw_valid;
      mst_req.w          = prec_req_i.w;
      mst_req.w_valid    = prec_req_i.w_valid;
      mst_req.b_ready    = prec_req_i.b_ready;
      prec_resp.aw_ready = mst_resp_i.aw_ready;
      prec_resp.w_ready  = mst_resp_i.w_ready;
      prec_resp.b_valid  = mst_resp_i.b_valid;
    end

    mst_ar_hs = mst_req.ar_valid && mst_resp_i.ar_ready;
    mst_r_hs  = mst_resp_i.r_valid && mst_req.r_ready && mst_resp_i.r.last;
    mst_aw_hs = mst_req.aw_valid && mst_resp_i.aw_ready;
    mst_b_hs  = mst_resp_i.b_valid && mst_req.b_ready;

    if (mst_ar_hs) rd_cnt_d = rd_cnt_d + 9'd1;
    if (mst_r_hs)  rd_cnt_d = rd_cnt_d - 9'd1;
    if (mst_aw_hs) wr_cnt_d = wr_cnt_d + 9'd1;
    if (mst_b_hs)  wr_cnt_d = wr_cnt_d - 9'd1;

    if (rd_own_d != OWN_NONE && rd_cnt_d == '0 && !mst_ar_hs &&
        !((rd_own_d == OWN_ARA) ? ara_req_i.ar_valid : prec_req_i.ar_valid)) begin
      rd_own_d = OWN_NONE;
    end
    if (wr_own_d != OWN_NONE && wr_cnt_d == '0 && !mst_aw_hs &&
        !((wr_own_d == OWN_ARA) ? (ara_req_i.aw_valid || ara_req_i.w_valid)
                               : (prec_req_i.aw_valid || prec_req_i.w_valid))) begin
      wr_own_d = OWN_NONE;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rd_own_q <= OWN_NONE;
      wr_own_q <= OWN_NONE;
      rd_cnt_q <= '0;
      wr_cnt_q <= '0;
    end else begin
      rd_own_q <= rd_own_d;
      wr_own_q <= wr_own_d;
      rd_cnt_q <= rd_cnt_d;
      wr_cnt_q <= wr_cnt_d;
    end
  end

endmodule
