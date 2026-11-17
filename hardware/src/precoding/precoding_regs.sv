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

// AXI-Lite register file for the precoding coprocessor. A write of CTRL[0]
// pulses start for one cycle. STATUS and CYCLES are owned by the datapath.

module precoding_regs import precoding_pkg::*; #(
  parameter int unsigned DataWidth = 64,
  parameter type lite_req_t  = logic,
  parameter type lite_resp_t = logic
) (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  lite_req_t    req_i,
  output lite_resp_t   resp_o,
  output prec_cfg_t    cfg_o,
  output logic         start_o,
  input  logic         busy_i,
  input  logic         done_i,
  input  logic         err_i,
  input  logic [63:0]  cycles_i
);

  localparam int unsigned StrbWidth = DataWidth / 8;

  prec_cfg_t cfg_q;
  logic aw_ok_q, w_ok_q, b_valid_q;
  logic [63:0] aw_addr_q;
  logic [DataWidth-1:0] wdata_q;
  logic [StrbWidth-1:0] wstrb_q;
  logic ar_ok_q, r_valid_q;
  logic [63:0] ar_addr_q;
  logic [DataWidth-1:0] rdata_q;

  logic [11:0] woff, roff;
  logic [DataWidth-1:0] wdata_d, cfg_bits;
  logic start_d;

  assign cfg_o   = cfg_q;
  assign start_o = start_d;
  assign woff    = aw_addr_q[11:0];
  assign roff    = ar_addr_q[11:0];

  function automatic logic [DataWidth-1:0] apply_strb(
    input logic [DataWidth-1:0] old_val,
    input logic [DataWidth-1:0] new_val,
    input logic [StrbWidth-1:0] strb
  );
    for (int b = 0; b < StrbWidth; b++) begin
      apply_strb[8*b +: 8] = strb[b] ? new_val[8*b +: 8] : old_val[8*b +: 8];
    end
  endfunction

  always_comb begin
    resp_o          = '0;
    resp_o.aw_ready = ~aw_ok_q;
    resp_o.w_ready  = ~w_ok_q;
    resp_o.b_valid  = b_valid_q;
    resp_o.b.resp   = axi_pkg::RESP_OKAY;
    resp_o.ar_ready = ~ar_ok_q && ~r_valid_q;
    resp_o.r_valid  = r_valid_q;
    resp_o.r.data   = rdata_q;
    resp_o.r.resp   = axi_pkg::RESP_OKAY;

    start_d  = 1'b0;
    wdata_d  = '0;
    cfg_bits = '0;

    if (aw_ok_q && w_ok_q) begin
      unique case (woff)
        PREC_H_RE:  cfg_bits = cfg_q.h_re;
        PREC_H_IM:  cfg_bits = cfg_q.h_im;
        PREC_W_RE:  cfg_bits = cfg_q.w_re;
        PREC_W_IM:  cfg_bits = cfg_q.w_im;
        PREC_S_RE:  cfg_bits = cfg_q.s_re;
        PREC_S_IM:  cfg_bits = cfg_q.s_im;
        PREC_X_RE:  cfg_bits = cfg_q.x_re;
        PREC_X_IM:  cfg_bits = cfg_q.x_im;
        PREC_P:     cfg_bits = cfg_q.p;
        PREC_ALPHA: cfg_bits = cfg_q.alpha;
        PREC_DIMS:  cfg_bits = {16'd0, cfg_q.m, cfg_q.n, cfg_q.k};
        default:    cfg_bits = '0;
      endcase
      wdata_d = apply_strb(cfg_bits, wdata_q, wstrb_q);
      if (woff == PREC_CTRL && wstrb_q[0] && wdata_q[0] && !busy_i) start_d = 1'b1;
    end

  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cfg_q      <= '0;
      aw_ok_q    <= 1'b0;
      w_ok_q     <= 1'b0;
      b_valid_q  <= 1'b0;
      aw_addr_q  <= '0;
      wdata_q    <= '0;
      wstrb_q    <= '0;
      ar_ok_q    <= 1'b0;
      r_valid_q  <= 1'b0;
      ar_addr_q  <= '0;
      rdata_q    <= '0;
    end else begin
      if (req_i.aw_valid && resp_o.aw_ready) begin
        aw_ok_q   <= 1'b1;
        aw_addr_q <= req_i.aw.addr;
      end
      if (req_i.w_valid && resp_o.w_ready) begin
        w_ok_q  <= 1'b1;
        wdata_q <= req_i.w.data;
        wstrb_q <= req_i.w.strb;
      end
      if (aw_ok_q && w_ok_q && !b_valid_q) begin
        b_valid_q <= 1'b1;
        aw_ok_q   <= 1'b0;
        w_ok_q    <= 1'b0;
        unique case (woff)
          PREC_H_RE:  cfg_q.h_re  <= wdata_d;
          PREC_H_IM:  cfg_q.h_im  <= wdata_d;
          PREC_W_RE:  cfg_q.w_re  <= wdata_d;
          PREC_W_IM:  cfg_q.w_im  <= wdata_d;
          PREC_S_RE:  cfg_q.s_re  <= wdata_d;
          PREC_S_IM:  cfg_q.s_im  <= wdata_d;
          PREC_X_RE:  cfg_q.x_re  <= wdata_d;
          PREC_X_IM:  cfg_q.x_im  <= wdata_d;
          PREC_P:     cfg_q.p     <= wdata_d;
          PREC_ALPHA: cfg_q.alpha <= wdata_d;
          PREC_DIMS: begin
            cfg_q.k <= wdata_d[15:0];
            cfg_q.n <= wdata_d[31:16];
            cfg_q.m <= wdata_d[47:32];
          end
          default: ;
        endcase
      end
      if (b_valid_q && req_i.b_ready) b_valid_q <= 1'b0;

      if (req_i.ar_valid && resp_o.ar_ready) begin
        ar_ok_q   <= 1'b1;
        ar_addr_q <= req_i.ar.addr;
      end
      if (ar_ok_q && !r_valid_q) begin
        ar_ok_q   <= 1'b0;
        r_valid_q <= 1'b1;
        unique case (roff)
          PREC_CTRL:   rdata_q <= '0;
          PREC_STATUS: rdata_q <= {61'd0, err_i, done_i, busy_i};
          PREC_H_RE:   rdata_q <= cfg_q.h_re;
          PREC_H_IM:   rdata_q <= cfg_q.h_im;
          PREC_W_RE:   rdata_q <= cfg_q.w_re;
          PREC_W_IM:   rdata_q <= cfg_q.w_im;
          PREC_S_RE:   rdata_q <= cfg_q.s_re;
          PREC_S_IM:   rdata_q <= cfg_q.s_im;
          PREC_X_RE:   rdata_q <= cfg_q.x_re;
          PREC_X_IM:   rdata_q <= cfg_q.x_im;
          PREC_DIMS:   rdata_q <= {16'd0, cfg_q.m, cfg_q.n, cfg_q.k};
          PREC_P:      rdata_q <= cfg_q.p;
          PREC_ALPHA:  rdata_q <= cfg_q.alpha;
          PREC_CYCLES: rdata_q <= cycles_i;
          default:     rdata_q <= '0;
        endcase
      end
      if (r_valid_q && req_i.r_ready) r_valid_q <= 1'b0;
    end
  end

endmodule
