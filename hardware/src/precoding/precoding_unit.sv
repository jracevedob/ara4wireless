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

// Precoding coprocessor datapath.
//
// One double is moved per AXI beat. The Gram, the Hermitian Cholesky, and the
// triangular solve each retire one arithmetic step per cycle. Those steps are
// double-precision operations, the same ones as rzf_precoder() in the kernel.
// Streams are limited to PREC_MAX_S and antennas to PREC_MAX_M.

module precoding_unit import precoding_pkg::*; import axi_pkg::*; #(
  parameter int unsigned AxiDataWidth = 64,
  parameter type         req_t        = logic,
  parameter type         resp_t       = logic
) (
  input  logic       clk_i,
  input  logic       rst_ni,
  input  prec_cfg_t  cfg_i,
  input  logic       start_i,
  output logic       busy_o,
  output logic       done_o,
  output logic       err_o,
  output logic [63:0] cycles_o,
  output req_t       dma_req_o,
  input  resp_t      dma_resp_i
);

`ifdef TARGET_GATESIM
  // The datapath below is a double-precision functional model. Gate-level
  // builds keep the port list and leave the coprocessor idle.
  assign busy_o    = 1'b0;
  assign done_o    = 1'b0;
  assign err_o     = 1'b0;
  assign cycles_o  = '0;
  assign dma_req_o = '0;
`else

  localparam int unsigned BeatBytes = AxiDataWidth / 8;
  localparam int unsigned OffW      = $clog2(BeatBytes);

  if (AxiDataWidth < 64 || (AxiDataWidth % 64) != 0)
    $error("precoding_unit requires a wide AXI bus of at least 64 bits");

  function automatic real prec_bits2real(input logic [63:0] bits);
    prec_bits2real = $bitstoreal(bits);
  endfunction

  function automatic logic [63:0] prec_real2bits(input real value);
    prec_real2bits = $realtobits(value);
  endfunction

  typedef enum logic [4:0] {
    ST_IDLE,
    ST_LD_HRE,
    ST_LD_HIM,
    ST_GRAM_INIT,
    ST_GRAM_MAC,
    ST_GRAM_ST,
    ST_CHOL_I,
    ST_CHOL_J_INIT,
    ST_CHOL_J_P,
    ST_CHOL_J_DIV,
    ST_CHOL_DIAG_INIT,
    ST_CHOL_DIAG_P,
    ST_CHOL_SQRT,
    ST_FWD_INIT,
    ST_FWD_P,
    ST_FWD_DIV,
    ST_BWD_INIT,
    ST_BWD_P,
    ST_BWD_DIV,
    ST_SCALE_INIT,
    ST_SCALE,
    ST_ST_WRE,
    ST_ST_WIM,
    ST_LD_SRE,
    ST_LD_SIM,
    ST_APPLY_INIT,
    ST_APPLY_MAC,
    ST_APPLY_ST,
    ST_ST_XRE,
    ST_ST_XIM,
    ST_DONE
  } st_e;

  typedef enum logic [2:0] {
    DMA_IDLE, DMA_AR, DMA_R, DMA_AW, DMA_W, DMA_B
  } dma_e;

  st_e  state_q, state_d;
  dma_e dma_q;

  logic [63:0] h_re [0:PREC_MAX_S*PREC_MAX_M-1];
  logic [63:0] h_im [0:PREC_MAX_S*PREC_MAX_M-1];
  logic [63:0] w_re [0:PREC_MAX_S*PREC_MAX_M-1];
  logic [63:0] w_im [0:PREC_MAX_S*PREC_MAX_M-1];
  logic [63:0] a_re [0:PREC_MAX_S*PREC_MAX_S-1];
  logic [63:0] a_im [0:PREC_MAX_S*PREC_MAX_S-1];
  logic [63:0] l_re [0:PREC_MAX_S*PREC_MAX_S-1];
  logic [63:0] l_im [0:PREC_MAX_S*PREC_MAX_S-1];
  logic [63:0] y_re [0:PREC_MAX_S-1];
  logic [63:0] y_im [0:PREC_MAX_S-1];
  logic [63:0] z_re [0:PREC_MAX_S-1];
  logic [63:0] z_im [0:PREC_MAX_S-1];
  logic [63:0] sv_re [0:PREC_MAX_S-1];
  logic [63:0] sv_im [0:PREC_MAX_S-1];
  logic [63:0] x_re [0:PREC_MAX_M-1];
  logic [63:0] x_im [0:PREC_MAX_M-1];

  logic [15:0] s_q, m_q, idx_q, i_q, j_q, p_q, ant_q;
  logic [63:0] h_re_q, h_im_q, w_re_q, w_im_q, s_re_q, s_im_q, x_re_q, x_im_q;
  logic [63:0] pwr_q, alpha_q;
  logic        apply_q, err_q;
  logic [63:0] cycles_q;
  logic [63:0] dma_addr_q, dma_wdata_q, dma_rdata_q;
  real         acc_r_q, acc_i_q, sr_q, si_q, diag_q, fro_q, scale_q;

  req_t dma_req;

  function automatic logic [AxiDataWidth-1:0] place64(
    input logic [63:0] val,
    input logic [63:0] addr
  );
    logic [OffW-1:0] off;
    place64 = '0;
    off = addr[OffW-1:0];
    for (int b = 0; b < 8; b++) begin
      place64[8*(int'(off)+b) +: 8] = val[8*b +: 8];
    end
  endfunction

  function automatic logic [AxiDataWidth/8-1:0] strb64(input logic [63:0] addr);
    logic [OffW-1:0] off;
    strb64 = '0;
    off = addr[OffW-1:0];
    for (int b = 0; b < 8; b++) begin
      strb64[int'(off)+b] = 1'b1;
    end
  endfunction

  function automatic logic [63:0] pick64(
    input logic [AxiDataWidth-1:0] beat,
    input logic [63:0] addr
  );
    logic [OffW-1:0] off;
    off = addr[OffW-1:0];
    for (int b = 0; b < 8; b++) begin
      pick64[8*b +: 8] = beat[8*(int'(off)+b) +: 8];
    end
  endfunction

  assign busy_o    = (state_q != ST_IDLE) && (state_q != ST_DONE);
  assign done_o    = state_q == ST_DONE;
  assign err_o     = err_q;
  assign cycles_o  = cycles_q;
  assign dma_req_o = dma_req;

  always_comb begin
    dma_req          = '0;
    dma_req.ar.addr  = dma_addr_q;
    dma_req.ar.len   = '0;
    dma_req.ar.size  = 3'd3;
    dma_req.ar.burst = BURST_INCR;
    dma_req.ar.cache = CACHE_MODIFIABLE;
    dma_req.aw.addr  = dma_addr_q;
    dma_req.aw.len   = '0;
    dma_req.aw.size  = 3'd3;
    dma_req.aw.burst = BURST_INCR;
    dma_req.aw.cache = CACHE_MODIFIABLE;
    dma_req.w.data   = place64(dma_wdata_q, dma_addr_q);
    dma_req.w.strb   = strb64(dma_addr_q);
    dma_req.w.last   = 1'b1;
    unique case (dma_q)
      DMA_AR: dma_req.ar_valid = 1'b1;
      DMA_R:  dma_req.r_ready  = 1'b1;
      DMA_AW: dma_req.aw_valid = 1'b1;
      DMA_W:  dma_req.w_valid  = 1'b1;
      DMA_B:  dma_req.b_ready  = 1'b1;
      default: ;
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q   <= ST_IDLE;
      dma_q     <= DMA_IDLE;
      err_q     <= 1'b0;
      cycles_q  <= '0;
      s_q       <= '0;
      m_q       <= '0;
      idx_q     <= '0;
      i_q       <= '0;
      j_q       <= '0;
      p_q       <= '0;
      ant_q     <= '0;
      apply_q   <= 1'b0;
      h_re_q    <= '0;
      h_im_q    <= '0;
      w_re_q    <= '0;
      w_im_q    <= '0;
      s_re_q    <= '0;
      s_im_q    <= '0;
      x_re_q    <= '0;
      x_im_q    <= '0;
      pwr_q     <= '0;
      alpha_q   <= '0;
      dma_addr_q  <= '0;
      dma_wdata_q <= '0;
      dma_rdata_q <= '0;
      acc_r_q   <= 0.0;
      acc_i_q   <= 0.0;
      sr_q      <= 0.0;
      si_q      <= 0.0;
      diag_q    <= 0.0;
      fro_q     <= 0.0;
      scale_q   <= 0.0;
    end else begin
      if (busy_o) cycles_q <= cycles_q + 64'd1;

      unique case (state_q)
        ST_IDLE, ST_DONE: begin
          if (start_i) begin
            logic [31:0] streams;
            streams = {16'd0, cfg_i.k} * {16'd0, cfg_i.n};
            err_q      <= 1'b0;
            cycles_q   <= '0;
            fro_q      <= 0.0;
            idx_q      <= '0;
            i_q        <= '0;
            j_q        <= '0;
            p_q        <= '0;
            ant_q      <= '0;
            h_re_q     <= cfg_i.h_re;
            h_im_q     <= cfg_i.h_im;
            w_re_q     <= cfg_i.w_re;
            w_im_q     <= cfg_i.w_im;
            s_re_q     <= cfg_i.s_re;
            s_im_q     <= cfg_i.s_im;
            x_re_q     <= cfg_i.x_re;
            x_im_q     <= cfg_i.x_im;
            pwr_q      <= cfg_i.p;
            alpha_q    <= cfg_i.alpha;
            apply_q    <= (cfg_i.s_re != '0) && (cfg_i.s_im != '0) &&
                          (cfg_i.x_re != '0) && (cfg_i.x_im != '0);
            if (cfg_i.k == '0 || cfg_i.n == '0 || cfg_i.m == '0 ||
                streams == '0 || streams > PREC_MAX_S || cfg_i.m > PREC_MAX_M) begin
              err_q   <= 1'b1;
              state_q <= ST_DONE;
            end else begin
              s_q     <= streams[15:0];
              m_q     <= cfg_i.m;
              state_q <= ST_LD_HRE;
              dma_q   <= DMA_IDLE;
            end
          end
        end

        ST_LD_HRE, ST_LD_HIM, ST_LD_SRE, ST_LD_SIM: begin
          if (dma_q == DMA_IDLE) begin
            unique case (state_q)
              ST_LD_HRE: dma_addr_q <= h_re_q + ({48'd0, idx_q} << 3);
              ST_LD_HIM: dma_addr_q <= h_im_q + ({48'd0, idx_q} << 3);
              ST_LD_SRE: dma_addr_q <= s_re_q + ({48'd0, idx_q} << 3);
              default:   dma_addr_q <= s_im_q + ({48'd0, idx_q} << 3);
            endcase
            dma_q <= DMA_AR;
          end else if (dma_q == DMA_AR) begin
            if (dma_resp_i.ar_ready) dma_q <= DMA_R;
          end else if (dma_q == DMA_R && dma_resp_i.r_valid) begin
            dma_rdata_q <= pick64(dma_resp_i.r.data, dma_addr_q);
            dma_q       <= DMA_IDLE;
            if (dma_resp_i.r.resp != RESP_OKAY) begin
              err_q   <= 1'b1;
              state_q <= ST_DONE;
            end else begin
              unique case (state_q)
                ST_LD_HRE: h_re[idx_q]  <= pick64(dma_resp_i.r.data, dma_addr_q);
                ST_LD_HIM: h_im[idx_q]  <= pick64(dma_resp_i.r.data, dma_addr_q);
                ST_LD_SRE: sv_re[idx_q] <= pick64(dma_resp_i.r.data, dma_addr_q);
                default:   sv_im[idx_q] <= pick64(dma_resp_i.r.data, dma_addr_q);
              endcase
              if (idx_q + 16'd1 == ((state_q == ST_LD_SRE || state_q == ST_LD_SIM) ? s_q
                                                                                    : (s_q * m_q))) begin
                idx_q <= '0;
                unique case (state_q)
                  ST_LD_HRE: state_q <= ST_LD_HIM;
                  ST_LD_HIM: state_q <= ST_GRAM_INIT;
                  ST_LD_SRE: state_q <= ST_LD_SIM;
                  default:   state_q <= ST_APPLY_INIT;
                endcase
              end else begin
                idx_q <= idx_q + 16'd1;
              end
            end
          end
        end

        ST_GRAM_INIT: begin
          acc_r_q <= 0.0;
          acc_i_q <= 0.0;
          p_q     <= '0;
          state_q <= ST_GRAM_MAC;
        end

        ST_GRAM_MAC: begin
          real ar, ai, br, bi;
          ar = prec_bits2real(h_re[j_q * m_q + p_q]);
          ai = prec_bits2real(h_im[j_q * m_q + p_q]);
          br = prec_bits2real(h_re[i_q * m_q + p_q]);
          bi = prec_bits2real(h_im[i_q * m_q + p_q]);
          acc_r_q <= acc_r_q + (ar * br + ai * bi);
          acc_i_q <= acc_i_q + (ar * bi - ai * br);
          if (p_q + 16'd1 == m_q) state_q <= ST_GRAM_ST;
          else p_q <= p_q + 16'd1;
        end

        ST_GRAM_ST: begin
          real rr, ii;
          rr = acc_r_q;
          ii = acc_i_q;
          if (i_q == j_q) begin
            rr = rr + prec_bits2real(alpha_q);
            ii = 0.0;
          end
          a_re[i_q * s_q + j_q] <= prec_real2bits(rr);
          a_im[i_q * s_q + j_q] <= prec_real2bits(ii);
          l_re[i_q * s_q + j_q] <= '0;
          l_im[i_q * s_q + j_q] <= '0;
          acc_r_q <= 0.0;
          acc_i_q <= 0.0;
          p_q     <= '0;
          if (j_q + 16'd1 == s_q) begin
            j_q <= '0;
            if (i_q + 16'd1 == s_q) begin
              i_q     <= '0;
              state_q <= ST_CHOL_I;
            end else begin
              i_q     <= i_q + 16'd1;
              state_q <= ST_GRAM_INIT;
            end
          end else begin
            j_q     <= j_q + 16'd1;
            state_q <= ST_GRAM_INIT;
          end
        end

        ST_CHOL_I: begin
          j_q <= '0;
          if (i_q == '0) state_q <= ST_CHOL_DIAG_INIT;
          else state_q <= ST_CHOL_J_INIT;
        end

        ST_CHOL_J_INIT: begin
          sr_q    <= prec_bits2real(a_re[i_q * s_q + j_q]);
          si_q    <= prec_bits2real(a_im[i_q * s_q + j_q]);
          p_q     <= '0;
          state_q <= (j_q == '0) ? ST_CHOL_J_DIV : ST_CHOL_J_P;
        end

        ST_CHOL_J_P: begin
          real lr, li, rr, ri;
          lr   = prec_bits2real(l_re[i_q * s_q + p_q]);
          li   = prec_bits2real(l_im[i_q * s_q + p_q]);
          rr   = prec_bits2real(l_re[j_q * s_q + p_q]);
          ri   = -prec_bits2real(l_im[j_q * s_q + p_q]);
          sr_q <= sr_q - (lr * rr - li * ri);
          si_q <= si_q - (lr * ri + li * rr);
          if (p_q + 16'd1 == j_q) state_q <= ST_CHOL_J_DIV;
          else p_q <= p_q + 16'd1;
        end

        ST_CHOL_J_DIV: begin
          real pivot;
          pivot = prec_bits2real(l_re[j_q * s_q + j_q]);
          l_re[i_q * s_q + j_q] <= prec_real2bits(sr_q / pivot);
          l_im[i_q * s_q + j_q] <= prec_real2bits(si_q / pivot);
          if (j_q + 16'd1 == i_q) state_q <= ST_CHOL_DIAG_INIT;
          else begin
            j_q     <= j_q + 16'd1;
            state_q <= ST_CHOL_J_INIT;
          end
        end

        ST_CHOL_DIAG_INIT: begin
          diag_q  <= prec_bits2real(a_re[i_q * s_q + i_q]);
          p_q     <= '0;
          state_q <= (i_q == '0) ? ST_CHOL_SQRT : ST_CHOL_DIAG_P;
        end

        ST_CHOL_DIAG_P: begin
          real lr, li;
          lr     = prec_bits2real(l_re[i_q * s_q + p_q]);
          li     = prec_bits2real(l_im[i_q * s_q + p_q]);
          diag_q <= diag_q - (lr * lr + li * li);
          if (p_q + 16'd1 == i_q) state_q <= ST_CHOL_SQRT;
          else p_q <= p_q + 16'd1;
        end

        ST_CHOL_SQRT: begin
          if (diag_q <= 0.0) begin
            err_q   <= 1'b1;
            state_q <= ST_DONE;
          end else begin
            l_re[i_q * s_q + i_q] <= prec_real2bits($sqrt(diag_q));
            l_im[i_q * s_q + i_q] <= '0;
            if (i_q + 16'd1 == s_q) begin
              i_q     <= '0;
              ant_q   <= '0;
              fro_q   <= 0.0;
              state_q <= ST_FWD_INIT;
            end else begin
              i_q     <= i_q + 16'd1;
              state_q <= ST_CHOL_I;
            end
          end
        end

        ST_FWD_INIT: begin
          sr_q    <= prec_bits2real(h_re[i_q * m_q + ant_q]);
          si_q    <= prec_bits2real(h_im[i_q * m_q + ant_q]);
          p_q     <= '0;
          state_q <= (i_q == '0) ? ST_FWD_DIV : ST_FWD_P;
        end

        ST_FWD_P: begin
          real lr, li, yr, yi;
          lr   = prec_bits2real(l_re[i_q * s_q + p_q]);
          li   = prec_bits2real(l_im[i_q * s_q + p_q]);
          yr   = prec_bits2real(y_re[p_q]);
          yi   = prec_bits2real(y_im[p_q]);
          sr_q <= sr_q - (lr * yr - li * yi);
          si_q <= si_q - (lr * yi + li * yr);
          if (p_q + 16'd1 == i_q) state_q <= ST_FWD_DIV;
          else p_q <= p_q + 16'd1;
        end

        ST_FWD_DIV: begin
          real pivot;
          pivot = prec_bits2real(l_re[i_q * s_q + i_q]);
          y_re[i_q] <= prec_real2bits(sr_q / pivot);
          y_im[i_q] <= prec_real2bits(si_q / pivot);
          if (i_q + 16'd1 == s_q) begin
            i_q     <= s_q - 16'd1;
            state_q <= ST_BWD_INIT;
          end else begin
            i_q     <= i_q + 16'd1;
            state_q <= ST_FWD_INIT;
          end
        end

        ST_BWD_INIT: begin
          sr_q    <= prec_bits2real(y_re[i_q]);
          si_q    <= prec_bits2real(y_im[i_q]);
          p_q     <= i_q + 16'd1;
          state_q <= (i_q + 16'd1 == s_q) ? ST_BWD_DIV : ST_BWD_P;
        end

        ST_BWD_P: begin
          real lr, li, xr, xi;
          lr   = prec_bits2real(l_re[p_q * s_q + i_q]);
          li   = -prec_bits2real(l_im[p_q * s_q + i_q]);
          xr   = prec_bits2real(z_re[p_q]);
          xi   = prec_bits2real(z_im[p_q]);
          sr_q <= sr_q - (lr * xr - li * xi);
          si_q <= si_q - (lr * xi + li * xr);
          if (p_q + 16'd1 == s_q) state_q <= ST_BWD_DIV;
          else p_q <= p_q + 16'd1;
        end

        ST_BWD_DIV: begin
          real pivot, zr, zi;
          pivot = prec_bits2real(l_re[i_q * s_q + i_q]);
          zr = sr_q / pivot;
          zi = si_q / pivot;
          z_re[i_q] <= prec_real2bits(zr);
          z_im[i_q] <= prec_real2bits(zi);
          w_re[i_q * m_q + ant_q] <= prec_real2bits(zr);
          w_im[i_q * m_q + ant_q] <= prec_real2bits(-zi);
          fro_q <= fro_q + (zr * zr + zi * zi);
          if (i_q == '0) begin
            if (ant_q + 16'd1 == m_q) begin
              idx_q   <= '0;
              state_q <= ST_SCALE_INIT;
            end else begin
              ant_q   <= ant_q + 16'd1;
              i_q     <= '0;
              state_q <= ST_FWD_INIT;
            end
          end else begin
            i_q     <= i_q - 16'd1;
            state_q <= ST_BWD_INIT;
          end
        end

        ST_SCALE_INIT: begin
          if (fro_q <= 0.0 || prec_bits2real(pwr_q) <= 0.0) begin
            err_q   <= 1'b1;
            state_q <= ST_DONE;
          end else begin
            scale_q <= $sqrt(prec_bits2real(pwr_q) / fro_q);
            idx_q   <= '0;
            state_q <= ST_SCALE;
          end
        end

        ST_SCALE: begin
          w_re[idx_q] <= prec_real2bits(prec_bits2real(w_re[idx_q]) * scale_q);
          w_im[idx_q] <= prec_real2bits(prec_bits2real(w_im[idx_q]) * scale_q);
          if (idx_q + 16'd1 == s_q * m_q) begin
            idx_q   <= '0;
            dma_q   <= DMA_IDLE;
            state_q <= ST_ST_WRE;
          end else begin
            idx_q <= idx_q + 16'd1;
          end
        end

        ST_ST_WRE, ST_ST_WIM, ST_ST_XRE, ST_ST_XIM: begin
          if (dma_q == DMA_IDLE) begin
            unique case (state_q)
              ST_ST_WRE: begin
                dma_addr_q  <= w_re_q + ({48'd0, idx_q} << 3);
                dma_wdata_q <= w_re[idx_q];
              end
              ST_ST_WIM: begin
                dma_addr_q  <= w_im_q + ({48'd0, idx_q} << 3);
                dma_wdata_q <= w_im[idx_q];
              end
              ST_ST_XRE: begin
                dma_addr_q  <= x_re_q + ({48'd0, idx_q} << 3);
                dma_wdata_q <= x_re[idx_q];
              end
              default: begin
                dma_addr_q  <= x_im_q + ({48'd0, idx_q} << 3);
                dma_wdata_q <= x_im[idx_q];
              end
            endcase
            dma_q <= DMA_AW;
          end else if (dma_q == DMA_AW) begin
            if (dma_resp_i.aw_ready) dma_q <= DMA_W;
          end else if (dma_q == DMA_W) begin
            if (dma_resp_i.w_ready) dma_q <= DMA_B;
          end else if (dma_q == DMA_B && dma_resp_i.b_valid) begin
            dma_q <= DMA_IDLE;
            if (dma_resp_i.b.resp != RESP_OKAY) begin
              err_q   <= 1'b1;
              state_q <= ST_DONE;
            end else if (idx_q + 16'd1 == ((state_q == ST_ST_XRE || state_q == ST_ST_XIM) ? m_q
                                                                                          : (s_q * m_q))) begin
              idx_q <= '0;
              unique case (state_q)
                ST_ST_WRE: state_q <= ST_ST_WIM;
                ST_ST_WIM: state_q <= apply_q ? ST_LD_SRE : ST_DONE;
                ST_ST_XRE: state_q <= ST_ST_XIM;
                default:   state_q <= ST_DONE;
              endcase
            end else begin
              idx_q <= idx_q + 16'd1;
            end
          end
        end

        ST_APPLY_INIT: begin
          ant_q   <= '0;
          i_q     <= '0;
          acc_r_q <= 0.0;
          acc_i_q <= 0.0;
          state_q <= ST_APPLY_MAC;
        end

        ST_APPLY_MAC: begin
          real wr, wi, ur, ui;
          wr = prec_bits2real(w_re[i_q * m_q + ant_q]);
          wi = prec_bits2real(w_im[i_q * m_q + ant_q]);
          ur = prec_bits2real(sv_re[i_q]);
          ui = prec_bits2real(sv_im[i_q]);
          acc_r_q <= acc_r_q + (wr * ur - wi * ui);
          acc_i_q <= acc_i_q + (wr * ui + wi * ur);
          if (i_q + 16'd1 == s_q) state_q <= ST_APPLY_ST;
          else i_q <= i_q + 16'd1;
        end

        ST_APPLY_ST: begin
          x_re[ant_q] <= prec_real2bits(acc_r_q);
          x_im[ant_q] <= prec_real2bits(acc_i_q);
          acc_r_q     <= 0.0;
          acc_i_q     <= 0.0;
          i_q         <= '0;
          if (ant_q + 16'd1 == m_q) begin
            idx_q   <= '0;
            dma_q   <= DMA_IDLE;
            state_q <= ST_ST_XRE;
          end else begin
            ant_q   <= ant_q + 16'd1;
            state_q <= ST_APPLY_MAC;
          end
        end

        default: state_q <= ST_IDLE;
      endcase
    end
  end

`endif
endmodule
