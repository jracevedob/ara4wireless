// Copyright 2026 TUD Technical University of Dresden
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Standalone check of the precoding datapath against a few closed-form ZF
// problems. The memory is a one-beat AXI slave on a 64-bit bus.

`include "axi/typedef.svh"

module precoding_unit_tb;
  localparam int unsigned AW = 64;
  localparam int unsigned DW = 64;
  localparam int unsigned IW = 4;
  localparam int unsigned MEM_BYTES = 4096;

  import axi_pkg::*;
  import precoding_pkg::*;

  typedef logic [AW-1:0] addr_t;
  typedef logic [DW-1:0] data_t;
  typedef logic [DW/8-1:0] strb_t;
  typedef logic [IW-1:0] id_t;
  typedef logic user_t;

  `AXI_TYPEDEF_ALL(axi, addr_t, id_t, data_t, strb_t, user_t)

  logic clk = 0;
  logic rst_n = 0;
  always #5 clk = ~clk;

  axi_req_t  dma_req;
  axi_resp_t dma_resp;
  prec_cfg_t cfg;
  logic      start, busy, done, err;
  logic [63:0] cycles;

  byte unsigned mem[0:MEM_BYTES-1] = '{default: '0};

  logic        r_valid_q, b_valid_q, host_we;
  logic [63:0] r_addr_q, aw_addr_q, host_addr, host_data;

  precoding_unit #(
    .AxiDataWidth(DW),
    .req_t       (axi_req_t),
    .resp_t      (axi_resp_t)
  ) dut (
    .clk_i     (clk),
    .rst_ni    (rst_n),
    .cfg_i     (cfg),
    .start_i   (start),
    .busy_o    (busy),
    .done_o    (done),
    .err_o     (err),
    .cycles_o  (cycles),
    .dma_req_o (dma_req),
    .dma_resp_i(dma_resp)
  );

  function automatic logic [63:0] load64(input logic [63:0] addr);
    for (int b = 0; b < 8; b++) load64[8*b +: 8] = mem[addr + b];
  endfunction

  always_comb begin
    dma_resp         = '0;
    dma_resp.ar_ready = 1'b1;
    dma_resp.aw_ready = 1'b1;
    dma_resp.w_ready  = 1'b1;
    dma_resp.r_valid  = r_valid_q;
    dma_resp.r.data   = load64(r_addr_q);
    dma_resp.r.resp   = RESP_OKAY;
    dma_resp.r.last   = 1'b1;
    dma_resp.b_valid  = b_valid_q;
    dma_resp.b.resp   = RESP_OKAY;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      r_valid_q <= 1'b0;
      b_valid_q <= 1'b0;
      r_addr_q  <= '0;
      aw_addr_q <= '0;
    end else begin
      if (host_we) begin
        for (int b = 0; b < 8; b++) mem[host_addr + b] <= host_data[8*b +: 8];
      end
      if (dma_req.ar_valid && dma_resp.ar_ready) begin
        r_addr_q  <= dma_req.ar.addr;
        r_valid_q <= 1'b1;
      end else if (r_valid_q && dma_req.r_ready) begin
        r_valid_q <= 1'b0;
      end

      if (dma_req.aw_valid && dma_resp.aw_ready) aw_addr_q <= dma_req.aw.addr;

      if (dma_req.w_valid && dma_resp.w_ready) begin
        for (int b = 0; b < 8; b++) begin
          if (dma_req.w.strb[b]) mem[aw_addr_q + b] <= dma_req.w.data[8*b +: 8];
        end
        b_valid_q <= 1'b1;
      end else if (b_valid_q && dma_req.b_ready) begin
        b_valid_q <= 1'b0;
      end
    end
  end

  task automatic poke(input logic [63:0] addr, input real value);
    @(negedge clk);
    host_addr = addr;
    host_data = $realtobits(value);
    host_we   = 1'b1;
    @(posedge clk);
    @(negedge clk);
    host_we   = 1'b0;
  endtask

  function automatic real peek(input logic [63:0] addr);
    peek = $bitstoreal(load64(addr));
  endfunction

  task automatic check_close(input real got, input real exp, input string what);
    real diff;
    diff = got - exp;
    if (diff < 0.0) diff = -diff;
    if (diff > 1.0e-9) begin
      $display("FAIL %s got %f exp %f", what, got, exp);
      $finish(1);
    end
  endtask

  task automatic run_and_wait(input bit expect_err);
    int unsigned guard;
    @(negedge clk);
    start = 1'b1;
    @(posedge clk);
    @(negedge clk);
    start = 1'b0;
    guard = 0;
    while (!done) begin
      @(posedge clk);
      guard++;
      if (guard > 200000) begin
        $display("FAIL timeout");
        $finish(1);
      end
    end
    if (err != expect_err) begin
      $display("FAIL err=%0b expected %0b", err, expect_err);
      $finish(1);
    end
  endtask

  localparam logic [63:0] HRE = 64'h100;
  localparam logic [63:0] HIM = 64'h200;
  localparam logic [63:0] WRE = 64'h300;
  localparam logic [63:0] WIM = 64'h400;
  localparam logic [63:0] SRE = 64'h500;
  localparam logic [63:0] SIM = 64'h580;
  localparam logic [63:0] XRE = 64'h600;
  localparam logic [63:0] XIM = 64'h680;

  real inv_sqrt3;

  initial begin
    start     = 0;
    host_we   = 0;
    host_addr = '0;
    host_data = '0;
    cfg       = '0;
    repeat (4) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // H = 2, one stream, one antenna, P = 1, alpha = 0 -> W = 1.
    poke(HRE, 2.0);
    poke(HIM, 0.0);
    cfg.h_re  = HRE;
    cfg.h_im  = HIM;
    cfg.w_re  = WRE;
    cfg.w_im  = WIM;
    cfg.s_re  = '0;
    cfg.s_im  = '0;
    cfg.x_re  = '0;
    cfg.x_im  = '0;
    cfg.p     = $realtobits(1.0);
    cfg.alpha = $realtobits(0.0);
    cfg.k     = 16'd1;
    cfg.n     = 16'd1;
    cfg.m     = 16'd1;
    run_and_wait(1'b0);
    check_close(peek(WRE), 1.0, "zf1.wre");
    check_close(peek(WIM), 0.0, "zf1.wim");

    // Same channel with alpha = 3. Gram becomes 4, and the scaled weight is 1.
    cfg.alpha = $realtobits(3.0);
    run_and_wait(1'b0);
    check_close(peek(WRE), 1.0, "mmse1.wre");
    check_close(peek(WIM), 0.0, "mmse1.wim");

    // H = [[1, 0], [1, 1]], P = 1. Closed form is in the test comments.
    inv_sqrt3 = 1.0 / $sqrt(3.0);
    poke(HRE + 0, 1.0);
    poke(HRE + 8, 0.0);
    poke(HRE + 16, 1.0);
    poke(HRE + 24, 1.0);
    poke(HIM + 0, 0.0);
    poke(HIM + 8, 0.0);
    poke(HIM + 16, 0.0);
    poke(HIM + 24, 0.0);
    poke(SRE + 0, 1.0);
    poke(SRE + 8, 0.0);
    poke(SIM + 0, 0.0);
    poke(SIM + 8, 0.0);
    cfg.s_re  = SRE;
    cfg.s_im  = SIM;
    cfg.x_re  = XRE;
    cfg.x_im  = XIM;
    cfg.p     = $realtobits(1.0);
    cfg.alpha = $realtobits(0.0);
    cfg.k     = 16'd2;
    cfg.n     = 16'd1;
    cfg.m     = 16'd2;
    run_and_wait(1'b0);
    check_close(peek(WRE + 0), inv_sqrt3, "zf2.w00");
    check_close(peek(WRE + 8), -inv_sqrt3, "zf2.w01");
    check_close(peek(WRE + 16), 0.0, "zf2.w10");
    check_close(peek(WRE + 24), inv_sqrt3, "zf2.w11");
    check_close(peek(WIM + 0), 0.0, "zf2.wi");
    // s = [1, 0], so x is the first column of W: [sqrt(1/3), -sqrt(1/3)].
    check_close(peek(XRE + 0), inv_sqrt3, "zf2.x0");
    check_close(peek(XRE + 8), -inv_sqrt3, "zf2.x1");
    check_close(peek(XIM + 0), 0.0, "zf2.xi");

    // A zero channel is not positive definite.
    poke(HRE, 0.0);
    cfg.s_re = '0;
    cfg.s_im = '0;
    cfg.x_re = '0;
    cfg.x_im = '0;
    cfg.k    = 16'd1;
    cfg.n    = 16'd1;
    cfg.m    = 16'd1;
    cfg.alpha = $realtobits(0.0);
    run_and_wait(1'b1);

    // Dimensions above the coprocessor limit are rejected before any DMA.
    cfg.m = 16'd65;
    run_and_wait(1'b1);

    $display("PASS precoding_unit");
    $finish(0);
  end
endmodule
