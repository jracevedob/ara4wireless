#!/usr/bin/env python3
# Copyright 2026 TUD Technical University of Dresden
#
# SPDX-License-Identifier: Apache-2.0

# Same downlink channel as the linear precoders, for dirty-paper coding.
# Optional args: M  K  N
# H is (K*N) x M. The N antennas of a user share that user's large-scale fading.

import numpy as np
import sys


def emit(name, array, alignment="8"):
    print(".global %s" % name)
    print(".balign " + alignment)
    print("%s:" % name)
    bs = array.tobytes()
    for i in range(0, len(bs), 4):
        s = ""
        for n in range(4):
            s += "%02x" % bs[i + 3 - n]
        print("    .word 0x%s" % s)


M = int(sys.argv[1]) if len(sys.argv) >= 2 else 16
K = int(sys.argv[2]) if len(sys.argv) >= 3 else 4
N = int(sys.argv[3]) if len(sys.argv) >= 4 else 2
P = 1.0
S = K * N

rng = np.random.default_rng(0)
dtype = np.float64
beta = np.array([1.0, 0.5, 0.25, 0.125], dtype=dtype)
if K > beta.size:
    beta = np.resize(beta, K)
beta = beta[:K]
g_re = rng.standard_normal((K, N, M)) * np.sqrt(0.5)
g_im = rng.standard_normal((K, N, M)) * np.sqrt(0.5)
scale = np.sqrt(beta)[:, None, None]
h_re = (g_re * scale).astype(dtype).reshape(-1)
h_im = (g_im * scale).astype(dtype).reshape(-1)
bits = rng.integers(0, 2, size=(S, 2))
s_re = ((2 * bits[:, 0] - 1) / np.sqrt(2.0)).astype(dtype)
s_im = ((2 * bits[:, 1] - 1) / np.sqrt(2.0)).astype(dtype)

print('.section .data,"aw",@progbits')
emit("M", np.array(M, dtype=np.uint64))
emit("K", np.array(K, dtype=np.uint64))
emit("N", np.array(N, dtype=np.uint64))
emit("P", np.array(P, dtype=dtype))
emit("h_re", h_re, "NR_LANES*8")
emit("h_im", h_im, "NR_LANES*8")
emit("s_re", s_re, "NR_LANES*8")
emit("s_im", s_im, "NR_LANES*8")
emit("q_re", np.zeros(S * M, dtype=dtype), "NR_LANES*8")
emit("q_im", np.zeros(S * M, dtype=dtype), "NR_LANES*8")
emit("rdiag", np.zeros(S, dtype=dtype), "NR_LANES*8")
emit("u_re", np.zeros(S, dtype=dtype), "NR_LANES*8")
emit("u_im", np.zeros(S, dtype=dtype), "NR_LANES*8")
emit("x_re", np.zeros(M, dtype=dtype), "NR_LANES*8")
emit("x_im", np.zeros(M, dtype=dtype), "NR_LANES*8")
emit("gq_re", np.zeros(S * M, dtype=dtype), "NR_LANES*8")
emit("gq_im", np.zeros(S * M, dtype=dtype), "NR_LANES*8")
emit("grdiag", np.zeros(S, dtype=dtype), "NR_LANES*8")
emit("gu_re", np.zeros(S, dtype=dtype), "NR_LANES*8")
emit("gu_im", np.zeros(S, dtype=dtype), "NR_LANES*8")
emit("gx_re", np.zeros(M, dtype=dtype), "NR_LANES*8")
emit("gx_im", np.zeros(M, dtype=dtype), "NR_LANES*8")
