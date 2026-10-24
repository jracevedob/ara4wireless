#!/usr/bin/env python3
# Copyright 2026 TUD Technical University of Dresden
#
# SPDX-License-Identifier: Apache-2.0

# Uplink multi-user MIMO detection vectors.
# Optional args: M  K  N  noise variance σ²  PIC iterations
# H is M x (K*N), stored by columns. Each of the K users has N transmit
# antennas that share that user's large-scale fading. y = H x + n, QPSK x.

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


M = int(sys.argv[1]) if len(sys.argv) >= 2 else 64
K = int(sys.argv[2]) if len(sys.argv) >= 3 else 4
N = int(sys.argv[3]) if len(sys.argv) >= 4 else 2
sigma_n2 = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.01
niter = int(sys.argv[5]) if len(sys.argv) >= 6 else 3
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
# h[s * M + m] = H[m, s], s = k*N + n
h_re = (g_re * scale).astype(dtype).reshape(S, M).reshape(-1)
h_im = (g_im * scale).astype(dtype).reshape(S, M).reshape(-1)

bits = rng.integers(0, 2, size=(S, 2))
x_re = ((2 * bits[:, 0] - 1) / np.sqrt(2.0)).astype(dtype)
x_im = ((2 * bits[:, 1] - 1) / np.sqrt(2.0)).astype(dtype)

H = (h_re + 1j * h_im).reshape(S, M).T
x = x_re + 1j * x_im
noise = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) * np.sqrt(sigma_n2 / 2.0)
y = H @ x + noise
y_re = np.real(y).astype(dtype)
y_im = np.imag(y).astype(dtype)

zeros_s = np.zeros(S, dtype=dtype)

print('.section .data,"aw",@progbits')
emit("M", np.array(M, dtype=np.uint64))
emit("K", np.array(K, dtype=np.uint64))
emit("N", np.array(N, dtype=np.uint64))
emit("niter", np.array(niter, dtype=np.uint64))
emit("sigma_n2", np.array(sigma_n2, dtype=dtype))
emit("h_re", h_re, "NR_LANES*8")
emit("h_im", h_im, "NR_LANES*8")
emit("y_re", y_re, "NR_LANES*8")
emit("y_im", y_im, "NR_LANES*8")
emit("x_re", x_re, "NR_LANES*8")
emit("x_im", x_im, "NR_LANES*8")
for name in ("zf", "pic", "sd"):
    emit("%s_re" % name, zeros_s.copy(), "NR_LANES*8")
    emit("%s_im" % name, zeros_s.copy(), "NR_LANES*8")
    emit("g_%s_re" % name, zeros_s.copy(), "NR_LANES*8")
    emit("g_%s_im" % name, zeros_s.copy(), "NR_LANES*8")
