#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Synthetic 5G NR DMRS channel-estimation vectors.
# arg1: number of pilot subcarriers (N)
# arg2: SNR in dB (optional, default 10)

import numpy as np
import sys


def emit(name, array, alignment='8'):
  print(".global %s" % name)
  print(".balign " + alignment)
  print("%s:" % name)
  bs = array.tobytes()
  for i in range(0, len(bs), 4):
    s = ""
    for n in range(4):
      s += "%02x" % bs[i + 3 - n]
    print("    .word 0x%s" % s)


if len(sys.argv) >= 2:
  N = int(sys.argv[1])
else:
  N = 256

if len(sys.argv) >= 3:
  snr_db = float(sys.argv[2])
else:
  snr_db = 10.0

np.random.seed(0)
dtype = np.float64

# Unit-energy QPSK DMRS pilots, with a 3 dB boost on half the REs so |x|
# is not constant and the MMSE weights vary across subcarriers.
bits = np.random.randint(0, 2, size=(N, 2))
xp_re = ((2 * bits[:, 0] - 1) / np.sqrt(2.0)).astype(dtype)
xp_im = ((2 * bits[:, 1] - 1) / np.sqrt(2.0)).astype(dtype)
xp_re[N // 2:] *= np.sqrt(2.0)
xp_im[N // 2:] *= np.sqrt(2.0)

# Rayleigh fading h ~ CN(0, 1)
h_re = (np.random.randn(N) * np.sqrt(0.5)).astype(dtype)
h_im = (np.random.randn(N) * np.sqrt(0.5)).astype(dtype)

# y = h * x + n, n ~ CN(0, sigma_n2)
sig_re = h_re * xp_re - h_im * xp_im
sig_im = h_re * xp_im + h_im * xp_re
sigma_h2 = np.array(1.0, dtype=dtype)
sigma_n2 = np.array(10.0 ** (-snr_db / 10.0), dtype=dtype)
n_re = (np.random.randn(N) * np.sqrt(sigma_n2 / 2.0)).astype(dtype)
n_im = (np.random.randn(N) * np.sqrt(sigma_n2 / 2.0)).astype(dtype)
yp_re = (sig_re + n_re).astype(dtype)
yp_im = (sig_im + n_im).astype(dtype)

zeros = np.zeros(N, dtype=dtype)

print(".section .data,\"aw\",@progbits")
emit("N", np.array(N, dtype=np.uint64))
emit("sigma_h2", sigma_h2)
emit("sigma_n2", sigma_n2)
emit("xp_re", xp_re, 'NR_LANES*8')
emit("xp_im", xp_im, 'NR_LANES*8')
emit("yp_re", yp_re, 'NR_LANES*8')
emit("yp_im", yp_im, 'NR_LANES*8')
emit("h_lse_re", zeros.copy(), 'NR_LANES*8')
emit("h_lse_im", zeros.copy(), 'NR_LANES*8')
emit("h_mmse_re", zeros.copy(), 'NR_LANES*8')
emit("h_mmse_im", zeros.copy(), 'NR_LANES*8')
emit("g_lse_re", zeros.copy(), 'NR_LANES*8')
emit("g_lse_im", zeros.copy(), 'NR_LANES*8')
emit("g_mmse_re", zeros.copy(), 'NR_LANES*8')
emit("g_mmse_im", zeros.copy(), 'NR_LANES*8')
