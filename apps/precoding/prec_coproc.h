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

// Memory map of the precoding coprocessor at PREC_BASE.
// Write the pointers and DIMS, then store 1 to CTRL. Poll STATUS until done.
// DIMS is {16'h0, M[15:0], N[15:0], K[15:0]}. P and ALPHA are IEEE-754 doubles.
// H, W, s and x are contiguous little-endian doubles. H and W are row-major
// in the stream index: element (stream, antenna) is at stream * M + antenna.
// Every pointer is 8-byte aligned. Streams K*N are limited to 16 and M to 64.
// A symbol apply runs when S_RE, S_IM, X_RE and X_IM are all non-zero.

#ifndef _PREC_COPROC_H_
#define _PREC_COPROC_H_

#include <stdint.h>

#define PREC_BASE   0xE0000000ull

#define PREC_CTRL   0x00u
#define PREC_STATUS 0x08u
#define PREC_H_RE   0x10u
#define PREC_H_IM   0x18u
#define PREC_W_RE   0x20u
#define PREC_W_IM   0x28u
#define PREC_S_RE   0x30u
#define PREC_S_IM   0x38u
#define PREC_X_RE   0x40u
#define PREC_X_IM   0x48u
#define PREC_DIMS   0x50u
#define PREC_P      0x58u
#define PREC_ALPHA  0x60u
#define PREC_CYCLES 0x68u

#define PREC_STAT_BUSY 1u
#define PREC_STAT_DONE 2u
#define PREC_STAT_ERR  4u

static inline void prec_write(uint32_t off, uint64_t val) {
  *(volatile uint64_t *)(uintptr_t)(PREC_BASE + off) = val;
}

static inline uint64_t prec_read(uint32_t off) {
  return *(volatile uint64_t *)(uintptr_t)(PREC_BASE + off);
}

#endif
