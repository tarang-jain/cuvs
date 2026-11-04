/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, Sirius Contributors.
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::detail::bounded_int8 {

// Adapted from sirius-db/sirius, vector-join, src/cuda/vss/bound_gemm.cu.
// A block scores 128 queries x 128 corpus rows, using eight 64 x 32 warp tiles.
// On sm80+ it pipelines 64-byte row segments through cp.async, ldmatrix, and k32 IMMA.
// Only surviving IDs leave registers; there is no dense score matrix.
constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kWarps = 8;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ inline void cp_async16(void* smem, void const* gmem, bool fill)
{
  auto const s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(fill ? 16 : 0));
}

__device__ inline void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

/// Waits until every committed group has landed.
__device__ inline void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ inline void ldmatrix_x4(unsigned (&r)[4], void const* smem)
{
  auto const s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}

// c (m16n8, 4 per lane) += a (m16 x 32 bytes) b (32 bytes x n8): k32 for int8, k16 for half.
__device__ inline void mma_m16n8(int* c, unsigned const (&a)[4], unsigned b0, unsigned b1)
{
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
    "{%8, %9}, {%0, %1, %2, %3};\n"
    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

struct score_frag {
  int x[8];
};
#endif

RAFT_KERNEL __launch_bounds__(kWarps * 32, 2) fused_filter_kernel(const int8_t* x,
                                                                  const double* x_sq,
                                                                  int64_t n,
                                                                  int64_t id_base,
                                                                  const int8_t* probe,
                                                                  const double* probe_sq,
                                                                  int64_t m,
                                                                  int d,
                                                                  const int64_t* limits,
                                                                  int32_t* out_rows,
                                                                  int64_t* out_ids,
                                                                  unsigned long long* count)
{
  int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int const wm = warp / 4, wn = warp % 4;
  int64_t const q0     = int64_t(blockIdx.x) * kTileM;
  int64_t x0           = int64_t(blockIdx.y) * kTileN;
  int64_t const x_step = int64_t(gridDim.y) * kTileN, x_stop = n;
#if __CUDA_ARCH__ >= 800
  constexpr int kStageBytes = 64, kPitchBytes = 80;
  __shared__ __align__(128) unsigned char stage_a[2][kTileM * kPitchBytes];
  __shared__ __align__(128) unsigned char stage_b[2][kTileN * kPitchBytes];
  __shared__ int64_t tile_rows[kTileM], tile_qsq[kTileM], tile_limit[kTileM];
  __shared__ int32_t tile_xsq[2][kTileN];
  int const row_bytes = d;
  auto const* x_bytes = reinterpret_cast<unsigned char const*>(x);
  auto const* p_bytes = reinterpret_cast<unsigned char const*>(probe);
  auto load_b         = [&](int64_t origin, int s, int buf) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      int const e = threadIdx.x + k * kWarps * 32, r = e / 4, at = s * kStageBytes + (e % 4) * 16;
      bool const ok   = origin + r < n && at < row_bytes;
      auto const* ptr = ok ? x_bytes + (origin + r) * row_bytes + at : x_bytes;
      cp_async16(&stage_b[buf][r * kPitchBytes + (e % 4) * 16], ptr, ok);
    }
  };
  auto load_xsq = [&](int64_t origin, int which) {
    for (int r = threadIdx.x; r < kTileN; r += blockDim.x) {
      tile_xsq[which][r] = origin + r < n ? static_cast<int32_t>(x_sq[origin + r]) : 0;
    }
  };
  auto load_a = [&](int s, int buf) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      int const e = threadIdx.x + k * kWarps * 32, r = e / 4, at = s * kStageBytes + (e % 4) * 16;
      int64_t const row = tile_rows[r];
      bool const ok     = row >= 0 && at < row_bytes;
      auto const* ptr   = ok ? p_bytes + row * row_bytes + at : p_bytes;
      cp_async16(&stage_a[buf][r * kPitchBytes + (e % 4) * 16], ptr, ok);
    }
  };
  load_b(x0, 0, 0);
  load_xsq(x0, 0);
  for (int r = threadIdx.x; r < kTileM; r += blockDim.x) {
    int64_t const row = q0 + r < m ? q0 + r : -1;
    tile_rows[r]      = row;
    tile_qsq[r]       = row >= 0 ? static_cast<int64_t>(probe_sq[row]) : 0;
    tile_limit[r]     = row >= 0 ? limits[row] : -1;
  }
  __syncthreads();
  load_a(0, 0);
  cp_async_commit();
  int const stages = (row_bytes + kStageBytes - 1) / kStageBytes;
  int buf          = 0;
  for (int tile = 0; x0 < x_stop; x0 += x_step, ++tile) {
    score_frag acc[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          acc[i][j].x[t] = 0;
        }
      }
    }
    for (int s = 0; s < stages; ++s, buf ^= 1) {
      // One barrier a stage: past it this stage is visible to every warp and every warp is done
      // with the previous one, whose buffer the next copies then fill -- this tile's next stage,
      // or after its last the next tile's first stage and norms, which land during the epilogue.
      cp_async_wait_all();
      __syncthreads();
      if (s + 1 < stages) {
        load_a(s + 1, buf ^ 1);
        load_b(x0, s + 1, buf ^ 1);
        cp_async_commit();
      } else if (x0 + x_step < x_stop) {
        load_a(0, buf ^ 1);
        load_b(x0 + x_step, 0, buf ^ 1);
        load_xsq(x0 + x_step, (tile + 1) & 1);
        cp_async_commit();
      }
      // 32 bytes of the rows per product; a stage's zero-filled tail past the row adds nothing.
      int const steps = min(kStageBytes, row_bytes - s * kStageBytes) > 32 ? 2 : 1;
      for (int ks = 0; ks < steps; ++ks) {
        // Lane l addresses row l % 8 of 8 x 16-byte matrix l / 8 of each ldmatrix.x4: for the
        // corpus, n8 tile 2p + l / 16 at byte 16 ((l / 8) % 2); for the probe, row l % 16 at byte
        // 16 (l / 16).
        unsigned b[4][2];
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          unsigned r[4];
          ldmatrix_x4(
            r,
            &stage_b[buf][(wn * 32 + (2 * p + (lane >> 4)) * 8 + (lane & 7)) * kPitchBytes +
                          ks * 32 + ((lane >> 3) & 1) * 16]);
          b[2 * p][0]     = r[0];
          b[2 * p][1]     = r[1];
          b[2 * p + 1][0] = r[2];
          b[2 * p + 1][1] = r[3];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          unsigned a[4];
          ldmatrix_x4(a,
                      &stage_a[buf][(wm * 64 + i * 16 + (lane & 15)) * kPitchBytes + ks * 32 +
                                    (lane >> 4) * 16]);
#pragma unroll
          for (int jn = 0; jn < 4; ++jn) {
            mma_m16n8(&acc[i][jn >> 1].x[(jn & 1) * 4], a, b[jn][0], b[jn][1]);
          }
        }
      }
    }  // Keep norms, subtraction, and threshold comparison integral. The dot product fits
    // INT32, while squared distances may exceed INT32 and FP32's exact-integer range.
    int const g = lane / 4, q = lane % 4;
    int64_t xs[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      int const col = wn * 32 + (c >> 2) * 16 + 8 * ((c & 3) >> 1) + 2 * q + (c & 1);
      xs[c]         = tile_xsq[tile & 1][col];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int const r0 = wm * 64 + i * 16 + g;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        unsigned keep = 0;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          int const h = (t >> 1) & 1, c = j * 4 + ((t >> 2) << 1) + (t & 1);
          int64_t const row      = tile_rows[r0 + 8 * h];
          int64_t const xj       = x0 + wn * 32 + j * 16 + 8 * (t >> 2) + 2 * q + (t & 1);
          int64_t const distance = tile_qsq[r0 + 8 * h] + xs[c] - 2 * int64_t(acc[i][j].x[t]);
          if (row >= 0 && xj < n && distance <= tile_limit[r0 + 8 * h]) { keep |= 1u << t; }
        }
        if (!__any_sync(0xffffffffu, keep != 0)) { continue; }
        // Ported warp compaction: one atomic reserves space for all this warp's survivors.
        int const own = __popc(keep);
        int before    = own;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
          int const value = __shfl_up_sync(0xffffffffu, before, offset);
          if (lane >= offset) { before += value; }
        }
        unsigned long long base = 0;
        if (lane == 31) { base = atomicAdd(count, static_cast<unsigned long long>(before)); }
        auto out = __shfl_sync(0xffffffffu, base, 31) + before - own;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          if ((keep >> t) & 1u) {
            int const h      = (t >> 1) & 1;
            int64_t const xj = x0 + wn * 32 + j * 16 + 8 * (t >> 2) + 2 * q + (t & 1);
            out_rows[out]    = static_cast<int32_t>(tile_rows[r0 + 8 * h]);
            out_ids[out]     = id_base + xj;
            ++out;
          }
        }
      }
    }
  }
#else
  // Older targets retain an integer-only fallback, with the same bounds and candidate format.
  for (; x0 < x_stop; x0 += x_step) {
    for (int p = threadIdx.x; p < kTileM * kTileN; p += blockDim.x) {
      int64_t const qi = q0 + p / kTileN, xi = x0 + p % kTileN;
      bool keep = false;
      if (qi < m && xi < n) {
        int dot       = 0;
        auto const* a = reinterpret_cast<const int*>(probe + qi * d);
        auto const* b = reinterpret_cast<const int*>(x + xi * d);
        for (int j = 0; j < d / 4; ++j) {
          dot = __dp4a(a[j], b[j], dot);
        }
        auto const distance = int64_t(probe_sq[qi]) + int64_t(x_sq[xi]) - 2 * int64_t(dot);
        keep                = distance <= limits[qi];
      }
      unsigned const mask = __ballot_sync(0xffffffffu, keep);
      if (mask) {
        unsigned long long base = 0;
        if (lane == 0) { base = atomicAdd(count, static_cast<unsigned long long>(__popc(mask))); }
        base = __shfl_sync(0xffffffffu, base, 0);
        if (keep) {
          auto const out = base + __popc(mask & ((1u << lane) - 1));
          out_rows[out]  = static_cast<int32_t>(qi);
          out_ids[out]   = id_base + xi;
        }
      }
    }
  }
#endif
}

}  // namespace cuvs::neighbors::detail::bounded_int8
