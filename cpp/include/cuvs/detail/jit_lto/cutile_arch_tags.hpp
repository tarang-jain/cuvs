/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef CUVS_CUTILE_ENABLED
#define CUVS_CUTILE_ENABLED 0
#endif

#ifndef CUVS_CUTILE_EMBED_SM_80
#define CUVS_CUTILE_EMBED_SM_80 1
#endif
#ifndef CUVS_CUTILE_EMBED_SM_86
#define CUVS_CUTILE_EMBED_SM_86 1
#endif
#ifndef CUVS_CUTILE_EMBED_SM_90
#define CUVS_CUTILE_EMBED_SM_90 1
#endif
#ifndef CUVS_CUTILE_EMBED_SM_100
#define CUVS_CUTILE_EMBED_SM_100 1
#endif
#ifndef CUVS_CUTILE_EMBED_SM_120
#define CUVS_CUTILE_EMBED_SM_120 1
#endif

namespace cuvs::detail::jit_lto {

#if CUVS_CUTILE_ENABLED

/** Must stay in sync with cuTile matrix _arch entries and planner add_static_fragment calls. */
struct cutile_arch_8_0 {
  static constexpr int cc_major = 8;
  static constexpr int cc_minor = 0;
};

struct cutile_arch_8_6 {
  static constexpr int cc_major = 8;
  static constexpr int cc_minor = 6;
};

struct cutile_arch_9_0 {
  static constexpr int cc_major = 9;
  static constexpr int cc_minor = 0;
};

struct cutile_arch_10_0 {
  static constexpr int cc_major = 10;
  static constexpr int cc_minor = 0;
};

struct cutile_arch_12_0 {
  static constexpr int cc_major = 12;
  static constexpr int cc_minor = 0;
};

inline bool is_embedded_cubin_arch(int cc_major, int cc_minor)
{
  if (cc_minor < 0) { return false; }
  return (cc_major == 8 &&
          (CUVS_CUTILE_EMBED_SM_80 || (CUVS_CUTILE_EMBED_SM_86 && cc_minor >= 6))) ||
         (CUVS_CUTILE_EMBED_SM_90 && cc_major == 9) ||
         (CUVS_CUTILE_EMBED_SM_100 && cc_major == 10) ||
         (CUVS_CUTILE_EMBED_SM_120 && cc_major == 12);
}

#else

inline bool is_embedded_cubin_arch(int, int) { return false; }

#endif

}  // namespace cuvs::detail::jit_lto
