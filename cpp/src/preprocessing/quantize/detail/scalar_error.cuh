/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/preprocessing/quantize/scalar.hpp>
#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cuvs::preprocessing::quantize::scalar::detail {

template <typename T>
struct sq_affine_parameters {
  double scale;
  double shift;
  T const* shifts;
};

template <typename T>
sq_affine_parameters<T> get_sq_affine_parameters(
  cuvs::preprocessing::quantize::scalar::quantizer_const_view<T> quantizer, int64_t dim)
{
  using view_type = cuvs::preprocessing::quantize::scalar::quantizer_const_view<T>;
  sq_affine_parameters<T> affine{};
  if (auto const* range = std::get_if<typename view_type::range_type>(&quantizer.value)) {
    auto const lo = static_cast<double>(range->min_);
    auto const hi = static_cast<double>(range->max_);
    RAFT_EXPECTS(std::isfinite(lo) && std::isfinite(hi) && hi >= lo,
                 "SQ bounds must be finite with min <= max");
    constexpr double code_min = std::numeric_limits<int8_t>::min();
    constexpr double code_max = std::numeric_limits<int8_t>::max();
    affine.scale              = hi > lo ? (hi - lo) / (code_max - code_min) : 1.0;
    affine.shift              = lo - code_min * affine.scale;
  } else {
    auto const& shifts = std::get<typename view_type::shifts_type>(quantizer.value);
    RAFT_EXPECTS(shifts.shifts.extent(0) == dim, "SQ shifts must match the dataset dimension");
    affine.scale  = static_cast<double>(shifts.scale);
    affine.shifts = shifts.shifts.data_handle();
  }
  RAFT_EXPECTS(std::isfinite(affine.scale) && affine.scale > 0.0 && std::isfinite(affine.shift),
               "SQ scale must be positive and finite, and its shift must be finite");
  return affine;
}

// Directed rounding bounds the mathematical affine reconstruction, including clipping and
// floating-point decoding. Measuring inverse_transform's saturated output would understate
// the error of the affine coordinates used by INT8 GEMM for extreme quantizers.
template <typename T>
RAFT_KERNEL quantization_error_kernel(const T* original,
                                      const int8_t* codes,
                                      int64_t rows,
                                      int64_t dim,
                                      sq_affine_parameters<T> affine,
                                      double* errors,
                                      double* maximum)
{
  int const lane = threadIdx.x % 32;
  for (int64_t row = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32; row < rows;
       row += int64_t(gridDim.x) * (blockDim.x / 32)) {
    double sum = 0.0;
    for (int64_t j = lane; j < dim; j += 32) {
      auto const value = static_cast<double>(original[row * dim + j]);
      auto const code  = static_cast<double>(codes[row * dim + j]);
      auto const shift = affine.shifts ? static_cast<double>(affine.shifts[j]) : affine.shift;
      auto const lo    = __fma_rd(affine.scale, code, shift);
      auto const hi    = __fma_ru(affine.scale, code, shift);
      auto const error = isfinite(value) && isfinite(lo) && isfinite(hi)
                           ? fmax(__dsub_ru(value, lo), __dsub_ru(hi, value))
                           : std::numeric_limits<double>::infinity();
      sum              = __dadd_ru(sum, __dmul_ru(error, error));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      sum = __dadd_ru(sum, __shfl_xor_sync(0xffffffffu, sum, offset));
    }
    if (lane == 0) {
      auto const error = __dsqrt_ru(sum);
      if (errors) { errors[row] = error; }
      if (maximum) {
        atomicMax(reinterpret_cast<unsigned long long*>(maximum),
                  static_cast<unsigned long long>(__double_as_longlong(error)));
      }
    }
  }
}

template <typename T>
void quantization_errors(raft::resources const& res,
                         quantizer_const_view<T> quantizer,
                         raft::device_matrix_view<const T, int64_t> original,
                         raft::device_matrix_view<const int8_t, int64_t> codes,
                         double* errors,
                         double* maximum)
{
  RAFT_EXPECTS(original.extent(0) == codes.extent(0) && original.extent(1) == codes.extent(1),
               "Original vectors and SQ codes must have the same shape");
  auto const affine = get_sq_affine_parameters(quantizer, original.extent(1));
  auto const stream = raft::resource::get_cuda_stream(res);
  if (raft::resource::get_dry_run_flag(res)) { return; }
  if (maximum) { RAFT_CUDA_TRY(cudaMemsetAsync(maximum, 0, sizeof(double), stream.get())); }
  if (original.extent(0) == 0) { return; }
  auto const blocks = static_cast<unsigned>(std::min<int64_t>((original.extent(0) + 7) / 8, 65535));
  quantization_error_kernel<<<blocks, 256, 0, stream.get()>>>(original.data_handle(),
                                                              codes.data_handle(),
                                                              original.extent(0),
                                                              original.extent(1),
                                                              affine,
                                                              errors,
                                                              maximum);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
double max_quantization_error(raft::resources const& res,
                              quantizer_const_view<T> quantizer,
                              raft::device_matrix_view<const T, int64_t> original,
                              raft::device_matrix_view<const int8_t, int64_t> codes)
{
  auto maximum = raft::make_device_vector<double, int64_t>(res, 1);
  quantization_errors(res, quantizer, original, codes, nullptr, maximum.data_handle());
  double result = std::numeric_limits<double>::infinity();
  raft::copy(res,
             raft::make_host_scalar_view(&result),
             raft::make_device_scalar_view<const double>(maximum.data_handle()));
  raft::resource::sync_stream(res);
  return result;
}

}  // namespace cuvs::preprocessing::quantize::scalar::detail
