/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/dense.hpp>
#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/export.hpp>
#include <cuvs/preprocessing/quantize/scalar.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace CUVS_EXPORT cuvs {

/**
 * Scalar-quantized dataset specification: every component of a vector becomes one code, so the
 * manifest holds the quantizer and the dimension is still a property
 * of the data.
 */
template <typename MathT, typename CodePolicy>
struct sq_spec {
  // One code per component, so the codes are laid out exactly like a dense dataset.
  template <typename IdxT>
  using storage_spec =
    typename mdarray_spec<raft::layout_c_contiguous, CodePolicy>::template apply<int8_t, IdxT>;

  using math_type = MathT;

  using quantizer_type = cuvs::preprocessing::quantize::scalar::quantizer<MathT>;

  /** Quantizer parameters and a bound on this dataset's Euclidean reconstruction error.
   * The quantizer base keeps existing transform/inverse_transform calls source-compatible. */
  struct manifest_view_type : quantizer_type::view_type {
    /** Maximum ||x - (scale * code + shift)|| over the encoded dataset, in original units.
     * Infinity means no error bound is available and bounded search cannot prune rows.
     * Modifying codes or quantizer parameters invalidates a previously computed bound. */
    double max_error = std::numeric_limits<double>::infinity();

    /** Construct empty quantizer parameters with no measured error bound. */
    manifest_view_type() = default;

    /** Borrow quantizer parameters and the associated codes' maximum reconstruction error.
     * The quantizer owner must outlive use of the view. */
    manifest_view_type(typename quantizer_type::view_type quantizer,
                       double error = std::numeric_limits<double>::infinity())
      : quantizer_type::view_type{quantizer}, max_error{error}
    {
    }
  };

  /** Owns the quantizer and the maximum reconstruction error for the associated codes.
   * Existing quantizer constructors and parameter access remain available. */
  struct manifest_type : quantizer_type {
    using quantizer_type::quantizer_type;
    using view_type  = manifest_view_type;
    double max_error = std::numeric_limits<double>::infinity();

    /** Construct an empty quantizer with no measured error bound. */
    manifest_type() = default;

    /** Take ownership of a quantizer and optionally a measured error bound for these codes. */
    manifest_type(quantizer_type&& q, double error = std::numeric_limits<double>::infinity())
      : quantizer_type{std::move(q)}, max_error{error}
    {
    }

    /** Borrow parameters and the error bound; this owner must outlive use of the view. */
    [[nodiscard]] auto view() const noexcept -> manifest_view_type
    {
      return {quantizer_type::view(), max_error};
    }

    /** Borrow a manifest view implicitly, preserving the associated error bound. */
    operator view_type() const noexcept { return view(); }  // NOLINT(google-explicit-constructor)
  };

  template <typename T, typename IdxT>
  struct apply : storage_spec<IdxT> {
    // The dataset stores codes, but it still represents vectors of `T`.
    using value_type = std::remove_cv_t<T>;
    using code_type  = int8_t;
    // NOTE: re-exported rather than declared here, so that they do not belong to one instantiation
    // of `apply`; qualified lookup into `apply` does not reach the enclosing spec on its own.
    using math_type          = sq_spec::math_type;
    using manifest_type      = sq_spec::manifest_type;
    using manifest_view_type = sq_spec::manifest_view_type;

    // NOTE: every accessor is inherited; the dimension is still one code per component, so even
    // `get_dim` is the dense one.
    /** Borrow read-only quantizer parameters from the owning manifest. */
    [[nodiscard]] static auto get_manifest_view(const manifest_type& manifest) noexcept
      -> manifest_view_type
    {
      return manifest.view();
    }
    /** Borrow read-only quantizer parameters; the manifest must outlive use of the view. */
    [[nodiscard]] static auto get_manifest_const_view(const manifest_type& manifest) noexcept
      -> manifest_view_type
    {
      return manifest.view();
    }
    /** Return the read-only view; its owner must outlive stream work using it. */
    [[nodiscard]] static auto get_manifest_const_view(manifest_view_type manifest) noexcept
      -> manifest_view_type
    {
      return manifest;
    }
  };
};

template <typename T, typename IdxT, typename MathT, typename CodePolicy>
using sq_dataset = dataset<T, IdxT, sq_spec<MathT, CodePolicy>>;
template <typename T, typename IdxT, typename MathT>
using device_sq_dataset =
  sq_dataset<T, IdxT, MathT, raft::device_accessor<raft::device_container_policy<int8_t>>>;
template <typename T, typename IdxT, typename MathT>
using host_sq_dataset =
  sq_dataset<T, IdxT, MathT, raft::host_accessor<raft::host_container_policy<int8_t>>>;

namespace preprocessing::quantize::scalar {

/**
 * @defgroup scalar_dataset Scalar-quantized dataset construction
 * @{
 */
/**
 * @brief Train and encode a device corpus and measure its maximum reconstruction error.
 * Synchronizes the stream to return the dataset-wide error bound on the host.
 * @param[in] res Resources for training, encoding, and error measurement.
 * @param[in] params Scalar quantizer parameters.
 * @param[in] dataset Original device corpus to train on and encode.
 * @return Dataset owning codes, quantizer, and a conservative maximum L2 reconstruction error.
 * Reuse result.const_manifest_view() with transform
 * for query batches. For separate training/encoding, use train and transform.
 */
template <typename T>
inline device_sq_dataset<T, int64_t, T> build(raft::resources const& res,
                                              params params,
                                              raft::device_matrix_view<const T, int64_t> dataset)
{
  auto quantizer = train(res, params, dataset);
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, dataset.extent(0), dataset.extent(1));
  transform(res, quantizer.view(), dataset, codes.view());
  auto const error =
    max_quantization_error(res, quantizer.view(), dataset, raft::make_const_mdspan(codes.view()));
  using result_type = device_sq_dataset<T, int64_t, T>;
  return result_type{std::move(codes),
                     typename result_type::manifest_type{std::move(quantizer), error}};
}

/** @} */

}  // namespace preprocessing::quantize::scalar

}  // namespace CUVS_EXPORT cuvs
