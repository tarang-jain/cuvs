/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>

#include <cstdint>
#include <type_traits>

namespace cuvs::distance::detail {

template <typename RawElementT, typename ElementT>
struct mapped_tile_load_op {
  CUTLASS_HOST_DEVICE
  ElementT operator()(RawElementT x) const
  {
    if constexpr (std::is_same_v<RawElementT, int8_t>) {
      return static_cast<ElementT>(x) / ElementT{128};
    } else if constexpr (std::is_same_v<RawElementT, uint8_t>) {
      return static_cast<ElementT>(x) / ElementT{256};
    } else {
      return static_cast<ElementT>(x);
    }
  }
};

/**
 * CUTLASS tile iterator adapter that reads quantized source elements but exposes
 * mapped math-type fragments to the MMA mainloop.
 */
template <typename RawElementT,
          typename Shape_,
          typename Element_,
          typename Layout_,
          int AdvanceRank,
          typename ThreadMap_,
          int AccessSize = ThreadMap_::kElementsPerAccess>
class MappedPredicatedTileIterator {
 public:
  using Shape                   = Shape_;
  using Element                 = Element_;
  using Layout                  = Layout_;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap               = ThreadMap_;

  using Index       = typename Layout::Index;
  using LongIndex   = typename Layout::LongIndex;
  using TensorRef   = cutlass::TensorRef<Element, Layout>;
  using TensorView  = cutlass::TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer         = Element*;
  using NonConstPointer = typename cutlass::platform::remove_const<Element>::type*;
  using AccessType      = cutlass::
    AlignedArray<Element, AccessSize, (AccessSize * cutlass::sizeof_bits<Element>::value / 8)>;
  using Fragment =
    cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

 private:
  using RawIterator = cutlass::transform::threadblock::
    PredicatedTileIterator<Shape, RawElementT, Layout, AdvanceRank, ThreadMap, AccessSize>;
  using RawFragment = typename RawIterator::Fragment;

 public:
  using Mask                          = typename RawIterator::Mask;
  static int const kAccessesPerVector = RawIterator::kAccessesPerVector;

  class Params {
   private:
    friend MappedPredicatedTileIterator;
    typename RawIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params(Layout const& layout) : params_(layout) {}

    Params() = default;
  };

 private:
  RawIterator raw_iterator_;
  mapped_tile_load_op<RawElementT, Element> map_op_;

 public:
  MappedPredicatedTileIterator() = default;

  CUTLASS_HOST_DEVICE
  MappedPredicatedTileIterator(Params const& params,
                               Pointer pointer,
                               TensorCoord extent,
                               int thread_id,
                               TensorCoord const& threadblock_offset)
    : raw_iterator_(params.params_,
                    reinterpret_cast<RawElementT*>(pointer),
                    extent,
                    thread_id,
                    threadblock_offset)
  {
  }

  CUTLASS_HOST_DEVICE
  MappedPredicatedTileIterator(Params const& params,
                               Pointer pointer,
                               TensorCoord extent,
                               int thread_id)
    : MappedPredicatedTileIterator(params, pointer, extent, thread_id, cutlass::make_Coord(0, 0))
  {
  }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset)
  {
    raw_iterator_.add_pointer_offset(pointer_offset);
  }

  CUTLASS_HOST_DEVICE
  MappedPredicatedTileIterator& operator++()
  {
    ++raw_iterator_;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  MappedPredicatedTileIterator operator++(int)
  {
    MappedPredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { raw_iterator_.clear_mask(enable); }

  CUTLASS_HOST_DEVICE
  void enable_mask() { raw_iterator_.enable_mask(); }

  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) { raw_iterator_.set_mask(mask); }

  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) { raw_iterator_.get_mask(mask); }

  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment& frag, Index pointer_offset)
  {
    RawFragment raw_frag;
    raw_frag.clear();
    raw_iterator_.load_with_pointer_offset(raw_frag, pointer_offset);
    convert_fragment(frag, raw_frag);
  }

  CUTLASS_DEVICE
  void load(Fragment& frag)
  {
    RawFragment raw_frag;
    raw_frag.clear();
    raw_iterator_.load(raw_frag);
    convert_fragment(frag, raw_frag);
  }

 private:
  CUTLASS_DEVICE
  void convert_fragment(Fragment& frag, RawFragment const& raw_frag) const
  {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Fragment::kElements; ++i) {
      frag[i] = map_op_(raw_frag[i]);
    }
  }
};

}  // namespace cuvs::distance::detail
