---
slug: api-reference/cpp-api-core-dataset-sq
---

# SQ

_Source header: `cuvs/core/dataset/sq.hpp`_

## Types

<a id="sq-spec"></a>
### sq_spec

Scalar-quantized dataset specification: every component of a vector becomes one code, so the

manifest holds the quantizer and the dimension is still a property of the data.

```cpp
template <typename MathT, typename CodePolicy>
struct sq_spec;
```

<a id="sq-spec-manifest-view-type"></a>
### sq_spec::manifest_view_type

Quantizer parameters and a bound on this dataset's Euclidean reconstruction error.

The quantizer base keeps existing transform/inverse_transform calls source-compatible.

```cpp
struct manifest_view_type;
```

<a id="sq-spec-manifest-type"></a>
### sq_spec::manifest_type

Owns the quantizer and the maximum reconstruction error for the associated codes.

Existing quantizer constructors and parameter access remain available.

```cpp
struct manifest_type;
```

## Scalar-quantized dataset construction

<a id="preprocessing-quantize-scalar-build"></a>
### preprocessing::quantize::scalar::build

Train and encode a device corpus and measure its maximum reconstruction error. Synchronizes the stream to return the dataset-wide error bound on the host.

```cpp
template <typename T>
inline device_sq_dataset<T, int64_t, T> build(raft::resources const& res,
params params,
raft::device_matrix_view<const T, int64_t> dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | Resources for training, encoding, and error measurement. |
| `params` | in | `params` | Scalar quantizer parameters. |
| `dataset` | in | `raft::device_matrix_view<const T, int64_t>` | Original device corpus to train on and encode. |

**Returns**

`inline device_sq_dataset<T, int64_t, T>`

Dataset owning codes, quantizer, and a conservative maximum L2 reconstruction error. Reuse result.const_manifest_view() with transform for query batches. For separate training/encoding, use train and transform.
