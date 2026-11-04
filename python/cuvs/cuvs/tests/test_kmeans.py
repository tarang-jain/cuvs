# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.cluster.kmeans import (
    KMeansParams,
    cluster_cost,
    fit,
    predict,
)
from cuvs.common import Resources
from cuvs.distance import pairwise_distance


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [5, 25])
@pytest.mark.parametrize("n_clusters", [5, 15])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("hierarchical", [True, False])
def test_kmeans_fit(n_rows, n_cols, n_clusters, dtype, hierarchical):
    if hierarchical and dtype == np.float64:
        pytest.skip("hierarchical kmeans doesn't support float64")

    # generate some random input points / centroids
    X_host = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    centroids = device_ndarray(X_host[:n_clusters])
    X = device_ndarray(X_host)

    # compute the inertia, before fitting centroids
    original_inertia = cluster_cost(X, centroids)

    params = KMeansParams(n_clusters=n_clusters, hierarchical=hierarchical)

    # fit the centroids, make sure inertia has gone down
    centroids, inertia, n_iter = fit(params, X, centroids)
    assert n_iter >= 1

    # balanced kmeans doesn't return inertia
    if not hierarchical:
        assert inertia < original_inertia
        assert np.allclose(cluster_cost(X, centroids), inertia, rtol=1e-6)

    # make sure the prediction for each centroid is the centroid itself
    labels, inertia = predict(params, centroids, centroids)
    assert np.all(labels.copy_to_host() == np.arange(labels.shape[0]))


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [5, 25])
@pytest.mark.parametrize("n_clusters", [4, 15])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cluster_cost(n_rows, n_cols, n_clusters, dtype):
    X = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    X_device = device_ndarray(X)

    centroids = X[:n_clusters]
    centroids_device = device_ndarray(centroids)

    inertia = cluster_cost(X_device, centroids_device)

    # compute the nearest centroid to each sample
    distances = pairwise_distance(
        X_device, centroids_device, metric="sqeuclidean"
    ).copy_to_host()
    cluster_ids = np.argmin(distances, axis=1)

    cluster_distances = np.take_along_axis(
        distances, cluster_ids[:, None], axis=1
    )

    # need reduced tolerance for float32
    tol = 1e-3 if dtype == np.float32 else 1e-6
    assert np.allclose(inertia, sum(cluster_distances), rtol=tol, atol=tol)


@pytest.mark.parametrize("n_rows", [1000, 5000])
@pytest.mark.parametrize("n_cols", [10, 100])
@pytest.mark.parametrize("n_clusters", [8, 16])
@pytest.mark.parametrize("device_buffer_samples", [0, 100, 239, 500])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("weighted", [False, True])
def test_fit_host_matches_fit_device(
    n_rows, n_cols, n_clusters, device_buffer_samples, dtype, weighted
):
    """
    Test that fit() with host (numpy) data produces the same centroids as
    fit() with device data, when given the same initial centroids.
    Optionally tests with non-uniform sample weights.
    """
    rng = np.random.default_rng(99)
    X_host = rng.random((n_rows, n_cols)).astype(dtype)

    centroid_indices = rng.choice(n_rows, size=n_clusters, replace=False)
    initial_centroids_host = X_host[centroid_indices].copy()

    if weighted:
        sample_weights_host = rng.uniform(0.5, 2.0, size=n_rows).astype(dtype)
        sample_weights_device = device_ndarray(sample_weights_host)
    else:
        sample_weights_host = None
        sample_weights_device = None

    params_device = KMeansParams(
        n_clusters=n_clusters,
        init_method="Array",
        max_iter=20,
        tol=1e-10,
    )
    centroids_regular, inertia_regular, _ = fit(
        params_device,
        device_ndarray(X_host),
        device_ndarray(initial_centroids_host.copy()),
        sample_weights=sample_weights_device,
    )
    centroids_regular = centroids_regular.copy_to_host()

    params_host = KMeansParams(
        n_clusters=n_clusters,
        init_method="Array",
        max_iter=20,
        tol=1e-10,
        device_buffer_samples=device_buffer_samples,
    )
    centroids_batched, inertia_batched, _ = fit(
        params_host,
        X_host,
        centroids=device_ndarray(initial_centroids_host.copy()),
        sample_weights=sample_weights_host,
    )
    centroids_batched = centroids_batched.copy_to_host()

    assert np.allclose(
        centroids_regular, centroids_batched, rtol=1e-3, atol=1e-3
    ), f"max diff: {np.max(np.abs(centroids_regular - centroids_batched))}"

    assert np.allclose(
        inertia_regular, inertia_batched, rtol=1e-3, atol=1e-3
    ), f"max diff: {np.max(np.abs(inertia_regular - inertia_batched))}"


def test_fit_host_with_stream_pool_setter():
    X = np.zeros((128, 2), dtype=np.float32)
    X[64:] = 10
    initial_centroids = device_ndarray(
        np.array([[0, 0], [10, 10]], dtype=np.float32)
    )
    params = KMeansParams(
        n_clusters=2,
        init_method="Array",
        device_buffer_samples=16,
        max_iter=10,
    )
    resources = Resources()
    resources.set_stream_pool(2)

    centroids, inertia, n_iter = fit(
        params, X, initial_centroids, resources=resources
    )
    resources.sync()

    assert n_iter >= 1
    assert np.isfinite(inertia)
    assert np.allclose(centroids.copy_to_host(), [[0, 0], [10, 10]], atol=1e-3)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("host_input", [False, True])
def test_native_byte_lloyd(dtype, host_input):
    origin = -110 if dtype == np.int8 else 170
    rows = np.arange(7)
    values = origin + np.where(rows < 3, rows, 40 + rows)
    samples = (values[:, None] + np.arange(17)[None, :] % 3).astype(dtype)
    weights = np.array([1, 2, 3, 1, 4, 2, 1], dtype=np.float32)
    initial = samples[[0, 3]].astype(np.float32)
    data = samples if host_input else device_ndarray(samples)
    sample_weights = weights if host_input else device_ndarray(weights)
    params = KMeansParams(
        n_clusters=2,
        init_method="Array",
        max_iter=1,
        batch_samples=3,
        device_buffer_samples=2,
    )
    centers, inertia, iterations = fit(
        params, data, device_ndarray(initial), sample_weights=sample_weights
    )
    expected = np.vstack(
        [
            np.average(samples[:3], axis=0, weights=weights[:3]),
            np.average(samples[3:], axis=0, weights=weights[3:]),
        ]
    )
    assert centers.dtype == np.float32
    assert iterations == 1
    np.testing.assert_allclose(centers.copy_to_host(), expected, atol=2e-5)
    distances = (
        (samples[:, None, :].astype(np.float64) - expected[None, :, :]) ** 2
    ).sum(axis=2)
    expected_cost = (
        (distances.min(axis=1) * weights).sum() * len(rows) / weights.sum()
    )
    np.testing.assert_allclose(inertia, expected_cost, rtol=1e-5, atol=1e-3)
    device_data = device_ndarray(samples)
    labels, predicted_cost = predict(
        params, device_data, centers, sample_weights=device_ndarray(weights)
    )
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(
        labels.copy_to_host(), distances.argmin(axis=1)
    )
    np.testing.assert_allclose(
        predicted_cost, expected_cost, rtol=1e-5, atol=1e-3
    )
    np.testing.assert_allclose(
        cluster_cost(device_data, centers),
        distances.min(axis=1).sum(),
        rtol=1e-5,
        atol=1e-3,
    )


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_native_byte_dtypes(dtype):
    data = device_ndarray(np.array([[0], [1], [9], [10]], dtype=dtype))
    params = KMeansParams(n_clusters=2)
    centers, _, _ = fit(params, data)
    assert centers.dtype == np.float32
    with pytest.raises(TypeError):
        fit(params, data, device_ndarray(np.zeros((2, 1), dtype=dtype)))
    with pytest.raises(TypeError):
        fit(
            params,
            data,
            sample_weights=device_ndarray(np.ones(4, dtype=dtype)),
        )
    with pytest.raises(TypeError):
        predict(
            params,
            data,
            centers,
            labels=device_ndarray(np.zeros(4, dtype=np.int32)),
        )
    with pytest.raises(ValueError):
        fit(KMeansParams(n_clusters=2, hierarchical=True), data)
