import pytest
import numpy as np
import pandas as pd
import os
from quallm.embedding_client import EmbeddingClient
from quallm.utils.clustering_utils import (
    get_embeddings, reduce_dimensions, cluster_embeddings, get_cluster_assignments
)

# Test Configuration
TEST_OLLAMA_MODEL = os.environ.get("QUALLM_TEST_OLLAMA_MODEL", "nomic-embed-text")

# Fixtures
@pytest.fixture
def embedding_client():
    return EmbeddingClient(provider="ollama", model=TEST_OLLAMA_MODEL)

@pytest.fixture
def sample_texts():
    return [
        "Apple is a fruit",
        "Orange is also a fruit", 
        "Banana is a yellow fruit",
        "Grapes are small fruits",
        "Strawberry is a red fruit",
        "Car is a vehicle",
        "Truck is also a vehicle",
        "Motorcycle is a two-wheeled vehicle",
        "Bus is a large vehicle",
        "Bicycle is a human-powered vehicle",
        "Pizza is food",
        "Hamburger is food",
        "Pasta is Italian food",
        "Sushi is Japanese food",
        "Tacos are Mexican food",
        "Dog is an animal",
        "Cat is a pet animal",
        "Lion is a wild animal",
        "Eagle is a bird",
        "Shark is a sea creature"
    ]

@pytest.fixture
def clustered_embeddings():
    """Create embeddings with clear cluster structure for testing."""
    return np.array([
        [1.0, 0.0], [1.1, 0.1],  # cluster 1
        [0.0, 1.0], [0.1, 1.1],  # cluster 2
        [-1.0, 0.0], [-1.1, -0.1]  # cluster 3
    ])

# Combined test for get_embeddings with multiple input types
@pytest.mark.parametrize("input_type", ["list", "single", "series"])
def test_get_embeddings_all_input_types(embedding_client, sample_texts, input_type):
    """Test get_embeddings with different input types."""
    if input_type == "list":
        texts = sample_texts
        expected_len = len(sample_texts)
    elif input_type == "single":
        texts = sample_texts[0]
        expected_len = 1
    else:  # series
        texts = pd.Series(sample_texts)
        expected_len = len(sample_texts)
    
    embeddings = get_embeddings(texts, embedding_client)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == expected_len
    assert embeddings.shape[1] > 0
    assert embeddings.dtype == np.float64

# Combined test for reduce_dimensions with various configurations
@pytest.mark.parametrize("n_samples,n_dims,n_components,n_neighbors", [
    (6, 10, 2, 5),     # Standard case
    (10, 20, 3, 5),    # Higher dimensions
    (4, 10, 2, 15),    # Small dataset (n_neighbors adjustment)
])
def test_reduce_dimensions_configurations(n_samples, n_dims, n_components, n_neighbors):
    """Test reduce_dimensions with various configurations."""
    high_dim = np.random.rand(n_samples, n_dims)
    reduced = reduce_dimensions(
        high_dim, 
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=42
    )
    assert isinstance(reduced, np.ndarray)
    assert reduced.shape == (n_samples, n_components)

# Combined test for cluster_embeddings with various scenarios
@pytest.mark.parametrize("scenario,min_cluster_size,expected_behavior", [
    ("clear_clusters", 2, "forms_clusters"),    # Clear clusters
    ("scattered", 5, "all_noise"),              # All noise
    ("mixed", 3, "some_clusters"),              # Mixed clusters and noise
])
def test_cluster_embeddings_scenarios(clustered_embeddings, scenario, min_cluster_size, expected_behavior):
    """Test cluster_embeddings with various data patterns."""
    if scenario == "clear_clusters":
        # Well-separated clusters
        data = np.array([
            [0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1],
            [5, 5], [5.1, 5], [5, 5.1], [5.1, 5.1],
        ])
    elif scenario == "scattered":
        # Random scattered points
        np.random.seed(42)
        data = np.random.rand(8, 2) * 10
    else:  # mixed
        data = clustered_embeddings
    
    labels = cluster_embeddings(data, min_cluster_size=min_cluster_size)
    
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(data)
    assert all(label >= -1 for label in labels)
    
    if expected_behavior == "forms_clusters":
        assert len(set(labels)) > 1  # Multiple clusters or clusters + noise
    elif expected_behavior == "all_noise":
        # May be all noise or form minimal clusters
        assert len(set(labels)) >= 1

# Comprehensive pipeline test with multiple scenarios
@pytest.mark.parametrize("text_count,min_cluster_size,clustering_kwargs", [
    (20, 5, {}),                                          # Standard case
    (15, 5, {}),                                          # Small dataset
    (20, 5, {"metric": "euclidean", "alpha": 1.0}),     # Custom parameters
])
def test_get_cluster_assignments_pipeline(embedding_client, sample_texts, text_count, min_cluster_size, clustering_kwargs):
    """Test the complete clustering pipeline with various configurations."""
    texts = sample_texts[:text_count]
    labels, metadata = get_cluster_assignments(
        texts, 
        embedding_client,
        min_cluster_size=min_cluster_size,
        random_state=42,
        **clustering_kwargs
    )
    
    # Core assertions for all scenarios
    assert isinstance(labels, np.ndarray)
    assert len(labels) == text_count
    assert all(label >= -1 for label in labels)
    
    # Metadata validation
    assert set(metadata.keys()) == {'reduced_embeddings', 'n_clusters', 'n_outliers'}
    assert metadata['reduced_embeddings'].shape == (text_count, 5)
    assert metadata['n_clusters'] >= 0
    assert metadata['n_outliers'] >= 0
    assert metadata['n_clusters'] + (1 if metadata['n_outliers'] > 0 else 0) == len(set(labels))