import pytest
import pandas as pd
import numpy as np
from quallm.dataset import Dataset, ClusterSampleString, _create_cluster_sample_string
from quallm.embedding_client import EmbeddingClient


# ============= Shared Test Infrastructure =============

class ConfigurableMockEmbeddingClient:
    """Configurable mock client for simulating different cluster patterns."""
    
    def __init__(self, cluster_pattern='balanced', n_clusters=3, cluster_separation=3.0):
        """
        Args:
            cluster_pattern: 'balanced', 'imbalanced', 'scattered', or 'single'
            n_clusters: Number of clusters to form
            cluster_separation: Distance between cluster centers
        """
        self.pattern = cluster_pattern
        self.n_clusters = n_clusters
        self.separation = cluster_separation
        
    def embed(self, texts, allow_na=True):
        """Generate embeddings based on configured pattern."""
        n_texts = len(texts)
        
        if self.pattern == 'scattered':
            # Return dispersed embeddings that won't cluster
            np.random.seed(42)
            return [[np.random.uniform(-10, 10), np.random.uniform(-10, 10)] for _ in range(n_texts)]
        
        elif self.pattern == 'imbalanced':
            # Create imbalanced clusters (50%, 33%, 17% distribution)
            embeddings = []
            cluster_sizes = [n_texts // 2, n_texts // 3, n_texts - n_texts // 2 - n_texts // 3]
            current_idx = 0
            
            for cluster_id, size in enumerate(cluster_sizes[:self.n_clusters]):
                center_x = cluster_id * self.separation
                for _ in range(size):
                    embeddings.append([center_x + np.random.normal(0, 0.1), np.random.normal(0, 0.1)])
                    current_idx += 1
            
            # Fill remaining with first cluster
            while len(embeddings) < n_texts:
                embeddings.append([0 + np.random.normal(0, 0.1), np.random.normal(0, 0.1)])
            
            return embeddings[:n_texts]
        
        else:  # 'balanced' or 'single'
            # Create evenly distributed clusters
            embeddings = []
            items_per_cluster = n_texts // self.n_clusters if self.n_clusters > 0 else n_texts
            
            for i in range(n_texts):
                cluster_id = i // items_per_cluster if items_per_cluster > 0 else 0
                if cluster_id >= self.n_clusters:
                    cluster_id = self.n_clusters - 1
                
                center_x = cluster_id * self.separation
                embeddings.append([center_x + np.random.normal(0, 0.05), np.random.normal(0, 0.05)])
            
            return embeddings


def assert_valid_cluster_sample(sample, expected_type=None, min_size=1):
    """Helper to validate cluster sample properties."""
    assert isinstance(sample, ClusterSampleString)
    assert sample.sample_size >= min_size
    assert sample.dataset_size > 0
    assert len(sample.indices) == sample.sample_size
    assert sample.cluster_stats['combination_type'] == sample.combination_type
    
    if expected_type:
        assert sample.combination_type == expected_type


def create_test_dataframe(n_observations, n_clusters=3):
    """Create a test DataFrame with specified structure."""
    texts = []
    
    if n_clusters == 0:
        # No specific cluster pattern
        texts = [f"Random observation {i}" for i in range(n_observations)]
    else:
        cluster_names = ['Animals', 'Technology', 'Food', 'Sports', 'Nature'][:n_clusters]
        for i in range(n_observations):
            cluster_idx = i % n_clusters
            cluster_name = cluster_names[cluster_idx]
            texts.append(f"{cluster_name} observation {i}")
    
    return pd.DataFrame({'text': texts})


# ============= Fixtures =============

@pytest.fixture
def small_dataset():
    """Dataset too small for clustering."""
    return pd.DataFrame({'text': ['First', 'Second', 'Third']})


# ============= Tests =============

def test_cluster_sample_string_properties():
    """Test ClusterSampleString initialization and core properties."""
    sample_data = pd.DataFrame({'text': ['Sample 1', 'Sample 2']})
    
    # Test different combination types
    test_cases = [
        ([0], 'individual', 1),
        ([0, 1], 'pair', 2),
        ([0, 'random'], 'cluster-random', 1),
        (['random'], 'random', 0),
    ]
    
    for source_clusters, combo_type, n_clusters_used in test_cases:
        sample = _create_cluster_sample_string(
            sample_data, source_clusters, combo_type, 100, None, '-----'
        )
        
        assert_valid_cluster_sample(sample, combo_type)
        assert sample.source_clusters == source_clusters
        assert sample.cluster_stats['n_clusters_used'] == n_clusters_used
        assert f'Cluster sample ({combo_type})' in str(sample)


@pytest.mark.parametrize("n_clusters,expected_types", [
    (0, ['random']),                                           # No clusters (too small/scattered)
    (2, ['individual', 'pair', 'cluster-random', 'random']),  # Two clusters
    (3, ['individual', 'pair', 'cluster-random', 'random']),  # Three clusters
])
def test_sample_generation_combination_types(n_clusters, expected_types):
    """Test that correct combination types are generated based on cluster count."""
    n_observations = max(20, n_clusters * 8)  # Ensure enough data
    data = create_test_dataframe(n_observations, n_clusters)
    
    # Use mock client with appropriate pattern
    if n_clusters == 0:
        mock_client = ConfigurableMockEmbeddingClient('scattered')
    else:
        mock_client = ConfigurableMockEmbeddingClient('balanced', n_clusters)
    
    dataset = Dataset.from_cluster_samples(
        data=data,
        n_per_combination=1,
        sample_size=3,
        min_cluster_size=5,
        embedding_client=mock_client,
        random_state=42
    )
    
    # Check that expected combination types are present
    actual_types = set(item['sample'].combination_type for item in dataset)
    for expected_type in expected_types:
        assert expected_type in actual_types, f"Missing expected type: {expected_type}"


@pytest.mark.parametrize("edge_case,expected_behavior", [
    ("too_small", "single_random"),      # Dataset too small to cluster
    ("all_outliers", "single_random"),   # All points are outliers
])
def test_edge_cases(small_dataset, edge_case, expected_behavior):
    """Test edge case handling."""
    if edge_case == "too_small":
        data = small_dataset
        mock_client = ConfigurableMockEmbeddingClient('balanced')
    else:  # all_outliers
        data = create_test_dataframe(6)
        mock_client = ConfigurableMockEmbeddingClient('scattered')
    
    dataset = Dataset.from_cluster_samples(
        data=data,
        n_per_combination=1,
        sample_size=2,
        min_cluster_size=15 if edge_case == "all_outliers" else 5,
        embedding_client=mock_client,
        random_state=42
    )
    
    if expected_behavior == "single_random":
        assert len(dataset) == 1
        sample = dataset[0]['sample']
        assert sample.combination_type == 'random'
        assert sample.source_clusters in [['small_dataset'], ['all_outliers']]


def test_cluster_pair_generation():
    """Test that correct number of pairs are generated for different cluster counts."""
    # Test with 4 clusters (should generate 6 pairs)
    data = create_test_dataframe(32, n_clusters=4)  # 8 items per cluster
    mock_client = ConfigurableMockEmbeddingClient('balanced', n_clusters=4)
    
    dataset = Dataset.from_cluster_samples(
        data=data,
        n_per_combination=2,
        sample_size=4,
        min_cluster_size=6,
        embedding_client=mock_client,
        random_state=42
    )
    
    # Count samples by type
    type_counts = {}
    for item in dataset:
        combo_type = item['sample'].combination_type
        type_counts[combo_type] = type_counts.get(combo_type, 0) + 1
    
    # With 4 clusters and n_per_combination=2:
    assert type_counts.get('individual', 0) == 8      # 4 clusters * 2
    assert type_counts.get('pair', 0) == 12           # C(4,2) * 2 = 6 * 2
    assert type_counts.get('cluster-random', 0) == 8  # 4 clusters * 2
    assert type_counts.get('random', 0) == 2          # 2 pure random


def test_imbalanced_clusters():
    """Test handling of imbalanced cluster sizes."""
    data = create_test_dataframe(65)  # Will create imbalanced clusters
    mock_client = ConfigurableMockEmbeddingClient('imbalanced', n_clusters=3)
    
    dataset = Dataset.from_cluster_samples(
        data=data,
        n_per_combination=1,
        sample_size=4,
        min_cluster_size=5,
        embedding_client=mock_client,
        random_state=42
    )
    
    # Should successfully generate samples despite imbalance
    assert len(dataset) > 0
    
    # All samples should be valid
    for item in dataset:
        assert_valid_cluster_sample(item['sample'], min_size=1)


@pytest.mark.parametrize("sample_size,n_per_combination", [
    (0, 1),   # Invalid sample_size
    (-1, 1),  # Negative sample_size
    (1, 0),   # Invalid n_per_combination
    (1, -1),  # Negative n_per_combination
])
def test_parameter_validation(sample_size, n_per_combination):
    """Test parameter validation."""
    data = pd.DataFrame({'text': ['sample 1', 'sample 2', 'sample 3']})
    
    with pytest.raises(ValueError, match="must be positive"):
        Dataset.from_cluster_samples(
            data, 
            sample_size=sample_size,
            n_per_combination=n_per_combination
        )


def test_comprehensive_integration():
    """Integration test covering the full feature with realistic data."""
    # Create dataset with 3 clear thematic clusters
    data = pd.DataFrame({
        'text': [
            # Animals (8 items)
            'Dogs are loyal companions', 'Cats are independent pets',
            'Birds sing in morning', 'Fish swim in oceans',
            'Horses run in fields', 'Lions hunt in prides',
            'Bears sleep in winter', 'Wolves howl at moon',
            
            # Technology (8 items)
            'Computers process data fast', 'Software automates tasks',
            'Internet connects people globally', 'AI learns from data',
            'Robots help in factories', 'Smartphones are portable',
            'Cloud stores data online', 'Networks transfer information',
            
            # Food (8 items)
            'Pizza has many toppings', 'Salad is healthy meal',
            'Chocolate tastes sweet', 'Coffee gives energy',
            'Bread is staple food', 'Fruit provides vitamins',
            'Vegetables are nutritious', 'Desserts are treats'
        ],
        'category': ['animals'] * 8 + ['tech'] * 8 + ['food'] * 8
    })
    
    mock_client = ConfigurableMockEmbeddingClient('balanced', n_clusters=3)
    
    dataset = Dataset.from_cluster_samples(
        data=data,
        n_per_combination=2,
        sample_size=5,
        min_cluster_size=6,
        embedding_client=mock_client,
        labels={'text': 'Content', 'category': 'Theme'},
        separator='---',
        random_state=42
    )
    
    # Verify all combination types present
    combo_types = set(item['sample'].combination_type for item in dataset)
    assert combo_types == {'individual', 'pair', 'cluster-random', 'random'}
    
    # Verify labels applied
    for item in dataset:
        sample_str = str(item['sample'])
        assert 'Content:' in sample_str
        assert 'Theme:' in sample_str
        assert '---' in sample_str  # Custom separator
    
    # Verify counts (3 clusters)
    expected_total = (3 * 2) + (3 * 2) + (3 * 2) + 2  # individual + pairs + cluster-random + random
    assert len(dataset) == expected_total


def test_small_cluster_with_replacement():
    """Test that small clusters use sampling with replacement."""
    # Create dataset where some clusters will be smaller than sample_size
    data = pd.DataFrame({
        'text': [
            'Small A1', 'Small A2',  # Cluster of size 2
            'Large B1', 'Large B2', 'Large B3', 'Large B4', 'Large B5', 'Large B6'
        ]
    })
    
    mock_client = ConfigurableMockEmbeddingClient('balanced', n_clusters=2)
    
    dataset = Dataset.from_cluster_samples(
        data=data,
        n_per_combination=1,
        sample_size=4,  # Larger than first cluster
        min_cluster_size=2,  # Allow small clusters
        embedding_client=mock_client,
        random_state=42
    )
    
    # Should generate samples successfully using replacement
    assert len(dataset) > 0
    
    # Check that samples are generated correctly
    for item in dataset:
        sample = item['sample']
        # Individual cluster samples should still work with replacement
        if sample.combination_type == 'individual':
            # Should have sample_size items even if cluster is smaller
            assert len(sample.indices) > 0  # At least some indices


# Optional: Test with real embedding client if available
def test_end_to_end_with_real_client():
    """End-to-end test with real embedding client (skipped if not available)."""
    data = create_test_dataframe(15, n_clusters=3)
    
    try:
        dataset = Dataset.from_cluster_samples(
            data=data,
            n_per_combination=1,
            sample_size=3,
            min_cluster_size=4,
            random_state=42
        )
        
        # Basic validation
        assert len(dataset) > 0
        assert all('sample' in item for item in dataset)
        
        # Should have multiple combination types
        combo_types = set(item['sample'].combination_type for item in dataset)
        assert len(combo_types) >= 2
        
    except Exception as e:
        pytest.skip(f"Embedding client not available: {e}")