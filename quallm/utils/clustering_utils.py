# quallm/utils/clustering_utils.py
from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import umap
import hdbscan
from quallm.embedding_client import EmbeddingClient

def get_embeddings(texts: Union[str, List[str], pd.Series], 
                   embedding_client: EmbeddingClient) -> np.ndarray:
    """Get embeddings for texts using the provided embedding client."""
    embeddings = embedding_client.embed(texts, allow_na=True)
    return np.array(embeddings)

def reduce_dimensions(embeddings: np.ndarray,
                     n_components: int = 5,
                     n_neighbors: int = 15,
                     min_dist: float = 0.1,
                     metric: str = 'euclidean',
                     random_state: int = 1234,
                     n_jobs: int = 1) -> np.ndarray:
    """Reduce embedding dimensions using UMAP."""
    n_samples = embeddings.shape[0]
    adjusted_n_neighbors = min(n_neighbors, max(2, n_samples // 2))
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=adjusted_n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=n_jobs
    )
    return reducer.fit_transform(embeddings)

def cluster_embeddings(reduced_embeddings: np.ndarray,
                      min_cluster_size: int = 5,
                      min_samples: Optional[int] = None,
                      metric: str = 'euclidean',
                      alpha: float = 1.0,
                      cluster_selection_epsilon: float = 0.0,
                      allow_single_cluster: bool = False,
                      cluster_selection_method: str = 'eom') -> np.ndarray:
    """Cluster reduced embeddings using HDBSCAN."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        alpha=alpha,
        cluster_selection_epsilon=cluster_selection_epsilon,
        allow_single_cluster=allow_single_cluster,
        cluster_selection_method=cluster_selection_method
    )
    return clusterer.fit_predict(reduced_embeddings)

def get_cluster_assignments(texts: Union[str, List[str], pd.Series],
                           embedding_client: EmbeddingClient,
                           min_cluster_size: int = 5,
                           random_state: int = 1234,
                           **clustering_kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """End-to-end clustering pipeline returning cluster labels and metadata.
    
    Returns:
        cluster_labels: Array of cluster assignments (-1 for outliers)
        metadata: Dict with 'reduced_embeddings', 'n_clusters', 'n_outliers'
    """
    embeddings = get_embeddings(texts, embedding_client)
    reduced = reduce_dimensions(embeddings, random_state=random_state)
    labels = cluster_embeddings(reduced, min_cluster_size=min_cluster_size, **clustering_kwargs)
    
    metadata = {
        'reduced_embeddings': reduced,
        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
        'n_outliers': np.sum(labels == -1)
    }
    
    return labels, metadata