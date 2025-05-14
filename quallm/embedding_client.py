from abc import ABC, abstractmethod
from typing import List, Union, Optional
import ollama
from litellm import embedding as litellm_embedding
import numpy as np
import pandas as pd
import umap
import hdbscan

default_ollama_model = "nomic-embed-text"
default_litellm_model = "text-embedding-ada-002"

class BaseEmbeddingProvider(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of text inputs.
        Returns:
            A list where each element is a list of floats representing a text embedding.
        """
        pass

class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str = default_ollama_model):
        super().__init__(model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = ollama.embed(model=self.model, input=texts).embeddings
        return embeddings

class LitellmEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str = default_litellm_model):
        super().__init__(model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = litellm_embedding(model=self.model, input=texts).data
        embeddings = [entry['embedding'] for entry in embeddings]
        return embeddings

class EmbeddingClient:
    def __init__(self, provider: str = "ollama", model: str = None):
        """
        Initializes the EmbeddingClient.

        Args:
            provider: The embedding provider to use ("ollama" or "litellm").
            model: The model identifier to use. If not provided, a default is used per provider.
        """
        self.provider_name = provider.lower()
        if self.provider_name == "ollama":
            self.embedding_provider = OllamaEmbeddingProvider(model if model else default_ollama_model)
        elif self.provider_name == "litellm":
            self.embedding_provider = LitellmEmbeddingProvider(model if model else default_litellm_model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def embed(self, texts: Union[str, List[str], pd.Series], allow_na: bool = False) -> List[List[float]]:
        """
        Processes a batch of texts to generate their embeddings using the selected provider.

        Args:
            texts: A list of text strings for which embeddings are requested (or a single string or a pandas Series).
            allow_na: Boolean flag indicating whether to allow NA values. 
                      If True, NaN or None values within texts are replaced with an empty string; 
                      if False, such values may cause errors.
                      Default is False.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        # Validate input
        # Coerce texts to List[str] if it's a single string or a pandas Series
        if isinstance(texts, str):
            texts = [texts]  # texts is now List[str]
        elif isinstance(texts, pd.Series):
            # If not allowing NAs, validate that the Series has no NAs
            if not allow_na and texts.isnull().any():
                raise ValueError("Input pandas Series contains NA/NaN values and 'allow_na' is False.")
            # Convert Series to list; Items retain their types (e.g., str, None, np.nan, int, float)
            texts = texts.tolist()
        elif not isinstance(texts, list):
            # If not a string, not a Series, and not a list, it's an unsupported type
            raise TypeError(
                f"Input 'texts' must be a string, list, or pandas Series. Got {type(texts)}."
            )

        # Validate the contents of the 'texts' list based on 'allow_na'
        if not allow_na:
            # If NAs are not allowed, every item in the list must be a string
            for i, item in enumerate(texts):
                if not isinstance(item, str):
                    if item is None or (isinstance(item, float) and np.isnan(item)):
                        raise ValueError(
                            f"Input list contains an NA value ('{item}') at index {i}, but 'allow_na' is False."
                        )
                    raise TypeError(
                        f"All items in 'texts' must be strings when 'allow_na' is False. "
                        f"Found item '{item}' (type: {type(item)}) at index {i}."
                    )
        else:
            # If NAs are allowed, items must be string, None, or float (for np.nan)
            for i, item in enumerate(texts):
                if not (isinstance(item, str) or item is None or (isinstance(item, float) and np.isnan(item))):
                    raise TypeError(
                        f"Input list 'texts' at index {i} contains an item of unsupported type '{type(item)}' ('{item}') "
                        f"when 'allow_na' is True. Allowed types are string, None, or float (for NaN)."
                    )
        if allow_na:
            processed_texts = []
            for item in texts:
                if item is None or (isinstance(item, float) and np.isnan(item)):
                    processed_texts.append('')
                else:
                    processed_texts.append(str(item))
        else:
            if any(item is None or (isinstance(item, float) and np.isnan(item)) for item in texts):
                raise ValueError("Input list contains NA/NaN values and allow_na is False.")
            processed_texts = [str(item) for item in texts]

        return self.embedding_provider.embed(processed_texts)
    
    def sort(self, data):
        """
        Sort a list of texts or embeddings by their semantic similarity.
        This version reduces the embeddings via UMAP (2 dimensions, fixed seed), clusters them with
        HDBSCAN (min cluster size 5), sorts points within each cluster by their distance from the
        cluster centroid, and then sorts clusters globally by the distance of their centroids from
        the overall (global) centroid.

        Args:
            data: Can be one of the following:
                - str or list of str or pd.Series: Text to embed
                - list of numeric lists/arrays: Pre-computed embeddings
                - pd.DataFrame: Multiple columns to embed separately and combine

        Returns:
            A DataFrame with columns:
            - "orig_index": original index of the input
            - "cluster": cluster label (noise is treated like any other cluster)
            - "global_order": position in the overall ordering
        """
        # Compute or retrieve embeddings
        if isinstance(data, pd.DataFrame):
            # Compute embeddings for the first column and convert to NumPy array
            first_col_embeddings = np.array(self.embed(data.iloc[:, 0], allow_na=True))
            combined_embeddings = np.zeros(first_col_embeddings.shape)
            for col in data.columns:
                col_embeddings = np.array(self.embed(data[col], allow_na=True))
                combined_embeddings += col_embeddings
            embeddings = combined_embeddings
        else:
            embeddings = self.embed(data, allow_na=True)
                
        order_df = self.sort_embeddings(embeddings)
        
        # Append the original data as additional columns in sorted order
        if isinstance(data, pd.DataFrame):
            # Reset index, then select rows using 'orig_index' and concatenate
            sorted_data = data.reset_index(drop=True).iloc[order_df["original_index"].values].reset_index(drop=True)
            result = pd.concat([order_df.reset_index(drop=True), sorted_data], axis=1)
        else:
            # For list or series inputs
            original_data = data if not isinstance(data, pd.Series) else data.tolist()
            sorted_data = [original_data[i] for i in order_df["original_index"]]
            order_df["sorted_data"] = sorted_data
            result = order_df
            
        return result.reset_index(drop=True)
    
    def _validate_array(self, array: np.ndarray) -> None:
        """Helper method to validate the type and shape of an input array."""
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if array.ndim != 2:
            raise ValueError("Input array must be 2-dimensional.")
        if array.shape[0] == 0:
            raise ValueError("Input array must have at least one row.")

    def _reduce_dimensions(self,
                           embeddings_array: np.ndarray,
                           n_components: int = 2,
                           n_neighbors: int = 15,
                           min_dist: float = 0.1,
                           metric: str = 'euclidean',
                           random_state: int = 1234) -> np.ndarray:
        """Helper method to reduce dimensionality of embeddings using UMAP."""
        self._validate_array(embeddings_array)
        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        umap_embeddings = umap_reducer.fit_transform(embeddings_array)
        return umap_embeddings

    def _cluster_embeddings(self,
                            embeddings_to_cluster: np.ndarray, # UMAP-reduced embeddings
                            min_cluster_size: int = 5,
                            min_samples: Optional[int] = None, # Defaults to min_cluster_size in HDBSCAN
                            cluster_metric: str = 'euclidean', # HDBSCAN's distance metric
                            alpha: float = 1.0,
                            cluster_selection_epsilon: float = 0.0,
                            allow_single_cluster: bool = False, # HDBSCAN parameter
                            cluster_selection_method: str = 'eom' # HDBSCAN parameter ('eom' or 'leaf')
                           ) -> np.ndarray:
        """Helper method to cluster embeddings using HDBSCAN."""
        self._validate_array(embeddings_to_cluster)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples, # Pass None to let HDBSCAN use its default (min_cluster_size)
            metric=cluster_metric,
            alpha=alpha,
            cluster_selection_epsilon=cluster_selection_epsilon,
            allow_single_cluster=allow_single_cluster,
            cluster_selection_method=cluster_selection_method
        )
        cluster_labels = clusterer.fit_predict(embeddings_to_cluster)
        return cluster_labels
                
    def sort_embeddings(self, embeddings: Union[List[List[float]], np.ndarray]) -> pd.DataFrame:
        """Sorts embeddings using UMAP and HDBSCAN."""
        # Validate input
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        self._validate_array(embeddings)
        
        # Reduce dimensionality with UMAP and cluster using HDBSCAN
        umap_embeddings = self._reduce_dimensions(embeddings)
        cluster_labels = self._cluster_embeddings(umap_embeddings)
        df = pd.DataFrame({
            "original_index": np.arange(len(embeddings)),
            "cluster": cluster_labels,
            "umap0": umap_embeddings[:, 0],
            "umap1": umap_embeddings[:, 1],
        })

        # Compute cluster centroids and global centroid in UMAP space
        centroid_cols = ["umap0", "umap1"]
        cluster_centroids = df.groupby("cluster")[centroid_cols].mean()
        global_centroid = df[centroid_cols].mean().values

        # For each point, compute distance from its cluster centroid
        def point_distance(row):
            centroid = cluster_centroids.loc[row["cluster"]].values
            return np.linalg.norm(row[centroid_cols].values - centroid)
        df["within_dist"] = df.apply(point_distance, axis=1)

        # Sort points within each cluster by their distance from the cluster centroid,
        # then sort clusters globally by their distance from the global centroid
        df = df.sort_values(by=["cluster", "within_dist"])
        cluster_centroids["global_dist"] = cluster_centroids.apply(
            lambda row: np.linalg.norm(row.values - global_centroid), axis=1
        )
        # Define a global ordering for clusters
        sorted_clusters = cluster_centroids.sort_values("global_dist").index.tolist()
        df["cluster"] = pd.Categorical(df["cluster"], categories=sorted_clusters, ordered=True)
        df = df.sort_values(by=["cluster", "within_dist"])
        df["global_order"] = range(len(df))
        df = df.reset_index(drop=True)

        # Return the ordering information: original index, cluster label, and global order
        return df[["original_index", "cluster", "global_order"]]

    def test(self):
        """Test the embedding client with a simple input."""
        data = [
            "Apple", "Toyota", "Orange", "Violin", "Honda", "Banana", "Ford", 
            "Piano", "Grape", "Trumpet", "Nissan", "Chevrolet", "Saxophone", 
            "Cello", "Strawberry", "Blueberry", "Flute", "Mazda", "Clarinet", 
            "Pear", "Guitar", "Kiwi", "Jeep", "Harp", "Dodge", "Mandolin", 
            "Peach", "Drums", "Subaru", "Mango", "Tesla", "Banjo", "Accordion"
        ]
        return self.embed(data)