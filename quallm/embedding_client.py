from abc import ABC, abstractmethod
from typing import List
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

    def embed(self, texts: List[str], allow_na: bool = False) -> List[List[float]]:
        """
        Processes a batch of texts to generate their embeddings using the selected provider.

        Args:
            texts: A list of text strings for which embeddings are requested.
            allow_na: Boolean flag indicating whether to allow NA values. 
                      If True, NaN or None values within texts are replaced with an empty string; 
                      if False, such values may cause errors.
                      Default is False.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        # Handle pandas Series/Column
        if isinstance(texts, pd.Series):
            if allow_na: # Convert NaN/None values to empty strings
                texts = texts.fillna('')
            texts = texts.tolist()
        return self.embedding_provider.embed(texts)
    
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
                
    def sort_embeddings(self, embeddings):
        """Sorts embeddings using UMAP and HDBSCAN."""
        
        # Reduce dimensionality with UMAP and cluster using HDBSCAN
        umap_reducer = umap.UMAP(n_components=2, random_state=1234)
        umap_embeddings = umap_reducer.fit_transform(embeddings)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        cluster_labels = clusterer.fit_predict(umap_embeddings)
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