import pytest
import ollama
import pandas as pd
import numpy as np
import os
from quallm.embedding_client import EmbeddingClient, OllamaEmbeddingProvider, default_ollama_model

# Test Configuration
# Tests will use this model. Ensure it's pulled in your Ollama instance.
# e.g., ollama pull nomic-embed-text
TEST_OLLAMA_MODEL = os.environ.get("QUALLM_TEST_OLLAMA_MODEL", default_ollama_model)

# Fixtures
@pytest.fixture
def ollama_provider_default():
    return OllamaEmbeddingProvider()

@pytest.fixture
def ollama_provider_custom():
    return OllamaEmbeddingProvider(model=TEST_OLLAMA_MODEL)

@pytest.fixture
def embedding_client_default_ollama():
    return EmbeddingClient(provider="ollama")

@pytest.fixture
def embedding_client_custom_ollama():
    return EmbeddingClient(provider="ollama", model=TEST_OLLAMA_MODEL)

SAMPLE_TEXTS = ["Hello world from pytest", "Ollama is great with pytest!"]
SINGLE_TEXT = "This is a single test text for pytest."

# Tests for OllamaEmbeddingProvider

def test_ollama_provider_init_default_model(ollama_provider_default):
    assert ollama_provider_default.model == default_ollama_model

def test_ollama_provider_init_custom_model(ollama_provider_custom):
    assert ollama_provider_custom.model == TEST_OLLAMA_MODEL

def test_ollama_provider_embed_single_text(ollama_provider_custom):
    embeddings = ollama_provider_custom.embed([SINGLE_TEXT])
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)
    assert all(isinstance(x, float) for x in embeddings[0])
    assert len(embeddings[0]) > 0

def test_ollama_provider_embed_multiple_texts(ollama_provider_custom):
    embeddings = ollama_provider_custom.embed(SAMPLE_TEXTS)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(SAMPLE_TEXTS)
    first_embedding_len = 0
    for i, emb in enumerate(embeddings):
        assert isinstance(emb, list)
        assert all(isinstance(x, float) for x in emb)
        assert len(emb) > 0
        if i == 0:
            first_embedding_len = len(emb)
        else:
            assert len(emb) == first_embedding_len

def test_ollama_provider_embed_empty_list(ollama_provider_custom):
    # Test that an empty list is returned when embedding an empty list of texts
    result = ollama_provider_custom.embed([])
    assert result == []
    assert isinstance(result, list)

def test_ollama_provider_embed_list_with_empty_string(ollama_provider_custom):
    embeddings = ollama_provider_custom.embed([""]) # Ollama should handle empty strings
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)
    assert all(isinstance(x, float) for x in embeddings[0])
    assert len(embeddings[0]) > 0

def test_ollama_provider_embed_non_existent_model():
    provider = OllamaEmbeddingProvider(model="this-model-does-not-exist-for-pytest-12345")
    with pytest.raises(ollama.ResponseError) as excinfo:
        provider.embed([SINGLE_TEXT])
    assert excinfo.value.status_code >= 400 # e.g., 404
    error_message_lower = str(excinfo.value.error).lower()
    assert "not found" in error_message_lower
    assert "this-model-does-not-exist-for-pytest-12345" in error_message_lower


# Tests for EmbeddingClient with Ollama Provider

def test_embedding_client_init_ollama_default_model(embedding_client_default_ollama):
    assert isinstance(embedding_client_default_ollama.embedding_provider, OllamaEmbeddingProvider)
    assert embedding_client_default_ollama.embedding_provider.model == default_ollama_model

def test_embedding_client_init_ollama_custom_model(embedding_client_custom_ollama):
    assert isinstance(embedding_client_custom_ollama.embedding_provider, OllamaEmbeddingProvider)
    assert embedding_client_custom_ollama.embedding_provider.model == TEST_OLLAMA_MODEL

def test_embedding_client_embed_single_string(embedding_client_custom_ollama):
    embeddings = embedding_client_custom_ollama.embed(SINGLE_TEXT)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)
    assert all(isinstance(x, float) for x in embeddings[0])
    assert len(embeddings[0]) > 0

def test_embedding_client_embed_list_of_strings(embedding_client_custom_ollama):
    embeddings = embedding_client_custom_ollama.embed(SAMPLE_TEXTS)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(SAMPLE_TEXTS)
    first_embedding_len = 0
    for i, emb in enumerate(embeddings):
        assert isinstance(emb, list)
        assert all(isinstance(x, float) for x in emb)
        if i == 0:
            first_embedding_len = len(emb)
        else:
            assert len(emb) == first_embedding_len

def test_embedding_client_embed_pandas_series(embedding_client_custom_ollama):
    series = pd.Series(SAMPLE_TEXTS)
    embeddings = embedding_client_custom_ollama.embed(series)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(SAMPLE_TEXTS)
    for emb in embeddings:
        assert isinstance(emb, list)
        assert all(isinstance(x, float) for x in emb)

def test_embedding_client_embed_input_validation_type_error(embedding_client_custom_ollama):
    with pytest.raises(TypeError):
        embedding_client_custom_ollama.embed(12345)
    with pytest.raises(TypeError):
        embedding_client_custom_ollama.embed({"text": "a dict"})

def test_embedding_client_embed_input_validation_na_disallowed(embedding_client_custom_ollama):
    with pytest.raises(ValueError):
        embedding_client_custom_ollama.embed([SINGLE_TEXT, None], allow_na=False)
    with pytest.raises(ValueError):
        embedding_client_custom_ollama.embed(pd.Series([SINGLE_TEXT, np.nan]), allow_na=False)

def test_embedding_client_embed_input_validation_na_allowed(embedding_client_custom_ollama):
    texts_with_na = [SINGLE_TEXT, None, "Another valid text.", np.nan]
    series_with_na = pd.Series(texts_with_na)

    # NA values are processed as empty strings by the client, Ollama should embed them
    embeddings_list = embedding_client_custom_ollama.embed(texts_with_na, allow_na=True)
    assert len(embeddings_list) == len(texts_with_na)
    assert isinstance(embeddings_list[1], list) and len(embeddings_list[1]) > 0
    assert isinstance(embeddings_list[3], list) and len(embeddings_list[3]) > 0

    embeddings_series = embedding_client_custom_ollama.embed(series_with_na, allow_na=True)
    assert len(embeddings_series) == len(series_with_na)
    assert isinstance(embeddings_series[1], list) and len(embeddings_series[1]) > 0
    assert isinstance(embeddings_series[3], list) and len(embeddings_series[3]) > 0

def test_embedding_client_embed_non_string_in_list_na_disallowed(embedding_client_custom_ollama):
    with pytest.raises(TypeError): # As per your client's implementation
        embedding_client_custom_ollama.embed([SINGLE_TEXT, 123], allow_na=False)

def test_embedding_client_test_method_executes(embedding_client_custom_ollama):
    # This test relies on the EmbeddingClient.test() method and its predefined inputs
    # and on Ollama being available for the TEST_OLLAMA_MODEL.
    embeddings = embedding_client_custom_ollama.test()
    assert isinstance(embeddings, list)
    assert len(embeddings) == 33 # Based on quallm.embedding_client.EmbeddingClient.test()
    for emb in embeddings:
        assert isinstance(emb, list)
        assert all(isinstance(x, float) for x in emb)
        assert len(emb) > 0

def test_embedding_client_sort_method_executes_list(embedding_client_custom_ollama):
    # This also requires Ollama for embedding the data before sorting.
    data_to_sort = ["apple", "banana", "cherry", "apricot", "blueberry", "cantaloupe", "date", "elderberry"]
    sorted_df = embedding_client_custom_ollama.sort(data_to_sort)
    assert isinstance(sorted_df, pd.DataFrame)
    assert "original_index" in sorted_df.columns
    assert "cluster" in sorted_df.columns
    assert "global_order" in sorted_df.columns
    assert "sorted_data" in sorted_df.columns
    assert len(sorted_df) == len(data_to_sort)

def test_embedding_client_sort_method_executes_dataframe(embedding_client_custom_ollama):
    data_df = pd.DataFrame({
        "text1": ["apple", "banana", "cherry", "apricot", "blueberry", "cantaloupe", "date", "elderberry"],
        "text2": ["red", "yellow", "red", "orange", "blue", "orange", "brown", "purple"]
    })
    sorted_df = embedding_client_custom_ollama.sort(data_df)
    assert isinstance(sorted_df, pd.DataFrame)
    assert "original_index" in sorted_df.columns
    assert "text1" in sorted_df.columns
    assert "text2" in sorted_df.columns
    assert len(sorted_df) == len(data_df)