"""
Embedding model wrapper.

Provides a `.encode()` interface compatible with sentence-transformers,
backed by either litellm (if AAAJ_EMBEDDING_MODEL env var is set) or
the local sentence-transformers model as a fallback.
"""

import logging
import os
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import litellm


class LiteLLMEmbedder:
    """Drop-in replacement for SentenceTransformer that calls litellm.embedding."""

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def encode(self, texts, convert_to_tensor: bool = False, batch_size: int = 96):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            kwargs = {"model": self.model, "input": chunk}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            response = litellm.embedding(**kwargs)
            for item in response.data:
                all_embeddings.append(item["embedding"])

        if convert_to_tensor:
            import torch
            tensor = torch.tensor(all_embeddings, dtype=torch.float32)
            return tensor[0] if single else tensor

        import numpy as np
        arr = np.array(all_embeddings, dtype=np.float32)
        return arr[0] if single else arr


def get_embedding_model():
    """Return an embedding model. Uses litellm if AAAJ_EMBEDDING_MODEL is set,
    otherwise falls back to local SentenceTransformer.
    """
    model_name = os.getenv("AAAJ_EMBEDDING_MODEL")
    if model_name:
        try:
            logging.info(f"Using litellm embedding model: {model_name}")
            return LiteLLMEmbedder(model_name)
        except Exception as e:
            logging.warning(
                f"Failed to initialize litellm embedder ({model_name}): {e}. "
                f"Falling back to sentence-transformers."
            )

    from sentence_transformers import SentenceTransformer
    logging.info("Using local sentence-transformers model: all-MiniLM-L6-v2")
    return SentenceTransformer("all-MiniLM-L6-v2")
