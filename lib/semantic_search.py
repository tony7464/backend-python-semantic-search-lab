"""
Semantic Search Lab Starter Code

You will build a small semantic retrieval workflow that prepares documents,
embeds documents, embeds a query, compares similarity scores, and returns
ranked top-k results with source information.

The test suite uses a deterministic fake embedding model so grading does not
depend on Ollama, internet access, or a specific model output. Your code should
work with any object that has an .embed(text) method returning a list of numbers.

Optional: You can use OllamaEmbeddingModel locally to try your workflow with
a real embedding model after your pytest tests pass.
"""

from __future__ import annotations

import math
from typing import Any, Protocol


REQUIRED_DOCUMENT_FIELDS = ("id", "title", "category", "summary", "source")


DEFAULT_DOCUMENTS = [
    {
        "id": "DOC-101",
        "title": "Resetting a Forgotten Password",
        "category": "account",
        "summary": (
            "Explains how users can change or reset their account password "
            "after identity verification."
        ),
        "source": "platform-docs/account/password-reset",
        "tags": ["password", "account", "verification"],
    },
    {
        "id": "DOC-102",
        "title": "Fixing Invalid API Authentication Tokens",
        "category": "api",
        "summary": (
            "Explains how to resolve API calls blocked by expired, missing, "
            "or malformed bearer credentials."
        ),
        "source": "platform-docs/api/authentication-tokens",
        "tags": ["api", "authentication", "credentials"],
    },
    {
        "id": "DOC-103",
        "title": "Understanding Monthly Billing Limits",
        "category": "billing",
        "summary": (
            "Explains how usage limits affect invoices, monthly plans, "
            "and account upgrades."
        ),
        "source": "platform-docs/billing/monthly-limits",
        "tags": ["billing", "invoice", "plan"],
    },
    {
        "id": "DOC-104",
        "title": "Troubleshooting Slow Dashboard Loading",
        "category": "performance",
        "summary": (
            "Explains common causes of slow page loads, dashboard latency, "
            "and client-side rendering delays."
        ),
        "source": "platform-docs/performance/dashboard-loading",
        "tags": ["dashboard", "performance", "latency"],
    },
    {
        "id": "DOC-105",
        "title": "Updating User Profile Settings",
        "category": "account",
        "summary": (
            "Explains how users can update email, display name, "
            "notification preferences, and profile details."
        ),
        "source": "platform-docs/account/profile-settings",
        "tags": ["profile", "settings", "email"],
    },
]


class EmbeddingModel(Protocol):
    """Protocol for any embedding model used by the retrieval workflow."""

    def embed(self, text: str) -> list[float]:
        """Return a vector embedding for the provided text."""
        ...


class OllamaEmbeddingModel:
    """
    Optional local embedding model wrapper.

    This class is not required by the pytest suite. It is provided so you can
    try the same workflow locally with Ollama after your core functions work.

    Example:
        embedder = OllamaEmbeddingModel(model_name="embeddinggemma")
        results = semantic_search(
            "Why does the mobile app say my token is expired?",
            DEFAULT_DOCUMENTS,
            embedder,
            top_k=3,
        )
    """

    def __init__(self, model_name: str = "embeddinggemma"):
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")

        try:
            import ollama
        except ImportError as exc:
            raise ImportError(
                "The ollama package is required to use OllamaEmbeddingModel. "
                "Install dependencies with pipenv install."
            ) from exc

        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response["embedding"]


def build_search_text(document: dict[str, Any]) -> str:
    """
    Build the text that should be sent to the embedding model for one document.

    Requirements:
    - Include the document title.
    - Include the document category.
    - Include the document summary.
    - Include tags when available.
    - Return one clean, non-empty string.

    Why:
    Embedding only the title may lose important meaning. Embedding title,
    category, summary, and tags gives the model more context.
    """
    # Combine title, category, summary, and tags into one searchable string.
    if not isinstance(document, dict):
        raise ValueError("Document must be a dictionary.")

    parts = []
    for field in ("title", "category", "summary"):
        value = document.get(field)
        if value is not None and str(value).strip():
            parts.append(str(value).strip())

    tags = document.get("tags")
    if isinstance(tags, str):
        tags = [tags]
    if tags:
        tag_text = ", ".join(str(tag).strip() for tag in tags if str(tag).strip())
        if tag_text:
            parts.append(f"Tags: {tag_text}")

    search_text = ". ".join(part.rstrip(".") for part in parts)
    if not search_text:
        raise ValueError("Document has no searchable text.")
    return search_text + "."


def prepare_documents(raw_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Validate and prepare raw documents for retrieval.

    Requirements:
    - Accept a list of document dictionaries.
    - Validate that each document includes REQUIRED_DOCUMENT_FIELDS.
    - Return a new list of document dictionaries.
    - Do not mutate the original input documents.
    - Add a 'text' field created by build_search_text(document).
    - Preserve useful metadata: id, title, category, summary, source, and tags.

    Raises:
    - ValueError if a required field is missing.
    - ValueError if raw_documents is empty.
    """
    # Copy so the caller's documents are never mutated.
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("raw_documents must be a non-empty list of documents.")

    prepared = []
    for index, document in enumerate(raw_documents):
        if not isinstance(document, dict):
            raise ValueError(f"Document at index {index} must be a dictionary.")

        for field in REQUIRED_DOCUMENT_FIELDS:
            value = document.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(
                    f"Document at index {index} is missing required field '{field}'."
                )

        prepared_document = dict(document)
        if isinstance(document.get("tags"), list):
            prepared_document["tags"] = list(document["tags"])
        prepared_document["text"] = build_search_text(document)
        prepared.append(prepared_document)

    return prepared


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Requirements:
    - Return a float.
    - Return 1.0 for identical non-zero vectors.
    - Return 0.0 for orthogonal vectors.
    - Return 0.0 if either vector has zero magnitude.
    - Raise ValueError if the vectors have different dimensions.

    Do not use numpy for this lab. Implement the math with basic Python.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"Vectors must have the same dimensions ({len(vector_a)} != {len(vector_b)})."
        )

    # Dot product of the two vectors.
    dot = math.fsum(a * b for a, b in zip(vector_a, vector_b))
    # Squared magnitudes of each vector.
    sum_sq_a = math.fsum(a * a for a in vector_a)
    sum_sq_b = math.fsum(b * b for b in vector_b)

    # Zero-vector guard: cosine is undefined when either magnitude is zero.
    if sum_sq_a == 0 or sum_sq_b == 0:
        return 0.0

    score = dot / math.sqrt(sum_sq_a * sum_sq_b)  # single sqrt => identical vectors give exactly 1.0
    # Clamp to [-1, 1] so floating-point drift cannot push the score outside the valid cosine range.
    return float(max(-1.0, min(1.0, score)))


def embed_documents(
    prepared_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
) -> list[dict[str, Any]]:
    """
    Embed each prepared document.

    Requirements:
    - Accept documents that already include a 'text' field.
    - Call embedding_model.embed(document["text"]) once per document.
    - Return a new list of document dictionaries.
    - Add an 'embedding' field to each returned document.
    - Preserve metadata needed for source traceability.

    Do not mutate the input documents.
    """
    # Copy so the caller's documents are never mutated.
    embedded = []
    for index, document in enumerate(prepared_documents):
        text = document.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Document at index {index} is missing a 'text' field.")

        embedded_document = dict(document)
        embedded_document["embedding"] = list(embedding_model.embed(text))
        embedded.append(embedded_document)

    return embedded


def rank_documents(
    query: str,
    embedded_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Rank embedded documents against a user query.

    Requirements:
    - Validate that query is a non-empty string.
    - Validate that top_k is a positive integer.
    - Embed the query with embedding_model.embed(query).
    - Compare the query embedding to every document embedding.
    - Add a 'score' field to each returned result.
    - Sort results by score from highest to lowest.
    - Return only the top_k results.
    - Preserve source metadata: id, title, category, summary, and source.

    The returned result format should look like:
        {
            "id": "DOC-102",
            "title": "Fixing Invalid API Authentication Tokens",
            "category": "api",
            "summary": "...",
            "source": "platform-docs/api/authentication-tokens",
            "score": 0.87
        }
    """
    # Validate inputs.
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    # Embed query.
    query_embedding = embedding_model.embed(query.strip())

    # Score each doc.
    results = []
    for index, document in enumerate(embedded_documents):
        if "embedding" not in document:
            raise ValueError(f"Document at index {index} is missing an 'embedding'.")

        result = {field: document.get(field) for field in REQUIRED_DOCUMENT_FIELDS}
        result["score"] = cosine_similarity(query_embedding, document["embedding"])
        results.append(result)

    # Sort descending.
    results.sort(key=lambda result: result["score"], reverse=True)
    # Slice top_k (slicing naturally handles top_k > len(documents)).
    return results[:top_k]


def semantic_search(
    query: str,
    raw_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Run the full semantic retrieval workflow.

    Required sequence:
    1. Prepare documents.
    2. Embed documents.
    3. Embed the query.
    4. Compute similarity scores.
    5. Rank results.
    6. Return top-k results with source metadata.

    This function should orchestrate the smaller helper functions.
    """
    # Prepare documents, embed them, then rank against the query.
    prepared_documents = prepare_documents(raw_documents)
    embedded_documents = embed_documents(prepared_documents, embedding_model)
    return rank_documents(query, embedded_documents, embedding_model, top_k=top_k)


def main() -> None:
    """
    Optional manual run.

    This is not used by pytest. It is here so you can test your workflow locally
    after implementing the required functions.
    """
    embedder = OllamaEmbeddingModel()
    query = "Why does the mobile app say my token is expired?"

    results = semantic_search(
        query=query,
        raw_documents=DEFAULT_DOCUMENTS,
        embedding_model=embedder,
        top_k=3,
    )

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['title']} | score={result['score']:.4f}")
        print(f"   Source: {result['source']}")
        print(f"   Summary: {result['summary']}")


if __name__ == "__main__":
    main()