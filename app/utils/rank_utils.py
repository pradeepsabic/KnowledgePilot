import numpy as np
from typing import List, Any

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None or np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def rerank_returnedchunks(query_embedding: np.ndarray, nodes: List[Any]) -> List[Any]:
    """
    Rerank retrieved document nodes based on cosine similarity with the query embedding.

    Args:
        query_embedding: The query embedding as a numpy array.
        nodes: List of LlamaIndex nodes (source_nodes).

    Returns:
        List of nodes, sorted by similarity descending.
    """
    ranked_nodes = []

    if query_embedding is None:
        print("Warning: query_embedding is None — skipping rerank.")
        return []

    for node in nodes:
        # Ensure node.embedding is not None and is a numpy array
        if not hasattr(node, "embedding") or node.embedding is None:
            continue

        node_embedding = np.array(node.embedding)

        similarity = cosine_similarity(query_embedding, node_embedding)
        node.similarity_score = similarity
        ranked_nodes.append(node)

    return sorted(ranked_nodes, key=lambda n: n.similarity_score, reverse=True)
