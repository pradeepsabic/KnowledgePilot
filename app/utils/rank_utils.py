import numpy as np
from typing import List, Dict, Any

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
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
    for node in nodes:
        node_embedding = np.array(node.embedding)
        similarity = cosine_similarity(query_embedding, node_embedding)
        node.similarity_score = similarity  # Store score for debugging
        ranked_nodes.append(node)

    return sorted(ranked_nodes, key=lambda n: n.similarity_score, reverse=True)
