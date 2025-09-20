import os
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from crewai.tools import tool
from typing import Dict, Union, Any
import logging
from utils.rank_utils import rerank_returnedchunks
from phoenix_config import tracer
import json


load_dotenv()


@tool("Document Retrieval Tool")
@tracer.chain
def document_retrieval_tool(query: str) -> str:
    """
    Retrieves relevant documents based on the provided query using a PGVector vector store and embedding model.
    """
    try:

        print(f"Tool received query parameter: {repr(query)}")

        search_query = None

        if not query:
            return "Error: No valid search query provided."

        search_query = query.strip()

        if search_query in ["The search query to find relevant documents", ""]:
            return "Error: Tool received schema placeholder instead of actual query."

        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            return "Error: DATABASE_URL is not available."

        db_url_parts = urlparse(DATABASE_URL)

        print(
            f"Parsed Postgres connection details - "
            f"host: {db_url_parts.hostname}, "
            f"port: {db_url_parts.port}, "
            f"database: {db_url_parts.path.lstrip('/')}, "
            f"user: {db_url_parts.username}, "
            f"password: {'******' if db_url_parts.password else None}"
        )

        contextual_table = "document_embeddings"

        #  Connect to vector store using contextual table
        vector_store = PGVectorStore.from_params(
            host=db_url_parts.hostname,
            port=db_url_parts.port,
            database=db_url_parts.path.lstrip("/"),
            user=db_url_parts.username,
            password=db_url_parts.password,
            table_name=contextual_table,
            embed_dim=768,
            hybrid_search=True,
            text_search_config="english",
        )
        print(f"table: {contextual_table}")

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text:v1.5",
            base_url=ollama_base_url,
            request_timeout=120.0,
        )

        # Creating  VectorStoreIndex using vector store and model
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )

        # Create a query engine with hybrid search mode
        query_engine = index.as_query_engine(
            vector_store_query_mode="hybrid", similarity_top_k=5, sparse_top_k=2,return_source=True
        )

        response = query_engine.query(search_query)
        print(f"Query response: {response}")
        
        if not response.source_nodes:
            return "No relevant documents found for this query."
        
        # without reranking
        retrieved_nodes = response.source_nodes 

        # Rerank the retrieved nodes based on cosine similarity
        query_embedding = embed_model.get_text_embedding(query)
        if query_embedding is None:
            return "Error: Could not compute query embedding."
        # retrieved_nodes = rerank_returnedchunks(query_embedding, response.source_nodes) #uncomment for reranking

        if not retrieved_nodes:
            return "No relevant documents found for this query."

        # Format the retrieved context with source metadata and contextual information
        formatted_chunks = []
        for i, node in enumerate(retrieved_nodes, 1):
            content = node.get_content()

            # Extract from metadata
            source_info = "Unknown source"
            context_info = ""
            page_info = ""

            if hasattr(node, "metadata") and node.metadata:

                file_name = node.metadata.get(
                    "source_file", node.metadata.get("file_name", "Unknown file")
                )
                file_path = node.metadata.get("file_path", "")
                if file_path:
                    source_info = f"Source: {os.path.basename(file_path)}"
                else:
                    source_info = f"Source: {file_name}"

                context = node.metadata.get("context", "")
                if context:
                    context_info = f"\nContext: {context}"

                page_num = node.metadata.get("page_number", "")
                if page_num:
                    page_info = f" (Page {page_num})"
                    
            score = getattr(node, "score", None)
            if score is None:
                score = getattr(node, "similarity_score", 0.0) or 0.0
            
            formatted_chunk = (
                f"**Document Chunk {i}**\n"
                f"Similarity Score: {score:.4f}\n"
                f"{source_info}{page_info}{context_info}\n\nContent:\n{content}"
            )

            formatted_chunks.append(formatted_chunk)

        context = "\n\n" + "=" * 50 + "\n\n".join(formatted_chunks)
        print("Final context returned to agent:\n", context)
        return context
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"
