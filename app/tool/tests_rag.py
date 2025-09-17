import os
from urllib.parse import urlparse
import psycopg2
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv
import logging
import sys

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TABLE_NAME = "data_document_embeddings"
EMBED_DIM = 768  


QUERY = (
    'How do the "Delivery Terms" and "Payment Terms" relate to a "Purchase Order" '
    "within the procurement process described in this document?"
)


TEXT_TO_CHECK = "Delivery Terms"


def main():
    logging.basicConfig(
    filename="debug.log",       
    level=logging.DEBUG,        
    format="%(asctime)s [%(levelname)s] %(message)s"
)

    try:
        
        db_parts = urlparse(DATABASE_URL)
        print(f"DEBUG: Connecting to DB host={db_parts.hostname}, db={db_parts.path.lstrip('/')}")
        logging.debug(f"Connecting to DB host={db_parts.hostname}, db={db_parts.path.lstrip('/')}")
        sys.stdout.write("DEBUG: Starting document retrieval process...\n")
        sys.stdout.flush()
        # Connect to Postgres
        conn = psycopg2.connect(
            host=db_parts.hostname,
            port=db_parts.port,
            database=db_parts.path.lstrip('/'),
            user=db_parts.username,
            password=db_parts.password
        )
        logging.debug("DEBUG: Successfully connected to DB")
        

        
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
            row_count = cur.fetchone()[0]
            logging.debug(f"DEBUG: Table '{TABLE_NAME}' contains {row_count} rows")
            if row_count == 0:
                logging.debug("WARNING: Table is empty, no data to retrieve!")

            
            cur.execute(f"SELECT id, text FROM {TABLE_NAME} WHERE text ILIKE %s LIMIT 5;", (f"%{TEXT_TO_CHECK}%",))
            rows = cur.fetchall()
            if rows:
                logging.debug(f"DEBUG: Found {len(rows)} rows containing '{TEXT_TO_CHECK}':")
                for r in rows:
                    logging.debug(f" - id: {r[0]}, text snippet: {r[1][:100]}...")
            else:
                logging.debug(f"DEBUG: No rows found containing '{TEXT_TO_CHECK}'")

        
        vector_store = PGVectorStore.from_params(
            host=db_parts.hostname,
            port=db_parts.port,
            database=db_parts.path.lstrip('/'),
            user=db_parts.username,
            password=db_parts.password,
            table_name=TABLE_NAME,
            embed_dim=EMBED_DIM,
            hybrid_search=True,
            text_search_config="english"
        )
        logging.debug(f"DEBUG: Vector store initialized with table '{TABLE_NAME}'")

        # Initialize Ollama embedding model
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text:v1.5",
            base_url=OLLAMA_BASE_URL,
            request_timeout=120.0
        )

        # Create LlamaIndex VectorStoreIndex
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )

        # Create query engine with hybrid search
        query_engine = index.as_query_engine(
            vector_store_query_mode="hybrid",
            similarity_top_k=5,
            sparse_top_k=2
        )

        # Run query
        logging.debug(f"DEBUG: Running query: {QUERY}")
        response = query_engine.query(QUERY)

        # Print full raw response for debugging
        logging.debug("\nDEBUG: Full response object:")
        logging.debug(response)

        retrieved_nodes = response.source_nodes

        if not retrieved_nodes:
            logging.debug("No relevant documents found for this query.")
            return

       
        for i, node in enumerate(retrieved_nodes, 1):
            content = node.get_content()
            metadata = getattr(node, "metadata", {})
            source_file = metadata.get("source_file", metadata.get("file_name", "Unknown source"))
            page = metadata.get("page_number", "")
            context = metadata.get("context", "")

            logging.debug(f"\n--- Document Chunk {i} ---")
            logging.debug(f"Source: {source_file} {f'(Page {page})' if page else ''}")
            if context:
                logging.debug(f"Context: {context}")
            logging.debug(f"Content: {content[:500]}")  

    except Exception as e:
        logging.debug(f"Error during document retrieval: {e}")


if __name__ == "__main__":
    main()
