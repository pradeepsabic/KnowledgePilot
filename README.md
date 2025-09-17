# KnowledgePilot
An AgenticRAG-powered backend integrated with OpenWebUI frontend for intelligent, chat-style document search.

#Problem Statement
Develop a document search platform where users interact through OpenWebUI's chat interface. The backend implements an AgenticRAG system exposing REST endpoints, enabling contextual retrieval over ingested PDF documents.

#Backend Features

- Document Ingestion 
  Preprocess, split, vectorize, and index PDFs from a specified location using Docling.
- Retrieval Workflow 
  Contextual AgenticRAG pipeline powered by LlamaIndex and Crew.AI multi-agent orchestration.
- Vector Store 
  PGVector with PostgreSQL as the scalable vector database.
- LLM Provider 
  Ollama (local or remote) for large language model inference.
- Observability & Debugging 
  Arize Phoenix integration for prompt externalization, inference tracing, and monitoring.
- Evaluation  
  RAGAs framework for Retrieval-Augmented Generation pipeline evaluation using a ground-truth JSON file.

#Deployment Instructions
-Clone the repository
-Set up Python environment
-Configure environment variables
-Run backend API uvicorn app.main:app --reload
-OpenWebUI should be configured separately to connect with the backend API
