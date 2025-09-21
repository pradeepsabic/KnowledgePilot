def formatchunk_citations(node, index, content):
    """
    Formats a document chunk with similarity score and citation metadata.

    Args:
        node: The document node object containing metadata.
        index: Chunk index number.
        content: Text content of the chunk.

    Returns:
        A formatted string with similarity score and citation metadata.
    """
    # Extract citation metadata
    metadata = getattr(node, "metadata", {}) or {}
    document_title = metadata.get("document_title", "Unknown Document")
    version = metadata.get("version", "N/A")
    date_published = metadata.get("date_published", "Unknown Date")
    author = metadata.get("author", "Unknown Author")
    approver = metadata.get("approver", "Unknown Approver")
    issued_by = metadata.get("issued_by", "Unknown Issuer")

    citation_info = (
        f"Document: {document_title} (Version: {version}, Published: {date_published})\n"
        f"Issued by: {issued_by}\n"
        f"Author: {author}\n"
        f"Approved by: {approver}"
    )

    # Get similarity score
    score = getattr(node, "score", None)
    if score is None:
        score = getattr(node, "similarity_score", 0.0) or 0.0

    # Format the chunk with citation info
    formatted_chunk = (
        f"**Document Chunk {index}**\n"
        f"Similarity Score: {score:.4f}\n"
        f"{citation_info}\n\n"
        f"Content:\n{content}"
    )

    return formatted_chunk
