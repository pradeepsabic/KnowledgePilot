from crews.knowledgepilot_crew import KnowledgePilotCrew
import logging
import os
from phoenix_config import tracer
from app.utils.chathistory import ChatHistoryManager


@tracer.chain
def run():
    chat_manager = ChatHistoryManager(max_history=5)
    user_query = "How do the Delivery Terms and Payment Terms relate to a Purchase Order within the procurement process described in this document?"

    # Rewrite query including previous context
    query_with_context = chat_manager.rewrite_query(user_query)

    # Run CrewAI
    crew_instance = KnowledgePilotCrew().crew()
    result = crew_instance.kickoff(inputs={"topic": query_with_context})

    # Save assistant response in chat history
    chat_manager.add_message("assistant", str(result))

    print("\nFinal Answer:\n", result)
    print("\n--- Chat History ---")
    for msg in chat_manager.get_history():
        print(f"{msg['role']}: {msg['content']}")


if __name__ == "__main__":
    run()
