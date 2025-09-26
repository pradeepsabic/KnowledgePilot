from crews.knowledgepilot_crew import KnowledgePilotCrew
from app.utils.chathistory import ChatHistoryManager
from phoenix_config import tracer
from guardrails import Guard

guard = Guard.from_path("guardrails.yaml")


@tracer.chain
def run(user_query):
    chat_manager = ChatHistoryManager(max_history=5)

    # validate using guardrails
    user_query = guard.validate_input({"query": user_query})["query"]

    # Rewrite query including previous context
    query_with_context = chat_manager.rewrite_query(user_query)
    print("\n--- Rewritten Query Sent to CrewAI ---")
    print(query_with_context)

    # Run CrewAI
    crew_instance = KnowledgePilotCrew().crew()
    result = crew_instance.kickoff(inputs={"topic": query_with_context})

    # Validate output using guardrails
    validated_response = guard.output({"response": result})["response"]

    # Save assistant response in chat history
    chat_manager.add_message("assistant", str(result))

    print("\nUser Query:", user_query)
    print("CrewAI Answer:", result)
    print("\n--- Chat History ---")
    for msg in chat_manager.get_history():
        print(f"{msg['role']}: {msg['content']}")

    return result


if __name__ == "__main__":
    # First query
    run("How do the Delivery Terms relate to a Purchase Order?")

    # Second query — should include context from the first
    run("How do the Payment Terms relate to a Purchase Order?")
