from crews.knowledgepilot_crew import KnowledgePilotCrew
from app.utils.chathistory import ChatHistoryManager
from phoenix_config import tracer
from guardrails import Guard
import requests
import os

base_url = os.getenv("API_BASE", "http://localhost:11434")
model_name = os.getenv("MODEL", "ollama/gemma:2b")

print("Checking Ollama connection at {base_url} ...")
try:
    res = requests.get(f"{base_url}/api/tags")
    if res.status_code == 200:
        print("Ollama is running.")
        models = [m["name"] for m in res.json().get("models", [])]
        if model_name.split("/")[-1] in models:
            print(f"Model '{model_name}' is available.")
        else:
            print(f"Model '{model_name}' not found locally. You may need to run:")
            print(f"ollama pull {model_name.split('/')[-1]}")
    else:
        print(f"Ollama responded with status {res.status_code}")
except Exception as e:
    print(f"Could not connect to Ollama at {base_url}: {e}")


@tracer.chain
def run(user_query):
    chat_manager = ChatHistoryManager(max_history=5)

    # validate using guardrails
    # user_query = guard.validate_input({"query": user_query})["query"]

    # Rewrite query including previous context
    query_with_context = chat_manager.rewrite_query(user_query)
    print("\n--- Rewritten Query Sent to CrewAI ---")
    print(query_with_context)

    # Run CrewAI
    crew_instance = KnowledgePilotCrew().crew()
    print(
        "CrewAI model config:",
        KnowledgePilotCrew().agents_config["research_agent"]["llm"],
    )
    result = crew_instance.kickoff(inputs={"topic": query_with_context})

    # Validate output using guardrails
    # validated_response = guard.output({"response": result})["response"]

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
