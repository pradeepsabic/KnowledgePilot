from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict,Literal
from crews.knowledgepilot_crew import KnowledgePilotCrew
import uvicorn
import time
from phoenix_config import tracer

app = FastAPI()

# Allow all origins so that OpenWebUI can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        
    allow_credentials=True,
    allow_methods=["*"],        
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    
#compatiable with openapi 
class ChatCompletionPayload(BaseModel):
    model: str
    messages: List[ChatMessage]

#for internal use testing or UI
class RagQueryPayload(BaseModel):
    query: str

#Model Discovery by OpenAI compatiable -return the model list to the openai 
@tracer.chain
@app.get("/v1/models")
def get_model_list():
    return {
        "object": "list",
        "data": [
            {
                "id": "knowledge-pilot",
                "object": "model",
                "created": 1677652288,
                "owned_by": "knowledge-pilot company",
                "permission": [],
                "root": "knowledge-pilot",
                "parent": None,
                "max_tokens": 131072,
                "context_length": 131072
            }
        ]
    }

#endpoint for openai compatiable chat completion
@tracer.chain
@app.post("/v1/chat/completions")
def handle_chat_completion_request(payload: ChatCompletionPayload):
    try:
        # Extract the latest user message
        user_message = next(
            (msg.content for msg in reversed(payload.messages) if msg.role == "user"),
            None
        )

        if not user_message:
            return {"error": "No user message found in 'messages'."}

        print(f"DEBUG: Received OpenAI-style query: {user_message}")
    
            #Calling the CrewAI RAG crew
        print("DEBUG: Initializing KnowledgePilotCrew")
        crew_instance = KnowledgePilotCrew()
        print("DEBUG: KnowledgePilotCrew instance created")
        crew = crew_instance.crew()
        print(f"DEBUG: Crew instance created: {crew_instance}")
        result = crew.kickoff({"topic": user_message})
        print(f"DEBUG: Crew kickoff result: {result}")

        return {
            "id": "chatcmpl-knowledgepilot-00",
            "object": "chat.completion",
            "created":int(time.time()),
            "model": payload.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": str(result)
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}
    
@tracer.chain
@app.post("/rag-chat")
async def handle_rag_query(payload: RagQueryPayload):
    try:
        print(f"DEBUG: Received internal query: {payload.query}")
        
        # Calling the CrewAI RAG crew
        crew_instance = KnowledgePilotCrew()
        crew = crew_instance.crew()
        result = crew.kickoff({"topic": payload.query})

        return {"response": str(result)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # http://127.0.0.1:8000/docs 