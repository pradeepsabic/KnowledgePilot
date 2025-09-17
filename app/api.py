from fastapi import FastAPI
from pydantic import BaseModel
from crews.knowledgepilot_crew import KnowledgePilotCrew
import uvicorn
from phoenix_config import tracer

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/rag-chat")
async def rag_chat(request: QueryRequest):
    try:
        crew_instance = KnowledgePilotCrew()
        crew = crew_instance.crew()
        result = crew.kickoff({"topic": request.query})
        return {"response": str(result)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # http://127.0.0.1:8000/docs 