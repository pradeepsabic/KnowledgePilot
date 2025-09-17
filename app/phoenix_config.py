from phoenix.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor
import os

# os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:4317" 


os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com/s/kumar-pradeep-sharma/v1/traces"  
os.environ["PHOENIX_API_KEY"] = "" 
# Registering tracer provider
tracer_provider = register(
    project_name="KnowledgePilot",
    auto_instrument=True,   # traces tools, LLMs, etc.
)

CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Creating a global tracer
tracer = tracer_provider.get_tracer("knowledgepilot")


__all__ = ["tracer", "tracer_provider"]



