import os
from phoenix.otel import register

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com/s/kumar-pradeep-sharma/v1/traces"  
os.environ["PHOENIX_API_KEY"] = ""
tracer_provider = register(
    project_name="KnowledgePilot",
    auto_instrument=True,
    batch=True,
  
)

tracer = tracer_provider.get_tracer("test")

with tracer.start_as_current_span("test-span"):
    print("Test span created")
