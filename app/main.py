from crews.knowledgepilot_crew import KnowledgePilotCrew
import logging
import os
from phoenix_config import tracer


@tracer.chain 
def run():
    
    inputs = {
         "topic": "How do the Delivery Terms and Payment Terms relate to a Purchase Order within the procurement process described in this document?"
    }
    KnowledgePilotCrew().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()
#execution python app/main.py