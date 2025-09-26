from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from app.tool.tools import document_retrieval_tool
from typing import List
from crewai.agents.agent_builder.base_agent import BaseAgent
from phoenix_config import tracer
import os
from crewai.memory import LongTermMemory
from crewai.utilities.paths import db_storage_path

persistent_memory = LongTermMemory()
print("CrewAI storage path:", db_storage_path())


@tracer.chain
@CrewBase
class KnowledgePilotCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["research_agent"],
            verbose=True,
            tools=[document_retrieval_tool],
        )

    @agent
    def validator_agent(self) -> Agent:
        return Agent(config=self.agents_config["validator_agent"], verbose=True)

    @agent
    def answer_agent(self) -> Agent:
        return Agent(config=self.agents_config["answer_agent"], verbose=True)

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @task
    def validator_task(self) -> Task:
        return Task(config=self.tasks_config["validator_task"])

    @task
    def answer_task(self) -> Task:
        return Task(config=self.tasks_config["answer_task"])

    @tracer.chain
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,  # Enables short-term, long-term, and entity memory
            long_term_memory=persistent_memory,
            iterations=1,
        )
