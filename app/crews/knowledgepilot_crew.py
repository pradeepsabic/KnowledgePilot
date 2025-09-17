from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from app.tool.tools import document_retrieval_tool
from typing import List
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class KnowledgePilotCrew:
    
    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['research_agent'],
            verbose=True,
            tools=[document_retrieval_tool]  
        )

    @agent
    def answer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['answer_agent'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task']
        )

    @task
    def answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['answer_task']
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            iterations=1
            
        )
