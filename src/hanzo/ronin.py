import os

from crewai import Agent, Crew, Task
from dotenv import load_dotenv

load_dotenv()
# Set Together AI as the LLM backend
BASE_URL = "https://api.together.xyz/v1"
# LLM_backend = {
#     "model": "together_ai/mistral-7b-instruct",  # Adjust model as needed
#     "api_key": os.getenv("TOGETHER_API_KEY"),
#     "base_url": BASE_URL,
# }

# Define an AI agent
research_agent = Agent(
    role="AI Research Assistant",
    goal="Gather the latest research on AI model optimization techniques",
    backstory="You are an AI expert who specializes in summarizing complex research papers.",
    verbose=True,
    allow_delegation=False,
    llm="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
)

# Define a task
research_task = Task(
    description="Summarize the key findings from recent research papers on AI model optimization.",
    expected_output="one paragraph summary",
    agent=research_agent,
)

# Create a Crew (team of agents)
crew = Crew(agents=[research_agent], tasks=[research_task])

# Execute the task
result = crew.kickoff()
print(result)
