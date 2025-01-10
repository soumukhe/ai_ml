import os
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
from dotenv import load_dotenv

"""
Output:

00:06:34.987 agent run prompt=What is the capital of the US?
00:06:34.988   preparing model and tools run_step=1
00:06:34.989   model request
Logfire project URL: https://logfire.pydantic.dev/soumukhe/my-first-project
00:06:37.588   handle model response
00:06:37.606 Results from LLM: name='Washington, D.C.' year_founded=1790 short_history="Washi...ite House, the Capitol Building, and the Washington Monument."
00:06:37.610 Year founded: 1790
00:06:37.610 Short history: Washington, D.C., the capital of the United States, was establ...hite House, the Capitol Building, and the Washington Monument.
name='Washington, D.C.' 
year_founded=1790 
short_history="Washington, D.C., the capital of the United States, was established as the nation's capital in 1790. It was chosen for its central location between the northern and southern states and is named after George Washington. The city has served as the seat of the federal government and is known for its iconic landmarks such as the White House, the Capitol Building, and the Washington Monument."

"""

load_dotenv()

# Configure logfire
logfire.configure()

# Define the model
model = OpenAIModel('gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))

# Define the output model
class Capital(BaseModel):
    """Capital city model - includes name and short history of the city"""
    name: str
    year_founded: int
    short_history: str
    

# Define the agent
agent = Agent(model=model, result_type=Capital)

# Run the agent
result = agent.run_sync("What is the capital of the US?")

logfire.notice('Results from LLM: {result}', result = str(result.data))
logfire.info('Year founded: {year}', year = result.data.year_founded)
logfire.info('Short history: {history}', history = result.data.short_history)

print(result.data)