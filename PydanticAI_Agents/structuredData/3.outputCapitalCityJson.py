import os
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
from dotenv import load_dotenv
import json

"""

json_output = result.data.model_dump_json(indent=4)

`model_dump_json()` is a method provided by Pydantic for converting a Pydantic model instance to a JSON string. 
This method is particularly useful when working with Pydantic AI to obtain JSON structured output from the agent’s results.

Output:

00:27:01.498 agent run prompt=What is the capital of the US?
00:27:01.500   preparing model and tools run_step=1
00:27:01.501   model request
Logfire project URL: https://logfire.pydantic.dev/soumukhe/my-first-project
00:27:03.552   handle model response
00:27:03.555 Results from LLM: name='Washington, D.C.' year_founded=1790 short_history="Washi...es of government and numerous national monuments and museums."
00:27:03.555 Year founded: 1790
00:27:03.556 Short history: Washington, D.C., the capital of the United States, was establ...hes of government and numerous national monuments and museums.

{
    "name": "Washington, D.C.",
    "year_founded": 1790,
    "short_history": "Washington, D.C., the capital of the United States, was established as the nation's capital in 1790. It was chosen by George Washington and built on land designated by the Residence Act. The city was purposefully designed by Pierre L'Enfant, incorporating wide avenues and ceremonial spaces. Washington, D.C. has served as the political center of the United States, housing all three branches of government and numerous national monuments and museums."
}
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

json_output = result.data.model_dump_json(indent=4)
print(json_output)