import os
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
from dotenv import load_dotenv

"""
make sure to create logfire account and create a project
logfire auth from terminal
logfire projects use the project name

In this example, we are using the OpenAI model gpt-4o-mini and the output model Calculation1.
The Calculation1 model is a Pydantic BaseModel that captures the result of a calculation.
The agent is defined using the Agent class from pydantic_ai.
The agent is run using the run_sync method, which runs the agent synchronously and returns the result.
The result is logged using logfire.

Output:

23:56:58.821 agent run prompt=What is 100 + 300?
23:56:58.823   preparing model and tools run_step=1
23:56:58.824   model request
Logfire project URL: https://logfire.pydantic.dev/soumukhe/my-first-project
23:56:59.581   handle model response
23:56:59.584 Output from LLM: result=400
23:56:59.585 Result type: <class '__main__.Calculation1'>
23:56:59.585 Result: 400
result=400

"""

load_dotenv()

# Configure logfire
logfire.configure()

# Define the model
model = OpenAIModel('gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))

# Define the output model
class Calculation1(BaseModel):
    """Captures the result of a calculation"""
    result: int

# Define the agent
agent = Agent(model=model, result_type=Calculation1)

# Run the agent
result = agent.run_sync("What is 100 + 300?")

logfire.notice('Output from LLM: {result}', result = str(result.data))
logfire.info('Result type: {result}', result = type(result.data))
logfire.info('Result: {result}', result = result.data.result)

print(result.data)