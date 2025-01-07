from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel   


from dotenv import load_dotenv
import os

load_dotenv()

# # Set Logfire to ignore warnings before any other imports
# os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

model = OpenAIModel('gpt-4o-mini')
agent = Agent(model=model)
print(agent.run_sync("What is the capital of the United States?").data)
