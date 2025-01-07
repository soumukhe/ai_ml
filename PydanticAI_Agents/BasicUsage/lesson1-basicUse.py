"""
This example is a basic example showing structure 

"""
import os
# Set Logfire to ignore warnings before any other imports
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel   


from dotenv import load_dotenv
import os

load_dotenv()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant",
    tools=[],
)

result = agent.run_sync("Hello")

print("--------------------------------")
print(f"result: {result}")
print("--------------------------------")
print(f"type: {type(result)}")
print("--------------------------------")
print(f"result.data: {result.data}")
