"""
This example shows conversation with the agent.
By default LLMs are stateless. You need to add conversation history with query for it to act stateful.

You: hello
Agent: Hello! How can I assist you today?
You: what is my name ?
Agent: I don't have access to your personal information, so I don’t know your name. If you'd like to share it, feel free! How can I help you today?
You: my name is Soumitra
Agent: Nice to meet you, Soumitra! How can I assist you today?
You: what is my name ?
Agent: Your name is Soumitra. How can I help you today?
You: exit

"""
import os
# Set Logfire to ignore warnings before any other imports
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel   


from dotenv import load_dotenv


load_dotenv()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant",
    tools=[],
)

history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    result = agent.run_sync(user_prompt = user_input, message_history = history)
    print(f"Agent: {result.data}")
    history = result.all_messages()


