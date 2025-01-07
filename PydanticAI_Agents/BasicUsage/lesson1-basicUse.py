"""
This example is a basic example showing structure 

Outputs:

--------------------------------
result: RunResult(_all_messages=[ModelRequest(parts=[SystemPromptPart(content='You are a helpful assistant', part_kind='system-prompt'), 
UserPromptPart(content='Hello', timestamp=datetime.datetime(2025, 1, 7, 15, 59, 43, 428102, tzinfo=datetime.timezone.utc), 
part_kind='user-prompt')], kind='request'), ModelResponse(parts=[TextPart(content='Hello! How can I assist you today?', 
part_kind='text')], timestamp=datetime.datetime(2025, 1, 7, 15, 59, 43, tzinfo=datetime.timezone.utc),
kind='response')], _new_message_index=0, data='Hello! How can I assist you today?',
_usage=Usage(requests=1, request_tokens=17, response_tokens=10, total_tokens=27, 
details={'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'cached_tokens': 0}))
--------------------------------
type: <class 'pydantic_ai.result.RunResult'>
--------------------------------
result.data: Hello! How can I assist you today?


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
