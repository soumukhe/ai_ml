"""
This example shows dynamic system prompt

- to update the system prompt dynamically first create @dataclass
- then call the data class in the agent:  agent = Agent(model, deps_type=User)
- then create a function that returns the system prompt and decorate it with @agent.system_prompt

Outputs:
Hello, Mr. Soumitra! How are you today? Would you like a drink?
Hello, Ms. Ria! How are you today? Would you like a drink?
Hi there, kid! How are you today?

"""
import os
# Set Logfire to ignore warnings before any other imports
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel   
from dataclasses import dataclass



from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class User:
    name: str
    age: int
    gender: str

model = OpenAIModel(model_name="gpt-4o-mini")    

agent = Agent(model, deps_type=User)

@agent.system_prompt
def system_prompt(ctx: RunContext[User]):
    return f"""You are a helpful assistant. You always address the user by their name.
    you are talking to {ctx.deps.name} at the moment and he/she is {ctx.deps.age} years old. if the person is a minor, you address them as 'kid'. Otherwise you ask them if they want a drink. The gender of the user is {ctx.deps.gender}. For male please address as Mr. and for female, please address as Ms."""
  

#deps = User(name="Soumitra", age=64, gender="male")
#deps = User(name="Ria", age=21, gender="female")
deps = User(name="Milly", age=12, gender="female")

result = agent.run_sync("Hello", deps=deps)

print(result.data)   
