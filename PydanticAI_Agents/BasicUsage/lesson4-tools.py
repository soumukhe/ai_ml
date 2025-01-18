"""
This example shows agent with tools

- to update the system prompt dynamically first create @dataclass
- then call the data class in the agent:  agent = Agent(model, deps_type=User)
- then create a function that returns the system prompt and decorate it with @agent.system_prompt


2 kinds of tool decorators:
- @agent.tool_plain: for plain text tools (no deps in the tool function)
- @agent.tool: for tools that return a value (use deps in the tool function)

also another one:
- @pydantic_ai_expert.tool

You can use both decorators in the same agent.
You can also pass tools as a list to the agent: agent = Agent(model, tools=[get_current_time, get_bank_balance])

Observation: (notice that the prompt (system prompt, + user prompt) makes it call both tools)
- System Prompt: def system_prompt(ctx: RunContext[User]):
- User Prompt: result = agent.run_sync("What is my bank balance?  and what is the current time? ", deps=deps)

Outputs:
- Mr. Soumitra, your bank balance is $100,000 and the current time is 09:39:57 on January 7, 2025. Would you like a drink?
- Ms. Ria, your bank balance is $200,000, and the current time is 09:47:23 on January 7, 2025. Would you like a drink?
- I'm sorry, Milly, but I am not authorized to access your bank balance. However, I can tell you that the current time is 09:49 AM on January 7, 2025. 
  Would you like a drink, kid?
"""
import os
# Set Logfire to ignore warnings before any other imports
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel   
from dataclasses import dataclass
from datetime import datetime

@dataclass
class User:
    name: str
    age: int
    gender: str
    bank_balance: int

model = OpenAIModel(model_name="gpt-4o-mini")    

agent = Agent(model, deps_type=User)

@agent.system_prompt
def system_prompt(ctx: RunContext[User]):
    return f"""You are a helpful assistant. You always address the user by their name.
    you are talking to {ctx.deps.name} at the moment and he/she is {ctx.deps.age} years old. if the person is a minor, you address them as 'kid'. Otherwise you ask them if they want a drink. The gender of the user is {ctx.deps.gender}. For male please address as Mr. and for female, please address as Ms."""
  
@agent.tool_plain
def get_current_time():
    return f" Today is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"

@agent.tool
def get_bank_balance(ctx: RunContext[User]):
    if ctx.deps.name == "Soumitra":
        return f" Your bank balance is ${ctx.deps.bank_balance}"
    elif ctx.deps.name == "Ria":
        return f" Your bank balance is ${ctx.deps.bank_balance}"
    elif ctx.deps.name == "Milly":
        return f" You are not authorized to access this information"
    else:
        return f" please check your name and try again"
    
#deps = User(name="Soumitra", age=64, gender="male", bank_balance=100000)
#deps = User(name="Ria", age=21, gender="female", bank_balance=200000)
deps = User(name="Milly", age=12, gender="female", bank_balance=300000)

result = agent.run_sync("What is my bank balance?  and what is the current time? ", deps=deps)

print(result.data)   


