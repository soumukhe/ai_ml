[ Skip to content ](https://ai.pydantic.dev/examples/bank-support/<#running-the-example>)
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/bank-support/<../..> "PydanticAI")
PydanticAI 
Bank support 
Initializing search 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/bank-support/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/bank-support/<../..> "PydanticAI") PydanticAI 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/bank-support/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
  * [ Introduction  ](https://ai.pydantic.dev/examples/bank-support/<../..>)
  * [ Installation  ](https://ai.pydantic.dev/examples/bank-support/install/>)
  * [ Getting Help  ](https://ai.pydantic.dev/examples/bank-support/help/>)
  * [ Contributing  ](https://ai.pydantic.dev/examples/bank-support/contributing/>)
  * [ Troubleshooting  ](https://ai.pydantic.dev/examples/bank-support/troubleshooting/>)
  * Documentation  Documentation 
    * [ Agents  ](https://ai.pydantic.dev/examples/bank-support/agents/>)
    * [ Models  ](https://ai.pydantic.dev/examples/bank-support/models/>)
    * [ Dependencies  ](https://ai.pydantic.dev/examples/bank-support/dependencies/>)
    * [ Function Tools  ](https://ai.pydantic.dev/examples/bank-support/tools/>)
    * [ Results  ](https://ai.pydantic.dev/examples/bank-support/results/>)
    * [ Messages and chat history  ](https://ai.pydantic.dev/examples/bank-support/message-history/>)
    * [ Testing and Evals  ](https://ai.pydantic.dev/examples/bank-support/testing-evals/>)
    * [ Debugging and Monitoring  ](https://ai.pydantic.dev/examples/bank-support/logfire/>)
    * [ Multi-agent Applications  ](https://ai.pydantic.dev/examples/bank-support/multi-agent-applications/>)
  * [ Examples  ](https://ai.pydantic.dev/examples/bank-support/<../>)
Examples 
    * [ Pydantic Model  ](https://ai.pydantic.dev/examples/bank-support/<../pydantic-model/>)
    * [ Weather agent  ](https://ai.pydantic.dev/examples/bank-support/<../weather-agent/>)
    * Bank support  [ Bank support  ](https://ai.pydantic.dev/examples/bank-support/<./>) Table of contents 
      * [ Running the Example  ](https://ai.pydantic.dev/examples/bank-support/<#running-the-example>)
      * [ Example Code  ](https://ai.pydantic.dev/examples/bank-support/<#example-code>)
    * [ SQL Generation  ](https://ai.pydantic.dev/examples/bank-support/<../sql-gen/>)
    * [ Flight booking  ](https://ai.pydantic.dev/examples/bank-support/<../flight-booking/>)
    * [ RAG  ](https://ai.pydantic.dev/examples/bank-support/<../rag/>)
    * [ Stream markdown  ](https://ai.pydantic.dev/examples/bank-support/<../stream-markdown/>)
    * [ Stream whales  ](https://ai.pydantic.dev/examples/bank-support/<../stream-whales/>)
    * [ Chat App with FastAPI  ](https://ai.pydantic.dev/examples/bank-support/<../chat-app/>)
  * API Reference  API Reference 
    * [ pydantic_ai.agent  ](https://ai.pydantic.dev/examples/bank-support/api/agent/>)
    * [ pydantic_ai.tools  ](https://ai.pydantic.dev/examples/bank-support/api/tools/>)
    * [ pydantic_ai.result  ](https://ai.pydantic.dev/examples/bank-support/api/result/>)
    * [ pydantic_ai.messages  ](https://ai.pydantic.dev/examples/bank-support/api/messages/>)
    * [ pydantic_ai.exceptions  ](https://ai.pydantic.dev/examples/bank-support/api/exceptions/>)
    * [ pydantic_ai.settings  ](https://ai.pydantic.dev/examples/bank-support/api/settings/>)
    * [ pydantic_ai.usage  ](https://ai.pydantic.dev/examples/bank-support/api/usage/>)
    * [ pydantic_ai.format_as_xml  ](https://ai.pydantic.dev/examples/bank-support/api/format_as_xml/>)
    * [ pydantic_ai.models  ](https://ai.pydantic.dev/examples/bank-support/api/models/base/>)
    * [ pydantic_ai.models.openai  ](https://ai.pydantic.dev/examples/bank-support/api/models/openai/>)
    * [ pydantic_ai.models.anthropic  ](https://ai.pydantic.dev/examples/bank-support/api/models/anthropic/>)
    * [ pydantic_ai.models.gemini  ](https://ai.pydantic.dev/examples/bank-support/api/models/gemini/>)
    * [ pydantic_ai.models.vertexai  ](https://ai.pydantic.dev/examples/bank-support/api/models/vertexai/>)
    * [ pydantic_ai.models.groq  ](https://ai.pydantic.dev/examples/bank-support/api/models/groq/>)
    * [ pydantic_ai.models.mistral  ](https://ai.pydantic.dev/examples/bank-support/api/models/mistral/>)
    * [ pydantic_ai.models.ollama  ](https://ai.pydantic.dev/examples/bank-support/api/models/ollama/>)
    * [ pydantic_ai.models.test  ](https://ai.pydantic.dev/examples/bank-support/api/models/test/>)
    * [ pydantic_ai.models.function  ](https://ai.pydantic.dev/examples/bank-support/api/models/function/>)


Table of contents 
  * [ Running the Example  ](https://ai.pydantic.dev/examples/bank-support/<#running-the-example>)
  * [ Example Code  ](https://ai.pydantic.dev/examples/bank-support/<#example-code>)


  1. [ Introduction  ](https://ai.pydantic.dev/examples/bank-support/<../..>)
  2. [ Examples  ](https://ai.pydantic.dev/examples/bank-support/<../>)


# Bank support
Small but complete example of using PydanticAI to build a support agent for a bank.
Demonstrates:
  * [dynamic system prompt](https://ai.pydantic.dev/examples/bank-support/agents/#system-prompts>)
  * [structured `result_type`](https://ai.pydantic.dev/examples/bank-support/results/#structured-result-validation>)
  * [tools](https://ai.pydantic.dev/examples/bank-support/tools/>)


## Running the Example
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/bank-support/<../#usage>), run:
[pip](https://ai.pydantic.dev/examples/bank-support/<#__tabbed_1_1>)[uv](https://ai.pydantic.dev/examples/bank-support/<#__tabbed_1_2>)
```
python-mpydantic_ai_examples.bank_support

```

```
uvrun-mpydantic_ai_examples.bank_support

```

(or `PYDANTIC_AI_MODEL=gemini-1.5-flash ...`)
## Example Code
bank_support.py```
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

class DatabaseConn:
"""This is a fake database for example purposes.
  In reality, you'd be connecting to an external database
  (e.g. PostgreSQL) to get information about customers.
  """
  @classmethod
  async def customer_name(cls, *, id: int) -> str | None:
    if id == 123:
      return 'John'
  @classmethod
  async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
    if id == 123:
      return 123.45
    else:
      raise ValueError('Customer not found')

@dataclass
class SupportDependencies:
  customer_id: int
  db: DatabaseConn

class SupportResult(BaseModel):
  support_advice: str = Field(description='Advice returned to the customer')
  block_card: bool = Field(description='Whether to block their')
  risk: int = Field(description='Risk level of query', ge=0, le=10)

support_agent = Agent(
  'openai:gpt-4o',
  deps_type=SupportDependencies,
  result_type=SupportResult,
  system_prompt=(
    'You are a support agent in our bank, give the '
    'customer support and judge the risk level of their query. '
    "Reply using the customer's name."
  ),
)

@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
  customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
  return f"The customer's name is {customer_name!r}"

@support_agent.tool
async def customer_balance(
  ctx: RunContext[SupportDependencies], include_pending: bool
) -> str:
"""Returns the customer's current account balance."""
  balance = await ctx.deps.db.customer_balance(
    id=ctx.deps.customer_id,
    include_pending=include_pending,
  )
  return f'${balance:.2f}'

if __name__ == '__main__':
  deps = SupportDependencies(customer_id=123, db=DatabaseConn())
  result = support_agent.run_sync('What is my balance?', deps=deps)
  print(result.data)
"""
  support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
  """
  result = support_agent.run_sync('I just lost my card!', deps=deps)
  print(result.data)
"""
  support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
  """

```

© Pydantic Services Inc. 2024 to present 
