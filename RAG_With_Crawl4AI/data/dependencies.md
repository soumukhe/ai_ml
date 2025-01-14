[ Skip to content ](https://ai.pydantic.dev/dependencies/<#dependencies>)
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/dependencies/<..> "PydanticAI")
PydanticAI 
Dependencies 
Initializing search 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/dependencies/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/dependencies/<..> "PydanticAI") PydanticAI 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/dependencies/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
  * [ Introduction  ](https://ai.pydantic.dev/dependencies/<..>)
  * [ Installation  ](https://ai.pydantic.dev/dependencies/<../install/>)
  * [ Getting Help  ](https://ai.pydantic.dev/dependencies/<../help/>)
  * [ Contributing  ](https://ai.pydantic.dev/dependencies/<../contributing/>)
  * [ Troubleshooting  ](https://ai.pydantic.dev/dependencies/<../troubleshooting/>)
  * Documentation  Documentation 
    * [ Agents  ](https://ai.pydantic.dev/dependencies/<../agents/>)
    * [ Models  ](https://ai.pydantic.dev/dependencies/<../models/>)
    * Dependencies  [ Dependencies  ](https://ai.pydantic.dev/dependencies/<./>) Table of contents 
      * [ Defining Dependencies  ](https://ai.pydantic.dev/dependencies/<#defining-dependencies>)
      * [ Accessing Dependencies  ](https://ai.pydantic.dev/dependencies/<#accessing-dependencies>)
        * [ Asynchronous vs. Synchronous dependencies  ](https://ai.pydantic.dev/dependencies/<#asynchronous-vs-synchronous-dependencies>)
      * [ Full Example  ](https://ai.pydantic.dev/dependencies/<#full-example>)
      * [ Overriding Dependencies  ](https://ai.pydantic.dev/dependencies/<#overriding-dependencies>)
      * [ Examples  ](https://ai.pydantic.dev/dependencies/<#examples>)
    * [ Function Tools  ](https://ai.pydantic.dev/dependencies/<../tools/>)
    * [ Results  ](https://ai.pydantic.dev/dependencies/<../results/>)
    * [ Messages and chat history  ](https://ai.pydantic.dev/dependencies/<../message-history/>)
    * [ Testing and Evals  ](https://ai.pydantic.dev/dependencies/<../testing-evals/>)
    * [ Debugging and Monitoring  ](https://ai.pydantic.dev/dependencies/<../logfire/>)
    * [ Multi-agent Applications  ](https://ai.pydantic.dev/dependencies/<../multi-agent-applications/>)
  * [ Examples  ](https://ai.pydantic.dev/dependencies/<../examples/>)
Examples 
    * [ Pydantic Model  ](https://ai.pydantic.dev/dependencies/<../examples/pydantic-model/>)
    * [ Weather agent  ](https://ai.pydantic.dev/dependencies/<../examples/weather-agent/>)
    * [ Bank support  ](https://ai.pydantic.dev/dependencies/<../examples/bank-support/>)
    * [ SQL Generation  ](https://ai.pydantic.dev/dependencies/<../examples/sql-gen/>)
    * [ Flight booking  ](https://ai.pydantic.dev/dependencies/<../examples/flight-booking/>)
    * [ RAG  ](https://ai.pydantic.dev/dependencies/<../examples/rag/>)
    * [ Stream markdown  ](https://ai.pydantic.dev/dependencies/<../examples/stream-markdown/>)
    * [ Stream whales  ](https://ai.pydantic.dev/dependencies/<../examples/stream-whales/>)
    * [ Chat App with FastAPI  ](https://ai.pydantic.dev/dependencies/<../examples/chat-app/>)
  * API Reference  API Reference 
    * [ pydantic_ai.agent  ](https://ai.pydantic.dev/dependencies/<../api/agent/>)
    * [ pydantic_ai.tools  ](https://ai.pydantic.dev/dependencies/<../api/tools/>)
    * [ pydantic_ai.result  ](https://ai.pydantic.dev/dependencies/<../api/result/>)
    * [ pydantic_ai.messages  ](https://ai.pydantic.dev/dependencies/<../api/messages/>)
    * [ pydantic_ai.exceptions  ](https://ai.pydantic.dev/dependencies/<../api/exceptions/>)
    * [ pydantic_ai.settings  ](https://ai.pydantic.dev/dependencies/<../api/settings/>)
    * [ pydantic_ai.usage  ](https://ai.pydantic.dev/dependencies/<../api/usage/>)
    * [ pydantic_ai.format_as_xml  ](https://ai.pydantic.dev/dependencies/<../api/format_as_xml/>)
    * [ pydantic_ai.models  ](https://ai.pydantic.dev/dependencies/<../api/models/base/>)
    * [ pydantic_ai.models.openai  ](https://ai.pydantic.dev/dependencies/<../api/models/openai/>)
    * [ pydantic_ai.models.anthropic  ](https://ai.pydantic.dev/dependencies/<../api/models/anthropic/>)
    * [ pydantic_ai.models.gemini  ](https://ai.pydantic.dev/dependencies/<../api/models/gemini/>)
    * [ pydantic_ai.models.vertexai  ](https://ai.pydantic.dev/dependencies/<../api/models/vertexai/>)
    * [ pydantic_ai.models.groq  ](https://ai.pydantic.dev/dependencies/<../api/models/groq/>)
    * [ pydantic_ai.models.mistral  ](https://ai.pydantic.dev/dependencies/<../api/models/mistral/>)
    * [ pydantic_ai.models.ollama  ](https://ai.pydantic.dev/dependencies/<../api/models/ollama/>)
    * [ pydantic_ai.models.test  ](https://ai.pydantic.dev/dependencies/<../api/models/test/>)
    * [ pydantic_ai.models.function  ](https://ai.pydantic.dev/dependencies/<../api/models/function/>)


Table of contents 
  * [ Defining Dependencies  ](https://ai.pydantic.dev/dependencies/<#defining-dependencies>)
  * [ Accessing Dependencies  ](https://ai.pydantic.dev/dependencies/<#accessing-dependencies>)
    * [ Asynchronous vs. Synchronous dependencies  ](https://ai.pydantic.dev/dependencies/<#asynchronous-vs-synchronous-dependencies>)
  * [ Full Example  ](https://ai.pydantic.dev/dependencies/<#full-example>)
  * [ Overriding Dependencies  ](https://ai.pydantic.dev/dependencies/<#overriding-dependencies>)
  * [ Examples  ](https://ai.pydantic.dev/dependencies/<#examples>)


  1. [ Introduction  ](https://ai.pydantic.dev/dependencies/<..>)
  2. [ Documentation  ](https://ai.pydantic.dev/dependencies/<../agents/>)


# Dependencies
PydanticAI uses a dependency injection system to provide data and services to your agent's [system prompts](https://ai.pydantic.dev/dependencies/<../agents/#system-prompts>), [tools](https://ai.pydantic.dev/dependencies/<../tools/>) and [result validators](https://ai.pydantic.dev/dependencies/<../results/#result-validators-functions>).
Matching PydanticAI's design philosophy, our dependency system tries to use existing best practice in Python development rather than inventing esoteric "magic", this should make dependencies type-safe, understandable easier to test and ultimately easier to deploy in production.
## Defining Dependencies
Dependencies can be any python type. While in simple cases you might be able to pass a single object as a dependency (e.g. an HTTP connection), [dataclasses](https://ai.pydantic.dev/dependencies/<https:/docs.python.org/3/library/dataclasses.html#module-dataclasses>) are generally a convenient container when your dependencies included multiple objects.
Here's an example of defining an agent that requires dependencies.
(**Note:** dependencies aren't actually used in this example, see [Accessing Dependencies](https://ai.pydantic.dev/dependencies/<#accessing-dependencies>) below)
unused_dependencies.py```
from dataclasses import dataclass
import httpx
from pydantic_ai import Agent

@dataclass
class MyDeps: 
Define a dataclass to hold dependencies.
[](https://ai.pydantic.dev/dependencies/<#__code_0_annotation_1>)
  api_key: str
  http_client: httpx.AsyncClient

agent = Agent(
  'openai:gpt-4o',
  deps_type=MyDeps, 
Pass the dataclass type to the deps_type argument of the Agent constructor[](https://ai.pydantic.dev/dependencies/<../api/agent/#pydantic_ai.agent.Agent.__init__>). **Note**: we're passing the type here, NOT an instance, this parameter is not actually used at runtime, it's here so we can get full type checking of the agent.
[](https://ai.pydantic.dev/dependencies/<#__code_0_annotation_2>)
)

async def main():
  async with httpx.AsyncClient() as client:
    deps = MyDeps('foobar', client)
    result = await agent.run(
      'Tell me a joke.',
      deps=deps, 
When running the agent, pass an instance of the dataclass to the deps parameter.
[](https://ai.pydantic.dev/dependencies/<#__code_0_annotation_3>)
    )
    print(result.data)
    #> Did you hear about the toothpaste scandal? They called it Colgate.

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
## Accessing Dependencies
Dependencies are accessed through the `RunContext`[](https://ai.pydantic.dev/dependencies/<../api/tools/#pydantic_ai.tools.RunContext>) type, this should be the first parameter of system prompt functions etc.
system_prompt_dependencies.py```
from dataclasses import dataclass
import httpx
from pydantic_ai import Agent, RunContext

@dataclass
class MyDeps:
  api_key: str
  http_client: httpx.AsyncClient

agent = Agent(
  'openai:gpt-4o',
  deps_type=MyDeps,
)

@agent.system_prompt 
RunContext[](https://ai.pydantic.dev/dependencies/<../api/tools/#pydantic_ai.tools.RunContext>) may optionally be passed to a system_prompt[](https://ai.pydantic.dev/dependencies/<../api/agent/#pydantic_ai.agent.Agent.system_prompt>) function as the only argument.
[](https://ai.pydantic.dev/dependencies/<#__code_1_annotation_1>)
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str: 
RunContext[](https://ai.pydantic.dev/dependencies/<../api/tools/#pydantic_ai.tools.RunContext>) is parameterized with the type of the dependencies, if this type is incorrect, static type checkers will raise an error.
[](https://ai.pydantic.dev/dependencies/<#__code_1_annotation_2>)
  response = await ctx.deps.http_client.get( 
Access dependencies through the .deps[](https://ai.pydantic.dev/dependencies/<../api/tools/#pydantic_ai.tools.RunContext.deps>) attribute.
[](https://ai.pydantic.dev/dependencies/<#__code_1_annotation_3>)
    'https://example.com',
    headers={'Authorization': f'Bearer {ctx.deps.api_key}'}, 
Access dependencies through the .deps[](https://ai.pydantic.dev/dependencies/<../api/tools/#pydantic_ai.tools.RunContext.deps>) attribute.
[](https://ai.pydantic.dev/dependencies/<#__code_1_annotation_4>)
  )
  response.raise_for_status()
  return f'Prompt: {response.text}'

async def main():
  async with httpx.AsyncClient() as client:
    deps = MyDeps('foobar', client)
    result = await agent.run('Tell me a joke.', deps=deps)
    print(result.data)
    #> Did you hear about the toothpaste scandal? They called it Colgate.

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
### Asynchronous vs. Synchronous dependencies
[System prompt functions](https://ai.pydantic.dev/dependencies/<../agents/#system-prompts>), [function tools](https://ai.pydantic.dev/dependencies/<../tools/>) and [result validators](https://ai.pydantic.dev/dependencies/<../results/#result-validators-functions>) are all run in the async context of an agent run.
If these functions are not coroutines (e.g. `async def`) they are called with `run_in_executor`[](https://ai.pydantic.dev/dependencies/<https:/docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor>) in a thread pool, it's therefore marginally preferable to use `async` methods where dependencies perform IO, although synchronous dependencies should work fine too.
`run` vs. `run_sync` and Asynchronous vs. Synchronous dependencies
Whether you use synchronous or asynchronous dependencies, is completely independent of whether you use `run` or `run_sync` — `run_sync` is just a wrapper around `run` and agents are always run in an async context.
Here's the same example as above, but with a synchronous dependency:
sync_dependencies.py```
from dataclasses import dataclass
import httpx
from pydantic_ai import Agent, RunContext

@dataclass
class MyDeps:
  api_key: str
  http_client: httpx.Client 
Here we use a synchronous httpx.Client instead of an asynchronous httpx.AsyncClient.
[](https://ai.pydantic.dev/dependencies/<#__code_2_annotation_1>)

agent = Agent(
  'openai:gpt-4o',
  deps_type=MyDeps,
)

@agent.system_prompt
def get_system_prompt(ctx: RunContext[MyDeps]) -> str: 
To match the synchronous dependency, the system prompt function is now a plain function, not a coroutine.
[](https://ai.pydantic.dev/dependencies/<#__code_2_annotation_2>)
  response = ctx.deps.http_client.get(
    'https://example.com', headers={'Authorization': f'Bearer {ctx.deps.api_key}'}
  )
  response.raise_for_status()
  return f'Prompt: {response.text}'

async def main():
  deps = MyDeps('foobar', httpx.Client())
  result = await agent.run(
    'Tell me a joke.',
    deps=deps,
  )
  print(result.data)
  #> Did you hear about the toothpaste scandal? They called it Colgate.

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
## Full Example
As well as system prompts, dependencies can be used in [tools](https://ai.pydantic.dev/dependencies/<../tools/>) and [result validators](https://ai.pydantic.dev/dependencies/<../results/#result-validators-functions>).
full_example.py```
from dataclasses import dataclass
import httpx
from pydantic_ai import Agent, ModelRetry, RunContext

@dataclass
class MyDeps:
  api_key: str
  http_client: httpx.AsyncClient

agent = Agent(
  'openai:gpt-4o',
  deps_type=MyDeps,
)

@agent.system_prompt
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:
  response = await ctx.deps.http_client.get('https://example.com')
  response.raise_for_status()
  return f'Prompt: {response.text}'

@agent.tool 
To pass RunContext to a tool, use the tool[](https://ai.pydantic.dev/dependencies/<../api/agent/#pydantic_ai.agent.Agent.tool>) decorator.
[](https://ai.pydantic.dev/dependencies/<#__code_3_annotation_1>)
async def get_joke_material(ctx: RunContext[MyDeps], subject: str) -> str:
  response = await ctx.deps.http_client.get(
    'https://example.com#jokes',
    params={'subject': subject},
    headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
  )
  response.raise_for_status()
  return response.text

@agent.result_validator 
RunContext may optionally be passed to a result_validator[](https://ai.pydantic.dev/dependencies/<../api/agent/#pydantic_ai.agent.Agent.result_validator>) function as the first argument.
[](https://ai.pydantic.dev/dependencies/<#__code_3_annotation_2>)
async def validate_result(ctx: RunContext[MyDeps], final_response: str) -> str:
  response = await ctx.deps.http_client.post(
    'https://example.com#validate',
    headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    params={'query': final_response},
  )
  if response.status_code == 400:
    raise ModelRetry(f'invalid response: {response.text}')
  response.raise_for_status()
  return final_response

async def main():
  async with httpx.AsyncClient() as client:
    deps = MyDeps('foobar', client)
    result = await agent.run('Tell me a joke.', deps=deps)
    print(result.data)
    #> Did you hear about the toothpaste scandal? They called it Colgate.

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
## Overriding Dependencies
When testing agents, it's useful to be able to customise dependencies.
While this can sometimes be done by calling the agent directly within unit tests, we can also override dependencies while calling application code which in turn calls the agent.
This is done via the `override`[](https://ai.pydantic.dev/dependencies/<../api/agent/#pydantic_ai.agent.Agent.override>) method on the agent.
joke_app.py```
from dataclasses import dataclass
import httpx
from pydantic_ai import Agent, RunContext

@dataclass
class MyDeps:
  api_key: str
  http_client: httpx.AsyncClient
  async def system_prompt_factory(self) -> str: 
Define a method on the dependency to make the system prompt easier to customise.
[](https://ai.pydantic.dev/dependencies/<#__code_4_annotation_1>)
    response = await self.http_client.get('https://example.com')
    response.raise_for_status()
    return f'Prompt: {response.text}'

joke_agent = Agent('openai:gpt-4o', deps_type=MyDeps)

@joke_agent.system_prompt
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:
  return await ctx.deps.system_prompt_factory() 
Call the system prompt factory from within the system prompt function.
[](https://ai.pydantic.dev/dependencies/<#__code_4_annotation_2>)

async def application_code(prompt: str) -> str: 
Application code that calls the agent, in a real application this might be an API endpoint.
[](https://ai.pydantic.dev/dependencies/<#__code_4_annotation_3>)
  ...
  ...
  # now deep within application code we call our agent
  async with httpx.AsyncClient() as client:
    app_deps = MyDeps('foobar', client)
    result = await joke_agent.run(prompt, deps=app_deps) 
Call the agent from within the application code, in a real application this call might be deep within a call stack. Note app_deps here will NOT be used when deps are overridden.
[](https://ai.pydantic.dev/dependencies/<#__code_4_annotation_4>)
  return result.data

```

_(This example is complete, it can be run "as is")_
test_joke_app.py```
from joke_app import MyDeps, application_code, joke_agent

class TestMyDeps(MyDeps): 
Define a subclass of MyDeps in tests to customise the system prompt factory.
[](https://ai.pydantic.dev/dependencies/<#__code_5_annotation_1>)
  async def system_prompt_factory(self) -> str:
    return 'test prompt'

async def test_application_code():
  test_deps = TestMyDeps('test_key', None) 
Create an instance of the test dependency, we don't need to pass an http_client here as it's not used.
[](https://ai.pydantic.dev/dependencies/<#__code_5_annotation_2>)
  with joke_agent.override(deps=test_deps): 
Override the dependencies of the agent for the duration of the with block, test_deps will be used when the agent is run.
[](https://ai.pydantic.dev/dependencies/<#__code_5_annotation_3>)
    joke = await application_code('Tell me a joke.') 
Now we can safely call our application code, the agent will use the overridden dependencies.
[](https://ai.pydantic.dev/dependencies/<#__code_5_annotation_4>)
  assert joke.startswith('Did you hear about the toothpaste scandal?')

```

## Examples
The following examples demonstrate how to use dependencies in PydanticAI:
  * [Weather Agent](https://ai.pydantic.dev/dependencies/<../examples/weather-agent/>)
  * [SQL Generation](https://ai.pydantic.dev/dependencies/<../examples/sql-gen/>)
  * [RAG](https://ai.pydantic.dev/dependencies/<../examples/rag/>)


© Pydantic Services Inc. 2024 to present 
