[ Skip to content ](https://ai.pydantic.dev/examples/stream-whales/<#running-the-example>)
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/stream-whales/<../..> "PydanticAI")
PydanticAI 
Stream whales 
Initializing search 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/stream-whales/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/stream-whales/<../..> "PydanticAI") PydanticAI 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/stream-whales/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
  * [ Introduction  ](https://ai.pydantic.dev/examples/stream-whales/<../..>)
  * [ Installation  ](https://ai.pydantic.dev/examples/stream-whales/install/>)
  * [ Getting Help  ](https://ai.pydantic.dev/examples/stream-whales/help/>)
  * [ Contributing  ](https://ai.pydantic.dev/examples/stream-whales/contributing/>)
  * [ Troubleshooting  ](https://ai.pydantic.dev/examples/stream-whales/troubleshooting/>)
  * Documentation  Documentation 
    * [ Agents  ](https://ai.pydantic.dev/examples/stream-whales/agents/>)
    * [ Models  ](https://ai.pydantic.dev/examples/stream-whales/models/>)
    * [ Dependencies  ](https://ai.pydantic.dev/examples/stream-whales/dependencies/>)
    * [ Function Tools  ](https://ai.pydantic.dev/examples/stream-whales/tools/>)
    * [ Results  ](https://ai.pydantic.dev/examples/stream-whales/results/>)
    * [ Messages and chat history  ](https://ai.pydantic.dev/examples/stream-whales/message-history/>)
    * [ Testing and Evals  ](https://ai.pydantic.dev/examples/stream-whales/testing-evals/>)
    * [ Debugging and Monitoring  ](https://ai.pydantic.dev/examples/stream-whales/logfire/>)
    * [ Multi-agent Applications  ](https://ai.pydantic.dev/examples/stream-whales/multi-agent-applications/>)
  * [ Examples  ](https://ai.pydantic.dev/examples/stream-whales/<../>)
Examples 
    * [ Pydantic Model  ](https://ai.pydantic.dev/examples/stream-whales/<../pydantic-model/>)
    * [ Weather agent  ](https://ai.pydantic.dev/examples/stream-whales/<../weather-agent/>)
    * [ Bank support  ](https://ai.pydantic.dev/examples/stream-whales/<../bank-support/>)
    * [ SQL Generation  ](https://ai.pydantic.dev/examples/stream-whales/<../sql-gen/>)
    * [ Flight booking  ](https://ai.pydantic.dev/examples/stream-whales/<../flight-booking/>)
    * [ RAG  ](https://ai.pydantic.dev/examples/stream-whales/<../rag/>)
    * [ Stream markdown  ](https://ai.pydantic.dev/examples/stream-whales/<../stream-markdown/>)
    * Stream whales  [ Stream whales  ](https://ai.pydantic.dev/examples/stream-whales/<./>) Table of contents 
      * [ Running the Example  ](https://ai.pydantic.dev/examples/stream-whales/<#running-the-example>)
      * [ Example Code  ](https://ai.pydantic.dev/examples/stream-whales/<#example-code>)
    * [ Chat App with FastAPI  ](https://ai.pydantic.dev/examples/stream-whales/<../chat-app/>)
  * API Reference  API Reference 
    * [ pydantic_ai.agent  ](https://ai.pydantic.dev/examples/stream-whales/api/agent/>)
    * [ pydantic_ai.tools  ](https://ai.pydantic.dev/examples/stream-whales/api/tools/>)
    * [ pydantic_ai.result  ](https://ai.pydantic.dev/examples/stream-whales/api/result/>)
    * [ pydantic_ai.messages  ](https://ai.pydantic.dev/examples/stream-whales/api/messages/>)
    * [ pydantic_ai.exceptions  ](https://ai.pydantic.dev/examples/stream-whales/api/exceptions/>)
    * [ pydantic_ai.settings  ](https://ai.pydantic.dev/examples/stream-whales/api/settings/>)
    * [ pydantic_ai.usage  ](https://ai.pydantic.dev/examples/stream-whales/api/usage/>)
    * [ pydantic_ai.format_as_xml  ](https://ai.pydantic.dev/examples/stream-whales/api/format_as_xml/>)
    * [ pydantic_ai.models  ](https://ai.pydantic.dev/examples/stream-whales/api/models/base/>)
    * [ pydantic_ai.models.openai  ](https://ai.pydantic.dev/examples/stream-whales/api/models/openai/>)
    * [ pydantic_ai.models.anthropic  ](https://ai.pydantic.dev/examples/stream-whales/api/models/anthropic/>)
    * [ pydantic_ai.models.gemini  ](https://ai.pydantic.dev/examples/stream-whales/api/models/gemini/>)
    * [ pydantic_ai.models.vertexai  ](https://ai.pydantic.dev/examples/stream-whales/api/models/vertexai/>)
    * [ pydantic_ai.models.groq  ](https://ai.pydantic.dev/examples/stream-whales/api/models/groq/>)
    * [ pydantic_ai.models.mistral  ](https://ai.pydantic.dev/examples/stream-whales/api/models/mistral/>)
    * [ pydantic_ai.models.ollama  ](https://ai.pydantic.dev/examples/stream-whales/api/models/ollama/>)
    * [ pydantic_ai.models.test  ](https://ai.pydantic.dev/examples/stream-whales/api/models/test/>)
    * [ pydantic_ai.models.function  ](https://ai.pydantic.dev/examples/stream-whales/api/models/function/>)


Table of contents 
  * [ Running the Example  ](https://ai.pydantic.dev/examples/stream-whales/<#running-the-example>)
  * [ Example Code  ](https://ai.pydantic.dev/examples/stream-whales/<#example-code>)


  1. [ Introduction  ](https://ai.pydantic.dev/examples/stream-whales/<../..>)
  2. [ Examples  ](https://ai.pydantic.dev/examples/stream-whales/<../>)


# Stream whales
Information about whales — an example of streamed structured response validation.
Demonstrates:
  * [streaming structured responses](https://ai.pydantic.dev/examples/stream-whales/results/#streaming-structured-responses>)


This script streams structured responses from GPT-4 about whales, validates the data and displays it as a dynamic table using `rich`[](https://ai.pydantic.dev/examples/stream-whales/<https:/github.com/Textualize/rich>) as the data is received.
## Running the Example
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/stream-whales/<../#usage>), run:
[pip](https://ai.pydantic.dev/examples/stream-whales/<#__tabbed_1_1>)[uv](https://ai.pydantic.dev/examples/stream-whales/<#__tabbed_1_2>)
```
python-mpydantic_ai_examples.stream_whales

```

```
uvrun-mpydantic_ai_examples.stream_whales

```

Should give an output like this:
## Example Code
stream_whales.py```
from typing import Annotated
import logfire
from pydantic import Field, ValidationError
from rich.console import Console
from rich.live import Live
from rich.table import Table
from typing_extensions import NotRequired, TypedDict
from pydantic_ai import Agent
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

class Whale(TypedDict):
  name: str
  length: Annotated[
    float, Field(description='Average length of an adult whale in meters.')
  ]
  weight: NotRequired[
    Annotated[
      float,
      Field(description='Average weight of an adult whale in kilograms.', ge=50),
    ]
  ]
  ocean: NotRequired[str]
  description: NotRequired[Annotated[str, Field(description='Short Description')]]

agent = Agent('openai:gpt-4', result_type=list[Whale])

async def main():
  console = Console()
  with Live('\n' * 36, console=console) as live:
    console.print('Requesting data...', style='cyan')
    async with agent.run_stream(
      'Generate me details of 5 species of Whale.'
    ) as result:
      console.print('Response:', style='green')
      async for message, last in result.stream_structured(debounce_by=0.01):
        try:
          whales = await result.validate_structured_result(
            message, allow_partial=not last
          )
        except ValidationError as exc:
          if all(
            e['type'] == 'missing' and e['loc'] == ('response',)
            for e in exc.errors()
          ):
            continue
          else:
            raise
        table = Table(
          title='Species of Whale',
          caption='Streaming Structured responses from GPT-4',
          width=120,
        )
        table.add_column('ID', justify='right')
        table.add_column('Name')
        table.add_column('Avg. Length (m)', justify='right')
        table.add_column('Avg. Weight (kg)', justify='right')
        table.add_column('Ocean')
        table.add_column('Description', justify='right')
        for wid, whale in enumerate(whales, start=1):
          table.add_row(
            str(wid),
            whale['name'],
            f'{whale["length"]:0.0f}',
            f'{w:0.0f}' if (w := whale.get('weight')) else '…',
            whale.get('ocean') or '…',
            whale.get('description') or '…',
          )
        live.update(table)

if __name__ == '__main__':
  import asyncio
  asyncio.run(main())

```

© Pydantic Services Inc. 2024 to present 
