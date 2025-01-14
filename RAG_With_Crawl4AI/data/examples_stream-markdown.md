[ Skip to content ](https://ai.pydantic.dev/examples/stream-markdown/<#running-the-example>)
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/stream-markdown/<../..> "PydanticAI")
PydanticAI 
Stream markdown 
Initializing search 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/stream-markdown/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/stream-markdown/<../..> "PydanticAI") PydanticAI 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/stream-markdown/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
  * [ Introduction  ](https://ai.pydantic.dev/examples/stream-markdown/<../..>)
  * [ Installation  ](https://ai.pydantic.dev/examples/stream-markdown/install/>)
  * [ Getting Help  ](https://ai.pydantic.dev/examples/stream-markdown/help/>)
  * [ Contributing  ](https://ai.pydantic.dev/examples/stream-markdown/contributing/>)
  * [ Troubleshooting  ](https://ai.pydantic.dev/examples/stream-markdown/troubleshooting/>)
  * Documentation  Documentation 
    * [ Agents  ](https://ai.pydantic.dev/examples/stream-markdown/agents/>)
    * [ Models  ](https://ai.pydantic.dev/examples/stream-markdown/models/>)
    * [ Dependencies  ](https://ai.pydantic.dev/examples/stream-markdown/dependencies/>)
    * [ Function Tools  ](https://ai.pydantic.dev/examples/stream-markdown/tools/>)
    * [ Results  ](https://ai.pydantic.dev/examples/stream-markdown/results/>)
    * [ Messages and chat history  ](https://ai.pydantic.dev/examples/stream-markdown/message-history/>)
    * [ Testing and Evals  ](https://ai.pydantic.dev/examples/stream-markdown/testing-evals/>)
    * [ Debugging and Monitoring  ](https://ai.pydantic.dev/examples/stream-markdown/logfire/>)
    * [ Multi-agent Applications  ](https://ai.pydantic.dev/examples/stream-markdown/multi-agent-applications/>)
  * [ Examples  ](https://ai.pydantic.dev/examples/stream-markdown/<../>)
Examples 
    * [ Pydantic Model  ](https://ai.pydantic.dev/examples/stream-markdown/<../pydantic-model/>)
    * [ Weather agent  ](https://ai.pydantic.dev/examples/stream-markdown/<../weather-agent/>)
    * [ Bank support  ](https://ai.pydantic.dev/examples/stream-markdown/<../bank-support/>)
    * [ SQL Generation  ](https://ai.pydantic.dev/examples/stream-markdown/<../sql-gen/>)
    * [ Flight booking  ](https://ai.pydantic.dev/examples/stream-markdown/<../flight-booking/>)
    * [ RAG  ](https://ai.pydantic.dev/examples/stream-markdown/<../rag/>)
    * Stream markdown  [ Stream markdown  ](https://ai.pydantic.dev/examples/stream-markdown/<./>) Table of contents 
      * [ Running the Example  ](https://ai.pydantic.dev/examples/stream-markdown/<#running-the-example>)
      * [ Example Code  ](https://ai.pydantic.dev/examples/stream-markdown/<#example-code>)
    * [ Stream whales  ](https://ai.pydantic.dev/examples/stream-markdown/<../stream-whales/>)
    * [ Chat App with FastAPI  ](https://ai.pydantic.dev/examples/stream-markdown/<../chat-app/>)
  * API Reference  API Reference 
    * [ pydantic_ai.agent  ](https://ai.pydantic.dev/examples/stream-markdown/api/agent/>)
    * [ pydantic_ai.tools  ](https://ai.pydantic.dev/examples/stream-markdown/api/tools/>)
    * [ pydantic_ai.result  ](https://ai.pydantic.dev/examples/stream-markdown/api/result/>)
    * [ pydantic_ai.messages  ](https://ai.pydantic.dev/examples/stream-markdown/api/messages/>)
    * [ pydantic_ai.exceptions  ](https://ai.pydantic.dev/examples/stream-markdown/api/exceptions/>)
    * [ pydantic_ai.settings  ](https://ai.pydantic.dev/examples/stream-markdown/api/settings/>)
    * [ pydantic_ai.usage  ](https://ai.pydantic.dev/examples/stream-markdown/api/usage/>)
    * [ pydantic_ai.format_as_xml  ](https://ai.pydantic.dev/examples/stream-markdown/api/format_as_xml/>)
    * [ pydantic_ai.models  ](https://ai.pydantic.dev/examples/stream-markdown/api/models/base/>)
    * [ pydantic_ai.models.openai  ](https://ai.pydantic.dev/examples/stream-markdown/api/models/openai/>)
    * [ pydantic_ai.models.anthropic  ](https://ai.pydantic.dev/examples/stream-markdown/api/models/anthropic/>)
    * [ pydantic_ai.models.gemini  ](https://ai.pydantic.dev/examples/stream-markdown/api/models/gemini/>)
    * [ pydantic_ai.models.vertexai  ](https://ai.pydantic.dev/examples/stream-markdown/api/models/vertexai/>)
    * [ pydantic_ai.models.groq  ](https://ai.pydantic.dev/examples/stream-markdown/api/models/groq/>)
    * [ pydantic_ai.models.mistral  ](https://ai.pydantic.dev/examples/stream-markdown/api/models/mistral/>)
    * [ pydantic_ai.models.ollama  ](https://ai.pydantic.dev/examples/stream-markdown/api/models/ollama/>)
    * [ pydantic_ai.models.test  ](https://ai.pydantic.dev/examples/stream-markdown/api/models/test/>)
    * [ pydantic_ai.models.function  ](https://ai.pydantic.dev/examples/stream-markdown/api/models/function/>)


Table of contents 
  * [ Running the Example  ](https://ai.pydantic.dev/examples/stream-markdown/<#running-the-example>)
  * [ Example Code  ](https://ai.pydantic.dev/examples/stream-markdown/<#example-code>)


  1. [ Introduction  ](https://ai.pydantic.dev/examples/stream-markdown/<../..>)
  2. [ Examples  ](https://ai.pydantic.dev/examples/stream-markdown/<../>)


# Stream markdown
This example shows how to stream markdown from an agent, using the `rich`[](https://ai.pydantic.dev/examples/stream-markdown/<https:/github.com/Textualize/rich>) library to highlight the output in the terminal.
It'll run the example with both OpenAI and Google Gemini models if the required environment variables are set.
Demonstrates:
  * [streaming text responses](https://ai.pydantic.dev/examples/stream-markdown/results/#streaming-text>)


## Running the Example
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/stream-markdown/<../#usage>), run:
[pip](https://ai.pydantic.dev/examples/stream-markdown/<#__tabbed_1_1>)[uv](https://ai.pydantic.dev/examples/stream-markdown/<#__tabbed_1_2>)
```
python-mpydantic_ai_examples.stream_markdown

```

```
uvrun-mpydantic_ai_examples.stream_markdown

```

## Example Code
```
import asyncio
import os
import logfire
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
agent = Agent()
# models to try, and the appropriate env var
models: list[tuple[KnownModelName, str]] = [
  ('google-gla:gemini-1.5-flash', 'GEMINI_API_KEY'),
  ('openai:gpt-4o-mini', 'OPENAI_API_KEY'),
  ('groq:llama-3.1-70b-versatile', 'GROQ_API_KEY'),
]

async def main():
  prettier_code_blocks()
  console = Console()
  prompt = 'Show me a short example of using Pydantic.'
  console.log(f'Asking: {prompt}...', style='cyan')
  for model, env_var in models:
    if env_var in os.environ:
      console.log(f'Using model: {model}')
      with Live('', console=console, vertical_overflow='visible') as live:
        async with agent.run_stream(prompt, model=model) as result:
          async for message in result.stream():
            live.update(Markdown(message))
      console.log(result.usage())
    else:
      console.log(f'{model} requires {env_var} to be set.')

def prettier_code_blocks():
"""Make rich code blocks prettier and easier to copy.
  From https://github.com/samuelcolvin/aicli/blob/v0.8.0/samuelcolvin_aicli.py#L22
  """
  class SimpleCodeBlock(CodeBlock):
    def __rich_console__(
      self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
      code = str(self.text).rstrip()
      yield Text(self.lexer_name, style='dim')
      yield Syntax(
        code,
        self.lexer_name,
        theme=self.theme,
        background_color='default',
        word_wrap=True,
      )
      yield Text(f'/{self.lexer_name}', style='dim')
  Markdown.elements['fence'] = SimpleCodeBlock

if __name__ == '__main__':
  asyncio.run(main())

```

© Pydantic Services Inc. 2024 to present 
