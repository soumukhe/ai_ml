[ Skip to content ](https://ai.pydantic.dev/examples/pydantic-model/<#pydantic-model>)
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/pydantic-model/<../..> "PydanticAI")
PydanticAI 
Pydantic Model 
Initializing search 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/pydantic-model/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/pydantic-model/<../..> "PydanticAI") PydanticAI 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/pydantic-model/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
  * [ Introduction  ](https://ai.pydantic.dev/examples/pydantic-model/<../..>)
  * [ Installation  ](https://ai.pydantic.dev/examples/pydantic-model/install/>)
  * [ Getting Help  ](https://ai.pydantic.dev/examples/pydantic-model/help/>)
  * [ Contributing  ](https://ai.pydantic.dev/examples/pydantic-model/contributing/>)
  * [ Troubleshooting  ](https://ai.pydantic.dev/examples/pydantic-model/troubleshooting/>)
  * Documentation  Documentation 
    * [ Agents  ](https://ai.pydantic.dev/examples/pydantic-model/agents/>)
    * [ Models  ](https://ai.pydantic.dev/examples/pydantic-model/models/>)
    * [ Dependencies  ](https://ai.pydantic.dev/examples/pydantic-model/dependencies/>)
    * [ Function Tools  ](https://ai.pydantic.dev/examples/pydantic-model/tools/>)
    * [ Results  ](https://ai.pydantic.dev/examples/pydantic-model/results/>)
    * [ Messages and chat history  ](https://ai.pydantic.dev/examples/pydantic-model/message-history/>)
    * [ Testing and Evals  ](https://ai.pydantic.dev/examples/pydantic-model/testing-evals/>)
    * [ Debugging and Monitoring  ](https://ai.pydantic.dev/examples/pydantic-model/logfire/>)
    * [ Multi-agent Applications  ](https://ai.pydantic.dev/examples/pydantic-model/multi-agent-applications/>)
  * [ Examples  ](https://ai.pydantic.dev/examples/pydantic-model/<../>)
Examples 
    * Pydantic Model  [ Pydantic Model  ](https://ai.pydantic.dev/examples/pydantic-model/<./>) Table of contents 
      * [ Running the Example  ](https://ai.pydantic.dev/examples/pydantic-model/<#running-the-example>)
      * [ Example Code  ](https://ai.pydantic.dev/examples/pydantic-model/<#example-code>)
    * [ Weather agent  ](https://ai.pydantic.dev/examples/pydantic-model/<../weather-agent/>)
    * [ Bank support  ](https://ai.pydantic.dev/examples/pydantic-model/<../bank-support/>)
    * [ SQL Generation  ](https://ai.pydantic.dev/examples/pydantic-model/<../sql-gen/>)
    * [ Flight booking  ](https://ai.pydantic.dev/examples/pydantic-model/<../flight-booking/>)
    * [ RAG  ](https://ai.pydantic.dev/examples/pydantic-model/<../rag/>)
    * [ Stream markdown  ](https://ai.pydantic.dev/examples/pydantic-model/<../stream-markdown/>)
    * [ Stream whales  ](https://ai.pydantic.dev/examples/pydantic-model/<../stream-whales/>)
    * [ Chat App with FastAPI  ](https://ai.pydantic.dev/examples/pydantic-model/<../chat-app/>)
  * API Reference  API Reference 
    * [ pydantic_ai.agent  ](https://ai.pydantic.dev/examples/pydantic-model/api/agent/>)
    * [ pydantic_ai.tools  ](https://ai.pydantic.dev/examples/pydantic-model/api/tools/>)
    * [ pydantic_ai.result  ](https://ai.pydantic.dev/examples/pydantic-model/api/result/>)
    * [ pydantic_ai.messages  ](https://ai.pydantic.dev/examples/pydantic-model/api/messages/>)
    * [ pydantic_ai.exceptions  ](https://ai.pydantic.dev/examples/pydantic-model/api/exceptions/>)
    * [ pydantic_ai.settings  ](https://ai.pydantic.dev/examples/pydantic-model/api/settings/>)
    * [ pydantic_ai.usage  ](https://ai.pydantic.dev/examples/pydantic-model/api/usage/>)
    * [ pydantic_ai.format_as_xml  ](https://ai.pydantic.dev/examples/pydantic-model/api/format_as_xml/>)
    * [ pydantic_ai.models  ](https://ai.pydantic.dev/examples/pydantic-model/api/models/base/>)
    * [ pydantic_ai.models.openai  ](https://ai.pydantic.dev/examples/pydantic-model/api/models/openai/>)
    * [ pydantic_ai.models.anthropic  ](https://ai.pydantic.dev/examples/pydantic-model/api/models/anthropic/>)
    * [ pydantic_ai.models.gemini  ](https://ai.pydantic.dev/examples/pydantic-model/api/models/gemini/>)
    * [ pydantic_ai.models.vertexai  ](https://ai.pydantic.dev/examples/pydantic-model/api/models/vertexai/>)
    * [ pydantic_ai.models.groq  ](https://ai.pydantic.dev/examples/pydantic-model/api/models/groq/>)
    * [ pydantic_ai.models.mistral  ](https://ai.pydantic.dev/examples/pydantic-model/api/models/mistral/>)
    * [ pydantic_ai.models.ollama  ](https://ai.pydantic.dev/examples/pydantic-model/api/models/ollama/>)
    * [ pydantic_ai.models.test  ](https://ai.pydantic.dev/examples/pydantic-model/api/models/test/>)
    * [ pydantic_ai.models.function  ](https://ai.pydantic.dev/examples/pydantic-model/api/models/function/>)


Table of contents 
  * [ Running the Example  ](https://ai.pydantic.dev/examples/pydantic-model/<#running-the-example>)
  * [ Example Code  ](https://ai.pydantic.dev/examples/pydantic-model/<#example-code>)


  1. [ Introduction  ](https://ai.pydantic.dev/examples/pydantic-model/<../..>)
  2. [ Examples  ](https://ai.pydantic.dev/examples/pydantic-model/<../>)


# Pydantic Model
Simple example of using PydanticAI to construct a Pydantic model from a text input.
Demonstrates:
  * [structured `result_type`](https://ai.pydantic.dev/examples/pydantic-model/results/#structured-result-validation>)


## Running the Example
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/pydantic-model/<../#usage>), run:
[pip](https://ai.pydantic.dev/examples/pydantic-model/<#__tabbed_1_1>)[uv](https://ai.pydantic.dev/examples/pydantic-model/<#__tabbed_1_2>)
```
python-mpydantic_ai_examples.pydantic_model

```

```
uvrun-mpydantic_ai_examples.pydantic_model

```

This examples uses `openai:gpt-4o` by default, but it works well with other models, e.g. you can run it with Gemini using:
[pip](https://ai.pydantic.dev/examples/pydantic-model/<#__tabbed_2_1>)[uv](https://ai.pydantic.dev/examples/pydantic-model/<#__tabbed_2_2>)
```
PYDANTIC_AI_MODEL=gemini-1.5-propython-mpydantic_ai_examples.pydantic_model

```

```
PYDANTIC_AI_MODEL=gemini-1.5-prouvrun-mpydantic_ai_examples.pydantic_model

```

(or `PYDANTIC_AI_MODEL=gemini-1.5-flash ...`)
## Example Code
pydantic_model.py```
import os
from typing import cast
import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

class MyModel(BaseModel):
  city: str
  country: str

model = cast(KnownModelName, os.getenv('PYDANTIC_AI_MODEL', 'openai:gpt-4o'))
print(f'Using model: {model}')
agent = Agent(model, result_type=MyModel)
if __name__ == '__main__':
  result = agent.run_sync('The windy city in the US of A.')
  print(result.data)
  print(result.usage())

```

© Pydantic Services Inc. 2024 to present 
