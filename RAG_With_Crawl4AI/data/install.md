[ Skip to content ](https://ai.pydantic.dev/install/<#installation>)
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/install/<..> "PydanticAI")
PydanticAI 
Installation 
Initializing search 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/install/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/install/<..> "PydanticAI") PydanticAI 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/install/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
  * [ Introduction  ](https://ai.pydantic.dev/install/<..>)
  * Installation  [ Installation  ](https://ai.pydantic.dev/install/<./>) Table of contents 
    * [ Use with Pydantic Logfire  ](https://ai.pydantic.dev/install/<#use-with-pydantic-logfire>)
    * [ Running Examples  ](https://ai.pydantic.dev/install/<#running-examples>)
    * [ Slim Install  ](https://ai.pydantic.dev/install/<#slim-install>)
  * [ Getting Help  ](https://ai.pydantic.dev/install/<../help/>)
  * [ Contributing  ](https://ai.pydantic.dev/install/<../contributing/>)
  * [ Troubleshooting  ](https://ai.pydantic.dev/install/<../troubleshooting/>)
  * Documentation  Documentation 
    * [ Agents  ](https://ai.pydantic.dev/install/<../agents/>)
    * [ Models  ](https://ai.pydantic.dev/install/<../models/>)
    * [ Dependencies  ](https://ai.pydantic.dev/install/<../dependencies/>)
    * [ Function Tools  ](https://ai.pydantic.dev/install/<../tools/>)
    * [ Results  ](https://ai.pydantic.dev/install/<../results/>)
    * [ Messages and chat history  ](https://ai.pydantic.dev/install/<../message-history/>)
    * [ Testing and Evals  ](https://ai.pydantic.dev/install/<../testing-evals/>)
    * [ Debugging and Monitoring  ](https://ai.pydantic.dev/install/<../logfire/>)
    * [ Multi-agent Applications  ](https://ai.pydantic.dev/install/<../multi-agent-applications/>)
  * [ Examples  ](https://ai.pydantic.dev/install/<../examples/>)
Examples 
    * [ Pydantic Model  ](https://ai.pydantic.dev/install/<../examples/pydantic-model/>)
    * [ Weather agent  ](https://ai.pydantic.dev/install/<../examples/weather-agent/>)
    * [ Bank support  ](https://ai.pydantic.dev/install/<../examples/bank-support/>)
    * [ SQL Generation  ](https://ai.pydantic.dev/install/<../examples/sql-gen/>)
    * [ Flight booking  ](https://ai.pydantic.dev/install/<../examples/flight-booking/>)
    * [ RAG  ](https://ai.pydantic.dev/install/<../examples/rag/>)
    * [ Stream markdown  ](https://ai.pydantic.dev/install/<../examples/stream-markdown/>)
    * [ Stream whales  ](https://ai.pydantic.dev/install/<../examples/stream-whales/>)
    * [ Chat App with FastAPI  ](https://ai.pydantic.dev/install/<../examples/chat-app/>)
  * API Reference  API Reference 
    * [ pydantic_ai.agent  ](https://ai.pydantic.dev/install/<../api/agent/>)
    * [ pydantic_ai.tools  ](https://ai.pydantic.dev/install/<../api/tools/>)
    * [ pydantic_ai.result  ](https://ai.pydantic.dev/install/<../api/result/>)
    * [ pydantic_ai.messages  ](https://ai.pydantic.dev/install/<../api/messages/>)
    * [ pydantic_ai.exceptions  ](https://ai.pydantic.dev/install/<../api/exceptions/>)
    * [ pydantic_ai.settings  ](https://ai.pydantic.dev/install/<../api/settings/>)
    * [ pydantic_ai.usage  ](https://ai.pydantic.dev/install/<../api/usage/>)
    * [ pydantic_ai.format_as_xml  ](https://ai.pydantic.dev/install/<../api/format_as_xml/>)
    * [ pydantic_ai.models  ](https://ai.pydantic.dev/install/<../api/models/base/>)
    * [ pydantic_ai.models.openai  ](https://ai.pydantic.dev/install/<../api/models/openai/>)
    * [ pydantic_ai.models.anthropic  ](https://ai.pydantic.dev/install/<../api/models/anthropic/>)
    * [ pydantic_ai.models.gemini  ](https://ai.pydantic.dev/install/<../api/models/gemini/>)
    * [ pydantic_ai.models.vertexai  ](https://ai.pydantic.dev/install/<../api/models/vertexai/>)
    * [ pydantic_ai.models.groq  ](https://ai.pydantic.dev/install/<../api/models/groq/>)
    * [ pydantic_ai.models.mistral  ](https://ai.pydantic.dev/install/<../api/models/mistral/>)
    * [ pydantic_ai.models.ollama  ](https://ai.pydantic.dev/install/<../api/models/ollama/>)
    * [ pydantic_ai.models.test  ](https://ai.pydantic.dev/install/<../api/models/test/>)
    * [ pydantic_ai.models.function  ](https://ai.pydantic.dev/install/<../api/models/function/>)


Table of contents 
  * [ Use with Pydantic Logfire  ](https://ai.pydantic.dev/install/<#use-with-pydantic-logfire>)
  * [ Running Examples  ](https://ai.pydantic.dev/install/<#running-examples>)
  * [ Slim Install  ](https://ai.pydantic.dev/install/<#slim-install>)


# Installation
PydanticAI is available on PyPI as `pydantic-ai`[](https://ai.pydantic.dev/install/<https:/pypi.org/project/pydantic-ai/>) so installation is as simple as:
[pip](https://ai.pydantic.dev/install/<#__tabbed_1_1>)[uv](https://ai.pydantic.dev/install/<#__tabbed_1_2>)
```
pipinstallpydantic-ai

```

```
uvaddpydantic-ai

```

(Requires Python 3.9+)
This installs the `pydantic_ai` package, core dependencies, and libraries required to use all the models included in PydanticAI. If you want to use a specific model, you can install the ["slim"](https://ai.pydantic.dev/install/<#slim-install>) version of PydanticAI.
## Use with Pydantic Logfire
PydanticAI has an excellent (but completely optional) integration with [Pydantic Logfire](https://ai.pydantic.dev/install/<https:/pydantic.dev/logfire>) to help you view and understand agent runs.
To use Logfire with PydanticAI, install `pydantic-ai` or `pydantic-ai-slim` with the `logfire` optional group:
[pip](https://ai.pydantic.dev/install/<#__tabbed_2_1>)[uv](https://ai.pydantic.dev/install/<#__tabbed_2_2>)
```
pipinstall'pydantic-ai[logfire]'

```

```
uvadd'pydantic-ai[logfire]'

```

From there, follow the [Logfire setup docs](https://ai.pydantic.dev/install/<../logfire/#using-logfire>) to configure Logfire.
## Running Examples
We distribute the `pydantic_ai_examples`[](https://ai.pydantic.dev/install/<https:/github.com/pydantic/pydantic-ai/tree/main/pydantic_ai_examples>) directory as a separate PyPI package (`pydantic-ai-examples`[](https://ai.pydantic.dev/install/<https:/pypi.org/project/pydantic-ai-examples/>)) to make examples extremely easy to customize and run.
To install examples, use the `examples` optional group:
[pip](https://ai.pydantic.dev/install/<#__tabbed_3_1>)[uv](https://ai.pydantic.dev/install/<#__tabbed_3_2>)
```
pipinstall'pydantic-ai[examples]'

```

```
uvadd'pydantic-ai[examples]'

```

To run the examples, follow instructions in the [examples docs](https://ai.pydantic.dev/install/<../examples/>).
## Slim Install
If you know which model you're going to use and want to avoid installing superfluous packages, you can use the `pydantic-ai-slim`[](https://ai.pydantic.dev/install/<https:/pypi.org/project/pydantic-ai-slim/>) package. For example, if you're using just `OpenAIModel`[](https://ai.pydantic.dev/install/<../api/models/openai/#pydantic_ai.models.openai.OpenAIModel>), you would run:
[pip](https://ai.pydantic.dev/install/<#__tabbed_4_1>)[uv](https://ai.pydantic.dev/install/<#__tabbed_4_2>)
```
pipinstall'pydantic-ai-slim[openai]'

```

```
uvadd'pydantic-ai-slim[openai]'

```

See the [models](https://ai.pydantic.dev/install/<../models/>) documentation for information on which optional dependencies are required for each model.
You can also install dependencies for multiple models and use cases, for example:
[pip](https://ai.pydantic.dev/install/<#__tabbed_5_1>)[uv](https://ai.pydantic.dev/install/<#__tabbed_5_2>)
```
pipinstall'pydantic-ai-slim[openai,vertexai,logfire]'

```

```
uvadd'pydantic-ai-slim[openai,vertexai,logfire]'

```

© Pydantic Services Inc. 2024 to present 
