[ Skip to content ](https://ai.pydantic.dev/contributing/<#installation-and-setup>)
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/contributing/<..> "PydanticAI")
PydanticAI 
Contributing 
Initializing search 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/contributing/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/contributing/<..> "PydanticAI") PydanticAI 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/contributing/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
  * [ Introduction  ](https://ai.pydantic.dev/contributing/<..>)
  * [ Installation  ](https://ai.pydantic.dev/contributing/<../install/>)
  * [ Getting Help  ](https://ai.pydantic.dev/contributing/<../help/>)
  * Contributing  [ Contributing  ](https://ai.pydantic.dev/contributing/<./>) Table of contents 
    * [ Installation and Setup  ](https://ai.pydantic.dev/contributing/<#installation-and-setup>)
    * [ Running Tests etc.  ](https://ai.pydantic.dev/contributing/<#running-tests-etc>)
    * [ Documentation Changes  ](https://ai.pydantic.dev/contributing/<#documentation-changes>)
    * [ Rules for adding new models to PydanticAI  ](https://ai.pydantic.dev/contributing/<#new-model-rules>)
  * [ Troubleshooting  ](https://ai.pydantic.dev/contributing/<../troubleshooting/>)
  * Documentation  Documentation 
    * [ Agents  ](https://ai.pydantic.dev/contributing/<../agents/>)
    * [ Models  ](https://ai.pydantic.dev/contributing/<../models/>)
    * [ Dependencies  ](https://ai.pydantic.dev/contributing/<../dependencies/>)
    * [ Function Tools  ](https://ai.pydantic.dev/contributing/<../tools/>)
    * [ Results  ](https://ai.pydantic.dev/contributing/<../results/>)
    * [ Messages and chat history  ](https://ai.pydantic.dev/contributing/<../message-history/>)
    * [ Testing and Evals  ](https://ai.pydantic.dev/contributing/<../testing-evals/>)
    * [ Debugging and Monitoring  ](https://ai.pydantic.dev/contributing/<../logfire/>)
    * [ Multi-agent Applications  ](https://ai.pydantic.dev/contributing/<../multi-agent-applications/>)
  * [ Examples  ](https://ai.pydantic.dev/contributing/<../examples/>)
Examples 
    * [ Pydantic Model  ](https://ai.pydantic.dev/contributing/<../examples/pydantic-model/>)
    * [ Weather agent  ](https://ai.pydantic.dev/contributing/<../examples/weather-agent/>)
    * [ Bank support  ](https://ai.pydantic.dev/contributing/<../examples/bank-support/>)
    * [ SQL Generation  ](https://ai.pydantic.dev/contributing/<../examples/sql-gen/>)
    * [ Flight booking  ](https://ai.pydantic.dev/contributing/<../examples/flight-booking/>)
    * [ RAG  ](https://ai.pydantic.dev/contributing/<../examples/rag/>)
    * [ Stream markdown  ](https://ai.pydantic.dev/contributing/<../examples/stream-markdown/>)
    * [ Stream whales  ](https://ai.pydantic.dev/contributing/<../examples/stream-whales/>)
    * [ Chat App with FastAPI  ](https://ai.pydantic.dev/contributing/<../examples/chat-app/>)
  * API Reference  API Reference 
    * [ pydantic_ai.agent  ](https://ai.pydantic.dev/contributing/<../api/agent/>)
    * [ pydantic_ai.tools  ](https://ai.pydantic.dev/contributing/<../api/tools/>)
    * [ pydantic_ai.result  ](https://ai.pydantic.dev/contributing/<../api/result/>)
    * [ pydantic_ai.messages  ](https://ai.pydantic.dev/contributing/<../api/messages/>)
    * [ pydantic_ai.exceptions  ](https://ai.pydantic.dev/contributing/<../api/exceptions/>)
    * [ pydantic_ai.settings  ](https://ai.pydantic.dev/contributing/<../api/settings/>)
    * [ pydantic_ai.usage  ](https://ai.pydantic.dev/contributing/<../api/usage/>)
    * [ pydantic_ai.format_as_xml  ](https://ai.pydantic.dev/contributing/<../api/format_as_xml/>)
    * [ pydantic_ai.models  ](https://ai.pydantic.dev/contributing/<../api/models/base/>)
    * [ pydantic_ai.models.openai  ](https://ai.pydantic.dev/contributing/<../api/models/openai/>)
    * [ pydantic_ai.models.anthropic  ](https://ai.pydantic.dev/contributing/<../api/models/anthropic/>)
    * [ pydantic_ai.models.gemini  ](https://ai.pydantic.dev/contributing/<../api/models/gemini/>)
    * [ pydantic_ai.models.vertexai  ](https://ai.pydantic.dev/contributing/<../api/models/vertexai/>)
    * [ pydantic_ai.models.groq  ](https://ai.pydantic.dev/contributing/<../api/models/groq/>)
    * [ pydantic_ai.models.mistral  ](https://ai.pydantic.dev/contributing/<../api/models/mistral/>)
    * [ pydantic_ai.models.ollama  ](https://ai.pydantic.dev/contributing/<../api/models/ollama/>)
    * [ pydantic_ai.models.test  ](https://ai.pydantic.dev/contributing/<../api/models/test/>)
    * [ pydantic_ai.models.function  ](https://ai.pydantic.dev/contributing/<../api/models/function/>)


Table of contents 
  * [ Installation and Setup  ](https://ai.pydantic.dev/contributing/<#installation-and-setup>)
  * [ Running Tests etc.  ](https://ai.pydantic.dev/contributing/<#running-tests-etc>)
  * [ Documentation Changes  ](https://ai.pydantic.dev/contributing/<#documentation-changes>)
  * [ Rules for adding new models to PydanticAI  ](https://ai.pydantic.dev/contributing/<#new-model-rules>)


# Contributing
We'd love you to contribute to PydanticAI!
## Installation and Setup
Clone your fork and cd into the repo directory
```
gitclonegit@github.com:<yourusername>/pydantic-ai.git
cdpydantic-ai

```

Install `uv` (version 0.4.30 or later) and `pre-commit`
We use pipx here, for other options see:
  * `uv`[ install docs](https://ai.pydantic.dev/contributing/<https:/docs.astral.sh/uv/getting-started/installation/>)
  * `pre-commit`[ install docs](https://ai.pydantic.dev/contributing/<https:/pre-commit.com/#install>)


To get `pipx` itself, see [these docs](https://ai.pydantic.dev/contributing/<https:/pypa.github.io/pipx/>)
```
pipxinstalluvpre-commit

```

Install `pydantic-ai`, all dependencies and pre-commit hooks
```
makeinstall

```

## Running Tests etc.
We use `make` to manage most commands you'll need to run.
For details on available commands, run:
```
makehelp

```

To run code formatting, linting, static type checks, and tests with coverage report generation, run:
```
make

```

## Documentation Changes
To run the documentation page locally, run:
```
uvrunmkdocsserve

```

## Rules for adding new models to PydanticAI
To avoid an excessive workload for the maintainers of PydanticAI, we can't accept all model contributions, so we're setting the following rules for when we'll accept new models and when we won't. This should hopefully reduce the chances of disappointment and wasted work.
  * To add a new model with an extra dependency, that dependency needs > 500k monthly downloads from PyPI consistently over 3 months or more
  * To add a new model which uses another models logic internally and has no extra dependencies, that model's GitHub org needs > 20k stars in total
  * For any other model that's just a custom URL and API key, we're happy to add a one-paragraph description with a link and instructions on the URL to use
  * For any other model that requires more logic, we recommend you release your own Python package `pydantic-ai-xxx`, which depends on `pydantic-ai-slim`[](https://ai.pydantic.dev/contributing/<../install/#slim-install>) and implements a model that inherits from our `Model`[](https://ai.pydantic.dev/contributing/<../api/models/base/#pydantic_ai.models.Model>) ABC


If you're unsure about adding a model, please [create an issue](https://ai.pydantic.dev/contributing/<https:/github.com/pydantic/pydantic-ai/issues>).
© Pydantic Services Inc. 2024 to present 
