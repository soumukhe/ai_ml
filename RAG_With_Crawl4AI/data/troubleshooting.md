[ Skip to content ](https://ai.pydantic.dev/troubleshooting/<#troubleshooting>)
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/troubleshooting/<..> "PydanticAI")
PydanticAI 
Troubleshooting 
Initializing search 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/troubleshooting/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/troubleshooting/<..> "PydanticAI") PydanticAI 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/troubleshooting/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
  * [ Introduction  ](https://ai.pydantic.dev/troubleshooting/<..>)
  * [ Installation  ](https://ai.pydantic.dev/troubleshooting/<../install/>)
  * [ Getting Help  ](https://ai.pydantic.dev/troubleshooting/<../help/>)
  * [ Contributing  ](https://ai.pydantic.dev/troubleshooting/<../contributing/>)
  * Troubleshooting  [ Troubleshooting  ](https://ai.pydantic.dev/troubleshooting/<./>) Table of contents 
    * [ Jupyter Notebook Errors  ](https://ai.pydantic.dev/troubleshooting/<#jupyter-notebook-errors>)
      * [ RuntimeError: This event loop is already running  ](https://ai.pydantic.dev/troubleshooting/<#runtimeerror-this-event-loop-is-already-running>)
    * [ API Key Configuration  ](https://ai.pydantic.dev/troubleshooting/<#api-key-configuration>)
      * [ UserError: API key must be provided or set in the [MODEL]_API_KEY environment variable  ](https://ai.pydantic.dev/troubleshooting/<#usererror-api-key-must-be-provided-or-set-in-the-model_api_key-environment-variable>)
  * Documentation  Documentation 
    * [ Agents  ](https://ai.pydantic.dev/troubleshooting/<../agents/>)
    * [ Models  ](https://ai.pydantic.dev/troubleshooting/<../models/>)
    * [ Dependencies  ](https://ai.pydantic.dev/troubleshooting/<../dependencies/>)
    * [ Function Tools  ](https://ai.pydantic.dev/troubleshooting/<../tools/>)
    * [ Results  ](https://ai.pydantic.dev/troubleshooting/<../results/>)
    * [ Messages and chat history  ](https://ai.pydantic.dev/troubleshooting/<../message-history/>)
    * [ Testing and Evals  ](https://ai.pydantic.dev/troubleshooting/<../testing-evals/>)
    * [ Debugging and Monitoring  ](https://ai.pydantic.dev/troubleshooting/<../logfire/>)
    * [ Multi-agent Applications  ](https://ai.pydantic.dev/troubleshooting/<../multi-agent-applications/>)
  * [ Examples  ](https://ai.pydantic.dev/troubleshooting/<../examples/>)
Examples 
    * [ Pydantic Model  ](https://ai.pydantic.dev/troubleshooting/<../examples/pydantic-model/>)
    * [ Weather agent  ](https://ai.pydantic.dev/troubleshooting/<../examples/weather-agent/>)
    * [ Bank support  ](https://ai.pydantic.dev/troubleshooting/<../examples/bank-support/>)
    * [ SQL Generation  ](https://ai.pydantic.dev/troubleshooting/<../examples/sql-gen/>)
    * [ Flight booking  ](https://ai.pydantic.dev/troubleshooting/<../examples/flight-booking/>)
    * [ RAG  ](https://ai.pydantic.dev/troubleshooting/<../examples/rag/>)
    * [ Stream markdown  ](https://ai.pydantic.dev/troubleshooting/<../examples/stream-markdown/>)
    * [ Stream whales  ](https://ai.pydantic.dev/troubleshooting/<../examples/stream-whales/>)
    * [ Chat App with FastAPI  ](https://ai.pydantic.dev/troubleshooting/<../examples/chat-app/>)
  * API Reference  API Reference 
    * [ pydantic_ai.agent  ](https://ai.pydantic.dev/troubleshooting/<../api/agent/>)
    * [ pydantic_ai.tools  ](https://ai.pydantic.dev/troubleshooting/<../api/tools/>)
    * [ pydantic_ai.result  ](https://ai.pydantic.dev/troubleshooting/<../api/result/>)
    * [ pydantic_ai.messages  ](https://ai.pydantic.dev/troubleshooting/<../api/messages/>)
    * [ pydantic_ai.exceptions  ](https://ai.pydantic.dev/troubleshooting/<../api/exceptions/>)
    * [ pydantic_ai.settings  ](https://ai.pydantic.dev/troubleshooting/<../api/settings/>)
    * [ pydantic_ai.usage  ](https://ai.pydantic.dev/troubleshooting/<../api/usage/>)
    * [ pydantic_ai.format_as_xml  ](https://ai.pydantic.dev/troubleshooting/<../api/format_as_xml/>)
    * [ pydantic_ai.models  ](https://ai.pydantic.dev/troubleshooting/<../api/models/base/>)
    * [ pydantic_ai.models.openai  ](https://ai.pydantic.dev/troubleshooting/<../api/models/openai/>)
    * [ pydantic_ai.models.anthropic  ](https://ai.pydantic.dev/troubleshooting/<../api/models/anthropic/>)
    * [ pydantic_ai.models.gemini  ](https://ai.pydantic.dev/troubleshooting/<../api/models/gemini/>)
    * [ pydantic_ai.models.vertexai  ](https://ai.pydantic.dev/troubleshooting/<../api/models/vertexai/>)
    * [ pydantic_ai.models.groq  ](https://ai.pydantic.dev/troubleshooting/<../api/models/groq/>)
    * [ pydantic_ai.models.mistral  ](https://ai.pydantic.dev/troubleshooting/<../api/models/mistral/>)
    * [ pydantic_ai.models.ollama  ](https://ai.pydantic.dev/troubleshooting/<../api/models/ollama/>)
    * [ pydantic_ai.models.test  ](https://ai.pydantic.dev/troubleshooting/<../api/models/test/>)
    * [ pydantic_ai.models.function  ](https://ai.pydantic.dev/troubleshooting/<../api/models/function/>)


Table of contents 
  * [ Jupyter Notebook Errors  ](https://ai.pydantic.dev/troubleshooting/<#jupyter-notebook-errors>)
    * [ RuntimeError: This event loop is already running  ](https://ai.pydantic.dev/troubleshooting/<#runtimeerror-this-event-loop-is-already-running>)
  * [ API Key Configuration  ](https://ai.pydantic.dev/troubleshooting/<#api-key-configuration>)
    * [ UserError: API key must be provided or set in the [MODEL]_API_KEY environment variable  ](https://ai.pydantic.dev/troubleshooting/<#usererror-api-key-must-be-provided-or-set-in-the-model_api_key-environment-variable>)


# Troubleshooting
Below are suggestions on how to fix some common errors you might encounter while using PydanticAI. If the issue you're experiencing is not listed below or addressed in the documentation, please feel free to ask in the [Pydantic Slack](https://ai.pydantic.dev/troubleshooting/<../help/>) or create an issue on [GitHub](https://ai.pydantic.dev/troubleshooting/<https:/github.com/pydantic/pydantic-ai/issues>).
## Jupyter Notebook Errors
### `RuntimeError: This event loop is already running`
This error is caused by conflicts between the event loops in Jupyter notebook and PydanticAI's. One way to manage these conflicts is by using `nest-asyncio`[](https://ai.pydantic.dev/troubleshooting/<https:/pypi.org/project/nest-asyncio/>). Namely, before you execute any agent runs, do the following: 
```
import nest_asyncio
nest_asyncio.apply()

```

Note: This fix also applies to Google Colab. 
## API Key Configuration
### `UserError: API key must be provided or set in the [MODEL]_API_KEY environment variable`
If you're running into issues with setting the API key for your model, visit the [Models](https://ai.pydantic.dev/troubleshooting/<../models/>) page to learn more about how to set an environment variable and/or pass in an `api_key` argument.
© Pydantic Services Inc. 2024 to present 
