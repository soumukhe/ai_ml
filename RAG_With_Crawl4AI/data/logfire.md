[ Skip to content ](https://ai.pydantic.dev/logfire/<#debugging-and-monitoring>)
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/logfire/<..> "PydanticAI")
PydanticAI 
Debugging and Monitoring 
Initializing search 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/logfire/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/logfire/<..> "PydanticAI") PydanticAI 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/logfire/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
  * [ Introduction  ](https://ai.pydantic.dev/logfire/<..>)
  * [ Installation  ](https://ai.pydantic.dev/logfire/<../install/>)
  * [ Getting Help  ](https://ai.pydantic.dev/logfire/<../help/>)
  * [ Contributing  ](https://ai.pydantic.dev/logfire/<../contributing/>)
  * [ Troubleshooting  ](https://ai.pydantic.dev/logfire/<../troubleshooting/>)
  * Documentation  Documentation 
    * [ Agents  ](https://ai.pydantic.dev/logfire/<../agents/>)
    * [ Models  ](https://ai.pydantic.dev/logfire/<../models/>)
    * [ Dependencies  ](https://ai.pydantic.dev/logfire/<../dependencies/>)
    * [ Function Tools  ](https://ai.pydantic.dev/logfire/<../tools/>)
    * [ Results  ](https://ai.pydantic.dev/logfire/<../results/>)
    * [ Messages and chat history  ](https://ai.pydantic.dev/logfire/<../message-history/>)
    * [ Testing and Evals  ](https://ai.pydantic.dev/logfire/<../testing-evals/>)
    * Debugging and Monitoring  [ Debugging and Monitoring  ](https://ai.pydantic.dev/logfire/<./>) Table of contents 
      * [ Pydantic Logfire  ](https://ai.pydantic.dev/logfire/<#pydantic-logfire>)
      * [ Using Logfire  ](https://ai.pydantic.dev/logfire/<#using-logfire>)
        * [ Debugging  ](https://ai.pydantic.dev/logfire/<#debugging>)
        * [ Monitoring Performance  ](https://ai.pydantic.dev/logfire/<#monitoring-performance>)
    * [ Multi-agent Applications  ](https://ai.pydantic.dev/logfire/<../multi-agent-applications/>)
  * [ Examples  ](https://ai.pydantic.dev/logfire/<../examples/>)
Examples 
    * [ Pydantic Model  ](https://ai.pydantic.dev/logfire/<../examples/pydantic-model/>)
    * [ Weather agent  ](https://ai.pydantic.dev/logfire/<../examples/weather-agent/>)
    * [ Bank support  ](https://ai.pydantic.dev/logfire/<../examples/bank-support/>)
    * [ SQL Generation  ](https://ai.pydantic.dev/logfire/<../examples/sql-gen/>)
    * [ Flight booking  ](https://ai.pydantic.dev/logfire/<../examples/flight-booking/>)
    * [ RAG  ](https://ai.pydantic.dev/logfire/<../examples/rag/>)
    * [ Stream markdown  ](https://ai.pydantic.dev/logfire/<../examples/stream-markdown/>)
    * [ Stream whales  ](https://ai.pydantic.dev/logfire/<../examples/stream-whales/>)
    * [ Chat App with FastAPI  ](https://ai.pydantic.dev/logfire/<../examples/chat-app/>)
  * API Reference  API Reference 
    * [ pydantic_ai.agent  ](https://ai.pydantic.dev/logfire/<../api/agent/>)
    * [ pydantic_ai.tools  ](https://ai.pydantic.dev/logfire/<../api/tools/>)
    * [ pydantic_ai.result  ](https://ai.pydantic.dev/logfire/<../api/result/>)
    * [ pydantic_ai.messages  ](https://ai.pydantic.dev/logfire/<../api/messages/>)
    * [ pydantic_ai.exceptions  ](https://ai.pydantic.dev/logfire/<../api/exceptions/>)
    * [ pydantic_ai.settings  ](https://ai.pydantic.dev/logfire/<../api/settings/>)
    * [ pydantic_ai.usage  ](https://ai.pydantic.dev/logfire/<../api/usage/>)
    * [ pydantic_ai.format_as_xml  ](https://ai.pydantic.dev/logfire/<../api/format_as_xml/>)
    * [ pydantic_ai.models  ](https://ai.pydantic.dev/logfire/<../api/models/base/>)
    * [ pydantic_ai.models.openai  ](https://ai.pydantic.dev/logfire/<../api/models/openai/>)
    * [ pydantic_ai.models.anthropic  ](https://ai.pydantic.dev/logfire/<../api/models/anthropic/>)
    * [ pydantic_ai.models.gemini  ](https://ai.pydantic.dev/logfire/<../api/models/gemini/>)
    * [ pydantic_ai.models.vertexai  ](https://ai.pydantic.dev/logfire/<../api/models/vertexai/>)
    * [ pydantic_ai.models.groq  ](https://ai.pydantic.dev/logfire/<../api/models/groq/>)
    * [ pydantic_ai.models.mistral  ](https://ai.pydantic.dev/logfire/<../api/models/mistral/>)
    * [ pydantic_ai.models.ollama  ](https://ai.pydantic.dev/logfire/<../api/models/ollama/>)
    * [ pydantic_ai.models.test  ](https://ai.pydantic.dev/logfire/<../api/models/test/>)
    * [ pydantic_ai.models.function  ](https://ai.pydantic.dev/logfire/<../api/models/function/>)


Table of contents 
  * [ Pydantic Logfire  ](https://ai.pydantic.dev/logfire/<#pydantic-logfire>)
  * [ Using Logfire  ](https://ai.pydantic.dev/logfire/<#using-logfire>)
    * [ Debugging  ](https://ai.pydantic.dev/logfire/<#debugging>)
    * [ Monitoring Performance  ](https://ai.pydantic.dev/logfire/<#monitoring-performance>)


  1. [ Introduction  ](https://ai.pydantic.dev/logfire/<..>)
  2. [ Documentation  ](https://ai.pydantic.dev/logfire/<../agents/>)


# Debugging and Monitoring
Applications that use LLMs have some challenges that are well known and understood: LLMs are **slow** , **unreliable** and **expensive**.
These applications also have some challenges that most developers have encountered much less often: LLMs are **fickle** and **non-deterministic**. Subtle changes in a prompt can completely change a model's performance, and there's no `EXPLAIN` query you can run to understand why.
Warning
From a software engineers point of view, you can think of LLMs as the worst database you've ever heard of, but worse.
If LLMs weren't so bloody useful, we'd never touch them.
To build successful applications with LLMs, we need new tools to understand both model performance, and the behavior of applications that rely on them.
LLM Observability tools that just let you understand how your model is performing are useless: making API calls to an LLM is easy, it's building that into an application that's hard.
## Pydantic Logfire
[Pydantic Logfire](https://ai.pydantic.dev/logfire/<https:/pydantic.dev/logfire>) is an observability platform developed by the team who created and maintain Pydantic and PydanticAI. Logfire aims to let you understand your entire application: Gen AI, classic predictive AI, HTTP traffic, database queries and everything else a modern application needs.
Pydantic Logfire is a commercial product
Logfire is a commercially supported, hosted platform with an extremely generous and perpetual [free tier](https://ai.pydantic.dev/logfire/<https:/pydantic.dev/pricing/>). You can sign up and start using Logfire in a couple of minutes.
PydanticAI has built-in (but optional) support for Logfire via the `logfire-api`[](https://ai.pydantic.dev/logfire/<https:/github.com/pydantic/logfire/tree/main/logfire-api>) no-op package.
That means if the `logfire` package is installed and configured, detailed information about agent runs is sent to Logfire. But if the `logfire` package is not installed, there's virtually no overhead and nothing is sent.
Here's an example showing details of running the [Weather Agent](https://ai.pydantic.dev/logfire/<../examples/weather-agent/>) in Logfire:
[![Weather Agent Logfire](https://ai.pydantic.dev/img/logfire-weather-agent.png)](https://ai.pydantic.dev/logfire/<../img/logfire-weather-agent.png>)
## Using Logfire
To use logfire, you'll need a logfire [account](https://ai.pydantic.dev/logfire/<https:/logfire.pydantic.dev>), and logfire installed:
[pip](https://ai.pydantic.dev/logfire/<#__tabbed_1_1>)[uv](https://ai.pydantic.dev/logfire/<#__tabbed_1_2>)
```
pipinstall'pydantic-ai[logfire]'

```

```
uvadd'pydantic-ai[logfire]'

```

Then authenticate your local environment with logfire:
[pip](https://ai.pydantic.dev/logfire/<#__tabbed_2_1>)[uv](https://ai.pydantic.dev/logfire/<#__tabbed_2_2>)
```
logfireauth

```

```
uvrunlogfireauth

```

And configure a project to send data to:
[pip](https://ai.pydantic.dev/logfire/<#__tabbed_3_1>)[uv](https://ai.pydantic.dev/logfire/<#__tabbed_3_2>)
```
logfireprojectsnew

```

```
uvrunlogfireprojectsnew

```

(Or use an existing project with `logfire projects use`)
The last step is to add logfire to your code:
adding_logfire.py```
import logfire
logfire.configure()

```

The [logfire documentation](https://ai.pydantic.dev/logfire/<https:/logfire.pydantic.dev/docs/>) has more details on how to use logfire, including how to instrument other libraries like Pydantic, HTTPX and FastAPI.
Since Logfire is build on [OpenTelemetry](https://ai.pydantic.dev/logfire/<https:/opentelemetry.io/>), you can use the Logfire Python SDK to send data to any OpenTelemetry collector.
Once you have logfire set up, there are two primary ways it can help you understand your application:
  * **Debugging** — Using the live view to see what's happening in your application in real-time.
  * **Monitoring** — Using SQL and dashboards to observe the behavior of your application, Logfire is effectively a SQL database that stores information about how your application is running.


### Debugging
To demonstrate how Logfire can let you visualise the flow of a PydanticAI run, here's the view you get from Logfire while running the [chat app examples](https://ai.pydantic.dev/logfire/<../examples/chat-app/>):
### Monitoring Performance
We can also query data with SQL in Logfire to monitor the performance of an application. Here's a real world example of using Logfire to monitor PydanticAI runs inside Logfire itself:
[![Logfire monitoring PydanticAI](https://ai.pydantic.dev/img/logfire-monitoring-pydanticai.png)](https://ai.pydantic.dev/logfire/<../img/logfire-monitoring-pydanticai.png>)
© Pydantic Services Inc. 2024 to present 
