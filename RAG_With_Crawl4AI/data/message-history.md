[ Skip to content ](https://ai.pydantic.dev/message-history/<#messages-and-chat-history>)
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/message-history/<..> "PydanticAI")
PydanticAI 
Messages and chat history 
Initializing search 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/message-history/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/message-history/<..> "PydanticAI") PydanticAI 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/message-history/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
  * [ Introduction  ](https://ai.pydantic.dev/message-history/<..>)
  * [ Installation  ](https://ai.pydantic.dev/message-history/<../install/>)
  * [ Getting Help  ](https://ai.pydantic.dev/message-history/<../help/>)
  * [ Contributing  ](https://ai.pydantic.dev/message-history/<../contributing/>)
  * [ Troubleshooting  ](https://ai.pydantic.dev/message-history/<../troubleshooting/>)
  * Documentation  Documentation 
    * [ Agents  ](https://ai.pydantic.dev/message-history/<../agents/>)
    * [ Models  ](https://ai.pydantic.dev/message-history/<../models/>)
    * [ Dependencies  ](https://ai.pydantic.dev/message-history/<../dependencies/>)
    * [ Function Tools  ](https://ai.pydantic.dev/message-history/<../tools/>)
    * [ Results  ](https://ai.pydantic.dev/message-history/<../results/>)
    * Messages and chat history  [ Messages and chat history  ](https://ai.pydantic.dev/message-history/<./>) Table of contents 
      * [ Accessing Messages from Results  ](https://ai.pydantic.dev/message-history/<#accessing-messages-from-results>)
      * [ Using Messages as Input for Further Agent Runs  ](https://ai.pydantic.dev/message-history/<#using-messages-as-input-for-further-agent-runs>)
      * [ Other ways of using messages  ](https://ai.pydantic.dev/message-history/<#other-ways-of-using-messages>)
      * [ Examples  ](https://ai.pydantic.dev/message-history/<#examples>)
    * [ Testing and Evals  ](https://ai.pydantic.dev/message-history/<../testing-evals/>)
    * [ Debugging and Monitoring  ](https://ai.pydantic.dev/message-history/<../logfire/>)
    * [ Multi-agent Applications  ](https://ai.pydantic.dev/message-history/<../multi-agent-applications/>)
  * [ Examples  ](https://ai.pydantic.dev/message-history/<../examples/>)
Examples 
    * [ Pydantic Model  ](https://ai.pydantic.dev/message-history/<../examples/pydantic-model/>)
    * [ Weather agent  ](https://ai.pydantic.dev/message-history/<../examples/weather-agent/>)
    * [ Bank support  ](https://ai.pydantic.dev/message-history/<../examples/bank-support/>)
    * [ SQL Generation  ](https://ai.pydantic.dev/message-history/<../examples/sql-gen/>)
    * [ Flight booking  ](https://ai.pydantic.dev/message-history/<../examples/flight-booking/>)
    * [ RAG  ](https://ai.pydantic.dev/message-history/<../examples/rag/>)
    * [ Stream markdown  ](https://ai.pydantic.dev/message-history/<../examples/stream-markdown/>)
    * [ Stream whales  ](https://ai.pydantic.dev/message-history/<../examples/stream-whales/>)
    * [ Chat App with FastAPI  ](https://ai.pydantic.dev/message-history/<../examples/chat-app/>)
  * API Reference  API Reference 
    * [ pydantic_ai.agent  ](https://ai.pydantic.dev/message-history/<../api/agent/>)
    * [ pydantic_ai.tools  ](https://ai.pydantic.dev/message-history/<../api/tools/>)
    * [ pydantic_ai.result  ](https://ai.pydantic.dev/message-history/<../api/result/>)
    * [ pydantic_ai.messages  ](https://ai.pydantic.dev/message-history/<../api/messages/>)
    * [ pydantic_ai.exceptions  ](https://ai.pydantic.dev/message-history/<../api/exceptions/>)
    * [ pydantic_ai.settings  ](https://ai.pydantic.dev/message-history/<../api/settings/>)
    * [ pydantic_ai.usage  ](https://ai.pydantic.dev/message-history/<../api/usage/>)
    * [ pydantic_ai.format_as_xml  ](https://ai.pydantic.dev/message-history/<../api/format_as_xml/>)
    * [ pydantic_ai.models  ](https://ai.pydantic.dev/message-history/<../api/models/base/>)
    * [ pydantic_ai.models.openai  ](https://ai.pydantic.dev/message-history/<../api/models/openai/>)
    * [ pydantic_ai.models.anthropic  ](https://ai.pydantic.dev/message-history/<../api/models/anthropic/>)
    * [ pydantic_ai.models.gemini  ](https://ai.pydantic.dev/message-history/<../api/models/gemini/>)
    * [ pydantic_ai.models.vertexai  ](https://ai.pydantic.dev/message-history/<../api/models/vertexai/>)
    * [ pydantic_ai.models.groq  ](https://ai.pydantic.dev/message-history/<../api/models/groq/>)
    * [ pydantic_ai.models.mistral  ](https://ai.pydantic.dev/message-history/<../api/models/mistral/>)
    * [ pydantic_ai.models.ollama  ](https://ai.pydantic.dev/message-history/<../api/models/ollama/>)
    * [ pydantic_ai.models.test  ](https://ai.pydantic.dev/message-history/<../api/models/test/>)
    * [ pydantic_ai.models.function  ](https://ai.pydantic.dev/message-history/<../api/models/function/>)


Table of contents 
  * [ Accessing Messages from Results  ](https://ai.pydantic.dev/message-history/<#accessing-messages-from-results>)
  * [ Using Messages as Input for Further Agent Runs  ](https://ai.pydantic.dev/message-history/<#using-messages-as-input-for-further-agent-runs>)
  * [ Other ways of using messages  ](https://ai.pydantic.dev/message-history/<#other-ways-of-using-messages>)
  * [ Examples  ](https://ai.pydantic.dev/message-history/<#examples>)


  1. [ Introduction  ](https://ai.pydantic.dev/message-history/<..>)
  2. [ Documentation  ](https://ai.pydantic.dev/message-history/<../agents/>)


# Messages and chat history
PydanticAI provides access to messages exchanged during an agent run. These messages can be used both to continue a coherent conversation, and to understand how an agent performed.
### Accessing Messages from Results
After running an agent, you can access the messages exchanged during that run from the `result` object.
Both `RunResult`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.RunResult>) (returned by `Agent.run`[](https://ai.pydantic.dev/message-history/<../api/agent/#pydantic_ai.agent.Agent.run>), `Agent.run_sync`[](https://ai.pydantic.dev/message-history/<../api/agent/#pydantic_ai.agent.Agent.run_sync>)) and `StreamedRunResult`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.StreamedRunResult>) (returned by `Agent.run_stream`[](https://ai.pydantic.dev/message-history/<../api/agent/#pydantic_ai.agent.Agent.run_stream>)) have the following methods:
  * `all_messages()`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.RunResult.all_messages>): returns all messages, including messages from prior runs. There's also a variant that returns JSON bytes, `all_messages_json()`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.RunResult.all_messages_json>).
  * `new_messages()`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.RunResult.new_messages>): returns only the messages from the current run. There's also a variant that returns JSON bytes, `new_messages_json()`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.RunResult.new_messages_json>).


StreamedRunResult and complete messages
On `StreamedRunResult`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.StreamedRunResult>), the messages returned from these methods will only include the final result message once the stream has finished.
E.g. you've awaited one of the following coroutines:
  * `StreamedRunResult.stream()`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.StreamedRunResult.stream>)
  * `StreamedRunResult.stream_text()`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.StreamedRunResult.stream_text>)
  * `StreamedRunResult.stream_structured()`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.StreamedRunResult.stream_structured>)
  * `StreamedRunResult.get_data()`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.StreamedRunResult.get_data>)


**Note:** The final result message will NOT be added to result messages if you use `.stream_text(delta=True)`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.StreamedRunResult.stream_text>) since in this case the result content is never built as one string.
Example of accessing methods on a `RunResult`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.RunResult>) :
run_result_messages.py```
from pydantic_ai import Agent
agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')
result = agent.run_sync('Tell me a joke.')
print(result.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.
# all messages from the run
print(result.all_messages())
"""
[
  ModelRequest(
    parts=[
      SystemPromptPart(
        content='Be a helpful assistant.',
        dynamic_ref=None,
        part_kind='system-prompt',
      ),
      UserPromptPart(
        content='Tell me a joke.',
        timestamp=datetime.datetime(...),
        part_kind='user-prompt',
      ),
    ],
    kind='request',
  ),
  ModelResponse(
    parts=[
      TextPart(
        content='Did you hear about the toothpaste scandal? They called it Colgate.',
        part_kind='text',
      )
    ],
    timestamp=datetime.datetime(...),
    kind='response',
  ),
]
"""

```

_(This example is complete, it can be run "as is")_
Example of accessing methods on a `StreamedRunResult`[](https://ai.pydantic.dev/message-history/<../api/result/#pydantic_ai.result.StreamedRunResult>) :
streamed_run_result_messages.py```
from pydantic_ai import Agent
agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

async def main():
  async with agent.run_stream('Tell me a joke.') as result:
    # incomplete messages before the stream finishes
    print(result.all_messages())
"""
    [
      ModelRequest(
        parts=[
          SystemPromptPart(
            content='Be a helpful assistant.',
            dynamic_ref=None,
            part_kind='system-prompt',
          ),
          UserPromptPart(
            content='Tell me a joke.',
            timestamp=datetime.datetime(...),
            part_kind='user-prompt',
          ),
        ],
        kind='request',
      )
    ]
    """
    async for text in result.stream():
      print(text)
      #> Did you hear
      #> Did you hear about the toothpaste
      #> Did you hear about the toothpaste scandal? They called
      #> Did you hear about the toothpaste scandal? They called it Colgate.
    # complete messages once the stream finishes
    print(result.all_messages())
"""
    [
      ModelRequest(
        parts=[
          SystemPromptPart(
            content='Be a helpful assistant.',
            dynamic_ref=None,
            part_kind='system-prompt',
          ),
          UserPromptPart(
            content='Tell me a joke.',
            timestamp=datetime.datetime(...),
            part_kind='user-prompt',
          ),
        ],
        kind='request',
      ),
      ModelResponse(
        parts=[
          TextPart(
            content='Did you hear about the toothpaste scandal? They called it Colgate.',
            part_kind='text',
          )
        ],
        timestamp=datetime.datetime(...),
        kind='response',
      ),
    ]
    """

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
### Using Messages as Input for Further Agent Runs
The primary use of message histories in PydanticAI is to maintain context across multiple agent runs.
To use existing messages in a run, pass them to the `message_history` parameter of `Agent.run`[](https://ai.pydantic.dev/message-history/<../api/agent/#pydantic_ai.agent.Agent.run>), `Agent.run_sync`[](https://ai.pydantic.dev/message-history/<../api/agent/#pydantic_ai.agent.Agent.run_sync>) or `Agent.run_stream`[](https://ai.pydantic.dev/message-history/<../api/agent/#pydantic_ai.agent.Agent.run_stream>).
If `message_history` is set and not empty, a new system prompt is not generated — we assume the existing message history includes a system prompt.
Reusing messages in a conversation```
from pydantic_ai import Agent
agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')
result1 = agent.run_sync('Tell me a joke.')
print(result1.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.
result2 = agent.run_sync('Explain?', message_history=result1.new_messages())
print(result2.data)
#> This is an excellent joke invent by Samuel Colvin, it needs no explanation.
print(result2.all_messages())
"""
[
  ModelRequest(
    parts=[
      SystemPromptPart(
        content='Be a helpful assistant.',
        dynamic_ref=None,
        part_kind='system-prompt',
      ),
      UserPromptPart(
        content='Tell me a joke.',
        timestamp=datetime.datetime(...),
        part_kind='user-prompt',
      ),
    ],
    kind='request',
  ),
  ModelResponse(
    parts=[
      TextPart(
        content='Did you hear about the toothpaste scandal? They called it Colgate.',
        part_kind='text',
      )
    ],
    timestamp=datetime.datetime(...),
    kind='response',
  ),
  ModelRequest(
    parts=[
      UserPromptPart(
        content='Explain?',
        timestamp=datetime.datetime(...),
        part_kind='user-prompt',
      )
    ],
    kind='request',
  ),
  ModelResponse(
    parts=[
      TextPart(
        content='This is an excellent joke invent by Samuel Colvin, it needs no explanation.',
        part_kind='text',
      )
    ],
    timestamp=datetime.datetime(...),
    kind='response',
  ),
]
"""

```

_(This example is complete, it can be run "as is")_
## Other ways of using messages
Since messages are defined by simple dataclasses, you can manually create and manipulate, e.g. for testing.
The message format is independent of the model used, so you can use messages in different agents, or the same agent with different models.
```
from pydantic_ai import Agent
agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')
result1 = agent.run_sync('Tell me a joke.')
print(result1.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.
result2 = agent.run_sync(
  'Explain?', model='gemini-1.5-pro', message_history=result1.new_messages()
)
print(result2.data)
#> This is an excellent joke invent by Samuel Colvin, it needs no explanation.
print(result2.all_messages())
"""
[
  ModelRequest(
    parts=[
      SystemPromptPart(
        content='Be a helpful assistant.',
        dynamic_ref=None,
        part_kind='system-prompt',
      ),
      UserPromptPart(
        content='Tell me a joke.',
        timestamp=datetime.datetime(...),
        part_kind='user-prompt',
      ),
    ],
    kind='request',
  ),
  ModelResponse(
    parts=[
      TextPart(
        content='Did you hear about the toothpaste scandal? They called it Colgate.',
        part_kind='text',
      )
    ],
    timestamp=datetime.datetime(...),
    kind='response',
  ),
  ModelRequest(
    parts=[
      UserPromptPart(
        content='Explain?',
        timestamp=datetime.datetime(...),
        part_kind='user-prompt',
      )
    ],
    kind='request',
  ),
  ModelResponse(
    parts=[
      TextPart(
        content='This is an excellent joke invent by Samuel Colvin, it needs no explanation.',
        part_kind='text',
      )
    ],
    timestamp=datetime.datetime(...),
    kind='response',
  ),
]
"""

```

## Examples
For a more complete example of using messages in conversations, see the [chat app](https://ai.pydantic.dev/message-history/<../examples/chat-app/>) example.
© Pydantic Services Inc. 2024 to present 
