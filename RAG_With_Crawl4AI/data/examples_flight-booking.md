[ Skip to content ](https://ai.pydantic.dev/examples/flight-booking/<#running-the-example>)
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/flight-booking/<../..> "PydanticAI")
PydanticAI 
Flight booking 
Initializing search 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/flight-booking/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/flight-booking/<../..> "PydanticAI") PydanticAI 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/flight-booking/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
  * [ Introduction  ](https://ai.pydantic.dev/examples/flight-booking/<../..>)
  * [ Installation  ](https://ai.pydantic.dev/examples/flight-booking/install/>)
  * [ Getting Help  ](https://ai.pydantic.dev/examples/flight-booking/help/>)
  * [ Contributing  ](https://ai.pydantic.dev/examples/flight-booking/contributing/>)
  * [ Troubleshooting  ](https://ai.pydantic.dev/examples/flight-booking/troubleshooting/>)
  * Documentation  Documentation 
    * [ Agents  ](https://ai.pydantic.dev/examples/flight-booking/agents/>)
    * [ Models  ](https://ai.pydantic.dev/examples/flight-booking/models/>)
    * [ Dependencies  ](https://ai.pydantic.dev/examples/flight-booking/dependencies/>)
    * [ Function Tools  ](https://ai.pydantic.dev/examples/flight-booking/tools/>)
    * [ Results  ](https://ai.pydantic.dev/examples/flight-booking/results/>)
    * [ Messages and chat history  ](https://ai.pydantic.dev/examples/flight-booking/message-history/>)
    * [ Testing and Evals  ](https://ai.pydantic.dev/examples/flight-booking/testing-evals/>)
    * [ Debugging and Monitoring  ](https://ai.pydantic.dev/examples/flight-booking/logfire/>)
    * [ Multi-agent Applications  ](https://ai.pydantic.dev/examples/flight-booking/multi-agent-applications/>)
  * [ Examples  ](https://ai.pydantic.dev/examples/flight-booking/<../>)
Examples 
    * [ Pydantic Model  ](https://ai.pydantic.dev/examples/flight-booking/<../pydantic-model/>)
    * [ Weather agent  ](https://ai.pydantic.dev/examples/flight-booking/<../weather-agent/>)
    * [ Bank support  ](https://ai.pydantic.dev/examples/flight-booking/<../bank-support/>)
    * [ SQL Generation  ](https://ai.pydantic.dev/examples/flight-booking/<../sql-gen/>)
    * Flight booking  [ Flight booking  ](https://ai.pydantic.dev/examples/flight-booking/<./>) Table of contents 
      * [ Running the Example  ](https://ai.pydantic.dev/examples/flight-booking/<#running-the-example>)
      * [ Example Code  ](https://ai.pydantic.dev/examples/flight-booking/<#example-code>)
    * [ RAG  ](https://ai.pydantic.dev/examples/flight-booking/<../rag/>)
    * [ Stream markdown  ](https://ai.pydantic.dev/examples/flight-booking/<../stream-markdown/>)
    * [ Stream whales  ](https://ai.pydantic.dev/examples/flight-booking/<../stream-whales/>)
    * [ Chat App with FastAPI  ](https://ai.pydantic.dev/examples/flight-booking/<../chat-app/>)
  * API Reference  API Reference 
    * [ pydantic_ai.agent  ](https://ai.pydantic.dev/examples/flight-booking/api/agent/>)
    * [ pydantic_ai.tools  ](https://ai.pydantic.dev/examples/flight-booking/api/tools/>)
    * [ pydantic_ai.result  ](https://ai.pydantic.dev/examples/flight-booking/api/result/>)
    * [ pydantic_ai.messages  ](https://ai.pydantic.dev/examples/flight-booking/api/messages/>)
    * [ pydantic_ai.exceptions  ](https://ai.pydantic.dev/examples/flight-booking/api/exceptions/>)
    * [ pydantic_ai.settings  ](https://ai.pydantic.dev/examples/flight-booking/api/settings/>)
    * [ pydantic_ai.usage  ](https://ai.pydantic.dev/examples/flight-booking/api/usage/>)
    * [ pydantic_ai.format_as_xml  ](https://ai.pydantic.dev/examples/flight-booking/api/format_as_xml/>)
    * [ pydantic_ai.models  ](https://ai.pydantic.dev/examples/flight-booking/api/models/base/>)
    * [ pydantic_ai.models.openai  ](https://ai.pydantic.dev/examples/flight-booking/api/models/openai/>)
    * [ pydantic_ai.models.anthropic  ](https://ai.pydantic.dev/examples/flight-booking/api/models/anthropic/>)
    * [ pydantic_ai.models.gemini  ](https://ai.pydantic.dev/examples/flight-booking/api/models/gemini/>)
    * [ pydantic_ai.models.vertexai  ](https://ai.pydantic.dev/examples/flight-booking/api/models/vertexai/>)
    * [ pydantic_ai.models.groq  ](https://ai.pydantic.dev/examples/flight-booking/api/models/groq/>)
    * [ pydantic_ai.models.mistral  ](https://ai.pydantic.dev/examples/flight-booking/api/models/mistral/>)
    * [ pydantic_ai.models.ollama  ](https://ai.pydantic.dev/examples/flight-booking/api/models/ollama/>)
    * [ pydantic_ai.models.test  ](https://ai.pydantic.dev/examples/flight-booking/api/models/test/>)
    * [ pydantic_ai.models.function  ](https://ai.pydantic.dev/examples/flight-booking/api/models/function/>)


Table of contents 
  * [ Running the Example  ](https://ai.pydantic.dev/examples/flight-booking/<#running-the-example>)
  * [ Example Code  ](https://ai.pydantic.dev/examples/flight-booking/<#example-code>)


  1. [ Introduction  ](https://ai.pydantic.dev/examples/flight-booking/<../..>)
  2. [ Examples  ](https://ai.pydantic.dev/examples/flight-booking/<../>)


# Flight booking
Example of a multi-agent flow where one agent delegates work to another, then hands off control to a third agent.
Demonstrates:
  * [agent delegation](https://ai.pydantic.dev/examples/flight-booking/multi-agent-applications/#agent-delegation>)
  * [programmatic agent hand-off](https://ai.pydantic.dev/examples/flight-booking/multi-agent-applications/#programmatic-agent-hand-off>)
  * [usage limits](https://ai.pydantic.dev/examples/flight-booking/agents/#usage-limits>)


In this scenario, a group of agents work together to find the best flight for a user.
The control flow for this example can be summarised as follows:
```
graph TD
 START --> search_agent("search agent")
 search_agent --> extraction_agent("extraction agent")
 extraction_agent --> search_agent
 search_agent --> human_confirm("human confirm")
 human_confirm --> search_agent
 search_agent --> FAILED
 human_confirm --> find_seat_function("find seat function")
 find_seat_function --> human_seat_choice("human seat choice")
 human_seat_choice --> find_seat_agent("find seat agent")
 find_seat_agent --> find_seat_function
 find_seat_function --> buy_flights("buy flights")
 buy_flights --> SUCCESS
```

## Running the Example
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/flight-booking/<../#usage>), run:
[pip](https://ai.pydantic.dev/examples/flight-booking/<#__tabbed_1_1>)[uv](https://ai.pydantic.dev/examples/flight-booking/<#__tabbed_1_2>)
```
python-mpydantic_ai_examples.flight_booking

```

```
uvrun-mpydantic_ai_examples.flight_booking

```

## Example Code
flight_booking.py```
import datetime
from dataclasses import dataclass
from typing import Literal
import logfire
from pydantic import BaseModel, Field
from rich.prompt import Prompt
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage, UsageLimits
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

class FlightDetails(BaseModel):
"""Details of the most suitable flight."""
  flight_number: str
  price: int
  origin: str = Field(description='Three-letter airport code')
  destination: str = Field(description='Three-letter airport code')
  date: datetime.date

class NoFlightFound(BaseModel):
"""When no valid flight is found."""

@dataclass
class Deps:
  web_page_text: str
  req_origin: str
  req_destination: str
  req_date: datetime.date

# This agent is responsible for controlling the flow of the conversation.
search_agent = Agent[Deps, FlightDetails | NoFlightFound](
  'openai:gpt-4o',
  result_type=FlightDetails | NoFlightFound, # type: ignore
  retries=4,
  system_prompt=(
    'Your job is to find the cheapest flight for the user on the given date. '
  ),
)

# This agent is responsible for extracting flight details from web page text.
extraction_agent = Agent(
  'openai:gpt-4o',
  result_type=list[FlightDetails],
  system_prompt='Extract all the flight details from the given text.',
)

@search_agent.tool
async def extract_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
"""Get details of all flights."""
  # we pass the usage to the search agent so requests within this agent are counted
  result = await extraction_agent.run(ctx.deps.web_page_text, usage=ctx.usage)
  logfire.info('found {flight_count} flights', flight_count=len(result.data))
  return result.data

@search_agent.result_validator
async def validate_result(
  ctx: RunContext[Deps], result: FlightDetails | NoFlightFound
) -> FlightDetails | NoFlightFound:
"""Procedural validation that the flight meets the constraints."""
  if isinstance(result, NoFlightFound):
    return result
  errors: list[str] = []
  if result.origin != ctx.deps.req_origin:
    errors.append(
      f'Flight should have origin {ctx.deps.req_origin}, not {result.origin}'
    )
  if result.destination != ctx.deps.req_destination:
    errors.append(
      f'Flight should have destination {ctx.deps.req_destination}, not {result.destination}'
    )
  if result.date != ctx.deps.req_date:
    errors.append(f'Flight should be on {ctx.deps.req_date}, not {result.date}')
  if errors:
    raise ModelRetry('\n'.join(errors))
  else:
    return result

class SeatPreference(BaseModel):
  row: int = Field(ge=1, le=30)
  seat: Literal['A', 'B', 'C', 'D', 'E', 'F']

class Failed(BaseModel):
"""Unable to extract a seat selection."""

# This agent is responsible for extracting the user's seat selection
seat_preference_agent = Agent[
  None, SeatPreference | Failed
](
  'openai:gpt-4o',
  result_type=SeatPreference | Failed, # type: ignore
  system_prompt=(
    "Extract the user's seat preference. "
    'Seats A and F are window seats. '
    'Row 1 is the front row and has extra leg room. '
    'Rows 14, and 20 also have extra leg room. '
  ),
)

# in reality this would be downloaded from a booking site,
# potentially using another agent to navigate the site
flights_web_page = """
1. Flight SFO-AK123
- Price: $350
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025
2. Flight SFO-AK456
- Price: $370
- Origin: San Francisco International Airport (SFO)
- Destination: Fairbanks International Airport (FAI)
- Date: January 10, 2025
3. Flight SFO-AK789
- Price: $400
- Origin: San Francisco International Airport (SFO)
- Destination: Juneau International Airport (JNU)
- Date: January 20, 2025
4. Flight NYC-LA101
- Price: $250
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025
5. Flight CHI-MIA202
- Price: $200
- Origin: Chicago O'Hare International Airport (ORD)
- Destination: Miami International Airport (MIA)
- Date: January 12, 2025
6. Flight BOS-SEA303
- Price: $120
- Origin: Boston Logan International Airport (BOS)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 12, 2025
7. Flight DFW-DEN404
- Price: $150
- Origin: Dallas/Fort Worth International Airport (DFW)
- Destination: Denver International Airport (DEN)
- Date: January 10, 2025
8. Flight ATL-HOU505
- Price: $180
- Origin: Hartsfield-Jackson Atlanta International Airport (ATL)
- Destination: George Bush Intercontinental Airport (IAH)
- Date: January 10, 2025
"""
# restrict how many requests this app can make to the LLM
usage_limits = UsageLimits(request_limit=15)

async def main():
  deps = Deps(
    web_page_text=flights_web_page,
    req_origin='SFO',
    req_destination='ANC',
    req_date=datetime.date(2025, 1, 10),
  )
  message_history: list[ModelMessage] | None = None
  usage: Usage = Usage()
  # run the agent until a satisfactory flight is found
  while True:
    result = await search_agent.run(
      f'Find me a flight from {deps.req_origin} to {deps.req_destination} on {deps.req_date}',
      deps=deps,
      usage=usage,
      message_history=message_history,
      usage_limits=usage_limits,
    )
    if isinstance(result.data, NoFlightFound):
      print('No flight found')
      break
    else:
      flight = result.data
      print(f'Flight found: {flight}')
      answer = Prompt.ask(
        'Do you want to buy this flight, or keep searching? (buy/*search)',
        choices=['buy', 'search', ''],
        show_choices=False,
      )
      if answer == 'buy':
        seat = await find_seat(usage)
        await buy_tickets(flight, seat)
        break
      else:
        message_history = result.all_messages(
          result_tool_return_content='Please suggest another flight'
        )

async def find_seat(usage: Usage) -> SeatPreference:
  message_history: list[ModelMessage] | None = None
  while True:
    answer = Prompt.ask('What seat would you like?')
    result = await seat_preference_agent.run(
      answer,
      message_history=message_history,
      usage=usage,
      usage_limits=usage_limits,
    )
    if isinstance(result.data, SeatPreference):
      return result.data
    else:
      print('Could not understand seat preference. Please try again.')
      message_history = result.all_messages()

async def buy_tickets(flight_details: FlightDetails, seat: SeatPreference):
  print(f'Purchasing flight {flight_details=!r}{seat=!r}...')

if __name__ == '__main__':
  import asyncio
  asyncio.run(main())

```

© Pydantic Services Inc. 2024 to present 
