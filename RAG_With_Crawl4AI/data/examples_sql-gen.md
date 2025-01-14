[ Skip to content ](https://ai.pydantic.dev/examples/sql-gen/<#sql-generation>)
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/sql-gen/<../..> "PydanticAI")
PydanticAI 
SQL Generation 
Initializing search 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/sql-gen/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
[ ![logo](https://ai.pydantic.dev/img/logo-white.svg) ](https://ai.pydantic.dev/examples/sql-gen/<../..> "PydanticAI") PydanticAI 
[ pydantic/pydantic-ai  ](https://ai.pydantic.dev/examples/sql-gen/<https:/github.com/pydantic/pydantic-ai> "Go to repository")
  * [ Introduction  ](https://ai.pydantic.dev/examples/sql-gen/<../..>)
  * [ Installation  ](https://ai.pydantic.dev/examples/sql-gen/install/>)
  * [ Getting Help  ](https://ai.pydantic.dev/examples/sql-gen/help/>)
  * [ Contributing  ](https://ai.pydantic.dev/examples/sql-gen/contributing/>)
  * [ Troubleshooting  ](https://ai.pydantic.dev/examples/sql-gen/troubleshooting/>)
  * Documentation  Documentation 
    * [ Agents  ](https://ai.pydantic.dev/examples/sql-gen/agents/>)
    * [ Models  ](https://ai.pydantic.dev/examples/sql-gen/models/>)
    * [ Dependencies  ](https://ai.pydantic.dev/examples/sql-gen/dependencies/>)
    * [ Function Tools  ](https://ai.pydantic.dev/examples/sql-gen/tools/>)
    * [ Results  ](https://ai.pydantic.dev/examples/sql-gen/results/>)
    * [ Messages and chat history  ](https://ai.pydantic.dev/examples/sql-gen/message-history/>)
    * [ Testing and Evals  ](https://ai.pydantic.dev/examples/sql-gen/testing-evals/>)
    * [ Debugging and Monitoring  ](https://ai.pydantic.dev/examples/sql-gen/logfire/>)
    * [ Multi-agent Applications  ](https://ai.pydantic.dev/examples/sql-gen/multi-agent-applications/>)
  * [ Examples  ](https://ai.pydantic.dev/examples/sql-gen/<../>)
Examples 
    * [ Pydantic Model  ](https://ai.pydantic.dev/examples/sql-gen/<../pydantic-model/>)
    * [ Weather agent  ](https://ai.pydantic.dev/examples/sql-gen/<../weather-agent/>)
    * [ Bank support  ](https://ai.pydantic.dev/examples/sql-gen/<../bank-support/>)
    * SQL Generation  [ SQL Generation  ](https://ai.pydantic.dev/examples/sql-gen/<./>) Table of contents 
      * [ Running the Example  ](https://ai.pydantic.dev/examples/sql-gen/<#running-the-example>)
      * [ Example Code  ](https://ai.pydantic.dev/examples/sql-gen/<#example-code>)
    * [ Flight booking  ](https://ai.pydantic.dev/examples/sql-gen/<../flight-booking/>)
    * [ RAG  ](https://ai.pydantic.dev/examples/sql-gen/<../rag/>)
    * [ Stream markdown  ](https://ai.pydantic.dev/examples/sql-gen/<../stream-markdown/>)
    * [ Stream whales  ](https://ai.pydantic.dev/examples/sql-gen/<../stream-whales/>)
    * [ Chat App with FastAPI  ](https://ai.pydantic.dev/examples/sql-gen/<../chat-app/>)
  * API Reference  API Reference 
    * [ pydantic_ai.agent  ](https://ai.pydantic.dev/examples/sql-gen/api/agent/>)
    * [ pydantic_ai.tools  ](https://ai.pydantic.dev/examples/sql-gen/api/tools/>)
    * [ pydantic_ai.result  ](https://ai.pydantic.dev/examples/sql-gen/api/result/>)
    * [ pydantic_ai.messages  ](https://ai.pydantic.dev/examples/sql-gen/api/messages/>)
    * [ pydantic_ai.exceptions  ](https://ai.pydantic.dev/examples/sql-gen/api/exceptions/>)
    * [ pydantic_ai.settings  ](https://ai.pydantic.dev/examples/sql-gen/api/settings/>)
    * [ pydantic_ai.usage  ](https://ai.pydantic.dev/examples/sql-gen/api/usage/>)
    * [ pydantic_ai.format_as_xml  ](https://ai.pydantic.dev/examples/sql-gen/api/format_as_xml/>)
    * [ pydantic_ai.models  ](https://ai.pydantic.dev/examples/sql-gen/api/models/base/>)
    * [ pydantic_ai.models.openai  ](https://ai.pydantic.dev/examples/sql-gen/api/models/openai/>)
    * [ pydantic_ai.models.anthropic  ](https://ai.pydantic.dev/examples/sql-gen/api/models/anthropic/>)
    * [ pydantic_ai.models.gemini  ](https://ai.pydantic.dev/examples/sql-gen/api/models/gemini/>)
    * [ pydantic_ai.models.vertexai  ](https://ai.pydantic.dev/examples/sql-gen/api/models/vertexai/>)
    * [ pydantic_ai.models.groq  ](https://ai.pydantic.dev/examples/sql-gen/api/models/groq/>)
    * [ pydantic_ai.models.mistral  ](https://ai.pydantic.dev/examples/sql-gen/api/models/mistral/>)
    * [ pydantic_ai.models.ollama  ](https://ai.pydantic.dev/examples/sql-gen/api/models/ollama/>)
    * [ pydantic_ai.models.test  ](https://ai.pydantic.dev/examples/sql-gen/api/models/test/>)
    * [ pydantic_ai.models.function  ](https://ai.pydantic.dev/examples/sql-gen/api/models/function/>)


Table of contents 
  * [ Running the Example  ](https://ai.pydantic.dev/examples/sql-gen/<#running-the-example>)
  * [ Example Code  ](https://ai.pydantic.dev/examples/sql-gen/<#example-code>)


  1. [ Introduction  ](https://ai.pydantic.dev/examples/sql-gen/<../..>)
  2. [ Examples  ](https://ai.pydantic.dev/examples/sql-gen/<../>)


# SQL Generation
Example demonstrating how to use PydanticAI to generate SQL queries based on user input.
Demonstrates:
  * [dynamic system prompt](https://ai.pydantic.dev/examples/sql-gen/agents/#system-prompts>)
  * [structured `result_type`](https://ai.pydantic.dev/examples/sql-gen/results/#structured-result-validation>)
  * [result validation](https://ai.pydantic.dev/examples/sql-gen/results/#result-validators-functions>)
  * [agent dependencies](https://ai.pydantic.dev/examples/sql-gen/dependencies/>)


## Running the Example
The resulting SQL is validated by running it as an `EXPLAIN` query on PostgreSQL. To run the example, you first need to run PostgreSQL, e.g. via Docker:
```
dockerrun--rm-ePOSTGRES_PASSWORD=postgres-p54320:5432postgres

```

_(we run postgres on port`54320` to avoid conflicts with any other postgres instances you may have running)_
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/sql-gen/<../#usage>), run:
[pip](https://ai.pydantic.dev/examples/sql-gen/<#__tabbed_1_1>)[uv](https://ai.pydantic.dev/examples/sql-gen/<#__tabbed_1_2>)
```
python-mpydantic_ai_examples.sql_gen

```

```
uvrun-mpydantic_ai_examples.sql_gen

```

or to use a custom prompt:
[pip](https://ai.pydantic.dev/examples/sql-gen/<#__tabbed_2_1>)[uv](https://ai.pydantic.dev/examples/sql-gen/<#__tabbed_2_2>)
```
python-mpydantic_ai_examples.sql_gen"find me errors"

```

```
uvrun-mpydantic_ai_examples.sql_gen"find me errors"

```

This model uses `gemini-1.5-flash` by default since Gemini is good at single shot queries of this kind.
## Example Code
sql_gen.py```
import asyncio
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Union
import asyncpg
import logfire
from annotated_types import MinLen
from devtools import debug
from pydantic import BaseModel, Field
from typing_extensions import TypeAlias
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.format_as_xml import format_as_xml
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()
DB_SCHEMA = """
CREATE TABLE records (
  created_at timestamptz,
  start_timestamp timestamptz,
  end_timestamp timestamptz,
  trace_id text,
  span_id text,
  parent_span_id text,
  level log_level,
  span_name text,
  message text,
  attributes_json_schema text,
  attributes jsonb,
  tags text[],
  is_exception boolean,
  otel_status_message text,
  service_name text
);
"""
SQL_EXAMPLES = [
  {
    'request': 'show me records where foobar is false',
    'response': "SELECT * FROM records WHERE attributes->>'foobar' = false",
  },
  {
    'request': 'show me records where attributes include the key "foobar"',
    'response': "SELECT * FROM records WHERE attributes ? 'foobar'",
  },
  {
    'request': 'show me records from yesterday',
    'response': "SELECT * FROM records WHERE start_timestamp::date > CURRENT_TIMESTAMP - INTERVAL '1 day'",
  },
  {
    'request': 'show me error records with the tag "foobar"',
    'response': "SELECT * FROM records WHERE level = 'error' and 'foobar' = ANY(tags)",
  },
]

@dataclass
class Deps:
  conn: asyncpg.Connection

class Success(BaseModel):
"""Response when SQL could be successfully generated."""
  sql_query: Annotated[str, MinLen(1)]
  explanation: str = Field(
    '', description='Explanation of the SQL query, as markdown'
  )

class InvalidRequest(BaseModel):
"""Response the user input didn't include enough information to generate SQL."""
  error_message: str

Response: TypeAlias = Union[Success, InvalidRequest]
agent: Agent[Deps, Response] = Agent(
  'google-gla:gemini-1.5-flash',
  # Type ignore while we wait for PEP-0747, nonetheless unions will work fine everywhere else
  result_type=Response, # type: ignore
  deps_type=Deps,
)

@agent.system_prompt
async def system_prompt() -> str:
  return f"""\
Given the following PostgreSQL table of records, your job is to
write a SQL query that suits the user's request.
Database schema:
{DB_SCHEMA}
today's date = {date.today()}
{format_as_xml(SQL_EXAMPLES)}
"""

@agent.result_validator
async def validate_result(ctx: RunContext[Deps], result: Response) -> Response:
  if isinstance(result, InvalidRequest):
    return result
  # gemini often adds extraneous backslashes to SQL
  result.sql_query = result.sql_query.replace('\\', '')
  if not result.sql_query.upper().startswith('SELECT'):
    raise ModelRetry('Please create a SELECT query')
  try:
    await ctx.deps.conn.execute(f'EXPLAIN {result.sql_query}')
  except asyncpg.exceptions.PostgresError as e:
    raise ModelRetry(f'Invalid query: {e}') from e
  else:
    return result

async def main():
  if len(sys.argv) == 1:
    prompt = 'show me logs from yesterday, with level "error"'
  else:
    prompt = sys.argv[1]
  async with database_connect(
    'postgresql://postgres:postgres@localhost:54320', 'pydantic_ai_sql_gen'
  ) as conn:
    deps = Deps(conn)
    result = await agent.run(prompt, deps=deps)
  debug(result.data)

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(server_dsn: str, database: str) -> AsyncGenerator[Any, None]:
  with logfire.span('check and create DB'):
    conn = await asyncpg.connect(server_dsn)
    try:
      db_exists = await conn.fetchval(
        'SELECT 1 FROM pg_database WHERE datname = $1', database
      )
      if not db_exists:
        await conn.execute(f'CREATE DATABASE {database}')
    finally:
      await conn.close()
  conn = await asyncpg.connect(f'{server_dsn}/{database}')
  try:
    with logfire.span('create schema'):
      async with conn.transaction():
        if not db_exists:
          await conn.execute(
            "CREATE TYPE log_level AS ENUM ('debug', 'info', 'warning', 'error', 'critical')"
          )
        await conn.execute(DB_SCHEMA)
    yield conn
  finally:
    await conn.close()

if __name__ == '__main__':
  asyncio.run(main())

```

© Pydantic Services Inc. 2024 to present 
