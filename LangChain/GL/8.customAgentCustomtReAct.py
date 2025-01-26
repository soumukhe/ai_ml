import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

from langchain_experimental.utilities.python import PythonREPL
from langchain.tools import Tool
from langchain import hub
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import yahoo_finance_news
import json
load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

"""
Summary of the code:

1. Set up the environment and load the API keys.
2. Define the LLM and tool DuckDuckGoSearchRun.
3. Create a custom ReAct prompt template.
4. Define the SearchAgent class with methods to run the agent and parse the output.
5. Create an instance of the SearchAgent and test it with two different questions.

"""


llm = ChatOpenAI(
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    model="deepseek-chat",
    #model="deepseek-reasoner",
    temperature=0,
    max_tokens=1000,
    max_completion_tokens=1000,
    max_retries=3,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
    seed=42,
    logprobs=False,
    
)

# 1. Tool: Now create duckduckgo search tool
duckduckgo_search_tool = Tool(
    name = "duckduckgo_search",
    description = "An interface to the DuckDuckGo search engine. Input should be a search query.",
    func = DuckDuckGoSearchRun().run
)

# 2. Let us now craft the ReAct prompt designed following langchain guidelines.
# custom ReAct prompt

react_prompt_template = """
Answer the following questions as best you can. You have access to the following tool:

search_tool: An interface to the DuckDuckGo search engine. Input should be a string.

The way you use the tool is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
search_tool: A function to search the DuckDuckGo search engine

The $JSON_BLOB should only contain a SINGLE action and MUST be formatted as markdown, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```
Make sure to have the $INPUT in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:
```json
$JSON_BLOB
```
Question: {user_question}


"""


# Note:
# The default react prompt is:

# # create a ReAct prompt - Response/Action/Thought
# ## built into langchain.  Make sure to get free api from langchain hub.
# react_prompt = hub.pull("hwchase17/react")


# print("ReAct Prompt: ")

# Output:
# -------
# ReAct Prompt: 
# Answer the following questions as best you can. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!

# Question: {input}
# Thought:{agent_scratchpad}

#### -------------

# below is a normal output from the llm to a user question

user_question = "What is the latest stock price of AAPL?"

response = llm.invoke(react_prompt_template.format(user_question=user_question))

print(f"\nResponse from llm to user question: {user_question} \n"
      f"is to format the user question to a Thought and a json blob:\n")
output = response.content
print(output)

print("--"*10 + "\n")

# As the  output indicates, 
# the tool choice and the input to the tool are delimited by ```json and ```. 
# This allows us to:
#  write an output parser to extract the action parameters (i.e., the tool and its inputs).


# 3. Now get the content ibetween the ```json and ```
start_index = output.find('```json')
end_index = output.rfind('```')

# extract the json blob
print("Extracted json blob:")
print(output[start_index+7:end_index]) # 7 comes from the length of the string "```json"

print("--"*10 + "\n")

# 4. We can now assemble our search agent with this tool.

class SearchAgent:
  """
  This agent will use the search tool to answer questions.
  """
  def __init__(self, llm, tool, prompt_template, verbose=True):
    self.llm = llm
    self.tool = tool
    self.tool_prompt_template = prompt_template
    self.verbose = verbose
    

  def run(self, question):
    # Format messages according to DeepSeek's API
    messages = [{"role": "user", "content": self.tool_prompt_template.format(user_question=question)}]
    response = self.llm.invoke(messages)
    output = response.content.strip()

    if self.verbose:
      print(output)

    # Parse Output
    start_index = output.find('```json')
    end_index = output.rfind('```')

    if start_index == -1 or end_index == -1:
        return "Could not parse tool choice from response"

    try:
        tool_choice = json.loads(output[start_index+7:end_index])
        tool_input = tool_choice["action_input"]

        # Execute search using the tool
        try:
            search_result = self.tool.invoke(tool_input)
        except Exception as e:
            search_result = "Cannot find relevant information"

        # Answer the question using llm given the search result
        answering_messages = [{"role": "user", "content": f"""Given the context:\n {search_result}, answer the following question based on this context:\n {question}. Do not include information about the context."""}]
        final_response = self.llm.invoke(answering_messages)
        return final_response.content

    except json.JSONDecodeError:
        return "Could not parse JSON response from model"
    except KeyError:
        return "Invalid tool choice format"

# Now we can create an instance of the agent
question = "What is the latest stock price of AAPL?"
search_agent = SearchAgent(llm, duckduckgo_search_tool, react_prompt_template)


# print the Final Answer
print(f"\nFinal Answer with verbose: ")
print(search_agent.run(question))

print("--"*10 + "\n")

# Now let us test the agent with a different question and without verbose
question = "What is the latest stock price of GOOGL?"

search_agent_no_verbose = SearchAgent(llm, duckduckgo_search_tool, react_prompt_template, verbose=False)


print(f"\nFinal Answer without verbose: ")
print(search_agent_no_verbose.run(question))

print("--"*10 + "\n")

"""
Output:
--------
Response from llm to user question: What is the latest stock price of AAPL? 
is to format the user question to a Thought and a json blob:

Thought: To find the latest stock price of AAPL, I will use the search tool to query the current stock price.

Action:
```json
{
  "action": "search_tool",
  "action_input": "latest stock price of AAPL"
}
```
--------------------

Extracted json blob:

{
  "action": "search_tool",
  "action_input": "latest stock price of AAPL"
}

--------------------


Final Answer with verbose: 
Thought: To find the latest stock price of AAPL, I will use the search tool to look up the current stock price.

Action:
```json
{
  "action": "search_tool",
  "action_input": "latest stock price of AAPL"
}
```
The latest stock price of AAPL is $222.78.
--------------------


Final Answer without verbose: 
The latest stock price of GOOGL is $200.21.
--------------------

"""