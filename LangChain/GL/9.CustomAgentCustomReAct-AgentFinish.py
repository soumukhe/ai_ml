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

Note:
- in this code, we are using the agent finish method to get the final answer.
- until the end of Step 3 the code is the same as the previous code.
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

# Example of response format (commented out to avoid extra output)
# user_question = "What is the latest stock price of AAPL?"
# response = llm.invoke(react_prompt_template.format(user_question=user_question))
# print(f"\nResponse from llm to user question: {user_question} \n"
#       f"is to format the user question to a Thought and a json blob:\n")
# output = response.content
# print(output)
# print("--"*10 + "\n")

# As the  output indicates, 
# the tool choice and the input to the tool are delimited by ```json and ```. 
# This allows us to:
#  write an output parser to extract the action parameters (i.e., the tool and its inputs).

# Example of parsing (commented out to avoid extra output)
# start_index = output.find('```json')
# end_index = output.rfind('```')
# print("Extracted json blob:")
# print(output[start_index+7:end_index])
# print("--"*10 + "\n")

# 4. We can now assemble our search agent with this tool.
# we will need to implement two classes 
# - one for the agent execution (creating a Thought - Action - Observation chain in each run) 
# and another that implements a stopping criterion that checks if the final answer has been reached

class SearchAgent:
    """
    This agent will use the search tool to answer questions.
    """
    def __init__(self, llm, tool, prompt_template, verbose=True):
        self.llm = llm
        self.tool = tool
        self.tool_prompt_template = prompt_template
        self.verbose = verbose
        self.observation = None  # Store the observation
        self.memory = ""  # Initialize memory as empty string
        self.final_answer = None  # variable initialized for later use at the end of the "while not finish_status" loop

    def run(self, question):
        # Format messages according to DeepSeek's API
        messages = [{"role": "user", "content": self.tool_prompt_template.format(user_question=question)}]
        response = self.llm.invoke(messages)
        output = response.content.strip()  # Raw output from the llm

        # commenting out these lines from previous code, because we don't need to print the output
        #if self.verbose:
            # print(f"\nResponse from llm to user question: {question} \nis to format the user question to a Thought and a json blob:\n")
            # print(output)
            # print("\n" + "-"*20 + "\n")
            # print("Extracted json blob:\n")

        # Parse Output (to be put in action_input)
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
            answering_messages = [{"role": "user", "content": f"""Given the context:\n {search_result}, 
                                answer the following question based on this context:\n {question}. 
                                Do not include information about the context."""}]
            answer = self.llm.invoke(answering_messages)

            self.observation = answer.content # Note, this is per run, so it will be updated each time the agent is run
            self.memory = self.memory + output + "\n\n" + f"Observation: {self.observation}" + "\n" # Note, every time the agent is run, the memory is updated with the new observation and the old observation is kept

            return None

        except json.JSONDecodeError:
            return "Could not parse JSON response from model"
        except KeyError:
            return "Invalid tool choice format"

# Now we can create an instance of the agent and check the observation and memory
question = "What is the latest stock price of AAPL?"
search_agent = SearchAgent(llm, duckduckgo_search_tool, react_prompt_template, verbose=False) # verbose is set to False
search_agent.run(question) # this will run the agent and store the final answer in memory

print(f"\nObservation shown below:\n {search_agent.observation}\n")
print("--"*10 + "\n")
print(f"\nMemory shown below:\n {search_agent.memory}\n")
print("==="*10 + "\n")


# running the agent again to show how the observation and memory are updated

search_agent.run(question) # this will run the agent and store the final answer in memory

print("Second run of the agent:\n")
print(f"\nObservation shown below after running the agent again: (this is not additive, it is per run)\n {search_agent.observation}\n")
print("---"*10 + "\n")
print(f"\nMemory shown below after running the agent again (this adds on every run):\n {search_agent.memory}\n")
print("--"*10 + "\n")

print("As the output above indicates, the memory of the agent stores the first Thought - Action - Observation chain. \n Repeated runs of this agent will keep adding these chains to memory.")

print("---"*10 + "\n")
print(f"search_agent.memory:\n {search_agent.memory}")
print("---"*10 + "\n")

# The final piece in agent creation is to implement a stopping criterion, 
# that is, a signal to the agent that the final answer has been reached 
# and that there is no further execution of tooling that is required.

class AgentFinish:
  """
  This class will check if the agent has reached the final solution.
  It accesses the memory of the agent to check if the final answer has been reached.
  """
  def __init__(self, llm, agent):
    self.llm = llm
    self.agent = agent

  def run(self, question):
    checking_prompt = f"""
    Given the following status of the thought process of a LLM agent:\n {self.agent.memory},
    verify if the observation of the agent is enough to answer the following question:\n {question}.
    Answer only as yes or no (single word). Do not include any other information."""

    answer = self.llm.invoke(
        checking_prompt)

    finish_status = answer.content.strip().lower().replace(".", '')

    return finish_status



# Let us run the checker on the agent 
# we ran for a couple of times in the previous stage.

finish_checker = AgentFinish(llm, search_agent)
finish_checker = AgentFinish(llm, search_agent)

print("running the finish checker on the agent couple of times to check if the agent has reached the final solution:")
print(finish_checker.run(question))
print("Notice the above output will be yes or no")

print("---"*10 + "\n")

########################################################
# Now that we have both the steps,
# we can compose a loop that executes 
# the ReAct thought chains till the final answer is reached.

question = "What is the latest stock price of AAPL?"
finish_status = False

# Create a new instance of SearchAgent
search_react_agent = SearchAgent(llm, duckduckgo_search_tool, react_prompt_template)

while not finish_status:
    search_react_agent.run(question)
    finish_checker = AgentFinish(llm, search_react_agent)
    finish_status = (finish_checker.run(question)=="yes")

# The final answer is the last observation of the agent
# A more involved implementation could look at all the observations of the agent
# and invoke the LLM to compose a more nuanced answer.

search_react_agent.final_answer = search_react_agent.observation

print(f"Input: {question}")
print(f"search_react_agent.final_answer:\n {search_react_agent.final_answer}")
print("---"*10 + "\n")

"""
Output:
-------
Observation shown below:
 The latest stock price of AAPL is $222.78.

--------------------


Memory shown below:
 Thought: To find the latest stock price of AAPL, I will use the search tool to query the current stock price.

Action:
```json
{
  "action": "search_tool",
  "action_input": "latest stock price of AAPL"
}
```

Observation: The latest stock price of AAPL is $222.78.


==============================

Second run of the agent:


Observation shown below after running the agent again: (this is not additive, it is per run)
 The latest stock price of AAPL is $223.58 USD.

------------------------------


Memory shown below after running the agent again (this adds on every run):
 Thought: To find the latest stock price of AAPL, I will use the search tool to query the current stock price.

Action:
```json
{
  "action": "search_tool",
  "action_input": "latest stock price of AAPL"
}
```

Observation: The latest stock price of AAPL is $222.78.
Thought: To find the latest stock price of AAPL, I will use the search tool to look up the current stock price.

Action:
```json
{
  "action": "search_tool",
  "action_input": "latest stock price of AAPL"
}
```

Observation: The latest stock price of AAPL is $223.58 USD.


--------------------

As the output above indicates, the memory of the agent stores the first Thought - Action - Observation chain. 
 Repeated runs of this agent will keep adding these chains to memory.
------------------------------

search_agent.memory:
 Thought: To find the latest stock price of AAPL, I will use the search tool to query the current stock price.

Action:
```json
{
  "action": "search_tool",
  "action_input": "latest stock price of AAPL"
}
```

Observation: The latest stock price of AAPL is $222.78.
Thought: To find the latest stock price of AAPL, I will use the search tool to look up the current stock price.

Action:
```json
{
  "action": "search_tool",
  "action_input": "latest stock price of AAPL"
}
```

Observation: The latest stock price of AAPL is $223.58 USD.

------------------------------

running the finish checker on the agent couple of times to check if the agent has reached the final solution:
yes
Notice the above output will be yes or no
------------------------------

Input: What is the latest stock price of AAPL?
search_react_agent.final_answer:
 The latest stock price of AAPL is $223.58 USD.
------------------------------

"""