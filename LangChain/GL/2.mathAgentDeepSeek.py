from langchain_experimental.utilities.python import PythonREPL
from langchain.tools import Tool
from langchain import hub
from dotenv import load_dotenv
import os
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    model="deepseek-reasoner"
)

# create the repl tool:
python_repl = PythonREPL()

repl_tool = Tool(
    name = "python_repl",
    description = "A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func = python_repl.run,
)

# create a ReAct prompt - Response/Action/Thought
## built into langchain.  Make sure to get free api from langchain hub.
react_prompt = hub.pull("hwchase17/react")


print("ReAct Prompt: ")
print(react_prompt.template)
print("--"*10 + "\n")



## Now create an agent with the prompt
react_agent = create_react_agent(
    llm=llm, # llm from deepseek
    tools=[repl_tool], # tool from python repl
    prompt=react_prompt, # prompt from langchain hub
)


print("Agent Prompts: description came from the repl_tool description")
print(react_agent.get_prompts())
print("--"*10 + "\n")

## now use AgentExecutor
react_executor = AgentExecutor(
    agent = react_agent,
    tools = [repl_tool],
    verbose = True,
)

result = react_executor.invoke({"input": "how many r's in strawberry?"})

print("Result:")
print(result)
print("--"*10 + "\n")

"""
Output:

ReAct Prompt: 
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
--------------------

Agent Prompts: description came from the repl_tool description
[PromptTemplate(input_variables=['agent_scratchpad', 'input'], input_types={}, partial_variables={'tools': 'python_repl(command: str, timeout: Optional[int] = None) -> str - A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.', 'tool_names': 'python_repl'}, metadata={'lc_hub_owner': 'hwchase17', 'lc_hub_repo': 'react', 'lc_hub_commit_hash': 'd15fe3c426f1c4b3f37c9198853e4a86e20c425ca7f4752ec0c9b0e97ca7ea4d'}, template='Answer the following questions as best you can. You have access to the following tools:\n\n{tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}')]
--------------------



> Entering new AgentExecutor chain...
Thought: I need to count the number of 'r's in the word "strawberry". Let's use Python to do this accurately.

Action: python_repl
Action Input: print('strawberry'.count('r'))
Python REPL can execute arbitrary code. Use with caution.
3
The word "strawberry" contains the letter 'r' three times.

Final Answer: 3

> Finished chain.
Result:
{'input': "how many r's in strawberry?", 'output': '3'}
--------------------
--------------------

"""