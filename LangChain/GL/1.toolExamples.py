from langchain_experimental.utilities.python import PythonREPL
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun


python_repl = PythonREPL()

result = python_repl.run("print('Hello, world!')")
print("REPL Output:", result)

print(python_repl.model_json_schema())
print("--"*10 + "\n")
"""
Output:

Python REPL can execute arbitrary code. Use with caution.
REPL Output: Hello, world!

{'description': 'Simulates a standalone Python REPL.', 'properties': {'_globals': {'anyOf': [{'type': 'object'}, {'type': 'null'}], 'title': 'Globals'}, '_locals': {'anyOf': [{'type': 'object'}, {'type': 'null'}], 'title': 'Locals'}}, 'title': 'PythonREPL', 'type': 'object'}

"""
## Tool 1: Python REPL
# Create a tool that uses the Python REPL
# description is how agent knows to use the too
repl_tool = Tool(
    name = "python_repl",
    description = "A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func = python_repl.run,
)

result = repl_tool.invoke({"input": "print('Hello, world!')"})
print("Tool Output:", result)
print("--"*10 + "\n")
"""
Output:
Tool Output: Hello, world!
"""

## Tool 2: duckduckgo search    
duckduckgo_tool = DuckDuckGoSearchRun()

result = duckduckgo_tool.invoke({"query": "What is the weather in San Francisco?"})
print("Tool Output:", result)
print("--"*10 + "\n")
"""
Output:
Tool Output: The weather in San Francisco is 60 degrees Fahrenheit.
"""

search_tool = Tool(
    name = "duckduckgo_search",
    description = "A tool that searches the web for information. Input should be a valid search query.",
    func = duckduckgo_tool.invoke,
)

result = search_tool.invoke({"query": "who is the president of the united states?"})
print("Tool Output:", result)
print("--"*10 + "\n")
"""
Output:
Tool Output: The current president of the United States is Joe Biden.
"""
print("--"*10 + "\n")





