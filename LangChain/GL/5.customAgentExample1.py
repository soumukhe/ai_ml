from langchain import hub
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI 


from langchain_experimental.utilities.python import PythonREPL
from langchain.tools import Tool

from dotenv import load_dotenv


import os

load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    model="deepseek-chat",
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

## Now instead of using the ReAct prompt, we can create a custom prompt:

tool_prompt_template = """
Create Python code to execute the following task:
{user_task}

Output only the code. Do not include anything else in your output.
"""

task = "Convert 39 degrees Celsius to Farenheit."

prompt = tool_prompt_template.format(user_task=task)

print(f"Final Prompt: {prompt}")
print("--"*10 + "\n")

# now get the response:
response = llm.invoke(prompt)

print(f"Response: \n{response.content}")
print("--"*10 + "\n")



"""
As the output above indicates, the LLM implements 
he Celsius to Farenheit conversion formula to a simple Python script 
enclosed within ```python and ```.

In order to execute this code, 
we will need to parse this output and extract the script. 
This amounts to finding these markers and indexing into the script using these markers.

"""

start_index = response.content.rfind('```python')
end_index = response.content.rfind('```')


print(f"Start Index: {start_index}")
print(f"End Index: {end_index}\n")
print("--"*10 + "\n")

code = response.content[start_index+9:end_index] # # ```python is 9 characters long, capture the code between

print(f"Code Extracted:\n {code}")
print("--"*10 + "\n")

# we've extracted the code, now we can execute it:

# Now we can create a tool to execute the code:
# create the repl tool:
python_repl = PythonREPL()

repl_tool = Tool(
    name = "python_repl",
    description = "A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func = python_repl.run,
)

# we can now use the repl tool to execute the code:

code_result = repl_tool.invoke(code)

print(f"Result: \n{code_result}")
print("--"*10 + "\n")

"""
Output:
--------------------

Final Prompt: 
Create Python code to execute the following task:
Convert 39 degrees Celsius to Farenheit.

Output only the code. Do not include anything else in your output.

--------------------

Response: 
```python
celsius = 39
fahrenheit = (celsius * 9/5) + 32
print(fahrenheit)
```
--------------------

Start Index: 0
End Index: 75

--------------------

Code Extracted:
 
celsius = 39
fahrenheit = (celsius * 9/5) + 32
print(fahrenheit)

--------------------

Python REPL can execute arbitrary code. Use with caution.
Result: 
102.2

--------------------


"""
