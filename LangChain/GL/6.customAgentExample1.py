import importlib.util
import os
import sys

"""
In this example: 
we will build a custom agent that can be used to execute tasks using the Python repl.

We will use the original module customToolExample1.py as a starting point.
For convenience, we just import the module and suppress the output of the module.

Remember: that the original module is a custom tool "repl_tool", and we will use it to build the agent.
In the original module:
- we used llm to get code from the user task
- we used repl_tool to execute the code

*** Note: the more complex task will fail here because the 
repl_tool does not have access to the namespace.

in the next example we will fix this issue by using the exec tool instead of the repl_tool.

"""

## First we will import the previous customToolExample1.py module, and suppress the output of the module.
## this way we can keep the code clean and not have to deal with the output of the original module.

# Import the module customToolExample1.py from file
spec = importlib.util.spec_from_file_location("myImportedModule", "./5.customToolExample1.py")
custom_tool = importlib.util.module_from_spec(spec) # import the original module

# Suppress output during module execution.  Not intrested in the output of the original module
with open(os.devnull, "w") as null_output:
    sys.stdout = null_output  # Redirect stdout to null
    try:
        spec.loader.exec_module(custom_tool)  # Execute the module
        sys.stderr = null_output  # Redirect stderr to null
    finally:
        sys.stdout = sys.__stdout__  # Restore original stdout
        sys.stderr = sys.__stderr__  # Restore original stderr



# redefine needed variables and functions from the original module ( we named it custom_tool)

LANGCHAIN_API_KEY = custom_tool.LANGCHAIN_API_KEY
DEEPSEEK_API_KEY = custom_tool.DEEPSEEK_API_KEY
llm = custom_tool.llm # importing the llm from the original module
tool_prompt_template = custom_tool.tool_prompt_template # importing the tool_prompt_template from the original module
repl_tool = custom_tool.repl_tool # importing the repl_tool from the original module

# These are redefined again so we can use the same code for different tasks.
task = "Convert 45 degrees Celsius to Farenheit." # notice, I changed the prompt to 45 degrees Celsius
prompt = tool_prompt_template.format(user_task=task)
response = llm.invoke(prompt)
start_index = response.content.rfind('```python')
end_index = response.content.rfind('```')
code = response.content[start_index+9:end_index] # # ```python is 9 characters long, capture the code between
code_result = repl_tool.invoke(code)

print(f"\n Below we are running the code from the original module with changed prompt")
print(f"code_result: {code_result}")

## done with the original module and running the code with new variables and functions.

print("-"*10 + "\n")


"""
We now have all the components needed to build the agent. 
Here is an implementation of this agent. 
Notice how the run method extracts the Python code from the LLM output and then uses the repl_tool to execute this code. 
The reasoning is performed by the LLM and the action is performed by the tool.
"""

# Now we will build the agent, 
# instead of using the ReactAgent from langchain, we will use our own agent.

class PythonAgent:
  """
  An agent that can be used to execute tasks using the Python repl.

  Args:
    llm (LLM): The LLM to use.
    tool (Tool): The tool to use.
  """
  def __init__(self, llm, tool):
    self.llm = llm
    self.tool = tool
    self.tool_prompt_template = """
    Create Python code to execute the following task:
    {user_task}
    
    Important:
    1. Output only the code between ```python and ``` markers
    2. Make sure to store the final result in a variable named 'result'
    3. Include any necessary helper functions
    """
    self.script = None # Place holder for the Python script that will be generated during runtime

  def run(self, user_task):
    # Format the message according to DeepSeek's API format
    messages = [{"role": "user", "content": self.tool_prompt_template.format(user_task=user_task)}]
    response = self.llm.invoke(messages)
    output = response.content.strip()

    # Parse Output
    start_index = output.find('```python')
    end_index = output.rfind('```')

    if start_index == -1:
        python_script = output  # If no code blocks found, use the entire output
    else:
        python_script = output[start_index+9:end_index].strip()

    self.script = python_script

    # Execute script using the repl tool
    try:
        result = self.tool.invoke(python_script)
    except Exception as e:
        result = str(e)

    return result
  
# Now we can create an instance of the agent
python_agent = PythonAgent(llm, repl_tool)  

# redefine the task
task = "Convert 55 degrees Celsius to Farenheit."


# Now we can run the agent
result = python_agent.run(task)
print(f"result: {result}")

print("-"*10 + "\n")

# now extracting the script from the agent object python_agent
print(python_agent.script)

print("-"*10 + "\n")

# Let us now focus the agent on a more difficult problem.
task = "Find the sum of the first 100 prime numbers."
result = python_agent.run(task)
print(f"result: {result}")

print("-"*10 + "\n")

# show the code that the llm generated
print("The code that the llm generated for the sum of the first 100 prime numbers:")
print(python_agent.script)

print("-"*10 + "\n")

"""
Output:
------
Python REPL can execute arbitrary code. Use with caution.

 Below we are running the code from the original module with changed prompt
code_result: 113.0

----------

result: 
----------

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

result = celsius_to_fahrenheit(55)
----------

result: NameError("name 'is_prime' is not defined")
----------

The code that the llm generated for the sum of the first 100 prime numbers:
def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def sum_of_first_n_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return sum(primes)

result = sum_of_first_n_primes(100)
----------

"""
