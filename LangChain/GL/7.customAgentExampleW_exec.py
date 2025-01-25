import importlib.util
import os
import sys

"""
In this example: 
we will build a custom agent that uses exec with namespace instead of the Python REPL tool.
This version maintains state between executions and handles function definitions better.
"""

## First we will import the previous customToolExample1.py module, and suppress the output
spec = importlib.util.spec_from_file_location("myImportedModule", "./5.customToolExample1.py")
custom_tool = importlib.util.module_from_spec(spec)

# Suppress output during module execution
with open(os.devnull, "w") as null_output:
    sys.stdout = null_output
    try:
        spec.loader.exec_module(custom_tool)
        sys.stderr = null_output
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

# Import needed variables from the original module
LANGCHAIN_API_KEY = custom_tool.LANGCHAIN_API_KEY
DEEPSEEK_API_KEY = custom_tool.DEEPSEEK_API_KEY
llm = custom_tool.llm

class PythonAgentWithNamespace:
    """
    An agent that executes Python code using exec with a namespace.
    This approach maintains state between executions and handles function definitions better.

    Args:
        llm (LLM): The LLM to use.
    """
    def __init__(self, llm):
        self.llm = llm
        self.namespace = {}  # Persistent namespace for code execution
        self.tool_prompt_template = """
        Create Python code to execute the following task:
        {user_task}
        
        Important:
        1. Output only the code between ```python and ``` markers
        2. Make sure to store the final result in a variable named 'result'
        3. Include any necessary helper functions
        """
        self.script = None

    def run(self, user_task):
        # Format the message according to DeepSeek's API format
        messages = [{"role": "user", "content": self.tool_prompt_template.format(user_task=user_task)}]
        response = self.llm.invoke(messages)
        output = response.content.strip()

        # Parse Output
        start_index = output.find('```python')
        end_index = output.rfind('```')

        if start_index == -1:
            python_script = output
        else:
            python_script = output[start_index+9:end_index].strip()

        self.script = python_script

        # Execute script in the persistent namespace
        try:
            exec(python_script, self.namespace)
            result = self.namespace.get('result', 'No result variable found')
            return result
        except Exception as e:
            return str(e)

# Create an instance of the agent
python_agent = PythonAgentWithNamespace(llm)

# Test with simple temperature conversion
task = "Convert 55 degrees Celsius to Fahrenheit."
result = python_agent.run(task)
print(f"Temperature conversion result: {result}")
print("-"*10)
print("Generated code:")
print(python_agent.script)
print("-"*10)

# Test with more complex prime numbers task
task = "Find the sum of the first 100 prime numbers."
result = python_agent.run(task)
print(f"\nPrime numbers sum result: {result}")
print("-"*10)
print("Generated code:")
print(python_agent.script)
print("-"*10) 


"""
Output:

Python REPL can execute arbitrary code. Use with caution.
Temperature conversion result: 131.0
----------
Generated code:
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

result = celsius_to_fahrenheit(55)
----------

Prime numbers sum result: 24133
----------
Generated code:
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