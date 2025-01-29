from langchain_experimental.utilities.python import PythonREPL
from langchain.tools import Tool
from langchain import hub
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI, OpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pydantic import BaseModel, Field, ValidationError
from IPython.display import display

load_dotenv()

import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the LLM (GPT-4o)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=1000,
)

# first look for the name of the file in the data folder
file_path = os.listdir('data')
print(file_path)

# Create ExcelFile object to get sheet names
excel_file = pd.ExcelFile('data/' + file_path[0])
print("Available sheets:", excel_file.sheet_names)

# Now load the data into a DataFrame
# If you want to load a specific sheet, specify the sheet name
df = pd.read_excel('data/' + file_path[0], sheet_name=excel_file.sheet_names[0])  # loads first sheet by default

# # Let's print the DataFrame info instead
# print("\nDataFrame Info:")
# print(df.info())

# Create Python REPL tool with name of python_repl_ast
# has to do this to make it work with the agent for openai
python_repl = PythonREPL()
python_repl_tool = Tool(
    name="python_repl_ast",  # Note: using the name the agent expects
    description="A Python shell. Use this to execute python commands. Input should be a valid python command.",
    func=python_repl.run
)

# Create a list to store agent interactions
agent_outputs = []

# Create the pandas agent
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    max_iterations=20,
    max_execution_time=300.0,
    allow_dangerous_code=True,
    return_intermediate_steps=True
)

# 4. do not show rows where the priincipal query field is empty.  As an example, if the query is "show me possible duplicates for data center networking", do not include rows with duplicates are empty

# Function to run agent and store output
def run_agent_query(query):
    try:
        # Add specific instruction to format the output
        formatted_query = f"""
        For this query: "{query}"
        Important instructions:
        1. Search across ALL solution domains in the dataframe
        2. Do not limit results to any specific solution domain
        3. Include ALL matching rows in the result
        4. For ANY filtering operation:
           - Include ALL rows that match the criteria
           - Do not truncate the results
           - Return the complete dataset that matches the criteria
           - Do not exclude any matching rows
           - Maintain all columns in the output
        5. Important: Return the COMPLETE result set, do not truncate
        6. Do not summarize or filter the results in the final answer
        7. Return the exact same dataframe that was found in the initial query
        
        If the result is a dataframe or table:
        1. First filter the dataframe based on the query criteria
        2. Convert the filtered dataframe to markdown using df_filtered.to_markdown(index=False)
        3. Print the complete markdown table
        4. Do not modify or summarize the results
        """
        
        response = agent.invoke({
            "input": formatted_query
        })
        
        # Extract output from the response
        output = response['output']
        
        # Clean up the output if needed
        if isinstance(output, str):
            # Remove any markdown code block markers
            output = output.replace('```python', '').replace('```', '').strip()
            # Remove any "Final Answer:" prefix
            if 'Final Answer:' in output:
                output = output.split('Final Answer:')[1].strip()
            # Remove any "DATAFRAME_RESULT:" prefix
            if 'DATAFRAME_RESULT:' in output:
                output = output.split('DATAFRAME_RESULT:')[1].strip()
        
        agent_outputs.append({
            'query': query,
            'response': output,
            'steps': response.get('intermediate_steps', [])
        })
        return output
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return f"Error: {str(e)}"




# # Example usage
# response = run_agent_query("show me the rows where the account name matches 'BU - MAN DE'")
# print("\nLatest Response:")
# print("```markdown")
# print(response)
# print("```")


# Example usage
response = run_agent_query("show me the records between March, 15th 2024 and March, 16th 2024")
print("\nLatest Response:")
print("```markdown")
print(response)
print("```")


# # Example usage
# response = run_agent_query("show me possible duplicates for data center networking. do not include rows with duplicates are empty")
# print("\nLatest Response:")
# print("```markdown")
# print(response)
# print("```")


# # Example usage
# response = run_agent_query("show me possible duplicates for data center networking. do not include rows with duplicates are empty")
# print("\nLatest Response:")
# print("```markdown")
# print(response)
# print("```")


# # Example usage
# response = run_agent_query("show me possible duplicates where solutions domain contains data center. ")
# print("\nLatest Response:")
# print("```markdown")
# print(response)
# print("```")

# # Example usage
# response = run_agent_query("show me ALL rows that have Negative sentiment across ALL solution domains")
# print("\nLatest Response:")
# print("```markdown")
# print(response)
# print("```")