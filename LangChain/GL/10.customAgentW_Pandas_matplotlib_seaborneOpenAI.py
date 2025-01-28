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

"""
Note:  using openAI instead of deepseek.  Seems like deepseek integration for pandas tool is still
not compatible with the latest version of langchain.

The tools are automaticlly chosen. 
- it uses the python_repl_ast tool
- it uses the pandas_dataframe_tool

"""

# Initialize the LLM (GPT-4o)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=1000,
)

# Import the dataset
df = pd.read_csv('https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data')

# Create Python REPL tool with name of python_repl_ast
# has to do this to make it work with the agent for openai
python_repl = PythonREPL()
python_repl_tool = Tool(
    name="python_repl_ast",  # Note: using the name the agent expects
    description="A Python shell. Use this to execute python commands. Input should be a valid python command.",
    func=python_repl.run
)

# Create the pandas agent
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    max_iterations=20,
    max_execution_time=20.0,
    allow_dangerous_code=True,
    handle_parsing_errors=True,
    
)

# Ask questions
def ask_agent(question):
    try:
        result = agent.invoke({"input": question})
        print(f"\nQuestion: {question}")
        
        # Special handling for table outputs
        if "first 5 rows" in question.lower():
            print("\nResult:")
            print(df.head().to_string())
        else:
            print("Result:", result["output"])
            
        print("---"*10)
    except Exception as e:
        print(f"\nError processing question: {question}")
        print(f"Error: {str(e)}")
        print("---"*10)

# Test different questions
questions = [
    "what is the mean age of the patients in the dataset?",
    "How many missing values are there in the dataset?",
    "can you display a brief summary of the columns in the dataset for me?"
]

for question in questions:
    ask_agent(question)


## now use matplotlib to plot the data

def create_plot(plot_request):
    """
    Function to handle plot creation and analysis using the agent
    Args:
        plot_request (str): Natural language request for plot creation and analysis
    """
    try:
        # Turn on interactive mode for non-blocking plots
        plt.ion()
        
        # Let the agent create and analyze the plot
        result = agent.invoke({"input": plot_request})
        print("\nAgent's Analysis:")
        print(result["output"])
        
        # Give time for the plot to display
        plt.pause(3)
        
        # Cleanup
        plt.ioff()
        plt.close('all')
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        plt.close('all')
    
    print("---"*10)

# Example plot requests - simplified for clarity
plot_requests = [
    """Create a KDE plot comparing obesity distribution between CHD and non-CHD patients. 
    Use fill=True in the kdeplot. Add appropriate labels and title.""",
    
    """Create a histogram of age distribution for people with CHD (chd=1). 
    Add appropriate labels and title.""",
    
    """Create a box plot comparing tobacco usage between CHD and non-CHD patients. 
    Add appropriate labels and title."""
]

# Create each plot with a pause between them
for request in plot_requests:
    create_plot(request)
    plt.pause(10)  # Add a pause between plots