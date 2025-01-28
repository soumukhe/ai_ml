from langchain_experimental.utilities.python import PythonREPL
from langchain.tools import Tool
from langchain import hub
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.llms.base import LLM
from typing import Any, List, Optional
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from langchain_together import Together
import os
import requests

from pydantic import BaseModel, Field, ValidationError


load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")

########### ssl proxy tried from corporate net, did not work ###########
# import certifi
# import requests

# # Configure proxy settings
# proxy_url = "corp_proxy"  # Replace with your proxy URL

# # Set environment variables for proxies
# os.environ['HTTPS_PROXY'] = proxy_url
# os.environ['HTTP_PROXY'] = proxy_url

############

llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.7,
    max_tokens=128,
    top_k=1,
    together_api_key=together_api_key,
    #client=session
)

# 1. # Importing the structured data. This could be any dataset of your choice, for now, let's work with some spotify data!
df = pd.read_csv('https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data')


#2. create the agent
agent = create_pandas_dataframe_agent(llm, df, verbose=True,allow_dangerous_code=True,max_iterations = 34000, max_execution_time = 240.0,
                                      )

text = 'what are the different columns in the dataset ?'
agent.invoke(text)

print("---"*10)

text = 'How many missing values are there in the dataset?'
agent.invoke(text)


print("---"*10)

text = 'what is the average obesity value?'
agent.invoke(text)

print("---"*10)


# Here we will show the ability of the model to create a plot and comment on the underlying trend in the data
# Observe how succinctly the model pin-points the trend!
text = """use matplotlib to Plot a kde plot of obesity among people with CHD and without CHD. Label both axes and add a legend.
Finally, comment on the trend which the plot uncovers"""
agent.invoke(text)

print("---"*10)

text= """show distribution of age of people with chd. Label the axes. Finally comment on the principal trend observed."""
agent.invoke(text)
