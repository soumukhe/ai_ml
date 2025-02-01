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

## BridgeIT

import os
import openai
import traceback
import requests
import base64
from langchain_openai import AzureChatOpenAI

import os
# Set the OpenAI API key from the environment variable
# app_key = os.getenv('app_key')
# client_id = os.getenv('client_id')
# client_secret = os.getenv('client_secret')
# # print(app_key)   # only for testing

# Get environment variables without defaults
app_key = os.getenv('app_key')  # app key is required
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')

## OAuth2 Authentication

client_id = client_id
client_secret = client_secret

url = "https://id.cisco.com/oauth2/default/v1/token"

payload = "grant_type=client_credentials"  # This specifies the type of OAuth2 grant being used, which in this case is client_credentials.

# The client ID and secret are combined, then Base64-encoded to be included in the Authorization header.
value = base64.b64encode(f'{client_id}:{client_secret}'.encode('utf-8')).decode('utf-8')
headers = {
    "Accept": "*/*",
    "Content-Type": "application/x-www-form-urlencoded",
    "Authorization": f"Basic {value}",
    "User": f'{{"appkey": "{app_key}"}}'  # Ensure the app_key is included as it worked in chat completions
}

token_response = requests.request("POST", url, headers=headers, data=payload)


print(token_response.json())