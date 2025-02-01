import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch
import os
import re
from tqdm import tqdm
import time
import shutil
import io
import warnings
import sys
import contextlib
from langchain_experimental.utilities.python import PythonREPL
from langchain.tools import Tool
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
import json
import base64
import requests
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

# Force reload of environment variables
load_dotenv(override=True)

# Get environment variables without defaults
app_key = os.getenv('app_key')  # app key is required
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')


# Validate required environment variables
if not all([app_key, client_id, client_secret, langsmith_api_key]):
    raise ValueError("Missing required environment variables. Please check your .env file contains: app_key, client_id, client_secret and LANGSMITH_API_KEY")

# Initialize OpenAI client
def init_azure_openai():
    """Initialize Azure OpenAI with hardcoded credentials"""
    url = "https://id.cisco.com/oauth2/default/v1/token"
    payload = "grant_type=client_credentials"
    value = base64.b64encode(f'{client_id}:{client_secret}'.encode('utf-8')).decode('utf-8')
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {value}",
        "User": f'{{"appkey": "{app_key}"}}'
    }
    
    token_response = requests.request("POST", url, headers=headers, data=payload)
    #print(token_response.json())
    
    llm = AzureChatOpenAI(
        azure_endpoint='https://chat-ai.cisco.com',
        api_key=token_response.json()["access_token"],
        api_version="2023-08-01-preview",
        temperature=0,
        max_tokens=1000,
        model="gpt-4o",
        model_kwargs={
            "user": f'{{"appkey": "{app_key}"}}'
        }
    )
    return llm, token_response.json()

llm, token = init_azure_openai()
print("---llm---")
print(llm)
print("---token---")
print(token)