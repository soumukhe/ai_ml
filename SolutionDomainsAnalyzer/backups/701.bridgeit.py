import openai

import os
import openai
import traceback
import requests
import base64
from openai import AzureOpenAI


import prompt_toolkit
from dotenv import load_dotenv, find_dotenv


# Load environment variables from .env file
load_dotenv('.env')


load_dotenv(override=True)

openai.api_type = "azure"
openai.api_version = "2024-07-01-preview"  # gpt-4o-mini



## OAuth2 Authentication

client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')

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


# Use access_token after regeneration in below api_key
openai.api_key = token_response.json()["access_token"]  # access_token is one of the keys obtained
openai.api_base = 'https://chat-ai.cisco.com'

messages = [
    {"role": "user", "content": "Find beachfront hotels in San Diego for less than $300 a month with free breakfast."}
]

client = AzureOpenAI(
      azure_endpoint='https://chat-ai.cisco.com',
          api_key=token_response.json()["access_token"],
              api_version="2023-08-01-preview"
              )

# Now, correctly pass the 'user' field in the request
response = client.chat.completions.create(
    model="gpt-4o-mini",  # or whatever model you're using
    messages=messages,
    temperature=0.7,  # Adds some creativity to the response
    max_tokens=2000,  # Limits the response to 2000 tokens
    user=f'{{"appkey": "{app_key}"}}'  # Include the user/app key for tracking
)

# Extract token usage and cost
usage = response.usage
total_tokens = usage.total_tokens

print(f"Total tokens used: {total_tokens}")
