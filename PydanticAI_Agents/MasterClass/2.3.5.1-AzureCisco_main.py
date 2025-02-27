import os
import asyncio
from dotenv import load_dotenv
from colorama import Fore
from ciscoPydanticHelper import CiscoCustomModel, Agent
import base64
import requests
from openai import AsyncAzureOpenAI


# Suppress Logfire warnings
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

# Load environment variables from .env file
load_dotenv()

### Authentication ###
# Retrieve environment variables
app_key = os.getenv('AZURE_OPEN_AI_APP_KEY')
client_id = os.getenv('AZURE_OPEN_AI_CLIENT_ID')
client_secret = os.getenv('AZURE_OPEN_AI_CLIENT_SECRET')

# OAuth2 Authentication to obtain access token
url = "https://id.cisco.com/oauth2/default/v1/token"
payload = "grant_type=client_credentials"  # OAuth2 client credentials grant type

# Encode client_id and client_secret for Authorization header
value = base64.b64encode(f'{client_id}:{client_secret}'.encode('utf-8')).decode('utf-8')

headers = {
    "Accept": "*/*",
    "Content-Type": "application/x-www-form-urlencoded",
    "Authorization": f"Basic {value}",
    "User": f'{{"appkey": "{app_key}"}}'  # Include app_key in User header
}

# Make the token request
token_response = requests.post(url, headers=headers, data=payload)

# Validate token response
if token_response.status_code != 200:
    raise Exception(f"Failed to obtain access token: {token_response.text}")

# Extract access token
access_token = token_response.json()["access_token"]

### Azure OpenAI Client ###
# Create AsyncAzureOpenAI client
client = AsyncAzureOpenAI(
    azure_endpoint='https://chat-ai.cisco.com',
    api_key=access_token,
    api_version="2023-08-01-preview"
)

### Model #### Create the Model
model = CiscoCustomModel('gpt-4o-mini', openai_client=client, app_key=app_key)

# Create the Agent
agent = Agent(model=model, tools=[])

### Removing the main function, and doing quick test ###
# def main():
#     try:
#         message = {
#             "role": "user",
#             "content": "what is the capital of the United States?"
#         }
#         # use agent.model.request here, instead of agent.run_sync
#         model_response, usage = asyncio.run(agent.model.request([message]))
        
#         # Extract the message and usage
#         if hasattr(model_response, 'parts') and len(model_response.parts) > 0:
#            print(Fore.GREEN + f"Result [{model_response.parts[0].role}]: {model_response.parts[0].content}")

#     except Exception as e:
#          if hasattr(e, '__traceback__'):
#             import traceback
#             print(Fore.RED + "Full traceback:")
#             traceback.print_tb(e.__traceback__)

# if __name__ == "__main__":
#     main()



message = {
    "role": "user",
    "content": "what is the capital of the United States?"
}

model_response, usage = asyncio.run(agent.model.request([message]))

print(Fore.GREEN + f"Result [{model_response.parts[0].role}]: {model_response.parts[0].content}")