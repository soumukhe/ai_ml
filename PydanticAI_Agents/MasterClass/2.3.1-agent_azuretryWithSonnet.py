from pydantic_ai import Agent
from openai import AsyncAzureOpenAI
from pydantic_ai.models.openai import OpenAIModel 
import os
from dotenv import load_dotenv
import base64
import requests
from colorama import Fore
import json


load_dotenv()

# Get environment variables
app_key = os.getenv('AZURE_OPEN_AI_APP_KEY')
client_id = os.getenv('AZURE_OPEN_AI_CLIENT_ID')
client_secret = os.getenv('AZURE_OPEN_AI_CLIENT_SECRET')

# ... your OAuth2 setup remains the same ...

## OAuth2 Authentication##############

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


# print(token_response.json())

access_token = token_response.json()["access_token"]


### Modify from here
from pydantic_ai import Agent, models
from openai import AsyncAzureOpenAI
import os
from dotenv import load_dotenv
import base64
import requests
from colorama import Fore
import json

class CiscoCustomModel(models.Model):
    def __init__(self, model_name: str, openai_client, app_key: str):
        self.model_name = model_name
        self.client = openai_client
        self.app_key = app_key

    def name(self) -> str:
        return self.model_name
    
    async def agent_model(self, function_tools=None, allow_text_result=False, **kwargs):
        return self

    async def request(self, messages, model_settings=None):
        print(Fore.YELLOW, "Debug - Entering request method")
        
        # Handle ModelRequest object
        if hasattr(messages, 'content'):
            message_content = messages.content
        elif isinstance(messages, list):
            message_content = messages[0].get('content') if messages else None
        else:
            message_content = messages.get('content') if messages else None

        try:
            request_messages = [{
                "role": "user",
                "content": message_content
            }]

            print(Fore.YELLOW, "Debug - Sending request with messages:", request_messages)

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=request_messages,
                temperature=0.7,
                max_tokens=2000,
                user=json.dumps({"appkey": self.app_key})
            )
            
            return response, {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        except Exception as e:
            print(Fore.RED, f"Debug - Full error: {str(e)}")
            if hasattr(e, 'response'):
                print(Fore.RED, "Debug - Response details:", e.response.text if hasattr(e.response, 'text') else str(e.response))
            raise

# Set environment variable for logfire warning
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

load_dotenv()

# Get environment variables
app_key = os.getenv('AZURE_OPEN_AI_APP_KEY')
client_id = os.getenv('AZURE_OPEN_AI_CLIENT_ID')
client_secret = os.getenv('AZURE_OPEN_AI_CLIENT_SECRET')

# OAuth2 Authentication
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

# Create AsyncAzureOpenAI client
client = AsyncAzureOpenAI(
    azure_endpoint='https://chat-ai.cisco.com',
    api_key=token_response.json()["access_token"],
    api_version="2023-08-01-preview"
)

# Create the agent
agent = Agent(CiscoCustomModel(
    model_name="gpt-4o-mini",
    openai_client=client,
    app_key=app_key
))

# Create message
message = {
    "role": "user",
    "content": "what is the capital of the United States?"
}

try:
    print(Fore.YELLOW, "Debug - Sending message:", json.dumps([message], indent=2))
    result = agent.run_sync([message])
    print(Fore.GREEN, "Result:", result)
except Exception as e:
    print(Fore.RED, f"Error: {str(e)}")
    if hasattr(e, '__traceback__'):
        import traceback
        print(Fore.RED, "Full traceback:")
        traceback.print_tb(e.__traceback__)