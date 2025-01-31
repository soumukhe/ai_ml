import os
import base64
import requests
import json
import asyncio
from dotenv import load_dotenv
from colorama import Fore

from pydantic_ai import Agent, models
from pydantic_ai.result import CompletionUsage
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

# Define a custom CompletionUsage class with 'requests' attribute
class CustomCompletionUsage(CompletionUsage):
    requests: int

class CiscoCustomModel(models.Model):
    """
    Custom model class inheriting from pydantic_ai.models.Model.
    Handles sending requests to Azure OpenAI and formatting the response.
    """
    def __init__(self, model_name: str, openai_client: AsyncAzureOpenAI, app_key: str):
        self.model_name = model_name
        self.client = openai_client
        self.app_key = app_key

    def name(self) -> str:
        return self.model_name

    async def agent_model(self, function_tools=None, allow_text_result=False, **kwargs):
        return self

    async def request(self, messages, model_settings=None):
        """
        Handles the request to Azure OpenAI.
        Extracts message content from PydanticAI's ModelRequest,
        sends it to Azure OpenAI, and returns the response along with usage information.
        """
        print(Fore.YELLOW + "Debug - Entering request method")
        
        try:
            # Debugging: Inspect the messages
            print(Fore.BLUE + "Debug - Messages:", messages)
            
            # Initialize message_content as None
            message_content = None
            
            # Extract the content from the first message
            if isinstance(messages, list) and len(messages) > 0:
                msg = messages[0]
                if hasattr(msg, 'parts') and len(msg.parts) > 0:
                    part = msg.parts[0]
                    if hasattr(part, 'content') and len(part.content) > 0:
                        # Assuming part.content is a list of dicts
                        message_content = part.content[0].get('content', None)
            
            if not message_content:
                print(Fore.RED + "Debug - No valid content found in messages.")
                raise ValueError("No valid content found in messages.")
            
            # Prepare the messages in the format expected by Azure OpenAI
            request_messages = [{
                "role": "user",
                "content": message_content
            }]

            print(Fore.YELLOW + "Debug - Sending request with messages:", request_messages)

            # Construct the 'user' parameter as a JSON string with only 'appkey'
            user_param = json.dumps({"appkey": self.app_key})

            print(Fore.BLUE + f"Debug - 'user' parameter set to: {user_param}")

            # Send the request to Azure OpenAI
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=request_messages,
                temperature=0.7,
                max_tokens=2000,
                user=user_param  # Ensure only 'appkey' is included
            )
            
            print(Fore.BLUE + "Debug - Received response:", response)
            
            # Construct the CustomCompletionUsage object
            usage = CustomCompletionUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                requests=1  # Set to 1 per request; adjust if multiple requests are made
            )

            # Return the response and usage as a tuple
            return response, usage
        except Exception as e:
            print(Fore.RED + f"Debug - Full error: {str(e)}")
            if hasattr(e, 'response'):
                print(Fore.RED + "Debug - Response details:", e.response.text if hasattr(e.response, 'text') else str(e.response))
            raise

async def main():
    try:
        print(Fore.YELLOW + "Debug - Sending message:", json.dumps([message], indent=2))
        result = await agent.run([message])
        # 'result' is the model_response
        if hasattr(result, 'choices') and len(result.choices) > 0:
            print(Fore.GREEN + "Result:", result.choices[0].message.content)
        else:
            print(Fore.RED + "Debug - No choices found in the response.")
    except Exception as e:
        print(Fore.RED + f"Error: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            print(Fore.RED + "Full traceback:")
            traceback.print_tb(e.__traceback__)

if __name__ == "__main__":
    # Suppress Logfire warnings
    os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

    # Load environment variables from .env file
    load_dotenv()

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

    # Create AsyncAzureOpenAI client
    client = AsyncAzureOpenAI(
        azure_endpoint='https://chat-ai.cisco.com',
        api_key=access_token,
        api_version="2023-08-01-preview"
    )

    # Create the Model
    model = models.OpenAIModel('gpt-4o-mini', openai_client=client)

    # Create the Agent
    agent = Agent(model=model)

    # Define the message to send
    message = {
        "role": "user",
        "content": "what is the capital of the United States?"
    }

    # Run the asynchronous main function
    asyncio.run(main())