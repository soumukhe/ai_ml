import os
import base64
import requests
import json
import asyncio
from dotenv import load_dotenv
from colorama import Fore

from pydantic_ai import Agent, models
from openai import AsyncAzureOpenAI

from pydantic import BaseModel
from typing import List, Optional

# Define the custom CompletionTokensDetails class
class CompletionTokensDetails(BaseModel):
    accepted_prediction_tokens: int
    audio_tokens: int
    reasoning_tokens: int
    rejected_prediction_tokens: int

# Define the custom PromptTokensDetails class
class PromptTokensDetails(BaseModel):
    audio_tokens: int
    cached_tokens: int

# Define the custom CompletionUsage class
class CompletionUsage(BaseModel):
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    requests: int
    request_tokens: int
    completion_tokens_details: CompletionTokensDetails
    prompt_tokens_details: PromptTokensDetails
    details: Optional[dict] = {}

    class Config:
        extra = "allow"

# Define the updated Part class with tool_name
class Part(BaseModel):
    content: str
    role: str
    tool_name: Optional[str] = None  # Added tool_name
    tool_call_id: Optional[str] = None # Make tool_call_id Optional

# Define the updated ModelResponse class
class ModelResponse(BaseModel):
    parts: List[Part]

    class Config:
        extra = "allow"

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
        try:
            # Initialize message_content as None
            message_content = None

            # Extract the content from the first message
            if isinstance(messages, list) and len(messages) > 0:
                 if isinstance(messages[0], dict) and 'content' in messages[0]:
                    message_content = messages[0].get('content', None)
                 elif hasattr(messages[0], 'parts') and len(messages[0].parts) > 0:
                     msg = messages[0]
                     if hasattr(msg, 'parts') and len(msg.parts) > 0:
                       part = msg.parts[0]
                       if hasattr(part, 'content') and len(part.content) > 0:
                        # Assuming part.content is a list of dicts
                         message_content = part.content[0].get('content', None)


            if not message_content:
                raise ValueError("No valid content found in messages.")

            # Prepare the messages in the format expected by Azure OpenAI
            request_messages = [{
                "role": "user",
                "content": message_content
            }]


            # Construct the 'user' parameter as a JSON string with only 'appkey'
            user_param = json.dumps({"appkey": self.app_key})

            # Send the request to Azure OpenAI
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=request_messages,
                temperature=0.7,
                max_tokens=2000,
                user=user_param  # Ensure only 'appkey' is included
            )

            # Construct the CompletionUsage object
            usage = CompletionUsage(
                prompt_tokens=response.usage.prompt_tokens,
                response_tokens=response.usage.completion_tokens,  # Mapping 'completion_tokens' to 'response_tokens'
                total_tokens=response.usage.total_tokens,
                requests=1,  # Set to 1 per request; adjust if multiple requests are made
                request_tokens=response.usage.prompt_tokens,  # Mapping 'prompt_tokens' to 'request_tokens'
                completion_tokens_details=CompletionTokensDetails(
                    accepted_prediction_tokens=response.usage.completion_tokens_details.accepted_prediction_tokens,
                    audio_tokens=response.usage.completion_tokens_details.audio_tokens,
                    reasoning_tokens=response.usage.completion_tokens_details.reasoning_tokens,
                    rejected_prediction_tokens=response.usage.completion_tokens_details.rejected_prediction_tokens
                ),
                prompt_tokens_details=PromptTokensDetails(
                    audio_tokens=response.usage.prompt_tokens_details.audio_tokens,
                    cached_tokens=response.usage.prompt_tokens_details.cached_tokens
                ),
                details={}  # Populate if there are additional details required
            )

            # Extract the assistant's message content
            assistant_message = response.choices[0].message.content

            # Construct the ModelResponse object
            model_response = ModelResponse(
                parts=[
                    Part(content=assistant_message, role='assistant')  # tool_name defaults to None
                ]
            )

            # Return the ModelResponse and CompletionUsage as a tuple
            return model_response, usage

        except Exception as e:
            if hasattr(e, 'response'):
                print(Fore.RED + "Debug - Response details:", e.response.text if hasattr(e.response, 'text') else str(e.response))
            raise

async def main():
    try:
        # Directly call the model to get the response without the tool processing
        model_response, usage = await agent.model.request([message])
        
        # Extract the message and usage
        if hasattr(model_response, 'parts') and len(model_response.parts) > 0:
           print(Fore.GREEN + f"Result [{model_response.parts[0].role}]: {model_response.parts[0].content}")

    except Exception as e:
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
    model = CiscoCustomModel('gpt-4o-mini', openai_client=client, app_key=app_key)

    # Create the Agent
    agent = Agent(model=model, tools=[])

    # Define the message to send
    message = {
        "role": "user",
        "content": "what is the capital of the United States?"
    }

    # Run the asynchronous main function
    asyncio.run(main())