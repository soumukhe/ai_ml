import os
import asyncio
from dotenv import load_dotenv
from colorama import Fore
from ciscoPydanticHelper import CiscoCustomModel, Agent
import base64
import requests
from openai import AsyncAzureOpenAI


import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

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


# Define the output model
class Invoice(BaseModel):
    invoice_number: str = Field(..., description="The unique identifier for the invoice.")
    date_issued: str = Field(..., description="The date when the invoice was issued.")
    due_date: str = Field(..., description="The date by which payment is expected to be made.")
    currency: str = Field(..., description="The currency in which the invoice is denominated.")

    customer_name: str = Field(..., description="The name of the customer.")
    company: str = Field(..., description="The company associated with the customer.")
    address: str = Field(..., description="The address of the customer.")

    services_provided: list[str] = Field([], description="A detailed breakdown of services provided.")
    subtotal: float = Field(..., description="The total amount before tax.")
    tax_rate: float = Field(..., description="The tax rate applied to the invoice.")
    tax_amount: float = Field(..., description="The calculated tax amount.")
    total_amount_due: float = Field(..., description="The final amount due after including tax.")

    payment_instructions: dict = Field({}, description="Instructions for making payment.")


### Azure OpenAI Client ###
# Create AsyncAzureOpenAI client
client = AsyncAzureOpenAI(
    azure_endpoint='https://chat-ai.cisco.com',
    api_key=access_token,
    api_version="2023-08-01-preview"
)

### Model Creation and Execution ###
async def generate_invoice():
    # Create the Model
    model = CiscoCustomModel('gpt-4o-mini', openai_client=client, app_key=app_key)

    # Get the JSON schema from the Invoice model
    invoice_schema = Invoice.model_json_schema()
    
    # Create a system prompt with an example
    system_prompt = f"""You are an invoice generation assistant that MUST output only valid JSON.

IMPORTANT: You must ONLY return a JSON object matching this schema - no other text or formatting:
{json.dumps(invoice_schema, indent=2)}

Example of correct response format:
{{
    "invoice_number": "INV-2023-001",
    "date_issued": "2023-01-01",
    "due_date": "2023-02-01",
    "currency": "USD",
    "customer_name": "John Doe",
    "company": "Example Corp",
    "address": "123 Main St, City, State 12345",
    "services_provided": ["Service 1", "Service 2"],
    "subtotal": 1000.00,
    "tax_rate": 0.20,
    "tax_amount": 200.00,
    "total_amount_due": 1200.00,
    "payment_instructions": {{
        "method": "bank transfer",
        "bank_name": "Example Bank",
        "account_number": "1234567890"
    }}
}}

DO NOT include any other text, markdown, or formatting. Return ONLY the JSON object."""

    # Create the Agent with the updated system prompt
    agent = Agent(model=model, result_type=Invoice, system_prompt=system_prompt)

    # Update the user message to be more explicit about requiring JSON
    message = {
        "role": "user",
        "content": """Generate a JSON invoice with these details:
- Customer: Acme Inc.
- Date issued: 2022-01-15
- Due date: 2022-02-15
- Services: 
  * 10 hours consulting at $100/hour
  * 5 hours research at $50/hour
  * 3 hours report writing at $75/hour
- Tax rate: 20%
- Payment method: bank transfer"""
    }

    model_response, usage = await agent.model.request([message])
    
    # Debug: Print raw response
    print("Raw response:")
    print(model_response.parts[0].content)
    
    # Parse the response content as JSON
    try:
        # Try to clean the response if needed
        content = model_response.parts[0].content.strip()
        if content.startswith('```json'):
            content = content.split('```json')[1]
        if content.endswith('```'):
            content = content.rsplit('```', 1)[0]
        content = content.strip()
        
        invoice_data = json.loads(content)
        # Create Invoice object from the parsed data
        invoice = Invoice(**invoice_data)
        return invoice
    except Exception as e:
        print(f"Error parsing response: {e}")
        print("Response content type:", type(model_response.parts[0].content))
        print("Response content length:", len(model_response.parts[0].content))
        return None

# Run the async function
if __name__ == "__main__":
    invoice = asyncio.run(generate_invoice())
    if invoice:
        # Print the invoice as JSON
        print(json.dumps(invoice.model_dump(), indent=2))



