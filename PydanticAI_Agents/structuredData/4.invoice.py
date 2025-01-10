import os
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

"""
For creating the base model, you can get help from claude, or local GPT4ALL with Quen-coding model

Output:

{
    "invoice_number": "INV-2024-00123",
    "date_issued": "2024-12-29",
    "due_date": "2025-01-15",
    "currency": "USD",
    "customer_name": "Jane Doe",
    "company": "Innovative Solutions LLC",
    "address": "123 Innovation Drive, Tech City, CA 90210",
    "services_provided": [
        "Custom Software Development",
        "System Architecture Consulting",
        "AI Model Integration",
        "Technical Documentation"
    ],
    "subtotal": 8350.0,
    "tax_rate": 10.0,
    "tax_amount": 835.0,
    "total_amount_due": 9185.0,
    "payment_instructions": {}
}

"""





load_dotenv()

# Configure logfire
logfire.configure()

# Define the model
model = OpenAIModel('gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))

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


# Define the agent
agent = Agent(model=model, result_type=Invoice)

# Run the agent
result = agent.run_sync("Can you generate an invoice for a consulting service provided to Acme Inc. on 2022-01-15 with a due date of 2022-02-15? The invoice should include the following services: 10 hours of consulting at $100 per hour, 5 hours of research at $50 per hour, and 3 hours of report writing at $75 per hour. The tax rate is 20 percent and the payment should be made via bank transfer.")

logfire.notice('Text prompt LLM results: {result}', result = str(result.data))
logfire.info('Result type: {result}', result = type(result.data))

# Read the markdown file
with open('data/invoice.md', 'r') as file:
    invoice_data = file.read()

# Run the agent
result = agent.run_sync(f"Can you extract the following information from the invoice? The raw data is {invoice_data}")

logfire.notice('Invoice markdown prompt LLM results: {result}', result = str(result.data))
logfire.info('Result type: {result}', result = type(result.data))

print("--------------------------------")

json_output = result.data.model_dump_json(indent=4)
print(json_output)