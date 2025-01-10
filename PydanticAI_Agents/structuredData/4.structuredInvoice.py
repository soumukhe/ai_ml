import os
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

"""
For creating the base model, you can get help from claude, or local GPT4ALL with Quen-coding model

The first part of the code and the second part of the code are independent in terms of how they operate.

1. First Part: Generating an Invoice

In the first part of the code:
	•	You provide a prompt asking the agent to generate an invoice for specific services (e.g., consulting, research, etc.) with the associated details (e.g., tax rate, payment terms).
	•	The agent uses the prompt, along with the OpenAIModel, to generate structured data. This output is then validated and parsed into the Invoice model.

The Invoice model only ensures that the output of the LLM conforms to the specified structure. The LLM itself is creating this data based on the text prompt and its own language generation capabilities—it is not dependent on any external file.

2. Second Part: Extracting Data from invoice.md

In the second part of the code:
	•	You provide the raw invoice data stored in invoice.md as a string.
	•	The agent processes this data using the same OpenAIModel and extracts information from it into the fields defined in the Invoice model.

The key difference here is the input data:
	•	In this part, the input comes from invoice.md, a text file containing invoice details.
	•	The agent uses its language understanding capabilities to parse the text, identify relevant fields (like invoice_number, date_issued), and match them to the fields in the Invoice model.


Output:

01:19:36.727 agent run prompt=Can you generate an invoice for a consulting service provided ...s 20 percent and the payment should be made via bank transfer.
01:19:36.729   preparing model and tools run_step=1
01:19:36.729   model request
Logfire project URL: https://logfire.pydantic.dev/soumukhe/my-first-project
01:19:39.273   handle model response
01:19:39.278 Text prompt LLM results: invoice_number='INV-2022-001' date_issued='2022-01-15' due_dat...x_amount=230.0 total_amount_due=1380.0 payment_instructions={}
01:19:39.279 Result type: <class '__main__.Invoice'>
--------------------------------
{
    "invoice_number": "INV-2022-001",
    "date_issued": "2022-01-15",
    "due_date": "2022-02-15",
    "currency": "USD",
    "customer_name": "Acme Inc.",
    "company": "Acme Inc.",
    "address": "Not Provided",
    "services_provided": [
        "Consulting (10 hours at $100/hour)",
        "Research (5 hours at $50/hour)",
        "Report Writing (3 hours at $75/hour)"
    ],
    "subtotal": 1150.0,
    "tax_rate": 20.0,
    "tax_amount": 230.0,
    "total_amount_due": 1380.0,
    "payment_instructions": {}
}
01:19:39.280 agent run prompt=Can you extract the following information from the invoice? Th...tions.com._

Would you like adjustments or additional details?
01:19:39.280   preparing model and tools run_step=1
01:19:39.281   model request
01:19:41.916   handle model response
01:19:41.923 Invoice markdown prompt LLM results: invoice_number='INV-2024-00123' date_issued='2024-12-29' due_d...x_amount=835.0 total_amount_due=9185.0 payment_instructions={}
01:19:41.924 Result type: <class '__main__.Invoice'>
--------------------------------
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


print("--------------------------------")

json_output = result.data.model_dump_json(indent=4)
print(json_output)

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
