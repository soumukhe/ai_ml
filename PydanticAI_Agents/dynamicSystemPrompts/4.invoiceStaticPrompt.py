import os
from datetime import date
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire

load_dotenv()

logfire.configure(send_to_logfire='if-token-present')

system_prompt="""You are an experienced accounting assistant tasked with generating detailed and precise invoices for a leading technology consulting firm. Your primary goal is to ensure that all invoices accurately reflect the services provided, including any applicable taxes or discounts.

Please follow these guidelines when creating each invoice:

1. **Invoice Number:** Ensure that each invoice has a unique number assigned sequentially.
2. **Date of Service:** Clearly state the date(s) on which the consulting services were rendered.
3. **Client Information:**
   - Name
   - Company (if applicable)
   - Address
   - Contact Details
4. **Service Description:**
   - Provide a detailed description of each service provided, including any specific projects or deliverables.
5. **Hourly Rate/Project Fee:** Clearly state the hourly rate for consulting services or the project fee if it's a fixed-price engagement.
6. **Hours Worked/Fees Incurred:** Include the total number of hours worked (if applicable) and the corresponding fees incurred, or list out specific project costs with their respective amounts.
7. **Taxes:**
   - Calculate any applicable taxes based on local regulations (e.g., VAT, GST).
   - Clearly indicate whether these are included in the invoice amount or if they need to be added separately.
8. **Discounts/Adjustments:** If there were any discounts applied or adjustments made during the engagement, clearly state them and reflect their impact on the total fee.
9. **Payment Terms:**
   - Specify the payment terms (e.g., due upon receipt, net 30 days).
10. **Invoice Data:** Use today's date {}.
11. **Due Date:** Add 30 days to today's date.
12. **Invoice Total:** Summarize all charges and taxes to provide a clear total amount owed by the client.

**Additional Notes:**
- Use professional language throughout the invoice.
- Ensure that all information is accurate and up-to-date.
- Attach any relevant documents or receipts supporting the services provided.
- Maintain consistency in formatting across all invoices generated for this business.

By following these guidelines, you will be able to generate highly detailed and precise invoices that accurately reflect the technology consulting services rendered by your firm. 
This will help ensure timely payments from clients while maintaining high standards of professionalism and accuracy.""".format(date.today())


# Define the model
model = OpenAIModel(os.getenv('LLM_MODEL'), api_key=os.getenv('OPENAI_API_KEY'))

# Define the agent
agent = Agent(model=model, system_prompt=system_prompt)

user_prompt= """Customer Name: John Doe\nServices Provided: Web Development, AI Consulting, Strategic Advisory,\nTotal Amount: $50000"""

# Run the agent
result = agent.run_sync(user_prompt=user_prompt)

print(result.data)


"""
Output:

18:58:27.203 agent run prompt=Customer Name: John Doe
Services Provided: Web Development, AI Consulting, Strategic Advisory,
Total Amount: $50000
18:58:27.204   preparing model and tools run_step=1
18:58:27.204   model request
18:58:35.047   handle model response
Below is a detailed invoice based on the provided information for Mr. John Doe.

---

**Invoice Number:** 0001  
**Date of Issue:** 2025-01-21  
**Due Date:** 2025-02-20  

---

**Bill To:**  
John Doe  
[Client's Address]  
[City, State, ZIP Code]  
[Client's Email Address]  
[Client's Phone Number]  

---

**Services Provided:**

| Description                       | Rate          | Hours/Project Fee | Amount     |
|-----------------------------------|---------------|--------------------|------------|
| Web Development                   | $20,000      | Fixed Fee           | $20,000    |
| AI Consulting                     | $15,000      | Fixed Fee           | $15,000    |
| Strategic Advisory                 | $15,000      | Fixed Fee           | $15,000    |
| **Subtotal**                      |               |                    | **$50,000**|

---

**Taxes:**  
- **Applicable Sales Tax (e.g. 0% VAT or GST)**: $0.00  
- **Total Amount Including Tax**: $50,000.00  

---

**Discounts/Adjustments:**  
- There are no discounts applied to this invoice.

---

**Payment Terms:**  
- Payment is due within 30 days from the date of the invoice (Due upon receipt).

---

**Invoice Total: $50,000.00**  

---

**Notes:**  
Thank you for choosing our services. If you have any questions regarding this invoice, please do not hesitate to contact us.

**Contact Information:**  
[Your Firm's Name]  
[Your Firm's Address]  
[City, State, ZIP Code]  
[Your Firm's Email Address]  
[Your Firm's Phone Number]  

---

*Please ensure that payments are made to [Your Firm's Bank Account/Payment Gateway Information].*

---

**Attachments:**  
- Copy of the service agreement.  
- Relevant project deliverables.

--- 

Make sure to fill in any missing information like the address and contact details, as well as attach any necessary supporting documents before sending this invoice to Mr. John Doe.

"""