from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]

# Non-streaming version
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)

print("\nReasoning Process:")
print(response.choices[0].message.reasoning_content)
print("\nFinal Answer:")
print(response.choices[0].message.content)

