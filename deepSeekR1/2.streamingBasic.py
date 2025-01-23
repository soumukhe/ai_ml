from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

messages = [{"role": "user", "content": "how many r's in the word strawberry?"}]

# Streaming version
print("\nStreaming Response:")
print("\nReasoning Process:")

reasoning_content = ""
final_content = ""

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.reasoning_content:
        reasoning_part = chunk.choices[0].delta.reasoning_content
        reasoning_content += reasoning_part
        print(reasoning_part, end="", flush=True)
    elif chunk.choices[0].delta.content:
        if not final_content:  # First content chunk
            print("\n\nFinal Answer:")
        content_part = chunk.choices[0].delta.content
        final_content += content_part
        print(content_part, end="", flush=True)

print("\n")  # Add final newline






