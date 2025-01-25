from langchain_experimental.utilities.python import PythonREPL
from langchain import hub
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv


import os

load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    model="deepseek-chat"
)

messages = [{"role": "user", "content": "what is the meaning of life?"}]

response = llm.invoke(messages)

print(response.content)

"""
requirements.txt
------------------
langchain>=0.3,<0.4
langchain-community>=0.3,<0.4
langchain-experimental>=0.3,<0.4
langchain-openai>=0.0.5
python-dotenv>=0.21.0 

"""

"""
Output:
--------

The meaning of life is a deeply philosophical and subjective question that has been pondered by humans for centuries. Different cultures, religions, philosophies, and individuals offer various perspectives on this question. Here are a few common interpretations:

1. **Religious Perspectives**: Many religions propose that the meaning of life is tied to a higher power or divine purpose. For example, in Christianity, the meaning of life might be to love and serve God and others. In Buddhism, it might be to achieve enlightenment and escape the cycle of suffering.

2. **Philosophical Perspectives**: Philosophers have offered numerous theories. Existentialists, like Jean-Paul Sartre, suggest that life has no inherent meaning, and it is up to each individual to create their own purpose. Absurdists, like Albert Camus, argue that while life may lack inherent meaning, we can still find joy and purpose in the struggle itself.

3. **Scientific Perspectives**: From a biological standpoint, the meaning of life might be seen as survival and reproduction, ensuring the continuation of one's genes. Evolutionary biology suggests that life's purpose is to adapt and thrive in one's environment.

4. **Personal Perspectives**: Many people find meaning through personal experiences, relationships, achievements, and contributions to society. This can include pursuing passions, helping others, seeking knowledge, or striving for personal growth.

5. **Humanistic Perspectives**: Humanists often emphasize the importance of human welfare, happiness, and flourishing. They might argue that the meaning of life is to seek fulfillment and well-being for oneself and others.

Ultimately, the meaning of life is a personal journey, and each individual may find their own unique answer based on their beliefs, experiences, and values.


"""
