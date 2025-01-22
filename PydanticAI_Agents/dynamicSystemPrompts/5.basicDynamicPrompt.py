import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
import logfire

load_dotenv()

logfire.configure(send_to_logfire='if-token-present')

# Define the model
model = OpenAIModel(os.getenv('LLM_MODEL'), api_key=os.getenv('OPENAI_API_KEY'))
#model = OpenAIModel("o1", api_key=os.getenv('OPENAI_API_KEY'))

# Define the output model
class Capital(BaseModel):
    """Capital city model - includes name, year founded, short history of the city and comparison to another city"""
    name: str
    year_founded: int
    short_history: str
    comparison: str



system_prompt="""You are an experienced historian and you are asked a question about the capital of a country. 
You are expected to provide the name of the capital city, the year it was founded, and a short history of the city. 
Provide an age and historical significance comparison of the cities."""

# Define the agent
agent = Agent(model=model, result_type=Capital, system_prompt=system_prompt) # note the pydantic model output is passed as result_type Capital:



@agent.system_prompt  
def add_comparison_city(ctx: RunContext[str]) -> str:
    return f"The city to compare is {ctx.deps['comparison_city']}"

user_prompt="What is the capital of France?"

result = agent.run_sync(user_prompt=user_prompt, deps={"comparison_city": "Paris"})

print(result.data)

print("--------------------------------")
json_output = result.data.model_dump_json(indent=4)
print(json_output)



"""
Output:

19:28:22.363 agent run prompt=What is the capital of France?
19:28:22.365   preparing model and tools run_step=1
19:28:22.365   model request
19:28:32.020   handle model response
name='Paris' year_founded=3 short_history='Paris, the capital of France, has a rich history dating back to its founding by a Celtic tribe known as the Parisii around the year 3 BC. It became an important center for commerce, culture, and learning during the Middle Ages, particularly after becoming the capital of the Frankish Empire. Over the centuries, Paris has played a pivotal role in numerous significant historical events, including the French Revolution and the establishment of modern democratic ideals, making it a cultural and political epicenter of Europe.' comparison="Paris is approximately 2026 years old, making it significantly older than many cities, including Washington, D.C., which was founded in 1790. This long history contributes to Paris's rich cultural heritage and its status as a global center for art, fashion, and science."
(crawl4ai_rag) (base) SOUMUKHE-M-2DJG:dynamicSystemPrompts soumukhe$ /Users/soumukhe/miniconda3/envs/crawl4ai_rag/bin/python /Users/soumukhe/pythonsScripts/agents/pydantic/dynamicSystemPrompts/5.basicDynamicPrompt.py
19:29:53.392 agent run prompt=What is the capital of France?
19:29:53.393   preparing model and tools run_step=1
19:29:53.394   model request
19:29:58.429   handle model response
name='Paris' year_founded=3 short_history='Paris, the capital of France, has a rich history dating back to its founding by a Gallic tribe called the Parisii in the 3rd century BC. The city became an essential political, cultural, and economic center in Europe, especially during the medieval and modern periods. Landmarks such as the Notre-Dame Cathedral, the Louvre Museum, and the Eiffel Tower symbolize its historical significance and architectural beauty.' comparison='Paris, with a founding year of approximately 3 BC, is over 2000 years old, highlighting its immense historical significance compared to many other cities, reflecting the evolution of culture, governance, and architecture through centuries.'
--------------------------------
{
    "name": "Paris",
    "year_founded": 3,
    "short_history": "Paris, the capital of France, has a rich history dating back to its founding by a Gallic tribe called the Parisii in the 3rd century BC. The city became an essential political, cultural, and economic center in Europe, especially during the medieval and modern periods. Landmarks such as the Notre-Dame Cathedral, the Louvre Museum, and the Eiffel Tower symbolize its historical significance and architectural beauty.",
    "comparison": "Paris, with a founding year of approximately 3 BC, is over 2000 years old, highlighting its immense historical significance compared to many other cities, reflecting the evolution of culture, governance, and architecture through centuries."
}


"""