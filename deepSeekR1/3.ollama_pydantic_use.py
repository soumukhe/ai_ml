from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.ollama import OllamaModel
from colorama import Fore


# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "deepseek-r1:8b"

# Initialize Ollama model with ModelSettings
model = OllamaModel(
    model_name=LLM_MODEL,
    base_url=f"{OLLAMA_BASE_URL}/v1"
)

# Set model parameters
model.settings = ModelSettings(
    temperature=0.1,  # Lower temperature for more consistent responses
    top_p=0.9,       # More focused sampling
    num_ctx=4096     # Larger context window
)

agent = Agent(model=model)

# Run the agent
result = agent.run_sync("how many r's are in the word 'strawberry'?")
print(Fore.RED, result.data) # using colorama to print in red


# Output:

#  <think>
# I need to count how many times the letter 'r' appears in the word "strawberry."

# First, I'll write down the letters in the word: S, T, R, A, W, B, E, R, R, Y.

# Next, I'll identify and mark all the 'R's: S, T, R, A, W, B, E, R, R, Y.

# Counting the marked letters, there are three 'R's in total.

# So, there are 3 r's in "strawberry."
# </think>

# To determine how many **'r'**s are in the word **"strawberry"**, follow these steps:

# 1. **Write down the word:**
   
#    S T R A W B E R R Y

# 2. **Identify and count each 'r':**

#    - 1st 'R': **T R**
#    - 2nd 'R': **E R**
#    - 3rd 'R': **R R** (both of these are consecutive **'r'**s)

# 3. **Total Count:**
   
#    There are **three** **'r'**s in "strawberry."

# \[
# \boxed{3}
# \]

