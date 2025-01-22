import os
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
import logfire

load_dotenv()

logfire.configure(send_to_logfire='if-token-present')

# Define the model
model = OpenAIModel(os.getenv('LLM_MODEL'), api_key=os.getenv('OPENAI_API_KEY'))

# Define the output model for use with the prompt_agent
class SystemPrompt(BaseModel):
    """System prompt for an agent to generate helpful responses"""
    prompt: str
    tags: list[str]

# Define the system prompts
## system prompt0 is the prompt that generates the system prompt for the agent
## system prompt1 is the prompt that uses the system prompt and tags to generate a helpful response to the user's input

system_prompt0 = """You an expert prompt writer. Create a system prompt to be used for an AI agent that will help a user based on the user's input. 
Must be very descriptive and include step by step instructions on how the agent can best answer user's question. 
Do not directly answer the question. Start with 'You are a helpful assistant specialized in...'. 
Include any relevant tags that will help the AI agent understand the context of the user's input."""

system_prompt1 = """Use the system prompt and tags provided to generate a helpful response to the user's input."""

# Define the agent
prompt_agent = Agent(model=model, result_type=SystemPrompt, system_prompt=system_prompt0)
agent = Agent(model=model, system_prompt=system_prompt1)

@agent.system_prompt  
def add_prompt(ctx: RunContext[str]) -> str:
    return ctx.deps.prompt

@agent.system_prompt
def add_tags(ctx: RunContext[str]) -> str:
    return f"Use these tags: {ctx.deps.tags}"

# Define the main loop
def main_loop():
    message_history = []
    prompt_generated = False
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not prompt_generated:
            system_prompt = prompt_agent.run_sync(user_input).data
            print("Prompt:", system_prompt.prompt)
            print("Tags:", system_prompt.tags)
            print("system prompt has been generated on first run")
            prompt_generated = True

        # Run the agent
        result = agent.run_sync(user_input, deps=system_prompt, message_history=message_history)
        message_history = result.all_messages()
        print(f"Assistant: {result.data}")

# Run the main loop
if __name__ == "__main__":
    main_loop()


"""
In the first run, the system prompt is generated and the agent is run with the system prompt.

Output:

>> teach me how to cook
20:32:54.596 prompt_agent run prompt=teach me how to cook
20:32:54.597   preparing model and tools run_step=1
20:32:54.598   model request
20:33:05.862   handle model response
Prompt: You are a helpful assistant specialized in cooking and culinary techniques. When a user asks for guidance on cooking, follow these step-by-step instructions to provide them with the best possible advice:

1. **Clarify User's Needs**: Begin by asking the user what specific dish or type of cuisine they are interested in cooking. This ensures that your advice is tailored to their request.

2. **Gather Ingredients**: Once you have their preferred dish, list out the necessary ingredients. If the user mentions any dietary restrictions or preferences, suggest suitable alternatives for key ingredients.

3. **Cooking Techniques**: Explain the fundamental cooking techniques required for the dish. For example, if the user wants to make pasta, describe boiling, sautéing, and possibly baking methods if relevant.

4. **Step-by-Step Instructions**: Provide a detailed, sequential guide on how to prepare the dish from start to finish. Include tips on timing, temperature settings, and presentation.

5. **Troubleshooting Tips**: Anticipate common issues the user may encounter while cooking and offer practical solutions. This could include how to fix overcooked food or what to do if the flavor is lacking.

6. **Encouragement and Suggestions**: Encourage the user to experiment with flavors and techniques, and suggest complementary side dishes or beverages to enhance their dining experience.

7. **Invite Questions**: After providing instructions, always invite the user to ask any further questions or seek clarification on any steps they find confusing.
Tags: ['cooking', 'culinary techniques', 'guided instructions', 'food preparation']
system prompt has been generated on first run
20:33:05.863 agent run prompt=teach me how to cook
20:33:05.864   preparing model and tools run_step=1
20:33:05.864   model request
20:33:07.076   handle model response
Assistant: I'd be happy to help you learn how to cook! To get started, could you let me know if you have a specific dish or cuisine in mind that you'd like to learn about? This way, I can provide tailored guidance that suits your interests!
>> beginner
20:33:18.642 agent run prompt=beginner
20:33:18.653   preparing model and tools run_step=1
20:33:18.656   model request
20:33:29.529   handle model response
Assistant: Great! Since you're a beginner, let's start with a simple and versatile dish that's perfect for learning fundamental cooking techniques: *scrambled eggs*. This dish will introduce you to basic cooking methods and is a great foundation for many other recipes.

Ingredients:
- 2 to 3 large eggs
- Salt, to taste
- Pepper, to taste
- 1 tablespoon butter (or oil, as a substitute)
- Optional: Milk or cream (about 1 tablespoon), cheese, herbs, or vegetables for extra flavor

Cooking Techniques:
1. **Whisking**: Mixing the eggs thoroughly to incorporate air, which makes them fluffy.
2. **Sautéing**: Cooking in a pan with heat and fat (like butter or oil).
3. **Seasoning**: Adding salt and pepper to enhance the flavor.

Step-by-Step Instructions:

1. **Prepare Your Ingredients**: 
   - Crack the eggs into a bowl and add a pinch of salt and pepper. If you're using milk or cream, add it now. 

2. **Whisk the Eggs**: 
   - Use a fork or a whisk to beat the eggs until the yolks and whites are fully combined and slightly frothy.

3. **Heat the Pan**: 
   - Place a non-stick skillet over medium-low heat. Add the butter or oil to the pan. Let the butter melt gently without browning.

4. **Cook the Eggs**: 
   - Once the butter is melted and bubbly, pour in the whisked eggs. Allow them to sit for a few seconds until they start to set around the edges.

5. **Stir Gently**: 
   - Using a spatula, gently push the eggs from the edges toward the center of the pan. Repeat this process, stirring occasionally until the eggs are mostly set but still slightly creamy. This usually takes about 2-4 minutes.

6. **Remove from Heat**: 
   - When the eggs are just set (still soft and a little runny), remove them from the heat. They will continue to cook slightly from the residual heat.

7. **Serve**: 
   - Transfer the scrambled eggs to a plate and season with additional salt and pepper if desired. You can also garnish with chopped herbs, cheese, or any other toppings you like.

### Troubleshooting Tips:
- **If the eggs are overcooked**: They will be dry and hard. To avoid this, cook on lower heat and remove them from the pan while they are still slightly undercooked.
- **Flavor is lacking**: Try adding herbs like chives or parsley, or cheese for richness.

### Encouragement and Suggestions:
Feel free to experiment with different add-ins or spices as you get comfortable. You can serve scrambled eggs with toast, avocado, or fruit for a complete meal.

If you have any further questions or need clarification on any steps, just let me know! Happy cooking!
>> quit
Goodbye!




"""    