# Langchain
- https://python.langchain.com/docs/
- https://python.langchain.com/docs/versions/v0_3/
- https://python.langchain.com/docs/versions/v0_3/modules/agents/
- https://python.langchain.com/docs/versions/v0_3/modules/agents/tools/
- https://python.langchain.com/docs/versions/v0_3/modules/agents/tools/python_repl/

# Langchain-experimental
- https://python.langchain.com/docs/versions/v0_3/modules/experimental/
- https://python.langchain.com/docs/versions/v0_3/modules/experimental/utilities/
- https://python.langchain.com/docs/versions/v0_3/modules/experimental/utilities/python/

# Langchain API Key: (free)
<get from: https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key>


# Python REPL
- The Python REPL stands for Read-Eval-Print Loop, a simple interactive programming environment. It’s a prompt where you can write and execute Python code line by line, and it immediately evaluates and displays the output.
- **7.customAgentExampleW_exec.py shows use of exec() with namespace**

# Comparison: Using Python REPL vs. `exec()` with a Namespace

| **Feature**                   | **Python REPL**                                         | **`exec()` with Namespace**                                 |
|--------------------------------|--------------------------------------------------------|------------------------------------------------------------|
| **Purpose**                   | Interactive testing and exploration of Python code.     | Dynamically execute Python code within a controlled scope.  |
| **Use Case**                  | Useful for debugging, quick experimentation, and testing. | Running dynamically generated or external Python code.     |
| **Execution Environment**     | Executes code directly in the interpreter’s global environment. | Executes code within a specified `globals` or `locals` dictionary. |
| **Namespace Control**         | Limited: Code modifies the global namespace directly.   | Full control over where variables and functions are stored (custom namespaces). |
| **Persistence of Variables**  | Variables persist in the REPL session.                  | Variables exist only within the specified namespace.        |
| **Output**                    | Displays results of expressions immediately.            | No output unless explicitly printed in the executed code.   |
| **Security**                  | Risky if arbitrary code is executed manually.           | Can be risky, but scope-limiting namespaces reduce the risk of unintended side effects. |
| **Error Handling**            | Immediate feedback on errors in the REPL.               | Requires wrapping `exec()` in a `try-except` block to handle errors. |
| **Execution Context**         | Interactive, step-by-step execution with live feedback. | Executes a block of code at once without interactivity.     |
| **Code Example**              | `>>> x = 5; print(x * 2)`                               | `exec('x = 5; print(x * 2)', {})`                          |
| **Scoping Example**           | All variables are stored in the global namespace.       | `exec('x = 5', my_namespace); print(my_namespace['x'])`    |

---

### Summary
- **Python REPL**: Best for interactive tasks, quick prototyping, and debugging.
- **`exec()` with Namespace**: Useful for controlled execution of dynamic or external code, especially when isolating or limiting the scope of execution.

# What Is a React Agent?

A React agent is a type of agent in AI systems (like LangChain) that combines:
1. **Reasoning**: The agent reasons about what needs to be done step-by-step.
2. **Acting**: The agent interacts with tools, APIs, or external systems to complete tasks based on its reasoning.

The term **“React”** often describes agents that:
- Use **chain-of-thought prompting** to break tasks into manageable steps.
- Decide dynamically when to call tools or stop reasoning.

---

## React Prompt and React Agent

A **React prompt** is a specific type of input designed to enable reasoning and acting in AI models. It includes:
- **Examples of reasoning steps**: How the model should reason about the problem.
- **Examples of actions**: How the model should use tools or interact with the environment.

When an agent uses a React prompt, it follows a structure where:
1. It reasons about the input.
2. Decides if a tool should be used or if it has the final answer.
3. Iteratively repeats reasoning and acting until the task is complete.

This creates a **React agent**, where the reasoning and acting process is central to the agent’s design.

---

## Characteristics of a React Agent:
- **Iterative**: It reasons about the task and acts in steps.
- **Tool-using**: It interacts with tools like APIs or databases.
- **Chain-of-thought reasoning**: It explains its reasoning at each step.
- **Dynamic**: Decides dynamically whether it needs to act further or stop.

---

## Summary:

If an agent is created using a **React prompt**, it is indeed a React agent because the core behavior—**reasoning + acting**—is defined by the structure of the prompt. It’s a design pattern for building agents that can reason through problems and take appropriate actions iteratively.

## Installation

1. **Clone the Repository**

   To clone only this specific project (sparse checkout):
   ```bash
   # Create and enter a new directory
   mkdir my_demo && cd my_demo

   # Initialize git
   git init

   # Add the remote repository
   git remote add -f origin https://github.com/soumukhe/ai_ml.git

   # Enable sparse checkout
   git config core.sparseCheckout true

   # Specify the subdirectory you want to clone
   echo "LangChain/GL" >> .git/info/sparse-checkout

   # Pull the subdirectory
   git pull origin master

   # Enter the project directory
   cd LangChain/GL
   ```

2. **Set Up Virtual Environment**

   Choose one of the following methods:

   **Using conda (recommended)**:
   ```bash
   # Create a new conda environment
   conda create -n langChain python=3.12
   
   # Activate the environment
   conda activate langChain
   ```

   **Using venv**:
   ```bash
   # Create a new virtual environment
   python -m venv venv
   
   # Activate the environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
```bash
# Install the required packages
pip install -r requirements.txt
```

# DeepSeek Models: 
- DeepSeek-chat and
- DeepSeek-reasoner (DeepSeek-R1)

DeepSeek-chat and DeepSeek-reasoner are two different models developed by DeepSeek, each designed for specific tasks and use cases. Below are their features and key differences:

---

# DeepSeek API (very cheap)
- https://platform.deepseek.com/

# DeepSeek API vs GPT-4o and o1 Models

DeepSeek offers two main models that compete with OpenAI’s GPT-4o and o1: **DeepSeek-V3** and **DeepSeek-R1**. Here’s a comparison of their features and pricing:

---

## Model Features

| **Feature**               | **DeepSeek-V3** | **DeepSeek-R1** | **GPT-4o**       | **o1**          |
|---------------------------|-----------------|-----------------|------------------|-----------------|
| **Input Context Window**  | 128K tokens     | 128K tokens     | 128K tokens      | 128K tokens     |
| **Maximum Output Tokens** | 8K tokens       | 32K tokens      | 16.4K tokens     | Unknown         |
| **Open Source**           | Yes             | Yes             | No               | No              |
| **Release Date**          | Dec 27, 2024    | Jan 21, 2025    | Aug 6, 2024      | Unknown         |

DeepSeek-R1 and DeepSeek-V3 are both **open-source models**, which sets them apart from OpenAI’s offerings. **DeepSeek-R1** has a larger maximum output token capacity compared to GPT-4o.

---

## Pricing Comparison

| **Model**     | **Input Cost (per million tokens)** | **Output Cost (per million tokens)** |
|---------------|------------------------------------|-------------------------------------|
| **DeepSeek-V3** | $0.14                             | $0.28                               |
| **DeepSeek-R1** | $0.55                             | $2.19                               |
| **GPT-4o**     | $2.50                             | $10.00                              |
| **o1**         | $15.00                            | $60.00                              |

---

## Key Insights

- **DeepSeek models** are significantly more affordable than OpenAI’s offerings:
  - **DeepSeek-V3** is approximately **29.8x cheaper** than GPT-4o for input and output tokens combined.
  - **DeepSeek-R1** is about **4.6x cheaper** than GPT-4o for input and output tokens.
  - Compared to **o1**, DeepSeek-R1 is even more cost-effective, estimated to cost only **2%** of what users would spend on o1.

## **DeepSeek-chat**
Also referred to as **DeepSeek-V3**, this is a general-purpose language model designed for a wide range of conversational tasks.

### **Key Features**
- **Open-source**: 671B parameter Mixture-of-Experts (MoE) model with 37B activated parameters per token.
- **Context Window**: 128K token context window.
- **Training Data**: Trained on 14.8T tokens.
- **Multi-token Prediction**: Supports multi-token prediction and incorporates innovative load balancing.
- **Pricing**: More affordable compared to DeepSeek-R1:
  - **Input Cost**: $0.14 per million tokens.
  - **Output Cost**: $0.28 per million tokens.

---

## **DeepSeek-reasoner (DeepSeek-R1)**
Also known as **DeepSeek-R1**, this is a specialized model focused on **advanced reasoning capabilities**.

### **Key Features**
- **Open-source**: 671B parameter Mixture-of-Experts (MoE) model with 37B activated parameters per token.
- **Reinforcement Learning**:
  - Trained via large-scale reinforcement learning with a focus on reasoning capabilities.
  - Includes two RL stages to discover improved reasoning patterns and align with human preferences.
- **Supervised Fine-Tuning (SFT)**:
  - Two SFT stages to seed reasoning and non-reasoning capabilities.
- **Pricing**: Higher cost compared to DeepSeek-V3:
  - **Input Cost**: $0.55 per million tokens.
  - **Output Cost**: $2.19 per million tokens.

---

## **Key Differences**
1. **Focus**:  
   - **DeepSeek-chat**: General-purpose conversational model.  
   - **DeepSeek-reasoner**: Specializes in advanced reasoning tasks.

2. **Training Approach**:  
   - **DeepSeek-reasoner**: Utilizes reinforcement learning for reasoning patterns.  
   - **DeepSeek-chat**: Relies on traditional training methods.

3. **Reasoning Capabilities**:  
   - **DeepSeek-reasoner**: Excels in complex reasoning, problem-solving, and logical deduction.  
   - **DeepSeek-chat**: Geared toward versatile and general conversational use.

4. **Pricing**:  
   - **DeepSeek-reasoner**: More expensive.  
   - **DeepSeek-chat**: Cost-effective for general use.

5. **Performance**:  
   - **DeepSeek-reasoner**: Comparable to OpenAI's o1 series in math, code, and reasoning tasks.

---

## **Summary**
- **DeepSeek-chat**: A versatile, cost-effective solution for general conversational tasks.  
- **DeepSeek-reasoner**: A powerful tool for specialized reasoning and problem-solving.

---

## **Model Names for Inferencing**
- For **DeepSeek-chat**:  
  `model="deepseek-chat"`

- For **DeepSeek-reasoner**:  
  `model="deepseek-reasoner"`

