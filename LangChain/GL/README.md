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

# Langchain API Key:
<get from: https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key


# DeepSeek API (very cheap)
- https://platform.deepseek.com/

# Python REPL
- The Python REPL stands for Read-Eval-Print Loop, a simple interactive programming environment. It’s a prompt where you can write and execute Python code line by line, and it immediately evaluates and displays the output.


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

