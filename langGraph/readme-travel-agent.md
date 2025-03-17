# Travel Agent Chatbot with Web Search and Memory

This repository contains code for an intelligent travel agent chatbot that combines web search capabilities with memory retention to provide personalized, up-to-date travel recommendations.

## Overview

This project demonstrates how to build an AI-driven support chatbot with three key capabilities:

1. **Web-Enabled Knowledge Retrieval** - Searches the web in real-time to provide current and relevant information
2. **User-Level Memory Retention** - Maintains context across user sessions for more natural and personalized interactions
3. **Seamless Automation** - Reduces burden on human support agents by automating responses to common queries

## Architecture

The system is built using LangGraph (for workflow orchestration), LangChain (for LLM integration), and Mem0 (for memory persistence).

### Key Components

#### State Management

The heart of the system is the `State` class, which defines the structure for tracking conversation state:

```python
class State(TypedDict):
    """
    Represents the state of the chatbot at any given moment.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Chat history
    query: str                                                # Current user query
    expanded_search: List[str]                                # Related search queries
    search_results: List[Dict[str, str]]                      # Web search results
    max_results: int                                          # Max search results
    filtered_urls: List[str]                                  # Filtered relevant URLs
    info: List[str]                                           # Extracted information
    feedback: str                                             # User feedback
    output: str                                               # Final response
```

This TypedDict defines the structure that LangGraph will use to create and maintain state throughout the conversation flow. The state is automatically initialized when the graph is invoked with initial inputs:

```python
inputs = {
    "query": user_query,
    "feedback": "",
    "max_results": 2
}
```

The remaining fields in the State TypedDict are initialized with default values by LangGraph.

#### Workflow Nodes

The system uses a series of specialized functions (nodes) that each perform a specific task in the information retrieval and response generation process:

1. **expand_search_queries** - Expands a user query into multiple search queries for better web coverage
   ```python
   def expand_search_queries(state):
       """
       Use the core LLM to expand a natural language user request into multiple related search queries.
       """
       # Uses the LLM to generate 3 related search queries based on the user's question
       # Returns a dictionary with expanded search queries to update the state
   ```

2. **get_duckduckgo_results** - Performs web searches using DuckDuckGo
   ```python
   def get_duckduckgo_results(state):
       """
       Use DuckDuckGo to search for a query and return a list of search results.
       """
       # Searches DuckDuckGo for each expanded query
       # Returns search results to update the state
   ```

3. **filter_urls** - Filters search results to retain only relevant URLs
   ```python
   def filter_urls(state):
       """
       Use the core LLM to filter out irrelevant URLs based on the context of the user's request.
       """
       # Uses the LLM to filter out irrelevant URLs
       # Returns filtered URLs to update the state
   ```

4. **fetch_site_content** - Fetches content from filtered URLs
   ```python
   def fetch_site_content(state):
       """
       Fetch and return the content of the given URL using the Jina API.
       """
       # Fetches content from each filtered URL using Jina's web content extraction API
       # Returns extracted information to update the state
   ```

5. **summarize_content** - Summarizes fetched content for relevance to the query
   ```python
   def summarize_content(query: str, content: str):
       """
       Summarize the content using the core LLM.
       """
       # Takes raw web content and extracts only information relevant to the user's query
       # Returns a concise summary
   ```

6. **final_output** - Generates a comprehensive answer from all sources
   ```python
   def final_output(state):
       """
       Summarize the content using the core LLM to create the final response.
       """
       # Creates a coherent response with proper citations
       # Returns the final answer to update the state
   ```

7. **grade_output** - Evaluates whether the answer is satisfactory
   ```python
   def grade_output(state):
       """
       Determines whether the retrieved information is relevant to the question.
       """
       # Grades the quality of the answer
       # Returns either "output" to proceed or "rewrite" to try again with feedback
   ```

8. **print_output** - Displays the final answer
   ```python
   def print_output(state):
       """
       Print the final output to the console.
       """
       # Simply prints the final output
   ```

#### Graph Construction

These nodes are connected in a workflow graph that defines the sequence of operations:

```python
search_agent = StateGraph(State)
search_agent.add_node("expand_search_queries", expand_search_queries)
search_agent.add_node("get_duckduckgo_results", get_duckduckgo_results)
search_agent.add_node("filter_urls", filter_urls)
search_agent.add_node("fetch_site_content", fetch_site_content)
search_agent.add_node("final_output", final_output)
search_agent.add_node("print_output", print_output)

# Connect nodes with edges
search_agent.add_edge(START, "expand_search_queries")
search_agent.add_edge("expand_search_queries", "get_duckduckgo_results")
search_agent.add_edge("get_duckduckgo_results", "filter_urls")
search_agent.add_edge("filter_urls", "fetch_site_content")
search_agent.add_edge("fetch_site_content", "final_output")
search_agent.add_conditional_edges("final_output", grade_output,
                                  {
                                      "output": "print_output", "rewrite": "expand_search_queries"})
search_agent.add_edge("print_output", END)

# Compile
graph = search_agent.compile()
```

This creates a flow where the system:
1. Expands the user query
2. Searches for information
3. Filters results
4. Extracts content
5. Generates an answer
6. Grades the answer quality
7. Either outputs the answer or reiterates with feedback

### Memory Integration (Mem0)

The `SupportChatbot` class wraps this search functionality and adds persistent memory:

```python
class SupportChatbot:
    def __init__(self):
        """
        Initialize the SupportChatbot class, setting up memory, the LLM client, tools, and the agent executor.
        """
        # Initialize memory, LLM client, tools and agent executor
        
    def store_customer_interaction(self, user_id: str, message: str, response: str, metadata: Dict = None):
        """
        Store customer interaction in memory for future reference.
        """
        # Stores conversations in Mem0 for later retrieval
        
    def get_relevant_history(self, user_id: str, query: str) -> List[Dict]:
        """
        Retrieve past interactions relevant to the current query.
        """
        # Searches for relevant past conversations
        
    def handle_customer_query(self, user_id: str, query: str) -> str:
        """
        Process a customer's query and provide a response, taking into account past interactions.
        """
        # Handles the complete flow, including retrieving relevant history
```

This allows the chatbot to maintain context across multiple sessions and provide personalized responses based on past interactions.

## Usage

To use the chatbot:

1. Initialize the `SupportChatbot` instance
2. Provide a user ID for identity tracking
3. Submit queries and receive responses

```python
chatbot = SupportChatbot()
user_id = "user123"  # In the actual code, this is input by the user
query = "What are some less crowded tourist places in Paris?"
response = chatbot.handle_customer_query(user_id, query)
```

## Key Features

1. **Persistent Memory**: The system remembers user preferences across sessions using Mem0
2. **Real-time Information**: Always provides up-to-date information by searching the web
3. **Intelligent Filtering**: Uses an LLM to filter and grade search results for relevance
4. **Personalized Responses**: Tailors recommendations based on past interactions
5. **Self-Correction**: Can reiterate with new search queries if answers are unsatisfactory

## Technical Details

### External Dependencies

- **LangChain**: For LLM integration and tool creation
- **LangGraph**: For workflow orchestration
- **Mem0**: For memory persistence
- **DuckDuckGo Search**: For web search functionality
- **Jina AI**: For web content extraction
- **OpenAI**: For LLM capabilities (GPT-4o)

### State Flow Explanation

The state dictionary is not explicitly created in the code but is managed internally by LangGraph. When you invoke the graph with initial inputs, LangGraph:

1. Creates a state dictionary based on the `State` TypedDict structure
2. Initializes it with the provided inputs and default values for other fields
3. Passes this state through each node in the graph
4. Each node returns updates to be applied to the state
5. The final state contains the complete conversation history and generated response

This approach allows clean separation of concerns while maintaining a shared context throughout the conversation flow.

## Example Usage Scenarios

### Scenario 1: New User with Specific Preferences

```
User: I'm Samay and I love dining at iconic city restaurants and internet-famous eateries. Can you suggest a 3-day itinerary for Paris?
```

The system will:
1. Search for information about iconic Paris restaurants
2. Generate a personalized itinerary
3. Remember Samay's preferences for future interactions

### Scenario 2: Returning User with New Query

```
User: What about budget hotels near these restaurants?
```

The system will:
1. Retrieve past interactions to recall Samay's preferences
2. Search for budget hotels near the previously suggested restaurants
3. Provide personalized recommendations that align with past preferences

## Conclusion

This travel agent chatbot demonstrates how combining web search capabilities with persistent memory can create a powerful, personalized user experience. The LangGraph architecture enables a flexible, modular approach to building complex conversation flows, while Mem0 ensures continuity across user sessions.
