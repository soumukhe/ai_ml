from langgraph.graph import StateGraph, START, END
from typing import Dict, List, Any, TypedDict

class AgentState(TypedDict):
    query: str
    expanded_query: str
    context: List[Dict[str, Any]]
    response: str
    precision_score: float
    groundedness_score: float
    groundedness_loop_count: int
    precision_loop_count: int
    feedback: str
    query_feedback: str
    loop_max_iter: int
    llm: Any
    retriever: Any

def create_workflow():
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("modify_query", lambda x: x)
    workflow.add_node("check_query_relevance", lambda x: x)
    workflow.add_node("retrieve_hypothetical_questions", lambda x: x)
    workflow.add_node("retrieve_context", lambda x: x)
    workflow.add_node("check_context_exists", lambda x: x)
    workflow.add_node("no_context_response", lambda x: x)
    workflow.add_node("craft_response", lambda x: x)
    workflow.add_node("score_groundedness", lambda x: x)
    workflow.add_node("refine_response", lambda x: x)
    workflow.add_node("check_precision", lambda x: x)
    workflow.add_node("refine_query", lambda x: x)
    workflow.add_node("max_iterations_reached", lambda x: x)
    
    # Define the workflow edges
    workflow.add_edge(START, "modify_query")
    workflow.add_edge("modify_query", "check_query_relevance")
    
    # Add conditional edges for query relevance check
    workflow.add_conditional_edges(
        "check_query_relevance",
        lambda state: "relevant" if True else "irrelevant",
        {
            "relevant": "retrieve_hypothetical_questions",
            "irrelevant": "no_context_response"
        }
    )
    
    workflow.add_edge("retrieve_hypothetical_questions", "retrieve_context")
    workflow.add_edge("retrieve_context", "check_context_exists")
    
    # Add conditional edges for context check
    workflow.add_conditional_edges(
        "check_context_exists",
        lambda x: "yes",
        {
            "yes": "craft_response",
            "no": "no_context_response"
        }
    )
    
    workflow.add_edge("no_context_response", END)
    workflow.add_edge("craft_response", "score_groundedness")
    
    # Update score_groundedness conditional edges to match app.py
    workflow.add_conditional_edges(
        "score_groundedness",
        lambda x: "check_precision",  # Default path for visualization
        {
            "not_grounded": "no_context_response",
            "check_precision": "check_precision",
            "refine_response": "refine_response",
            "max_iterations_reached": "max_iterations_reached"
        }
    )
    
    workflow.add_edge("refine_response", "craft_response")
    
    # Update check_precision conditional edges to match app.py
    workflow.add_conditional_edges(
        "check_precision",
        lambda x: "pass",  # Default path for visualization
        {
            "pass": END,
            "refine_query": "refine_query",
            "max_iterations_reached": "max_iterations_reached"
        }
    )
    
    workflow.add_edge("refine_query", "retrieve_hypothetical_questions")
    workflow.add_edge("max_iterations_reached", END)
    
    return workflow

def main():
    # Create and compile workflow
    workflow = create_workflow()
    app = workflow.compile()
    
    # Get Mermaid syntax
    mermaid_syntax = app.get_graph().draw_mermaid()
    
    # Print the Mermaid syntax
    print("\nMermaid Graph Syntax:\n")
    print(mermaid_syntax)
    
    # Save to a file
    with open("workflow_graph.mmd", "w") as f:
        f.write(mermaid_syntax)
    print("\nGraph saved to workflow_graph.mmd")

if __name__ == "__main__":
    main() 


# copy paste the mermaid syntax into the workflow_graph.mmd file 
# https://mermaid.live   