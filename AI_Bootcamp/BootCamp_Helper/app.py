import os
# Set PyTorch thread settings via environment variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import streamlit as st
import platform
import requests
import base64
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing import Dict, List, Any, TypedDict
import re
import time
from datetime import datetime, timedelta

# --- Configuration ---
st.set_page_config(
    page_title="AI Knowledge Assistant from CX Delivered AI/ML bootcamp material",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(override=True)

app_key = os.getenv('app_key')
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')

required_vars = {'app_key': app_key, 'client_id': client_id, 'client_secret': client_secret}
missing_vars = [k for k, v in required_vars.items() if not v]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file.")
    st.stop()

# Set device and log PyTorch info
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class EmbeddingWrapper:
    def __init__(self, embed_func):
        self.embed_func = embed_func
    def embed_documents(self, texts):
        return self.embed_func(texts)
    def embed_query(self, text):
        if isinstance(text, str):
            text = [text]
        return self.embed_func(text)[0]

# Add token refresh tracking
last_token_refresh = datetime.now()
TOKEN_REFRESH_INTERVAL = timedelta(hours=4)  # Refresh token every 4 hours

def get_fresh_token():
    """Get a fresh token from Azure OpenAI"""
    global last_token_refresh
    try:
        url = "https://id.cisco.com/oauth2/default/v1/token"
        payload = "grant_type=client_credentials"
        value = base64.b64encode(f'{client_id}:{client_secret}'.encode('utf-8')).decode('utf-8')
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {value}",
            "User": f'{{"appkey": "{app_key}"}}'
        }
        
        token_response = requests.post(url, headers=headers, data=payload, timeout=30)
        token_response.raise_for_status()
        last_token_refresh = datetime.now()
        return token_response.json()["access_token"]
    except Exception as e:
        logger.error(f"Error getting token: {str(e)}")
        raise

def init_azure_openai():
    """Initialize Azure OpenAI with token refresh"""
    try:
        # Check if token needs refresh
        if datetime.now() - last_token_refresh > TOKEN_REFRESH_INTERVAL:
            logger.info("Refreshing Azure OpenAI token...")
            access_token = get_fresh_token()
        else:
            access_token = get_fresh_token()  # Get initial token
        
        llm = AzureChatOpenAI(
            azure_endpoint='https://chat-ai.cisco.com',
            api_key=access_token,
            api_version="2023-08-01-preview",
            temperature=0,
            max_tokens=16000,
            model="gpt-4o-mini",
            model_kwargs={"user": f'{{"appkey": "{app_key}"}}'}
        )
        return llm
    except Exception as e:
        logger.error(f"Error initializing Azure OpenAI: {str(e)}")
        raise

@st.cache_resource
def init_vector_store():
    """Initialize the vector store with embeddings"""
    try:
        model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device=DEVICE,
            cache_folder="./model_cache"
        )
        
        def embed_function(texts):
            if isinstance(texts, str):
                texts = [texts]
            batch_size = 32
            all_embeddings = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    embeddings = model.encode(batch, convert_to_tensor=True, device=DEVICE)
                    if DEVICE == "cuda":
                        embeddings = embeddings.cpu()
                    all_embeddings.extend(embeddings.numpy())
            return all_embeddings
            
        embedding_wrapper = EmbeddingWrapper(embed_function)
        vector_store = Chroma(
            collection_name="hypothetical_questions",
            persist_directory="./rag_db",
            embedding_function=embedding_wrapper
        )
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        raise

def create_chain(llm, retriever):
    """Create a RAG chain with the given LLM and retriever"""
    try:
        system_message = '''You are an expert in AI and machine learning. Using the provided context, generate a clear, detailed, and accurate response to the query. 
        For general questions about AI concepts:
        1. Provide a clear definition
        2. Explain key characteristics
        3. Give relevant examples
        4. Use analogies when helpful
        5. Keep the explanation accessible while maintaining technical accuracy
        Ensure your answer addresses key aspects of the query and incorporates the context effectively.'''
        
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "Query: {query}\nContext: {context}\n\nFeedback: {feedback}")
        ])
        
        chain = response_prompt | llm | StrOutputParser()
        return chain
    except Exception as e:
        logger.error(f"Error creating chain: {str(e)}")
        raise

@st.cache_resource
def initialize_components():
    """Initialize and cache the model, vector store, and other components"""
    try:
        # Initialize Azure OpenAI with fresh token
        llm = init_azure_openai()
        
        # Initialize vector store
        vector_store = init_vector_store()
        
        # Create retriever with fresh LLM instance
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Create the chain with fresh components
        chain = create_chain(llm, retriever)
        
        return {
            "llm": llm,
            "vector_store": vector_store,
            "retriever": retriever,
            "chain": chain
        }
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        # Clear the cache to force reinitialization on next attempt
        st.cache_resource.clear()
        raise

try:
    components = initialize_components()
except Exception as e:
    st.error("Application failed to start due to initialization errors.")
    st.stop()

@st.cache_data(show_spinner=False)
def get_all_questions_by_source():
    try:
        all_docs = components["vector_store"].get()
        questions_by_source = {}
        documents = all_docs.get('documents', [])
        metadatas = all_docs.get('metadatas', [])
        for doc, metadata in zip(documents, metadatas):
            source = metadata.get('source', 'Unknown Source')
            if source not in questions_by_source:
                questions_by_source[source] = []
            questions_by_source[source].append({
                'question': doc,
                'original_chunk_id': metadata.get('original_chunk_id', 'Unknown ID')
            })
        return dict(sorted(questions_by_source.items()))
    except Exception as e:
        logger.error(f"Error loading questions by source: {str(e)}")
        st.error(f"Error loading questions by source: {str(e)}")
        return {}
    



# --- RAG AgentState and Workflow Functions (from rag_answers.py, adapted for Streamlit) ---

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

# Helper: get llm and retriever from state

def retrieve_hypothetical_questions(state: AgentState) -> AgentState:
    """
    Retrieves hypothetical questions based on the user's query using a self-querying retriever.
    
    Args:
        state (AgentState): The current state containing the user query.
        
    Returns:
        AgentState: The updated state with the retrieved hypothetical questions.
    """
    try:
        # Get fresh LLM instance
        llm = init_azure_openai()
        
        # Create new retriever with fresh LLM
        vector_store = components["vector_store"]
        metadata_field_info = [
            AttributeInfo(
                name="source",
                description="The source of the hypothetical question",
                type="string"
            ),
            AttributeInfo(
                name="original_chunk_id",
                description="The ID of the original chunk from which the question was generated",
                type="string"
            )
        ]
        document_content_description = "This document contains hypothetical questions generated from the original content."
        retriever = SelfQueryRetriever.from_llm(
            llm,
            vector_store,
            document_content_description,
            metadata_field_info
        )
        
        # Retrieve hypothetical questions using the self-querying retriever
        hypothetical_questions_retrieved = retriever.invoke(state['query'])
        
        # Extract the page content from each retrieved document
        questions = [doc.page_content for doc in hypothetical_questions_retrieved]
        
        # Add the questions to the state
        state['hypothetical_questions'] = questions
        
        # If the next node expects an expanded_query, create one from the hypothetical questions
        if questions:
            # Join hypothetical questions into a single expanded query
            state['expanded_query'] = " ".join(questions)
        else:
            # If no questions found, use the original query
            state['expanded_query'] = state['query']
        
        return state
    except Exception as e:
        logger.error(f"Error in retrieve_hypothetical_questions: {str(e)}")
        state['expanded_query'] = state['query']  # Fallback to original query
        return state

def retrieve_context(state: AgentState) -> AgentState:
    retriever = state['retriever']
    query = state['expanded_query']
    docs = retriever.invoke(query)
    state['context'] = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    return state

# def refresh_llm_token(state: AgentState) -> AgentState:
#     """
#     Refreshes the LLM token and updates the state.
    
#     Args:
#         state (AgentState): The current state
        
#     Returns:
#         AgentState: The state with refreshed LLM
#     """
#     try:
#         access_token = get_fresh_token()
#         state['llm'] = AzureChatOpenAI(
#             azure_endpoint='https://chat-ai.cisco.com',
#             api_key=access_token,
#             api_version="2023-08-01-preview",
#             temperature=0,
#             max_tokens=16000,
#             model="gpt-4o-mini",
#             model_kwargs={"user": f'{{"appkey": "{app_key}"}}'}
#         )
#         logger.info("Successfully refreshed LLM token")
#     except Exception as e:
#         logger.error(f"Error refreshing LLM token: {str(e)}")
#         st.error(f"Error refreshing LLM token: {str(e)}")
#     return state

def craft_response(state: AgentState) -> AgentState:
    llm = init_azure_openai()
    
    system_message = '''You are an expert in AI and machine learning. Using the provided context, generate a clear, detailed, and accurate response to the query. 
    For general questions about AI concepts:
    1. Provide a clear definition
    2. Explain key characteristics
    3. Give relevant examples
    4. Use analogies when helpful
    5. Keep the explanation accessible while maintaining technical accuracy
    Ensure your answer addresses key aspects of the query and incorporates the context effectively.'''
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Query: {query}\nContext: {context}\n\nFeedback: {feedback}")
    ])
    chain = response_prompt | llm
    response = chain.invoke({
        "query": state['query'],
        "context": "\n".join([doc["content"] for doc in state['context']]),
        "feedback": state.get('feedback', 'No additional feedback provided.')
    })
    state['response'] = response.content if hasattr(response, 'content') else str(response)
    return state

def score_groundedness(state: AgentState) -> AgentState:
    llm = init_azure_openai()
    
    system_message = '''You are an expert evaluator.
    Given the provided context and the response, assign a groundedness score between 0 and 1.
    A score of 1 indicates that the response is completely supported by the context,
    while 0 indicates no support at all.
    For general questions about AI concepts, be more lenient in scoring as long as the response provides a reasonable explanation based on the context. Consider partial matches and conceptual understanding, not just exact matches.'''
    groundedness_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Context: {context}\nResponse: {response}\n\nGroundedness score:")
    ])
    chain = groundedness_prompt | llm | StrOutputParser()
    response_text = chain.invoke({
        "context": "\n".join([doc["content"] for doc in state['context']]),
        "response": state['response']
    })
    match = re.search(r"([01](?:\.\d+)?)", str(response_text))
    if match:
        groundedness_score = float(match.group(1))
    else:
        groundedness_score = 0.0  # fallback if no number found
    state['groundedness_score'] = groundedness_score
    state['groundedness_loop_count'] += 1
    return state

def check_precision(state: AgentState) -> AgentState:
    llm = init_azure_openai()
    
    system_message = '''You are an expert evaluator.
    Evaluate the given response for how precisely it addresses the user's query.
    Provide a precision score between 0 and 1,
    where 1 indicates the response fully and accurately addresses the query,
    and 0 indicates it does not address it at all.'''
    precision_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Query: {query}\nResponse: {response}\n\nPrecision score:")
    ])
    chain = precision_prompt | llm | StrOutputParser()
    response_text = chain.invoke({
        "query": state['query'],
        "response": state['response']
    })
    match = re.search(r"([01](?:\.\d+)?)", str(response_text))
    if match:
        precision_score = float(match.group(1))
    else:
        precision_score = 0.0  # fallback if no number found
    state['precision_score'] = precision_score
    state['precision_loop_count'] += 1
    return state

def refine_response(state: AgentState) -> AgentState:
    llm = init_azure_openai()
    
    system_message = '''You are a constructive evaluator.
    Evaluate the given response in the context of the query and identify any potential gaps, ambiguities, or missing details.
    Provide suggestions for improvements to enhance accuracy and completeness.
    Do not rewrite the response; simply offer constructive feedback.'''
    refine_response_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Query: {query}\nResponse: {response}\n\nWhat improvements can be made to enhance accuracy and completeness?")
    ])
    chain = refine_response_prompt | llm | StrOutputParser()
    feedback = f"Previous Response: {state['response']}\nSuggestions: {chain.invoke({'query': state['query'], 'response': state['response']})}"
    state['feedback'] = feedback
    return state

def refine_query(state: AgentState) -> AgentState:
    llm = init_azure_openai()
    
    system_message = '''You are an expert in search query optimization.
    Evaluate the provided original and expanded queries and identify any missing details,
    specific keywords, or scope refinements that could improve search precision.
    Provide structured suggestions for improvement without replacing the original expanded query.'''
    refine_query_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Original Query: {query}\nExpanded Query: {expanded_query}\n\nWhat improvements can be made for a better search?")
    ])
    chain = refine_query_prompt | llm | StrOutputParser()
    query_feedback = f"Previous Expanded Query: {state['expanded_query']}\nSuggestions: {chain.invoke({'query': state['query'], 'expanded_query': state['expanded_query']})}"
    state['query_feedback'] = query_feedback
    return state

def should_continue_groundedness(state: AgentState) -> str:
    if state['groundedness_score'] >= 0.8:
        return "check_precision"
    elif state['groundedness_loop_count'] >= 3:
        return "max_iterations_reached"
    else:
        return "refine_response"

def should_continue_precision(state: AgentState) -> str:
    if state['precision_score'] >= 0.8:
        return "pass"
    elif state['precision_loop_count'] >= 3:
        return "max_iterations_reached"
    else:
        return "refine_query"

def max_iterations_reached(state: AgentState) -> AgentState:
    state['response'] = "This context is not relevant to your question. Please try rephrasing your query or ask relating to bootcamp AI/ML material."
    return state

def has_context(state: AgentState) -> str:
    if 'context' in state and state['context'] and len(state['context']) > 0:
        return "yes"
    else:
        return "no"

def check_context_exists(state: AgentState) -> AgentState:
    return state

def no_context_response(state: AgentState) -> AgentState:
    """
    Generates a response when no relevant context is found.
    
    Args:
        state (AgentState): The current state
        
    Returns:
        AgentState: The state with a context-specific response
    """
    state['response'] = "I'm here to help with questions about AI and machine learning concepts, particularly in the context of network architectures and infrastructure. Could you please rephrase your question to focus on AI/ML topics, or ask about a different aspect of AI/ML technology?"
    return state

def not_relevant_response(state: AgentState) -> AgentState:
    llm = init_azure_openai()
    
    system_message = '''You are an expert in AI and machine learning. Using the provided context, generate a clear, detailed, and accurate response to the query. 
    If the context doesn't fully answer the question, acknowledge this and provide the best possible answer based on the available information.'''
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Query: {query}\nContext: {context}\n\nPlease provide the best possible answer based on the available context.")
    ])
    chain = response_prompt | llm
    response = chain.invoke({
        "query": state['query'],
        "context": "\n".join([doc["content"] for doc in state['context']])
    })
    state['response'] = response.content if hasattr(response, 'content') else str(response)
    return state

def check_if_grounded(state: AgentState) -> str:
    if state.get('groundedness_score', 0.0) == 0.0:
        return "not_grounded"
    elif state.get('groundedness_loop_count', 0) >= state.get('loop_max_iter', 3):
        return "max_iterations_reached"
    elif state.get('groundedness_score', 0.0) >= 0.5:  # Lowered threshold from 0.7 to 0.5
        return "check_precision"
    else:
        return "refine_response"

def modify_query_for_context(state: AgentState) -> AgentState:
    """
    Modifies the query to include AI/ML network context if needed.
    
    Args:
        state (AgentState): The current state containing the query
        
    Returns:
        AgentState: The updated state with potentially modified query
    """
    # Check if query already contains context-specific keywords
    context_keywords = [
        'ai/ml', 'ai/ml network', 'ai network', 'ml network',
        'machine learning', 'artificial intelligence', 'nvidia',
        'fabric', 'architecture', 'network architecture',
        'networking', 'nvme', 'storage'
    ]
    
    # Keywords that indicate completely unrelated topics
    unrelated_keywords = [
        'eagle', 'bird', 'animal', 'sports', 'weather', 'food',
        'music', 'movie', 'game', 'sport', 'travel', 'history',
        'geography', 'politics', 'religion', 'art', 'literature'
    ]
    
    original_query = state['query']
    query = original_query.lower()
    
    # First check if the query is completely unrelated
    if any(keyword in query for keyword in unrelated_keywords):
        logger.info(f"Query is unrelated to AI/ML: {original_query}")
        state['query'] = "This is an unrelated topic. Please ask about AI/ML concepts."
        state['expanded_query'] = state['query']
        return state
    
    # Check for multiple keywords or specific phrases
    keyword_count = sum(1 for keyword in context_keywords if keyword in query)
    needs_context = keyword_count < 2  # Require at least 2 keywords or a specific phrase
    
    # Log debug information to console
    logger.info(f"Original query: {original_query}")
    logger.info(f"Keyword count: {keyword_count}")
    logger.info(f"Needs context: {needs_context}")
    
    # Append context if needed
    if needs_context:
        modified_query = f"{original_query} in the context of AI/ML network architecture and infrastructure"
        logger.info(f"Modified query: {modified_query}")
        state['query'] = modified_query
        state['expanded_query'] = modified_query  # Update expanded_query as well
    else:
        logger.info("Query already contains sufficient context keywords, no modification needed")
        state['expanded_query'] = original_query  # Set expanded_query to original if no modification
    
    return state

def create_workflow(llm, retriever):
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("modify_query", modify_query_for_context)
    workflow.add_node("retrieve_hypothetical_questions", retrieve_hypothetical_questions)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("check_context_exists", check_context_exists)
    workflow.add_node("no_context_response", no_context_response)
    workflow.add_node("craft_response", craft_response)
    workflow.add_node("score_groundedness", score_groundedness)
    workflow.add_node("not_relevant_response", not_relevant_response)
    workflow.add_node("refine_response", refine_response)
    workflow.add_node("check_precision", check_precision)
    workflow.add_node("refine_query", refine_query)
    workflow.add_node("max_iterations_reached", max_iterations_reached)
    
    # Define the workflow edges
    workflow.add_edge(START, "modify_query")
    workflow.add_edge("modify_query", "retrieve_hypothetical_questions")
    workflow.add_edge("retrieve_hypothetical_questions", "retrieve_context")
    workflow.add_edge("retrieve_context", "check_context_exists")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "check_context_exists",
        has_context,
        {
            "yes": "craft_response",
            "no": "no_context_response"
        }
    )
    
    workflow.add_edge("no_context_response", END)
    workflow.add_edge("craft_response", "score_groundedness")
    
    workflow.add_conditional_edges(
        "score_groundedness",
        check_if_grounded,
        {
            "not_grounded": "not_relevant_response",
            "check_precision": "check_precision",
            "refine_response": "refine_response",
            "max_iterations_reached": "max_iterations_reached"
        }
    )
    
    workflow.add_edge("not_relevant_response", END)
    workflow.add_edge("refine_response", "craft_response")
    
    workflow.add_conditional_edges(
        "check_precision",
        should_continue_precision,
        {
            "pass": END,
            "refine_query": "refine_query",
            "max_iterations_reached": "max_iterations_reached"
        }
    )
    
    workflow.add_edge("refine_query", "retrieve_hypothetical_questions")
    workflow.add_edge("max_iterations_reached", END)
    
    return workflow

def cleanup_resources():
    """Clean up resources to prevent memory leaks"""
    try:
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear any cached resources
        st.cache_resource.clear()
        st.cache_data.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during resource cleanup: {str(e)}")

def main():
    try:
        # Clean up resources periodically
        cleanup_resources()
        
        # Initialize components
        try:
            components = initialize_components()
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}", exc_info=True)
            st.error("Failed to initialize application components. Please try refreshing the page.")
            return

        tab1, tab2 = st.tabs(["AI Knowledge Assistant", "Hypothetical Questions Explorer"])
        with tab1:
            st.title("🤖 AI Knowledge Assistant from CX Delivered AI/ML bootcamp material")
            st.markdown("""
                Welcome to the AI Knowledge Assistant! This tool uses advanced RAG (Retrieval-Augmented Generation) 
                to provide accurate and context-aware answers about AI and machine learning concepts.
            """)

            try:
                # Use the initialized components
                llm = init_azure_openai()  # Get fresh LLM instance
                retriever = components["retriever"]

                # User input
                user_query = st.text_input("Ask a question about AI or machine learning:", 
                                           placeholder="e.g., What is a neural network?")

                if user_query:
                    try:
                        with st.spinner("Processing your question..."):
                            # Initialize state
                            initial_state: AgentState = {
                                "query": user_query,
                                "expanded_query": user_query,
                                "context": [],
                                "response": "",
                                "precision_score": 0.0,
                                "groundedness_score": 0.0,
                                "groundedness_loop_count": 0,
                                "precision_loop_count": 0,
                                "feedback": "",
                                "query_feedback": "",
                                "loop_max_iter": 3,
                                "llm": llm,
                                "retriever": retriever
                            }

                            workflow = create_workflow(llm, retriever)
                            app = workflow.compile()
                            result = app.invoke(initial_state)

                            # Display results
                            st.subheader("Answer")
                            st.write(result["response"])

                            # Display metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Groundedness Score", f"{result['groundedness_score']:.2f}")
                            with col2:
                                st.metric("Precision Score", f"{result['precision_score']:.2f}")

                            # --- Show source PPT and chunk info ---
                            if result.get("context"):
                                st.markdown("#### Sources Used")
                                for i, ctx in enumerate(result["context"], 1):
                                    meta = ctx.get("metadata", {})
                                    source = meta.get("source", "Unknown")
                                    chunk_id = meta.get("original_chunk_id", meta.get("chunk_id", "Unknown"))
                                    st.write(f"**{i}. Source:** {source}  \n**Chunk ID:** {chunk_id}")
                    except IOError as e:
                        logger.error(f"Input/Output error occurred: {str(e)}", exc_info=True)
                        st.error("An error occurred while processing your question. The application will attempt to recover...")
                        # Clear any cached resources
                        st.cache_resource.clear()
                        # Reinitialize components
                        components = initialize_components()
                        st.experimental_rerun()
                    except Exception as e:
                        logger.error(f"Unexpected error occurred: {str(e)}", exc_info=True)
                        st.error("An unexpected error occurred. Please try again or refresh the page.")
            except Exception as e:
                logger.error(f"Application error: {str(e)}", exc_info=True)
                st.error(f"An error occurred: {str(e)}")
        with tab2:
            st.title("Hypothetical Questions Explorer")
            st.markdown("""
                Explore hypothetical questions generated from documents. Browse questions by their source file to understand key concepts and implications.
            """)
            with st.spinner("Loading questions from database..."):
                all_questions_grouped = get_all_questions_by_source()
            if all_questions_grouped:
                source_list = list(all_questions_grouped.keys())
                if not source_list:
                    st.warning("No sources found in the database.")
                else:
                    source_search = st.text_input(
                        "Filter sources:",
                        placeholder="Type to filter source documents...",
                        help="Enter text to filter source document names"
                    )
                    filtered_sources = source_list
                    if source_search:
                        filtered_sources = [s for s in source_list if source_search.lower() in s.lower()]
                    if not filtered_sources:
                        st.warning("No sources match your filter. Try a different search term.")
                    else:
                        selected_source = st.selectbox(
                            "Select a source document to view its questions:",
                            options=filtered_sources,
                            key="source_selectbox"
                        )
                        if selected_source:
                            questions_for_source = all_questions_grouped[selected_source]
                            st.subheader(f"Questions from: {selected_source}")
                            st.caption(f"Total questions: {len(questions_for_source)}")
                            
                            # Add global select boxes
                            col1, col2 = st.columns(2)
                            with col1:
                                num_chunks = st.selectbox(
                                    "Number of chunks to display:",
                                    options=list(range(1, len(questions_for_source) + 1)),
                                    index=min(4, len(questions_for_source) - 1),  # Default to 5 or max available
                                    key="num_chunks_select"
                                )
                            with col2:
                                questions_per_chunk = st.selectbox(
                                    "Questions per chunk:",
                                    options=list(range(1, 11)),
                                    index=4,  # Default to 5
                                    key="questions_per_chunk_select"
                                )
                            
                            for i, q in enumerate(questions_for_source[:num_chunks], 1):
                                with st.expander(f"Question Set {i}", expanded=True):
                                    questions = q['question']
                                    if isinstance(questions, str):
                                        questions = [qq for qq in questions.split('\n') if qq.strip()]
                                    elif isinstance(questions, list):
                                        pass
                                    else:
                                        questions = [str(questions)]
                                    
                                    st.markdown("**Questions:**")
                                    for idx, question in enumerate(questions[:questions_per_chunk]):
                                        # Remove leading number/period/space (e.g., "1. ", "2. ", etc.)
                                        clean_question = re.sub(r"^\d+\.\s*", "", question)
                                        st.markdown(f"{idx+1}. {clean_question}")
                                    st.caption(f"Original Chunk ID: {q.get('original_chunk_id', 'Unknown')}")
                                    st.divider()
            else:
                st.error("No questions found in the database. Please ensure the database has been properly initialized.")
            st.markdown("---")
            st.markdown("""
                <div style='text-align: center; color: grey;'>
                    <p>Powered by LangChain, ChromaDB, Sentence Transformers, and Azure OpenAI</p>
                    <p><small>Hypothetical Questions Explorer v1.0</small></p>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        cleanup_resources()  # Attempt cleanup on fatal error
        st.error("A fatal error occurred. Please refresh the page.")

if __name__ == "__main__":
    main()
