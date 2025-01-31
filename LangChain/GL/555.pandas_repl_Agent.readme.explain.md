# Solutions Domain Analyzer Implementation Details

## Overview
This document explains the implementation details of the Solutions Domain Analyzer, comparing the old and new approaches to fuzzy search, and detailing how the application works with Streamlit.

## Fuzzy Search Implementation

### Current Implementation (Efficient Approach)
The current implementation uses a two-step process that separates the concerns between the LLM (query understanding) and Python/Pandas (data processing).

```python
# Step 1: Create simplified DataFrame with row numbers
simplified_df = df.copy()
simplified_df['row_number'] = df.index

# Step 2: Create pandas agent for finding matches
finder_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=simplified_df,
    verbose=True,
    max_iterations=3,
    max_execution_time=30.0,
    allow_dangerous_code=True,
    include_df_in_prompt=True,
    prefix="""You are working with a pandas dataframethat has these columns: Solution Domain, Account Name, Created Date, Product, Use Case, Created By, Status, Closed Date, Solution Domain, Next Step, Original_Row_Number, Reason_W_AddDetails, RequestFeatureImportance, Sentiment, possibleDuplicates, CrossDomainDuplicates.

IMPORTANT: DO NOT CREATE SAMPLE DATA. Use the actual DataFrame 'df' that is provided to you.

To find matching rows:
1. Use df.index or df['row_number'] to get the correct row numbers
2. For text comparisons, use case-insensitive matching (str.contains() or str.lower())
3. Return ONLY the list of matching row numbers
4. Format: [1, 2, 3]
5. No explanations or code blocks needed
6. For multiple conditions, combine them with & operator and proper parentheses grouping
7. For duplicate checks, use ((df['possibleDuplicates'].fillna('').str.len() > 0) | (df['CrossDomainDuplicates'].fillna('').str.len() > 0))
8. For sentiment, use exact match: df['Sentiment'] == 'Negative'
9. For date ranges, use pd.to_datetime(df['Created Date']).dt.strftime('%Y-%m') == '2024-03'

Example commands:
- For duplicates: df[((df['possibleDuplicates'].fillna('').str.len() > 0) | (df['CrossDomainDuplicates'].fillna('').str.len() > 0))]['row_number'].tolist()
- For sentiment: df[df['Sentiment'] == 'Negative']['row_number'].tolist()
- For domain search: df[df['Solution Domain'].str.lower().str.contains('campus', na=False)]['row_number'].tolist()
- For date range: df[pd.to_datetime(df['Created Date']).dt.strftime('%Y-%m') == '2024-03']['row_number'].tolist()
- For multiple conditions: df[((df['possibleDuplicates'].fillna('').str.len() > 0) | (df['CrossDomainDuplicates'].fillna('').str.len() > 0)) & (df['Sentiment'] == 'Negative') & (df['Solution Domain'].str.lower().str.contains('campus', na=False)) & (pd.to_datetime(df['Created Date']).dt.strftime('%Y-%m') == '2024-03')]['row_number'].tolist()"""
            )
            
            log_file.write(f"Finding matching row numbers...\n")
            
            # Get matching row numbers
            response = finder_agent.invoke({
                "input": f"Find row numbers where: {query}. IMPORTANT: Use the actual DataFrame 'df', do not create sample data."
            })
            
            # Extract and clean output
            output = response['output']
            if isinstance(output, str):
                # Clean the output string
                output = output.replace('Final Answer:', '').strip()
                
                # Try to extract numbers whether they're in a list or text format
                try:
                    # First try to evaluate as a Python list
                    matching_rows = eval(output)
                except:
                    # If that fails, extract numbers from text
                    import re
                    matching_rows = [int(num) for num in re.findall(r'\d+', output)]
                
                if matching_rows:
                    log_file.write(f"Found {len(matching_rows)} matching rows\n")
                    
                    # Step 2: Get complete data for matching rows
                    result_df = df.iloc[matching_rows]
                    log_file.write(f"Retrieved complete data for matching rows\n")
                    return result_df.to_markdown(index=False)
            
            log_file.write("No matching rows found\n")
            return "No matching results found."
                
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        with open('terminal_output.txt', 'a') as log_file:
            log_file.write(f"\n{error_msg}\n")
        return f"Error: {str(e)}"
"""
)
```

Key features:
1. **Focus on Row Numbers**: The LLM only needs to return matching row numbers
2. **Structured Examples**: Clear patterns for common operations
3. **Local Data Processing**: Actual data handling happens in Python/Pandas
4. **Efficient Token Usage**: Minimizes LLM involvement in data processing

### Previous Implementation (Less Efficient)
The old implementation tried to handle both query understanding and data processing within the LLM:

```python
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    max_iterations=3,
    allow_dangerous_code=True
)

formatted_query = f"""
For this query: "{query}"
1. Use the existing DataFrame 'df' - do not create sample data
2. Generate appropriate pandas code to filter and display the data
3. Use case-insensitive string operations
4. Execute the code using python_repl_ast
5. Display results using df.to_markdown(index=False)
"""
```

Issues with this approach:
1. **LLM Data Processing**: Made the LLM handle data formatting and display
2. **Token Intensive**: More complex operations within the LLM
3. **Less Structured**: Fewer examples and patterns to follow
4. **Slower Performance**: Due to LLM handling data processing

## Understanding `include_df_in_prompt`

The `include_df_in_prompt=True` parameter controls what DataFrame information is included in the LLM prompt:

```python
# What gets included:
1. DataFrame schema (column names and data types)
2. Sample of the data (usually first few rows)
3. Basic statistics:
   - Number of rows and columns
   - Data types
   - Null value information
```

Example of what the LLM receives:
```
DataFrame Info:
- Shape: (1000, 16) [showing first few rows]
- Columns: Solution Domain, Account Name, Created Date, ...
- Data Types: 
  - Solution Domain: object
  - Created Date: datetime64
  - ...

Sample Data (first 5 rows):
| Solution Domain | Account Name | Created Date |
|----------------|--------------|--------------|
| Campus Network | Company A    | 2024-03-01  |
...
```

Setting `include_df_in_prompt=False` would only send column names, making it more token-efficient but potentially less accurate.

## Python REPL Tool in Pandas DataFrame Agent

An interesting aspect of the implementation is how the Python REPL tool is handled:

```python
# This explicit REPL tool creation is actually not needed
python_repl = PythonREPL()
tools = [
    Tool(
        name="python_repl",
        description="A Python shell...",
        func=python_repl.run
    )
]
```

The `create_pandas_dataframe_agent` function actually comes with an implicit REPL tool. This means:
1. No need to explicitly create a REPL tool
2. The agent automatically handles Python code execution
3. Built-in safety checks and execution limits

## Streamlit Implementation

The application uses Streamlit for its web interface. Here's how different components are implemented:

### Page Configuration
```python
st.set_page_config(
    page_title="Solutions Domain Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### State Management
Streamlit uses session state to maintain data between reruns:
```python
# Initialize session state
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'selected_domain' not in st.session_state:
    st.session_state.selected_domain = None
```

### Interface Components

1. **Sidebar Controls**:
```python
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel File (up to 1GB)", 
    type=['xlsx']
)
```

2. **Tabs**:
```python
main_tab, fuzzy_search_tab = st.tabs(["📊 Main Analysis", "🔍 Fuzzy Search"])
```

3. **Progress Indicators**:
```python
progress_bar = st.progress(0)
status_text = st.empty()
```

4. **Data Display**:
```python
st.dataframe(styled_df, use_container_width=True)
```

### Custom Styling
The application includes custom CSS for a professional look:
```python
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        margin-top: 1em;
    }
    /* ... more styles ... */
    </style>
""", unsafe_allow_html=True)
```

### Key Streamlit Features Used

1. **File Handling**:
   - File upload with type restrictions
   - Excel file processing
   - Download buttons for results

2. **Layout**:
   - Wide layout for better data visibility
   - Sidebar for controls
   - Tabs for organizing different functions

3. **Interactivity**:
   - Progress bars
   - Status messages
   - Dynamic updates

4. **Data Display**:
   - DataFrame display with custom styling
   - Formatted text and markdown
   - Download options for results

5. **State Management**:
   - Session state for persistent data
   - Progress tracking
   - User selections

The Streamlit implementation provides a clean, professional interface while handling complex data processing and user interactions efficiently. 
