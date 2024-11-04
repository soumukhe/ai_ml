# Install Packages
"""
pip install streamlit
pip install streamlit-option-menu
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install plotly

"""

import os
import time
import io       # needed to load the dataframe from streamlit uploaded file


#import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#st.set_option('deprecation.showPyplotGlobalUse', False)

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Command to tell Python to actually display the graphs
# %matplotlib inline
pd.set_option("display.float_format", lambda x: f"{x:.2f}") #to display values upto 2 decimal places

# plotly interactive graphs
import plotly.express as px

## genAI
from dotenv import load_dotenv
import os
import PyPDF2
from PyPDF2 import PdfReader
#from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter


#from langchain_community.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory

# # langchain_community stuff
# from langchain_community.embeddings.openai import OpenAIEmbeddings
# from langchain_community.vectorstores.faiss import FAISS
# from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings

# # import HuggingFaceHub
# from langchain_community.llms import HuggingFaceHub

#####################################ITEMS******************************

#Items:

#(1) st.button("**Show Data Frame Head**"):
#(2) st.button("**Show SForce Incidents with Service Now Tickets**"):
#(3) st.button("**Show Task Types**"):
#(4) st.button("**Show Opened Duration Statistics**"):
#(5) st.button("**Show Customers**"):
#(6) st.button("Search Accounts by Customer Name"):
#(7) st.button("**Show Customers who have SForce Incidents more than 90 days for Status=Open**")
#(8) st.button("**Show Customers who have SForce Incidents less than 30 days for Status=Open**"):
#(9) st.button("**CS Consle Cases Opened with Adpoption Barrier, for state=open in AB**"):
#(10) st.button("**CS Consle Cases NOT Opened with Adpoption Barrier, for state=open in AB**"):
#(11) st.button("**CS Consle Cases: Opened/UnOpened Plot**"):
#(12) st.button("**Draw Bar Plot highlghting Months where Cases were created**"):
#(13) st.button("** Analysis on `Action Taken by:` for SForce Incidents**"):
#(14) st.button("****Show SForce Incidents Workload for `Action Taken by:`**"):
#(15) st.button("**Analyze SForce Incidents for `Action Comments:`. NOTE: These are for Status == Open**"
#(16) st.button("**Search Records using Keywords**"):


##########################################################################

# Common Functions

# LLM Query
import json
load_dotenv()
import prompts
import openai
import time
import random
import os
import traceback
import requests
import base64
from openai import AzureOpenAI

# Changed to Cisco BridgeIT
def analyze(json_data, model, temperature, max_tokens):
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)  

    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds                     

    if "conversation" not in st.session_state or st.session_state.conversation is None:
            st.session_state.conversation = None 


    if "chat_history" not in st.session_state:
            st.session_state.chat_history = []   

    data_list = json.loads(json_data)       
    raw_text = [str(item) for item in data_list]
    
    
    # Set the OpenAI API key from the environment variable
    app_key = os.getenv('app_key')
    client_id = os.getenv('client_id')
    client_secret = os.getenv('client_secret')
    
    # OAuth2Authentication
    client_id = client_id 
    client_secret = client_secret 

    url = "https://id.cisco.com/oauth2/default/v1/token"

    payload = "grant_type=client_credentials"  # This specifies the type of OAuth2 grant being used, which in this case is client_credentials.

    # The client ID and secret are combined, then Base64-encoded to be included in the Authorization header.
    value = base64.b64encode(f'{client_id}:{client_secret}'.encode('utf-8')).decode('utf-8')
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {value}",
        "User": f'{{"appkey": "{app_key}"}}'  # Ensure the app_key is included as it worked in chat completions
    }

    token_response = requests.request("POST", url, headers=headers, data=payload)  
    
    # use the access_token after regeneration
    openai.api_key = token_response.json()["access_token"]  # access_token is one of the keys obtained
    openai.api_base = 'https://chat-ai.cisco.com'
    
  
 
    client = AzureOpenAI(
    azure_endpoint='https://chat-ai.cisco.com',
    api_key=token_response.json()["access_token"],
    api_version="2023-08-01-preview"
)
 
 

    # prompts
    system_message = prompts.system_messages # this is getting the system_messages block from prompts.py
    prompt = prompts.generate_prompt(raw_text)
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt} ]

    response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens,
    user=f'{{"appkey": "{app_key}"}}'  # Include the user/app key for tracking
    )

    # Extract token usage and cost
    usage = response.usage
    total_tokens = usage.total_tokens

    # Assume the cost per token for the model
    # Replace these values with the actual costs from OpenAI pricing
    #cost_per_token = 0.06 / 1000  # Example cost for gpt-4-turbo (adjust accordingly)
    cost_per_token = 0.005/1000
    total_cost = total_tokens * cost_per_token   

    # Extract the response content
    #content = response['choices'][0]['message']['content'] 
    content = response.choices[0].message.content 

    st.write(f"**Using BridgeIT API provided by Cisco**")
    st.write(f"Total tokens used: {total_tokens}")
    st.write(f"Estimated cost: ${total_cost:.6f}")            
                        
    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

        
    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")            
    return content


# Old: Deprecated.  Using openAI.  .env has also been modified
#def analyze(json_data, model, temperature, max_tokens):
#     # Start the timer
#     start_time = time.time()

#     # Progress bar implementation, Placeholder for updating progress
#     latest_iteration = st.empty()
#     bar = st.progress(0)  

#     # Initially, show some progress to indicate the start of the process
#     for i in range(10): # This quickly increments progress to 90%
#         latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
#         bar.progress(i * 10)
#         time.sleep(0.1)  # Simulate delay 0.1 seconds                     

#     if "conversation" not in st.session_state or st.session_state.conversation is None:
#             st.session_state.conversation = None 


#     if "chat_history" not in st.session_state:
#             st.session_state.chat_history = []   

#     data_list = json.loads(json_data)       
#     raw_text = [str(item) for item in data_list]


#     # prompts
#     system_message = prompts.system_messages # this is getting the system_messages block from prompts.py
#     prompt = prompts.generate_prompt(raw_text)
#     messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt} ]
#     #response = openai.chat.completions.create(
#     #response =openai.ChatCompletion.create(
#     response = openai.chat.completions.create(
#     model=model,
#     messages=messages,
#     temperature=temperature,
#     max_tokens=max_tokens,
#     )

#     # Extract token usage and cost
#     usage = response.usage
#     total_tokens = usage.total_tokens

#     # Assume the cost per token for the model
#     # Replace these values with the actual costs from OpenAI pricing
#     #cost_per_token = 0.06 / 1000  # Example cost for gpt-4-turbo (adjust accordingly)
#     cost_per_token = 0.005/1000
#     total_cost = total_tokens * cost_per_token   

#     # Extract the response content
#     #content = response['choices'][0]['message']['content'] 
#     content = response.choices[0].message.content 

#     st.write(f"**Using BridgeIT API provided by Cisco**")
#     st.write(f"Total tokens used: {total_tokens}")
#     st.write(f"Estimated cost: ${total_cost:.6f}")            
                        
#     # Complete the progress bar when the API call is done
#     latest_iteration.text('Progress: 100%')
#     bar.progress(100)

#     # End the timer and calculate duration
#     end_time = time.time()
#     duration = end_time - start_time

        
#     # Display the duration
#     st.write(f"Process completed in {duration:.2f} seconds.")            
#     return content


def getpdf():
    st.title("Download LLM Report")

    #  download
    if st.session_state['pdf_generated']:
        file_path = st.session_state['pdf_file_path']

    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            content = file.read()

        # Provide the download button
        st.download_button(
            label="Download LLM PDF Report",
            data=content,
            file_name=os.path.basename(file_path),
            mime="application/pdf"
        )                    
    else:
        st.write("Download Report button will show up after running the LLM Report")
        
        
        
# Creating function for: dictionary to hold DataFrames for each category by solutions domain
def break_by_soldom(df1):

    # break by category
    unique_values = df1['Solution Domain: Solution Domain Name'].unique()

    dfs = {value: df1[df1['Solution Domain: Solution Domain Name'] == value] for value in unique_values}

    # Printing each DataFrame
    for key, dataframe in dfs.items():
        st.write(f"**DataFrame for Solutions Domain:** {key}")
        st.write(f"**record count:** {dataframe.shape[0]:,}")
        st.write(dataframe)
        #print("\n")           
                




#######################################

# Update default rc settings
import matplotlib as mpl
mpl.rcParams['savefig.format'] = 'svg'  # Use SVG for vector-based output
mpl.rcParams.update({
    'font.size': 10,  # sets the default font size for all elements
    'axes.titlesize': 10,  # specific font size for titles
    'axes.labelsize': 10,  # specific font size for axis labels
    'xtick.labelsize': 10,  # font size for the x ticks
    'ytick.labelsize': 10  # font size for the y ticks
})

########################################################################################################
# empty stuff
data = pd.DataFrame()   # empty dataframe


########################################################################################################
# Streamlit configuration wide page
st.set_page_config(
    page_title='Adoption Barrier Insights',
    page_icon=':rocket:',
    layout='wide',
    initial_sidebar_state='expanded'

)


#################
# Set the maximum upload size for Streamlit (e.g., 512 MB)
os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '512'  # MB, default is 200MB




# App framework
#st.title('🦜🔗 Adoption Barrier Insights')
st.title("🚧 Adoption Barrier Insights") 
st.write(f"**Please read the instructions below. Upload the CSV file to the side pane and then proceed**")





# Using st.expander with a Markdown styled label for bold text
with st.expander('''**Generate report in Sales Force and export to csv:**
Click Here for Instructions
'''):
    st.markdown("""
    ## Instructions:
    Generate a report in Sales Force and export to CSV.

    1. Please first create a Report from Sales Force with the fields shown below.
    2. From Sales Force, Run the report, and then export the report as a csv file
    3.  Upload the CSV file to ABI

    ### Report Type (Custom) RAs, ABs & Activities with Case Linkage





    Data should contain atleast these columns:
    If you have more columns, that's not a problem.  More the merrier.
    Order of the columns don't matter.
    

    * C360 CS Task ID
    * Created Date
    * Name
    * Account Name: Account Name
    * Solution Domain: Solution Domain
    * Product
    * Created By: Full Name
    * Additional Details
    * Action Taken By: Full Name
    * Action Comments
    * Use Case
    * Use Case Stage at Creation
    * Lifecycle Stage
    * Status
    * Creator Role
    * Age in Days
    * Task Type

    from `RA To Case Linkage`

    * Related Case Number: Case Number
    * Case Status
    * Created By: Full Name



    filter on:

    * Show Me: All actions, ab's and activities
    * Created Date: All Time
    * Record Type: equals Adoption Barrier
    """)

st.markdown('---')

# remember to initialize session state
if "data" not in st.session_state:
  st.session_state.data = None  

# Sidebar for file upload
with st.sidebar:
    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")


# After Uploading file
if uploaded_file is not None:
    try:
        # First try decoding with UTF-8
        decoded_content = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        try:
            # Fallback to ISO-8859-1 if UTF-8 fails
            decoded_content = uploaded_file.getvalue().decode("ISO-8859-1")
        except UnicodeDecodeError:
            st.error("Failed to decode the file with ISO-8859-1 encoding. Please check the file encoding.")
            raise



    # Use StringIO to turn decoded content into a stream
    data_stream = io.StringIO(decoded_content)

    # Read the content into a DataFrame, assuming it's a CSV file
    data = pd.read_csv(data_stream, low_memory=False)

    # Rename some columns
    data.rename(columns={'Use Case Stage at Creation': 'OriginalPitStop', 'Lifecycle Stage': 'CurrentPitstop'}, inplace=True)

    # Regular expression to capture 'INC' which is the prefix for Service Now Tickets followed by digits
    regex_pattern = r'(INC\d+)'

    # Extract the pattern and create a new column 'extracted_INC'
    data['ServiceNowID'] = data['Additional Details'].str.extract(regex_pattern)

    # Analysis for Jessee
    status = data[data['Status'].isin(['Open', 'Closed', 'Resolved', 'Closed - Customer Engaged', 'Closed - Other','Pending CSE Confirmation' ])]

    # Base URL format
    base_url = "https://ciscosales.lightning.force.com/lightning/r/C360_CS_Task__c/{}/view?"

    # Create new column 'SF_URL' with the formatted URLs
    data['SF_URL'] = data['C360 CS Task ID'].apply(lambda x: base_url.format(x))
    data.insert(1, 'SF_URL', data.pop('SF_URL')) # move the column to 2nd column
    
    # Create Case# to string
    column_name = 'Related Case Number: Case Number'
    if column_name in data.columns:
        data['Related Case Number: Case Number'] = data['Related Case Number: Case Number'].astype('Int64').astype(str)
        


    # Assuming df is your dataframe
    data = data.replace('nan', None)
        
    
    # preserve data session
    st.session_state.data = data
    

# Sidebar for navigation
st.sidebar.header("Navigation")
options = ["Overview of Data", "Service Now Tickets Associations", "Show Task Types", "Show Opened Duration Statistics",
           'Show & Search Customers', 'Show SForce Incidents more than or equal to 90 days for Status=Open',
           'Show SForce Incidents less than or equal to 30 days for Status=Open', 'CS Console Cases assoicated with Open State Adoption Barrier Incidents',
           'Draw Bar Plot highlghting Months where Cases were created', "Analysis (Top5) on `Action Taken by:` for SForce Incidents",
           "Who did What `Action Taken by:` ", "Analyze for Action Comments Status. Use BridgeIT by Cisco for Recommendations", "Search Records with Cisco BridgeIT using Keywords"]
choice = st.sidebar.radio("Go to", options)


 #######(1) st.button("**Show Data Frame Head**"):

def show_dfhead():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds

    # Code Block
    if not data.empty:
      dh = data.head()
      ds = data.shape[0]
    else:
      st.markdown('**have you uploaded the csv file ?**')
      dh = pd.DataFrame()
      ds = data.shape[0]





    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return dh, ds


if choice == "Overview of Data":

    # Create a button to trigger the processing
    if st.button("**Show Data Frame Head & Counts of Records**"):
        # Call the function when the button is clicked
        dh, ds = show_dfhead()
        st.write(dh)
        st.write(f'**Number of Records:**  {ds:,}')
        st.markdown(f'**Breakdown of Solution Domains:**')
        st.write(data['Solution Domain: Solution Domain Name'].value_counts().to_frame())




######(2) st.button("**Service Now Tickets Associated with the SForce Incidents**"):

def show_serviceNow():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds

    # Code Block
    if not data.empty:
      open_data = data[data['Status'] == 'Open']
      svcNow = data[data['ServiceNowID'].notna()]
      svcNowO = data[(data['ServiceNowID'].notna()) & (data['Status'] == 'Open')]
      svcNowNO = data[(data['ServiceNowID'].notna()) & (data['Status'] != 'Open')]
    else:
      st.markdown('**have you uploaded the csv file ?**')
      svcNow = pd.DataFrame()





    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return svcNowO, svcNowNO

if choice == 'Service Now Tickets Associations':
    

    # Create a button to trigger the processing
    if st.button("**Service Now Tickets Associated with the SForce Incidents**"):
        # Call the function when the button is clicked
        svcNowO, svcNowNO = show_serviceNow()
        st.markdown('**These Service Now Tickets were opened for AB incidents that still are in Open State, broken down by Solutions Domain:**')
        st.write('Scroll to **last column**  to see the Service Now Ticket ID')
        #st.write(svcNowO)

        # break by category
        unique_values = svcNowO['Solution Domain: Solution Domain Name'].unique() 
        
        # Creating a dictionary to hold DataFrames for each category
        dfs = {value: svcNowO[svcNowO['Solution Domain: Solution Domain Name'] == value] for value in unique_values}

        # Printing each DataFrame
        for key, dataframe in dfs.items():
            st.write(f"**DataFrame for Solutions Domain:** {key}")
            st.write(f"**record count:** {dataframe.shape[0]:,}")
            st.write(dataframe)
            #print("\n")        
        
        st.write(f'**Number of Records:**  {svcNowO.shape[0]:,}')
        
        st.markdown("---")
        
        st.markdown('**These Service Now Tickets were opened for AB incidents that are NOT any more in Open State**')
        st.write('Scroll to **last column** to see the Service Now Ticket ID')
        st.write(svcNowNO)
        st.write(f'**Number of Records:**  {svcNowNO.shape[0]:,}')





######(3) st.button("**Show Task Types**"):

def show_taskType():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds


    # Code Block
    if not data.empty:
      taskT =  data['Task Type']
    else:
      st.markdown('**have you uploaded the csv file ?**')
      taskT = pd.DataFrame()





    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return taskT

if choice == 'Show Task Types':
        

    # Create a button to trigger the processing
    if st.button("**Show Task Types**"):
        # Call the function when the button is clicked
        taskT = show_taskType()
        st.write(taskT)
        st.markdown('**taskTypes broken down by counts**')
        taskTCounts = (data['Task Type'].value_counts().to_frame())
        st.write(taskTCounts)


######(4) st.button("**Show Opened Duration Statistics**"):

def show_openFor():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds


    # Code Block
    if not data.empty:
      # openfor =  data['Age in Days'].describe()
      # dataStat = data['Age in Days']
      open_data = data[data['Status'] == 'Open']
      openfor =  open_data['Age in Days'].describe()
      dataStat = open_data['Age in Days']

    else:
      st.markdown('**have you uploaded the csv file ?**')
      openfor = pd.DataFrame()
      dataStat = openfor['Age in Days']






    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return openfor, dataStat, open_data

if choice == 'Show Opened Duration Statistics':


    # Create a button to trigger the processing
    if st.button("**Show Opened Duration Statistics**"):
        openfor, dataStat, open_data = show_openFor()
        # Call the function when the button is clicked
        st.markdown('**Note:** These are for Open Sales Forces Incidents only')
        st.write(openfor)


        # Create a box plot with plotly
        box1 = px.box(open_data, y='Age in Days', points='all', title='Distribution of Age in Days for Status=Open')

        # Update the layout to add a title
        box1.update_layout(
            title='Box Plot: Distribution of Age in Days',
            title_x=0.5,  # Centers the title
            width = 1600, # pixels
            height = 800  # pixels
        )

        st.plotly_chart(box1)

        
        # Create a figure and one subplot using sns new method
        fig, ax = plt.subplots(figsize=(5,2))
        # Create the histplot
        sns.histplot(data=open_data, x='Age in Days', color='green', kde=True, ax=ax);
        # Customize the plot
        ax.set_title('Histogram:Age Of SForce Cases')
        # display in streamlit like plt.show()                             
        st.pyplot(fig)


##

######(5) st.button("**Show Customers**"):

def show_customer():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds


    # Code Block
    if not data.empty:
      cust =  data['Account Name: Account Name'].unique()

    else:
      st.markdown('**have you uploaded the csv file ?**')
      cust = pd.DataFrame()






    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return cust


# if choice == 'Show Customers':

#     # Create a button to trigger the processing
#     if st.button("**Show Customers**"):
#         cust = show_customer()
#         # Call the function when the button is clicked
#         st.write(cust)



######(6) st.button("Search Accounts by Customer Name"):

# st.markdown('**Search By Customer Name Below:**')
# # Create an input box to get the customer's name outside the function
# custname = st.text_input("Please enter the customer's name:")

def search_customer(custname):
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)


    search_accounts  = pd.DataFrame()
    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds

    # Code Block
    custname = custname
    if not data.empty:
        if custname:  # Ensure the input is not empty
            search_accounts = data[data['Account Name: Account Name'] == custname]
        if search_accounts.empty:
            st.markdown('**No account found with this name.**')
    else:
        st.markdown('**Have you uploaded the CSV file?**')



    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return search_accounts



if choice == 'Show & Search Customers':
    # Create a button to trigger the processing
    if st.button("**Show Customers**"):
        cust = show_customer()
        # Call the function when the button is clicked
        st.write(cust)


    # Create a button to trigger the processing
    st.markdown('**Search By Customer Name Below:**')
    # # Create an input box to get the customer's name outside the function
    custname = st.text_input("Please enter the customer's name:")
    if st.button("Search Accounts by Customer Name"):
        acct = search_customer(custname)  # Call the function when the button is clicked
        st.write("**No results to display.**")
        if not acct.empty:
            st.write(acct)
        else:
            st.write("**No results to display.**")

###

######(7) st.button("**Show Customers who have SForce Incidents more than 90 days for Status=Open**")

def show_age90more():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds


    # Code Block
    if not data.empty:
      open_data = data[data['Status'] == 'Open']
      # data90 = open_data[open_data['Age in Days'] > 90][[ 'Account Name: Account Name', 'Age in Days', 'Use Case' , 'OriginalPitStop', 'CurrentPitstop' ]]
      data90 = open_data[open_data['Age in Days'] >= 90]
    else:
      st.markdown('**have you uploaded the csv file ?**')
      data90 = pd.DataFrame()






    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return data90

if choice == 'Show SForce Incidents more than or equal to 90 days for Status=Open':

    # Create a button to trigger the processing
    if st.button("**Show SForce Incidents more than or equal to 90 days for Status=Open**"):
        data90 = show_age90more()
        # Call the function when the button is clicked
        st.write(f"**SForce Incidents more than or equal to 90 days for Status=Open, Count: {data90.shape[0]:,}**")
        #st.write(data90)
        
        # break by category
        break_by_soldom(data90)                   


        # loop for seaborne countplot(barplot)
        # Define the order of categories
        category_order = ['Purchase', 'Onboard', 'Implement', 'Use', 'Engage', 'Adopt', 'Recommend']

        cols = ['OriginalPitStop', 'CurrentPitstop']

        # Determine the global maximum count for y-axis limit
        max_count = 0
        for variable in cols:
            max_count = max(max_count, data90[variable].value_counts().max())

        # Create a figure to host the subplots
        # plt.figure(figsize=(25, 19))
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 12))  # Adjusted for two plots side by side

        # Loop through the numeric columns and create a subplot for each one
        for i, variable in enumerate(cols):
            # plt.subplot(4, 4, i + 1)  # Create a subplot in a 4x4 layout, i+1 means where to put the grid
            ax = axes[i] # changed here
            # sns.countplot(x=data90[variable], hue= data90[variable], order=category_order)  # Use Seaborn's boxplot function # plt.boxplot(data[variable], whis=1.5) -would be if using matplotlib
            sns.countplot(data=data90, x=variable, hue=variable, order=category_order, ax=ax) # changed here
            # plt.title(f'{variable} for SForce Cases more than 90 days', fontsize = 15)  # Set the title for each subplot
            # plt.ylabel("") # moving ylabel, because by default, seems like seaborn puts it in and it's distracting here
            ax.set_title(f'{variable} for SForce Cases more than 90 days', fontsize = 30) # changed here
            ax.set_ylabel("") # changed here
            ax.set_xlabel(f'{variable}', fontsize = 25) # changed the xlablel
            ax.set_ylim(0, max_count)  # Set the same y-axis limit for all plots, changed here
            ax.yaxis.set_tick_params(width=3, length=3, color='gray', labelsize=9, labelcolor='black')
            ax.tick_params(axis='y', labelsize=20)  # Set y-axis tick labels to larger font size # new
            ax.tick_params(axis='x', labelsize=20)  # Set x-axis tick labels to larger font size # new



        plt.tight_layout()  # Adjust subplots to fit into figure area.

        # Display the figure in Streamlit
        # st.pyplot()
        st.pyplot(fig)



######(8) st.button("**Show SForce Incidents less than or equal to 30 days for Status=Open**"):

def show_age30less():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds


    # Code Block
    if not data.empty:
      open_data = data[data['Status'] == 'Open']
      #data30 = open_data[open_data['Age in Days'] < 30][[ 'Account Name: Account Name', 'Age in Days', 'Use Case' , 'OriginalPitStop', 'CurrentPitstop' ]]
      data30 = open_data[open_data['Age in Days'] <= 30]

    else:
      st.markdown('**have you uploaded the csv file ?**')
      data30 = pd.DataFrame()






    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return data30

if choice == 'Show SForce Incidents less than or equal to 30 days for Status=Open':
    # Create a button to trigger the processing
    if st.button("**Show SForce Incidents less than or equal to 30 days for Status=Open**"):
        data30 = show_age30less()
        # Call the function when the button is clicked
        st.write(f"Show SForce Incidents less than or equal to 30 days for Status=Open, Count: {data30.shape[0]:,}")
        #st.write(data30)
        
        # break by category
        break_by_soldom(data30)             
        

        # loop for seaborne countplot(barplot)
        # Define the order of categories
        category_order = ['Purchase', 'Onboard', 'Implement', 'Use', 'Engage', 'Adopt', 'Recommend']

        cols = ['OriginalPitStop', 'CurrentPitstop']

        # Determine the global maximum count for y-axis limit, New code added here to get the maximum value of y axis
        max_count = 0
        for variable in cols:
            max_count = max(max_count, data30[variable].value_counts().max())

        # Create a figure to host the subplots
        # plt.figure(figsize=(30, 24))
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 12))  # Changed from above, Adjusted for two plots side by side

        # Loop through the numeric columns and create a subplot for each one
        for i, variable in enumerate(cols):
            # plt.subplot(4, 4, i + 1)  # Create a subplot in a 4x4 layout, i+1 means where to put the grid
            ax = axes[i] # changed from above
            # sns.countplot(x=data30[variable], hue= data30[variable], order=category_order)  # Use Seaborn's boxplot function # plt.boxplot(data[variable], whis=1.5) -would be if using matplotlib
            sns.countplot(data=data30, x=variable, hue=variable, order=category_order, ax=ax) # changed from above
            # plt.title(f'{variable} for SForce Cases less than 30 days')  # Set the title for each subplot
            # plt.ylabel("") # moving ylabel, because by default, seems like seaborn puts it in and it's distracting here
            # ax.set_title(f'{variable} for SForce Cases less than 30 days') # changed from above
            # ax.set_ylabel("") # changed from above
            # ax.set_ylim(0, max_count)  # Set the same y-axis limit for all plots
            ax.set_title(f'{variable} for SForce Cases less than 30 days', fontsize = 30) # changed here
            ax.set_ylabel("") # changed here
            ax.set_xlabel(f'{variable}', fontsize = 25) # changed the xlablel
            ax.set_ylim(0, max_count)  # Set the same y-axis limit for all plots, changed here
            ax.yaxis.set_tick_params(width=3, length=3, color='gray', labelsize=9, labelcolor='black')
            ax.tick_params(axis='y', labelsize=20)  # Set y-axis tick labels to larger font size # new
            ax.tick_params(axis='x', labelsize=20)  # Set x-axis tick labels to larger font size # new




        plt.tight_layout()  # Adjust subplots to fit into figure area.

        # Display the figure in Streamlit
        # st.pyplot()
        st.pyplot(fig) # changed from above


######(9) st.button("**CS Consle Cases Opened for Adpoption Barrier Incident,  AB state=open in AB**"):

def show_casesO():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds


    # Code Block
    open = data[(data['Status'] =='Open')]
    column_name = 'Related Case Number: Case Number'
    if column_name in open.columns:
      case_opened = open[open[column_name] != '<NA>']
      

    else:
      st.markdown('**have you uploaded the csv file ?**')
      case_opened = 'your CSV file did not include Column: Related Case Number: Case Number '






    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return case_opened




# # Create a button to trigger the processing
# if st.button("**CS Consle Cases Opened for Adpoption Barrier Incident,  AB state=open in AB**"):
#     case_opened = show_casesO()
#     # Call the function when the button is clicked
#     st.write((f"**CS Console Cases for OPEN State Adoption Barrier**: {len(case_opened):,}"))
#     st.write(case_opened)



######(10) st.button("**CS Consle Cases NOT Opened for Adpoption Barrier, AB state=open in AB**"):

def show_casesNO():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds


    # Code Block
    open = data[(data['Status'] =='Open')]
    column_name = 'Related Case Number: Case Number'
    if column_name in open.columns:
      case_unopened = open[open[column_name].isna()]
      case_unopened = open[open[column_name] == '<NA>']

    else:
      st.markdown('**have you uploaded the csv file ?**')
      case_unopened = 'your CSV file did not include Column: Related Case Number: Case Number '






    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return case_unopened




# # Create a button to trigger the processing
# if st.button("**CS Consle Cases NOT Opened with Adpoption Barrier, for state=open in AB**"):

#     # Call the function when the button is clicked
#     case_unopened = show_casesNO()
#     st.write(f"**CS Console Cases for NOT OPEN State Adoption Barrier**: {len(case_unopened)}")
#     st.write(case_unopened)




######(11) (11) st.button("**CS Consle Cases: Opened/UnOpened Plot**"):

def show_casesO_NO():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds


    # Code Block
    open = data[(data['Status'] =='Open')]
    column_name = 'Related Case Number: Case Number'
    if column_name in open.columns:
      case_opened = open[open[column_name] != '<NA>']
      case_unopened = open[open[column_name] == '<NA>']

    else:
      st.markdown('**have you uploaded the csv file ?**')
      case_unopened = 'your CSV file did not include Column: Related Case Number: Case Number '
      case_opened = 'your CSV file did not include Column: Related Case Number: Case Number '






    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return case_opened, case_unopened

if choice == 'CS Console Cases assoicated with Open State Adoption Barrier Incidents':
    
    
    
    # Create a button to trigger the processing
    if st.button("**CS Consle Cases Opened for Adpoption Barrier Incident,  AB state=open in AB**"):
        case_opened = show_casesO()
        if type(case_opened) !=str:
            # Call the function when the button is clicked
            st.write((f"**CS Console Cases for OPEN State Adoption Barrier**: {len(case_opened)}"))
            st.write(case_opened)
        else:
            st.write(case_opened)

    # Create a button to trigger the processing
    if st.button("**CS Consle Cases NOT Opened with Adpoption Barrier, for state=open in AB**"):

        # Call the function when the button is clicked
        case_unopened = show_casesNO()
        if type(case_unopened) !=str:
            st.write(f"**CS Console Cases for NOT OPEN State Adoption Barrier**: {len(case_unopened)}")
            st.write(case_unopened)
        else:
            st.write(case_unopened)            


    # Create a button to trigger the processing
    if st.button("**CS Consle Cases: Opened/UnOpened Plot**"):
        opened, not_opened = show_casesO_NO()
        if type(opened) != str:
            # Call the function when the button is clicked

            dict1 = {'case': ['opened', 'not_opened'], "count": [len(opened), len(not_opened)]}
            cases = pd.DataFrame(dict1)

            # Plotly Graph
            pxbar = px.bar(cases,x='case', y='count' ,color ='count')

            # Update the layout to add a title
            pxbar.update_layout(
                title='Bar Plot showing cases opened/unopened',
                title_x=0.5,  # Centers the title
                width = 1600,
                height = 800
            )

            # Update axes with customized font for titles
            pxbar.update_yaxes(title_text='Count of Cases', title_font=dict(family='Arial', size=14, color='Green'))

            st.plotly_chart(pxbar)
        else:
            st.write(opened)




###
######(12) st.button("**Draw Bar Plot highlghting Months where Cases were created**"):

def show_CreateMonth():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds


    # Code Block
    if not data.empty:

      # Convert 'Created Date' from string to datetime
      data['Created Date'] = pd.to_datetime(data['Created Date']) # convert data string to datetime64
      # Extract the month name from the 'Created Date'
      data['CreatedMonth'] = data['Created Date'].dt.strftime('%B')
      # Filter the DataFrame to keep rows where 'Status' is neither 'Closed' nor 'Pending'
      data1 = data[~data['Status'].isin(['Closed', 'Pending'])]
      # st.write(data1)

    else:
      st.markdown('**have you uploaded the csv file ?**')
      data1 = pd.DataFrame()





    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return data1


if choice == 'Draw Bar Plot highlghting Months where Cases were created':

    # Create a button to trigger the processing
    if st.button("**Draw Bar Plot highlghting Months where Cases were created**"):
        # Call the function when the button is clicked
        data1 = show_CreateMonth()


        # Create the countplot using Plotly Express
        fig = px.histogram(data1, x='Status', color='CreatedMonth',
                        title='Count of Status by Created Month',
                        labels={'count': 'Frequency'},  # Custom label for legend
                        category_orders={"Status": ["Open", "Resolved", "Cancelled"]},  # Optional: custom order for x-axis categories
                        barmode='group')  # 'group' for grouped bar chart

        # Adjust figure size via the layout
        fig.update_layout(
            xaxis_title="Status",
            yaxis_title="Frequency",
            legend_title="Created Month",
            plot_bgcolor='white',
            width=1800,  # Set the width of the figure
            height=900   # Set the height of the figure
        )


        # Update axes and gridline styles
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='grey')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

        # Show the figure
        # fig.show()
        st.plotly_chart(fig) # changed from above

################
################



######(13) st.button("** Analysis on `Action Taken by:` for SForce Incidents**"):

def show_action():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds


    # Code Block
    if not data.empty:
      status_counts = status.groupby(['Action Taken By: Full Name', 'Status', ]).size().unstack(fill_value=0).reset_index()
      open_sort         = status_counts[status_counts['Open'] !=0].sort_values('Open', ascending=False)
      # Check if the 'Resolved' column is present
      if 'Resolved' in status_counts.columns:
          resolved_sort = status_counts.sort_values(by='Resolved', ascending=False)
      else:
          resolved_sort = 'Your data does not have any State that contains Resolved. You probably only captured data with State == Open.'
          
      # Check if the 'Pending CSE Confirmation' column is present
      if 'Pending CSE Confirmation' in status_counts.columns:
          pending_cse_sort = status_counts.sort_values(by='Pending CSE Confirmation', ascending=False)
      else:
          pending_cse_sort = 'Your data does not have any State that contains pending_cse. You probably only captured data with State == Open.'     

    else:
      st.markdown('**have you uploaded the csv file ?**')
      open_sort = pd.DataFrame()
      resolved_sort = pd.DataFrame()
      pending_cse_sort =  pd.DataFrame()






    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return open_sort, resolved_sort, pending_cse_sort




# Initialize session state variables if they don't exist
if 'show_analysis' not in st.session_state:
    st.session_state['show_analysis'] = False
if 'action_taker_name_open' not in st.session_state:
    st.session_state['action_taker_name_open'] = ""
if 'action_taker_name_resolved' not in st.session_state:
    st.session_state['action_taker_name_resolved'] = ""
if 'action_taker_name_pending' not in st.session_state:
    st.session_state['action_taker_name_pending'] = ""


if choice == "Analysis (Top5) on `Action Taken by:` for SForce Incidents":
    

    # Main button to trigger the processing
    if st.button("**Analysis (Top5) on `Action Taken by:` for SForce Incidents**", key='main_analysis'):
        st.session_state.show_analysis = not st.session_state.show_analysis

    # Display the analysis section if the state is active
    if st.session_state.show_analysis:
        open_sort, resolved_sort, pending_cse_sort = show_action()

        st.markdown("**Top 5 for `Action Taken by:` State is Open:**")
        st.write(open_sort.head())
        #st.write(f"**Total Count: `Action Taken by:` State is Open** {open_sort.shape[0]} records")
        st.write(f"**Total Count: `Action Taken by:` State is Open** {open_sort['Open'].sum()} records")
        st.markdown("---")

        #---
        st.markdown("**Top 5 for `Action Taken by:` State is resolved:**")
        if type(resolved_sort) == str:
            st.write(resolved_sort)
        else:
            st.write(resolved_sort.head())
            st.write(f"**Total Count: `Action Taken by:` State is resolved**  {resolved_sort['Resolved'].sum()} records")
        st.markdown("---")


        #---

        st.markdown("**Top 5 for `Action Taken by:` State is pending:**")
        if type(pending_cse_sort) == str:
            st.write(pending_cse_sort)
        else:
            st.write(pending_cse_sort.head())
            st.write(f"**Total Count: `Action Taken by:` State is Pending CSE Confirmation**  {pending_cse_sort['Pending CSE Confirmation'].sum()} records")
        st.markdown("---")

        #---


    st.markdown('---')  



######(14) st.button("****Show SForce Incidents Workload for `Action Taken by:`**"):

def show_workload():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds


    # Code Block
    if not data.empty:
      status_counts = status.groupby(['Action Taken By: Full Name', 'Status', ]).size().unstack(fill_value=0).reset_index()
      open_sort = status_counts.sort_values(by='Open', ascending=False)
      
      # Check if the 'Resolved' column is present
      if 'Resolved' in status_counts.columns:
          resolved_sort = status_counts.sort_values(by='Resolved', ascending=False)
      else:
          resolved_sort = 'Your data does not have any State that contains Resolved.  You probably only captured data with State == Open.'
          
      # Check if the 'Pending CSE Confirmation' column is present
      if 'Pending CSE Confirmation' in status_counts.columns:
          pending_cse_sort = status_counts.sort_values(by='Pending CSE Confirmation', ascending=False)
      else:
          pending_cse_sort = 'Your data does not have any State that contains pending_cse. You probably only captured data with State == Open.' 
             

      folks = pd.DataFrame(status_counts['Action Taken By: Full Name'].unique(), columns=['ActionTaker'])

    else:
      st.markdown('**have you uploaded the csv file ?**')
      open_sort = pd.DataFrame()
      resolved_sort = pd.DataFrame()
      pending_cse_sort =  pd.DataFrame()
      folks = pd.DataFrame()


    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return open_sort, resolved_sort, pending_cse_sort, folks




# Initialize session state variables if they don't exist
if 'show_folks_main' not in st.session_state:
    st.session_state['show_folks_main'] = False
if 'at_string' not in st.session_state:
    st.session_state['at_string'] = False
if 'at_string_O' not in st.session_state:
    st.session_state['at_string_O'] = False
if 'at_string_R' not in st.session_state:
    st.session_state['at_string_R'] = False
if 'at_string_P' not in st.session_state:
    st.session_state['at_string_P'] = False


if choice == "Who did What `Action Taken by:` ":
    

    # Main button to trigger the processing
    if st.button("**Show SForce Incidents Workload for `Action Taken by:`**", key='main_show_folks'):
        st.session_state.show_folks_main = not st.session_state.show_folks_main

    # Display the analysis section if the state is active
    if st.session_state.show_folks_main:
        open_sort, resolved_sort, pending_cse_sort, folks = show_workload()

        st.markdown("**Names of `Action Takers`**")
        st.write(folks)
        st.write(f"**Total Count of : `Action Taken Names:`** {folks.shape[0]:,}")
        st.markdown("   ")
        st.markdown("**SearchAction Taker name by partial string match:**")
        #---

        # Partial String Search for Action Taker (not keeping state for button)
        q_AT_String = st.text_input("Please enter `partial string of Action Taker's name for search:", value=st.session_state.at_string)
        if st.button("**Search by Action Taker  Name by `partial string match`**"):
            result = folks[folks['ActionTaker'].str.contains(f'{q_AT_String}', case=False)]
            st.write(result)

        #---

        # Show Open SF Incidents for Action Taker (not keeping state for button)
        q_AT_String_O = st.text_input("Please enter `Full Name of Action Taker's to see open incidents:", value=st.session_state.at_string_O)
        if st.button("**Search `Open SForce Incidents` for  Action Taker**"):   
            result1 = data[(data['Action Taken By: Full Name'] == f'{q_AT_String_O}')  &  (data['Status'] == 'Open')].reset_index()
            st.write(f'**Total open for {q_AT_String_O}:** {len(result1)}')
            st.write(result1)

        # Show Resolved SF Incidents for Action Taker (not keeping state for button)
        q_AT_String_R = st.text_input("Please enter `Full Name of Action Taker's to see Resolved incidents:", value=st.session_state.at_string_R)
        if st.button("**Search `Resolved SForce Incidents` for  Action Taker**"):
            if type(resolved_sort) == str:
                st.write(resolved_sort)
            else:
                result2 = data[(data['Action Taken By: Full Name'] == f'{q_AT_String_R}')  &  (data['Status'] == 'Resolved')].reset_index()
                st.write(f'**Total Resolved for {q_AT_String_R}:** {len(result2)}')
                st.write(result2)


        # Show Pending SF Incidents for Action Taker (not keeping state for button)
        q_AT_String_P = st.text_input("Please enter `Full Name of Action Taker's to see Pending incidents:", value=st.session_state.at_string_P)
        if st.button("**Search `Pending SForce Incidents` for  Action Taker**"):
            if type(pending_cse_sort) == str:
                st.write(pending_cse_sort)
            else:
                result2 = data[(data['Action Taken By: Full Name'] == f'{q_AT_String_P}')  &  (data['Status'] == 'Pending CSE Confirmation')].reset_index()
                st.write(f'**Total Pending for {q_AT_String_P}:** {len(result2)}')
                st.write(result2)


        #---


    st.markdown('---')      
 

#######(15)  st.button("**Analyze SForce Incidents for `Action Comments:`. NOTE: These are for Status == Open**"

def show_action_comments():
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds


    # Code Block
    open1 = data[(data['Status'] =='Open')]
    column_name = 'Action Comments'
    if column_name in open1.columns:
        filtered_data = open1[open1['Action Comments'].str.contains('Status:', na=False) ].reset_index(drop=True)
        statusFD_P=filtered_data[filtered_data['Action Comments'].str.contains('Pending feature implementation', case=False, na=False)] # Pending feature implementation
        statusFD_IP=filtered_data[filtered_data['Action Comments'].str.contains(r'in\s*progress', case=False,  na=False)] # In Progress
        statusFD_NM=filtered_data[filtered_data['Action Comments'].str.contains(r'need\s*more\s*information', case=False, na=False)] # need more information
        statusFD_NF=filtered_data[filtered_data['Action Comments'].str.contains(r'No\s*Further\s*Action', case=False,  na=False)] # No Further Action
        statusFD_RO=filtered_data[filtered_data['Action Comments'].str.contains(r'Reached\s*out', case=False,  na=False)] # Reached Out
        statusFD_PBF=filtered_data[filtered_data['Action Comments'].str.contains(r'Pending\s*Bugfix', case=False,  na=False)] # Pending Bugfix
        statusFD_RES=filtered_data[filtered_data['Action Comments'].str.contains('Resolved', case=False,  na=False)] # Resolved
        statusFD_OH=filtered_data[filtered_data['Action Comments'].str.contains(r'On\s*Hold', case=False,  na=False)] # On Hold

        
        # Now create DF whre none of the above conditions match
        # Combine all the filtered data indices to exclude them for statusFD_NONE
        combined_indices = pd.concat([
        statusFD_P, statusFD_IP, statusFD_NM, statusFD_NF, statusFD_RO, 
        statusFD_PBF, statusFD_RES, statusFD_OH
        ]).index.unique()

        # Create statusFD_NONE by excluding the combined indices
        statusFD_NONE = filtered_data.drop(combined_indices).reset_index(drop=True)

     

    else:
      st.markdown('**have you uploaded the csv file ?**')
      open1 = pd.DataFrame()
      statusFD_P = "no Action Comments Column was in your CSV File"
      statusFD_IP =  "no Action Comments Column was in your CSV File"
      filtered_data = "no Action Comments Column was in your CSV File"
      statusFD_NF= "no Action Comments Column was in your CSV File"
      statusFD_NM ="no Action Comments Column was in your CSV File"
      statusFD_RO = "no Action Comments Column was in your CSV File"
      statusFD_PBF = "no Action Comments Column was in your CSV File"
      statusFD_RES = "no Action Comments Column was in your CSV File"
      statusFD_OH = "no Action Comments Column was in your CSV File"
      statusFD_NONE = "no Action Comments Column was in your CSV File"
      



    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return filtered_data, statusFD_P, statusFD_IP,  statusFD_NF, statusFD_RO, statusFD_NM, statusFD_PBF, statusFD_RES, statusFD_OH, statusFD_NONE, open1



if choice == "Analyze for Action Comments Status. Use BridgeIT by Cisco for Recommendations":
    

    

    # Initialize session state variables if they don't exist
    if 'ActionComments_main' not in st.session_state:
        st.session_state['ActionComments_main'] = False

    # Main button to trigger the processing
    if st.button("**Analyze SForce Incidents for `Action Comments Status:` NOTE: These are for Sales Force Status == Open**", key='main_ActionComments'):
        st.session_state.ActionComments_main = not st.session_state.ActionComments_main
                      

    # Display the analysis section if the state is active
    if st.session_state.ActionComments_main:
        filtered_data, statusFD_P, statusFD_IP,  statusFD_NF, statusFD_RO, statusFD_NM, statusFD_PBF, statusFD_RES, statusFD_OH, statusFD_NONE, open1 = show_action_comments()
        
        if type(filtered_data) != str:

            st.markdown("**Action Comments Status:`Status is present`:**")
            #st.write(filtered_data)
            st.write(f"**Total Count of : Action Comments Status: `Status is present`** {filtered_data.shape[0]:,}") 
            
            # break by category
            break_by_soldom(filtered_data)          
            
            st.markdown("---")

            st.markdown("**Action Comments Status:`Pending Feature Implementation`**")
            #st.write(statusFD_P)
            st.write(f"**Total Count of : `Action Comments Status: Pending Feature Implementation`** {statusFD_P.shape[0]:,}")
            
            # break by category
            break_by_soldom(statusFD_P)      
            
            
            st.markdown("---")

            st.markdown("**Action Comments Status:`In Progress`**")
            # st.write(statusFD_IP)
            st.write(f"**Total Count of : Action Comments Status: `In Progress`** {statusFD_IP.shape[0]:,}")
            
            # break by category
            break_by_soldom(statusFD_IP)                
                        
            
            st.markdown("---")

            st.markdown("**Action Comments Status:`Need more information`**")
            # st.write(statusFD_NM)
            st.write(f"**Total Count of : Action Comments Status:  `Need more information`** {statusFD_NM.shape[0]}")
            
            # break by category
            break_by_soldom(statusFD_NM)              
            
            
            st.markdown("---")

            st.markdown("**Action Comments Status:`Reached Out`**")
            # st.write(statusFD_RO)
            st.write(f"**Total Count of : Action Comments Status: `Reached Out`** {statusFD_RO.shape[0]}")
            
            # break by category
            break_by_soldom(statusFD_RO)              
                        
            
            st.markdown("---")
            
            st.markdown("**Action Comments Status:`Pending Bugfix`**")
            #st.write(statusFD_PBF)
            st.write(f"**Total Count of : Action Comments Status: `Pending Bugfix`** {statusFD_PBF.shape[0]:,}")
            
            # break by category
            break_by_soldom(statusFD_PBF)              
                                    
                                    
            st.markdown("---")
            
            st.markdown("**Action Comments Status:`Resolved`**")
            #st.write(statusFD_RES)
            st.write(f"**Total Count of : Action Comments: with Status `Resolved`** {statusFD_RES.shape[0]:,}")
            
            # break by category
            break_by_soldom(statusFD_RES)         
            
            st.markdown("---")

            st.markdown("**Action Comments that have Status:`On Hold`**")
            #st.write(statusFD_OH)
            st.write(f"**Total Count of : Action Comments Status: `On Hold`** {statusFD_OH.shape[0]:,}")

            # break by category
            break_by_soldom(statusFD_OH)             
            
            st.markdown("---")

            st.markdown("**Action Comments Status:`No Further Action`**")
            #st.write(statusFD_NF)
            st.write(f"**Total Count of : Action Comments Status:  `No Further Action`** {statusFD_NF.shape[0]:,}")
            
            # break by category
            break_by_soldom(statusFD_NF)             
                        
            
            st.markdown("---")
            
            
            st.markdown("**Action Comments Status:` present, but none of the above creiteras match`**")
            #st.write(statusFD_NONE)
            st.write(f"**Total Count of : Action Comments Status:  `present, but none of the above criterias match`** {statusFD_NONE.shape[0]}")
            
            # break by category
            break_by_soldom(statusFD_NONE)              
            
            st.markdown("---")
            
            st.markdown("**Action Comments Status: `not present`:**")
            no_action = open1[open1['Action Comments'].isnull()]
            #st.write(no_action)
            st.write(f"**Total Count of : Action Comments Status: `not present`** {no_action.shape[0]:,}")
            
            # break by category
            break_by_soldom(no_action)                 
            
            st.markdown("---")
        
        else:
            st.write("**The CSV File did not have column `Action Comments` Please upload CSV file with that column for analysis**")

        ########## LLM Query #################

        # set Preferences, Topic
        # model = 'gpt-4-turbo-preview'  # got this from openAI Model documentation
        model = 'gpt-4o'
        temperature = 0.3 # I don't want it to be creative.  Can be from 0 to 1.  1 is very creative
        max_tokens = 4096  # 500 is pretty standard

                                  
        # Create a button to trigger the processing
        st.markdown("**Run LLM Report: Get Summary by  C360 CS Task ID Using BridgeIT LLM: model = 'gpt-4o'.**'")
        st.markdown("**Kindly do not enter more than 3 comma separated CSS Task IDs for now, since the max tokens is 4096.**'")
        st.markdown("**I will modify this code using langchain at a later time, so we could chain and raise the limit**'")
        # # Create an input box to get the C360 CS Task ID
        c360 = st.text_input("**Please enter the C360 CS Task IDs , separated by commas**")
        llm_access = st.text_input("Please enter the  LLM Access password provided by Soumitra", type="password")
        
        # Initialize session state variables if they don't exist
        if 'summarize_main' not in st.session_state:
            st.session_state['summarize_main'] = False 
        if 'pdf_generated' not in st.session_state:
            st.session_state['pdf_generated'] = False
        if 'pdf_file_path' not in st.session_state:
            st.session_state['pdf_file_path'] = None
        
        random_number = random.randint(1000, 9999)           
        md_file_path = f'llm_report_{random_number}.md'
        pdf_file_path = f'llm_report_{random_number}.pdf'
                
        if st.button("summarize C360 CS Task ID Using BridgeIT by Cisco, model = 'gpt-4o'", key='main_summarize'):
            st.session_state.summarize_main = not st.session_state.summarize_main
            if llm_access == 'soumu101!':
                c360_list = [x.strip() for x in c360.split(',')]
                #query = data.loc[data['C360 CS Task ID'] == c360]
                query = data[data['C360 CS Task ID'].isin(c360_list)]
                json1 = query.to_json(orient='records')
                results = analyze(json1, model, temperature, max_tokens)
                st.write(results)


                # Clean up older reports
                for filename in os.listdir():
                    if filename.startswith('llm_'):
                        os.remove(filename)

                # write the markdown file
                with open(md_file_path, 'a') as f:
                    f.write(results)
                    
                # Clean up \n characters
                with open(md_file_path, 'r') as f:
                    lines = f.readlines()
                    
                
                with open(md_file_path, 'w') as f:
                    for line in lines:
                        new_line = line.replace('\\n', '' )
                        f.write(new_line)
                        
                    
                       
                    
                # chmod 644 for llm_
                for filename in os.listdir():
                    if filename.startswith('llm_'):
                        os.chmod(filename, 0o644)
                                        
                    
                # put in logic here to execute markdown2pdf-pypandoc.py
                import os
                import convert2pdf
                from convert2pdf import convert_md_to_pdf
                # pdf_file_path = f'llm_report_{ramdom_number}.pdf'
                convert_md_to_pdf(md_file_path, pdf_file_path)
                
                # Update Session State
                st.session_state['pdf_generated'] = True
                st.session_state['pdf_file_path'] = pdf_file_path

            else:
                st.write("**You put the wrong OpenAI LLM Access Password: Please Contact Soumitra Mukherji for LLM Access Password**")
                
                
            # download the pdf
            if st.session_state['pdf_generated']:
                getpdf()
                    
        



 #######(16) st.button("**Search Records with Cisco BridgeIT using Keywords**"):

def search_records(user_input):
    import re
    
    # Start the timer
    start_time = time.time()

    # Progress bar implementation, Placeholder for updating progress
    latest_iteration = st.empty()
    bar = st.progress(0)



    # Initially, show some progress to indicate the start of the process
    for i in range(10): # This quickly increments progress to 90%
        latest_iteration.text(f'Progress: {i*10}%') # the % is just printing out Percentage symbol
        bar.progress(i * 10)
        time.sleep(0.1)  # Simulate delay 0.1 seconds

    # Code Block
    if not data.empty:
        if user_input:
            open1 = data[(data['Status'] =='Open')]
            open1_str = open1.astype(str)
            
                        
            all_others = data[(data['Status'] !='Open')]
            all_others_str =  all_others.astype(str)
            #st.write(all_others_str[contains_string.any(axis=1)])
            
            
            #escaped_input = re.escape(user_input)
            search_string = re.sub(r'\s+', r'\\s+', user_input)
            contains_string = open1_str.apply(lambda x: x.str.contains(search_string, case=False, na=False, regex=True))
            matching_rows_open = open1_str[contains_string.any(axis=1)]
            
            
            contains_string1 = all_others_str.apply(lambda x: x.str.contains(search_string, case=False, na=False, regex=True))
            matching_rows_all_others = all_others_str[contains_string1.any(axis=1)]
            


        else:
            st.write(f"**Please Enter Search String**")
            

    # Complete the progress bar when the API call is done
    latest_iteration.text('Progress: 100%')
    bar.progress(100)

    # End the timer and calculate duration
    end_time = time.time()
    duration = end_time - start_time

    # Display the duration
    st.write(f"Process completed in {duration:.2f} seconds.")

    return matching_rows_open, matching_rows_all_others
  
    


if choice ==  "Search Records with Cisco BridgeIT using Keywords":
    
    user_input = st.text_input(f"**Enter search string. For example to find all cases with bugs, search  csc**")
    
    # Initialize session state variables if they don't exist
    if 'search_main' not in st.session_state:
            st.session_state['search_main'] = False 

    # Create a button to trigger the processing
    if st.button("**Search Records using Keywords**", key='main_search'):
        st.session_state.search_main = not st.session_state.search_main
        # Call the function when the button is clicked
        matching_rows_open, matching_rows_all_others = search_records(user_input) 
       
        
        if matching_rows_open is not None and not matching_rows_open.empty:
            matching_rows_open = matching_rows_open.replace('nan', 'None')
            matching_rows_open['Solution Domain: Solution Domain Name'] = matching_rows_open['Solution Domain: Solution Domain Name'].replace('nan', 'None')
            st.write(f"**Matching Rows with ' {user_input} ' for Sales Force Cases still in Open State, broken down by Solutions Domain:**")
            st.write(f"**Number of Total Open Records: {matching_rows_open.shape[0]:,}**")
            # st.write(matching_rows_open)

            # break by category
            break_by_soldom(matching_rows_open)


            
        else:
            st.write(f"**No matching records found for {user_input} where SF cases are in Open State**")
        
        st.markdown('---') 
   

        if matching_rows_open is not None and not matching_rows_open.empty:
            matching_rows_all_others = matching_rows_all_others.replace('nan', 'None')
            matching_rows_all_others['Solution Domain: Solution Domain Name'] = matching_rows_all_others['Solution Domain: Solution Domain Name'].replace('nan', 'None')
            st.write(f"**Matching Rows with {user_input} for Sales Force Cases not in Open State:**")
            st.write(f"**Number of records: {matching_rows_all_others.shape[0]:,}**")
            #st.write(matching_rows_all_others)
            
            # break by category
            break_by_soldom(matching_rows_all_others)            
            
        else:
            st.write(f"**No matching records found for {user_input} where SF cases are not in Open State**")  

    ############## LLM Query ##############
    # set Preferences, Topic
    # model = 'gpt-4-turbo-preview'  # got this from openAI Model documentation
    model = 'gpt-4o'
    temperature = 0.3 # I don't want it to be creative.  Can be from 0 to 1.  1 is very creative
    max_tokens = 4096  # 500 is pretty standard        

                    
    # Create a button to trigger the processing
    st.markdown("**Run LLM Report: Get Summary by  C360 CS Task ID Using OpenAI LLM: model = 'gpt-4o'.**'")
    st.markdown("**Kindly do not enter more than 3 comma separated CSS Task IDs for now, since the max tokens is 4096.**'")
    st.markdown("**I will modify this code using langchain at a later time, so we could chain and raise the limit**'")
    # # Create an input box to get the C360 CS Task ID
    c360 = st.text_input("**Please enter the C360 CS Task IDs , separated by commas**")
    llm_access = st.text_input("Please enter the OpenAI LLM Access password provided by Soumitra", type="password")
    
    #Initialize session state variables if they don't exist
    if 'summarize_main1' not in st.session_state:
        st.session_state['summarize1_main'] = False 

    
    random_number = random.randint(1000, 9999)           
    md_file_path = f'llm_report_{random_number}.md'
    pdf_file_path = f'llm_report_{random_number}.pdf'
            
    if st.button("a summarize C360 CS Task ID Using OpenAI LLM,  model = 'gpt-4o' ", key='main_summarize1'):
        #st.session_state.summarize1_main = not st.session_state.summarize1_main
        if llm_access == 'soumu101!':
            c360_list = [x.strip() for x in c360.split(',')]
            #query = data.loc[data['C360 CS Task ID'] == c360]
            query = data[data['C360 CS Task ID'].isin(c360_list)]
            json1 = query.to_json(orient='records')
            results = analyze(json1, model, temperature, max_tokens)
            st.write(results)


            # Clean up older reports
            for filename in os.listdir():
                if filename.startswith('llm_'):
                    os.remove(filename)

            # write the markdown file
            with open(md_file_path, 'a') as f:
                f.write(results)
                
            # Clean up \n characters
            with open(md_file_path, 'r') as f:
                lines = f.readlines()                           
            with open(md_file_path, 'w') as f:
                for line in lines:
                    new_line = line.replace('\\n', '' )
                    f.write(new_line)
                    
            # chmod 644 for llm_
            for filename in os.listdir():
                if filename.startswith('llm_'):
                    os.chmod(filename, 0o644)
                                    
                
            # put in logic here to execute markdown2pdf-pypandoc.py
            import os
            import convert2pdf
            from convert2pdf import convert_md_to_pdf
            # pdf_file_path = f'llm_report_{ramdom_number}.pdf'
            convert_md_to_pdf(md_file_path, pdf_file_path)
            
            # Update Session State
            st.session_state['pdf_generated'] = True
            st.session_state['pdf_file_path'] = pdf_file_path

        else:
            st.write("**You put the wrong OpenAI LLM Access Password: Please Contact Soumitra Mukherji for LLM Access Password**")
            
                      
                    
        # download the pdf
        if st.session_state['pdf_generated']:
            getpdf()
                    
        
                  
            
            
