import os
# sqlite3 is part of the python standard library
import sqlite3
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from langchain_experimental.utilities.python import PythonREPL
from langchain.tools import Tool
from langchain.agents import load_tools
from langchain import hub
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import yahoo_finance_news

# for the manual agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent


load_dotenv()

from pydantic import BaseModel, Field, ValidationError
from IPython.display import display


load_dotenv()

import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# create manuaul db
# Creating a dummy SQL database for our example. If you have a database file on hand, you can skip this part
conn = sqlite3.connect('sample.db')
# Create a cursor object to interact with the database
cursor = conn.cursor()

# Create a table to store employee information
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Employees (
        EmployeeID INTEGER PRIMARY KEY,
        FirstName TEXT,
        LastName TEXT,
        Birthdate DATE,
        Department TEXT,
        Salary REAL
    )
''')

# Insert some sample data into the Employees table
cursor.executemany('''
    INSERT INTO Employees (FirstName, LastName, Birthdate, Department, Salary)
    VALUES (?, ?, ?, ?, ?)
''', [
    ('John', 'Doe', '1990-05-15', 'HR', 50000.00),
    ('Jane', 'Smith', '1985-12-10', 'Sales', 55000.00),
    ('Bob', 'Johnson', '1992-08-25', 'Engineering', 60000.00),
    ('Alice', 'Brown', '1988-04-03', 'Marketing', 52000.00)
])

# Commit the changes and close the connection
conn.commit()
conn.close()


# In place of file, you can enter the path to your database file
file = "sample.db"
db = SQLDatabase.from_uri(f"sqlite:///{file}")





# Initialize the LLM (GPT-4o)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=1000,
)


tools = ['llm-math']
tools = load_tools(tools,llm)


toolkit = SQLDatabaseToolkit(db=db, llm=llm)
db_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    tools = tools,
    verbose=True
)


# agent_executor.run("Describe the Order related table and how they are related")
db_agent.run("Name the columns present in the database?")

print("-"*100)
db_agent.run("What was the maximum salary paid to an employee ? To whom was it paid?")

print("-"*100)
# Now lets see if our LLM is profecient at arithemtic too
db_agent.run('can you find the square root of the maximum salary paid to an employee?')


#### Outputs: 


# > Entering new SQL Agent Executor chain...
# Action: sql_db_list_tables
# Action Input: ""EmployeesThe database contains a table named "Employees". I should now query the schema of this table to find out the columns present in it.

# Action: sql_db_schema
# Action Input: Employees
# CREATE TABLE "Employees" (
#         "EmployeeID" INTEGER, 
#         "FirstName" TEXT, 
#         "LastName" TEXT, 
#         "Birthdate" DATE, 
#         "Department" TEXT, 
#         "Salary" REAL, 
#         PRIMARY KEY ("EmployeeID")
# )

# /*
# 3 rows from Employees table:
# EmployeeID      FirstName       LastName        Birthdate       Department      Salary
# 1       John    Doe     1990-05-15      HR      50000.0
# 2       Jane    Smith   1985-12-10      Sales   55000.0
# 3       Bob     Johnson 1992-08-25      Engineering     60000.0
# */I now know the final answer.

# Final Answer: The columns present in the "Employees" table are: EmployeeID, FirstName, LastName, Birthdate, Department, and Salary.

# > Finished chain.
# ----------------------------------------------------------------------------------------------------


# > Entering new SQL Agent Executor chain...
# Action: sql_db_list_tables
# Action Input: ""EmployeesI should check the schema of the "Employees" table to find the relevant columns for employee names and salaries. Then, I can construct a query to find the maximum salary and the employee who received it. 

# Action: sql_db_schema
# Action Input: Employees
# CREATE TABLE "Employees" (
#         "EmployeeID" INTEGER, 
#         "FirstName" TEXT, 
#         "LastName" TEXT, 
#         "Birthdate" DATE, 
#         "Department" TEXT, 
#         "Salary" REAL, 
#         PRIMARY KEY ("EmployeeID")
# )

# /*
# 3 rows from Employees table:
# EmployeeID      FirstName       LastName        Birthdate       Department      Salary
# 1       John    Doe     1990-05-15      HR      50000.0
# 2       Jane    Smith   1985-12-10      Sales   55000.0
# 3       Bob     Johnson 1992-08-25      Engineering     60000.0
# */The "Employees" table contains the columns "FirstName", "LastName", and "Salary", which are relevant for finding the maximum salary and the employee who received it. I will construct a query to find the maximum salary and the corresponding employee's name.

# Action: sql_db_query_checker
# Action Input: "SELECT FirstName, LastName, Salary FROM Employees WHERE Salary = (SELECT MAX(Salary) FROM Employees) LIMIT 10;"```sql
# SELECT FirstName, LastName, Salary FROM Employees WHERE Salary = (SELECT MAX(Salary) FROM Employees) LIMIT 10;
# ```The query is correct, and I can now execute it to find the maximum salary and the employee who received it.

# Action: sql_db_query
# Action Input: SELECT FirstName, LastName, Salary FROM Employees WHERE Salary = (SELECT MAX(Salary) FROM Employees) LIMIT 10;[('Bob', 'Johnson', 60000.0), ('Bob', 'Johnson', 60000.0), ('Bob', 'Johnson', 60000.0), ('Bob', 'Johnson', 60000.0)]I now know the final answer. The maximum salary paid to an employee was 60,000, and it was paid to Bob Johnson. 

# Final Answer: The maximum salary paid to an employee was 60,000, and it was paid to Bob Johnson.

# > Finished chain.
# ----------------------------------------------------------------------------------------------------


# > Entering new SQL Agent Executor chain...
# Action: sql_db_list_tables
# Action Input: ""EmployeesThe table "Employees" likely contains information about employee salaries. I should check the schema of the "Employees" table to find the relevant column for salaries. 
# Action: sql_db_schema
# Action Input: Employees
# CREATE TABLE "Employees" (
#         "EmployeeID" INTEGER, 
#         "FirstName" TEXT, 
#         "LastName" TEXT, 
#         "Birthdate" DATE, 
#         "Department" TEXT, 
#         "Salary" REAL, 
#         PRIMARY KEY ("EmployeeID")
# )

# /*
# 3 rows from Employees table:
# EmployeeID      FirstName       LastName        Birthdate       Department      Salary
# 1       John    Doe     1990-05-15      HR      50000.0
# 2       Jane    Smith   1985-12-10      Sales   55000.0
# 3       Bob     Johnson 1992-08-25      Engineering     60000.0
# */The "Salary" column in the "Employees" table contains the salary information. I need to find the maximum salary and then calculate its square root. I will construct a query to do this.

# Action: sql_db_query_checker
# Action Input: SELECT SQRT(MAX(Salary)) AS MaxSalarySqrt FROM Employees;```sql
# SELECT SQRT(MAX(Salary)) AS MaxSalarySqrt FROM Employees;
# ```The query is correct and ready to be executed to find the square root of the maximum salary. 

# Action: sql_db_query
# Action Input: SELECT SQRT(MAX(Salary)) AS MaxSalarySqrt FROM Employees;[(244.94897427831782,)]I now know the final answer.

# Final Answer: The square root of the maximum salary paid to an employee is approximately 244.95.

# > Finished chain.