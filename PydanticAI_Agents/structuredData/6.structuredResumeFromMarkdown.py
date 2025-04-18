import os
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
from dotenv import load_dotenv


"""
This code is used to generate a structured resume for a software engineer.
It uses the OpenAI API to generate the resume based on the data in the markdown file.

Notice here how the agent is generating the resume and then extracting the data from the markdown file.
the main class is Resume and the agent is generating the data and then extracting the data from the markdown file.

agent = Agent(model=model, result_type=Resume, system_prompt=f'You are a technical...')

Output:
16:25:46.536 agent run prompt=Write a resume.
16:25:46.538   preparing model and tools run_step=1
16:25:46.538   model request
Logfire project URL: https://logfire.pydantic.dev/soumukhe/my-first-project
16:26:01.907   handle model response
16:26:01.918 Resume markdown prompt LLM results: full_name='Johnathan M. Smith' contact_email='john.smith@examp...ubernetes Administrator (CKA)', 'Certified ScrumMaster (CSM)']
16:26:01.921 Result type: <class '__main__.Resume'>
--------------------------------
{
    "full_name": "Johnathan M. Smith",
    "contact_email": "john.smith@example.com",
    "phone_number": "+1-555-123-4567",
    "summary": "Seasoned software engineer with over 20 years of experience designing and implementing scalable software solutions. Expertise in leading engineering teams, building distributed systems, and delivering innovative applications across industries such as finance, healthcare, and e-commerce. Proficient in cloud computing, AI/ML integrations, and microservices architecture with a proven track record of driving business value through technology.",
    "experience": [
        {
            "company": "Tech Innovators Inc.",
            "position": "Senior Software Engineer",
            "start_date": "June 2015",
            "end_date": "Present"
        },
        {
            "company": "NextGen Solutions",
            "position": "Software Architect",
            "start_date": "March 2010",
            "end_date": "May 2015"
        },
        {
            "company": "Alpha Development Corp.",
            "position": "Lead Developer",
            "start_date": "January 2005",
            "end_date": "February 2010"
        },
        {
            "company": "CodeSphere LLC",
            "position": "Software Engineer",
            "start_date": "June 2000",
            "end_date": "December 2004"
        }
    ],
    "education": [
        {
            "institution_name": "Massachusetts Institute of Technology",
            "degree": "Master of Science in Computer Science",
            "start_date": "August 1998",
            "end_date": "May 2000"
        },
        {
            "institution_name": "University of California, Berkeley",
            "degree": "Bachelor of Science in Computer Science",
            "start_date": "August 1994",
            "end_date": "May 1998"
        }
    ],
    "skills": [
        "Python",
        "Java",
        "C++",
        "JavaScript",
        "AWS",
        "Azure",
        "Google Cloud",
        "Microservices",
        "Distributed Systems",
        "RESTful APIs",
        "Docker",
        "Kubernetes",
        "Terraform",
        "Agile Development",
        "DevOps",
        "AI/ML Integration"
    ],
    "certifications": [
        "AWS Certified Solutions Architect – Professional",
        "Certified Kubernetes Administrator (CKA)",
        "Certified ScrumMaster (CSM)"
    ]
}
"""

load_dotenv()

# Configure logfire
logfire.configure()

# Define the model
model = OpenAIModel('gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))

# Define the output model
class Experience(BaseModel):
    company: str = Field(..., description="The name of the company.")
    position: str = Field(..., description="The job title held at the company.")
    start_date: str = Field(..., description="The date when you started working at the company.")
    end_date: str = Field(..., description="The date when you left the company. If still employed, use 'Present'.")

class Education(BaseModel):
    institution_name: str = Field(..., description="The name of the educational institution.")
    degree: str = Field(..., description="The degree obtained from the institution.")
    start_date: str = Field(..., description="The date when you started attending school at the institution.")
    end_date: str = Field(..., description="The date when you graduated. If still enrolled, use 'Present'.")

class Resume(BaseModel):
    full_name: str = Field(..., description="The full name of the person on the resume.")
    contact_email: str = Field(..., description="The email address for contacting the person.")
    phone_number: str = Field(..., description="The phone number for contacting the person.")
    
    summary: str = Field(..., description="A brief summary of the person's career highlights.")

    experience: list[Experience] = Field([], description="List of experiences held by the person.")
    education: list[Education] = Field([], description="List of educational institutions attended by the person.")
    
    skills: list[str] = Field([], description="Skills possessed by the person.")
    certifications: list[str] = Field([], description="Certifications obtained by the person.")





# Read the markdown file
with open('data/resume.md', 'r') as file:
    resume_data = file.read()

# Define the agent
agent = Agent(model=model, result_type=Resume, system_prompt=f'You are a technical writer and an HR expert specialized in writing resumes. Write a resume for a Software Engineer based on the following data: {resume_data}')

# Run the agent
# result = agent.run_sync(f"Can you extract the following information from the resume? The raw data is {resume_data}")
result = agent.run_sync("Write a resume.")
logfire.notice('Resume markdown prompt LLM results: {result}', result = str(result.data))
logfire.info('Result type: {result}', result = type(result.data))

print("--------------------------------")

json_output = result.data.model_dump_json(indent=4)
print(json_output)