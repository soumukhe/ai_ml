import datetime
import os
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
from pyppeteer import launch
from pyppeteer_stealth import stealth
from bs4 import BeautifulSoup
from html_to_markdown import convert_to_markdown
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
import bs4

"""
This script scrapes the HTML from the property listing and then parses the data using the agent.

Some Notes:
- I used chatgpt to create the pydantic model.
- what's on this page ?
https://www.homes.com/property/1803-fernald-point-ln-santa-barbara-ca/9wr35zkk9mxjq
- can you parse that page and give me a pydantic model giving real estate listing

Reply from chatgpt:

from pydantic import BaseModel, Field
from typing import List, Optional

class RealEstateListing(BaseModel):
    property_id: str = Field(..., description="Unique identifier for the property")
    address: str = Field(..., description="Full address of the property")
    city: str = Field(..., description="City where the property is located")
    state: str = Field(..., description="State where the property is located")
    zip_code: str = Field(..., description="ZIP code of the property's location")
    price: int = Field(..., description="Listing price of the property in USD")
    bedrooms: int = Field(..., description="Number of bedrooms")
    bathrooms: float = Field(..., description="Number of bathrooms")
    square_feet: int = Field(..., description="Living area size in square feet")
    lot_size_acres: float = Field(..., description="Lot size in acres")
    year_built: int = Field(..., description="Year the property was built")
    description: str = Field(..., description="Detailed description of the property")
    features: List[str] = Field(..., description="List of notable features and amenities")
    photos: List[str] = Field(..., description="URLs of property photos")
    listing_url: str = Field(..., description="URL of the property's listing page")
    mls_number: Optional[str] = Field(None, description="Multiple Listing Service number")


- pyppeteer is a Python port of Puppeteer, a Node.js library that provides a high-level API to control headless Chrome or Chromium browsers. 
  It enables tasks such as web scraping, automated testing, and browser automation by allowing programmatic control over browser actions. ￼
- pyppeteer_stealth is a Python package designed to make headless browsers less detectable to anti-bot measures employed by websites. 
  It achieves this by applying various techniques to mask the automated nature of the browser, reducing the likelihood of detection. ￼   
- lxml (more advanced than BeautifulSoup) is a Python library that provides a fast and flexible way to parse and manipulate XML and HTML data. 
  It is widely used for web scraping and data extraction tasks.
- html-to-markdown is a Python library that converts HTML content to Markdown format. 
  It is useful for converting web page content into a more structured text format for further processing.

We are using asyncio to run the browser in a non-blocking way.

asyncio.new_event_loop().run_until_complete(html_to_markdown(url, output_path))

Output:

17:29:17.757   preparing model and tools run_step=1
17:29:17.757   model request
17:29:25.180   handle model response
17:29:25.195 Property markdown prompt LLM results: property_id='1803-fernald-point-ln-santa-barbara-ca' address='.../1803-fernald-point-ln-santa-barbara-ca/' mls_number='24-3770'
17:29:25.196 Result type: <class '__main__.RealEstateListing'>
--------------------------------
{
    "property_id": "1803-fernald-point-ln-santa-barbara-ca",
    "address": "1803 Fernald Point Ln",
    "city": "Santa Barbara",
    "state": "CA",
    "zip_code": "93108",
    "price": 34500000,
    "bedrooms": 5,
    "bathrooms": 5.5,
    "square_feet": 5945,
    "lot_size_acres": 0.65,
    "year_built": 1995,
    "description": "Enveloped by the ocean's timeless beauty, this exquisite Montecito villa blends the charm of Provencal design with the breathtaking allure of the California coastline. Extensively renovated from 2020-2023 and located in the exclusive, gated Fernald Cove Point, a tapestry of exceptional craftsmanship and timeless functionality converge in spacious, bright living areas that open up to expansive patios with sweeping ocean & island views. A spectacular great room, chef's kitchen with La Cornue range & elegant dining room await ocean view entertainment while 2 primary & 3 guest suites serve as secluded retreats. With 101 feet of beach frontage just moments from Montecito, Summerland & Carpinteria, seize the opportunity for a seaside sanctuary unlike any other.",
    "features": [
        "Ocean Front",
        "Updated Kitchen",
        "Gated Home",
        "Fireplace in Primary Bedroom",
        "Hot Property"
    ],
    "photos": [
        "https://images.homes.com/listings/102/6896357604-702166981/1803-fernald-point-ln-santa-barbara-ca-primaryphoto.jpg",
        "https://images.homes.com/listings/214/9007357604-702166981/1803-fernald-point-ln-santa-barbara-ca-buildingphoto-2.jpg",
        "https://images.homes.com/listings/214/5107357604-702166981/1803-fernald-point-ln-santa-barbara-ca-buildingphoto-3.jpg",
        "https://images.homes.com/listings/214/7207357604-702166981/1803-fernald-point-ln-santa-barbara-ca-buildingphoto-4.jpg",
        "https://images.homes.com/listings/214/5307357604-702166981/1803-fernald-point-ln-santa-barbara-ca-buildingphoto-5.jpg",
        "https://images.homes.com/listings/117/4407357604-702166981/1803-fernald-point-ln-santa-barbara-ca-buildingphoto-6.jpg?t=p",
        "https://images.homes.com/listings/214/5507357604-702166981/1803-fernald-point-ln-santa-barbara-ca-buildingphoto-7.jpg"
    ],
    "listing_url": "https://www.homes.com/property/1803-fernald-point-ln-santa-barbara-ca/",
    "mls_number": "24-3770"
}

"""

load_dotenv()

# Configure logfire
logfire.configure()

# Define the model
model = OpenAIModel('gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))

# Define the output models
# class Address(BaseModel):
#     street: str
#     city: str
#     state: str
#     zip_code: str

# class PropertyFeatures(BaseModel):
#     bedrooms: int
#     bathrooms: int
#     square_footage: float
#     lot_size: float  # in acres, but we can convert to sqft if needed

# class AdditionalInfo(BaseModel):
#     price: float
#     listing_agent: str
#     last_updated: datetime.date

# class Property(BaseModel):
#     address: Address
#     info: AdditionalInfo
#     type: str  # Single Family Home
#     mls_id: int
#     features: PropertyFeatures
#     garage_spaces: int


class RealEstateListing(BaseModel):
    property_id: str = Field(..., description="Unique identifier for the property")
    address: str = Field(..., description="Full address of the property")
    city: str = Field(..., description="City where the property is located")
    state: str = Field(..., description="State where the property is located")
    zip_code: str = Field(..., description="ZIP code of the property's location")
    price: int = Field(..., description="Listing price of the property in USD")
    bedrooms: int = Field(..., description="Number of bedrooms")
    bathrooms: float = Field(..., description="Number of bathrooms")
    square_feet: int = Field(..., description="Living area size in square feet")
    lot_size_acres: float = Field(..., description="Lot size in acres")
    year_built: int = Field(..., description="Year the property was built")
    description: str = Field(..., description="Detailed description of the property")
    features: List[str] = Field(..., description="List of notable features and amenities")
    photos: List[str] = Field(..., description="URLs of property photos")
    listing_url: str = Field(..., description="URL of the property's listing page")
    mls_number: Optional[str] = Field(None, description="Multiple Listing Service number")


# Function to scrape HTML into markdown
async def html_to_markdown(url, output_path):
    browser = await launch(headless=True)
    requestHeaders = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        'Referer': 'https://www.google.com/',
    }
    page = await browser.newPage()
    await stealth(page)
    await page.setExtraHTTPHeaders(requestHeaders)
    await page.goto(url, {'waitUntil': 'load'})
    content = await page.content()
    
    # Try different parsers in order of preference
    for parser in ['lxml', 'html.parser', 'html5lib']:
        try:
            soup = BeautifulSoup(content, parser)
            markdown = convert_to_markdown(str(soup))
            # Write the markdown to a file
            with open(output_path, 'w') as f:
                f.write(markdown)
            await browser.close()
            return
        except (ImportError, bs4.FeatureNotFound):
            continue
    
    # If no parser worked, raise an error
    raise RuntimeError("No suitable HTML parser found. Please install 'lxml' or 'html5lib'")

# Get the real estate data
# url = 'https://www.redfin.com/VA/Sterling/47516-Anchorage-Cir-20165/home/11931811'
# url = 'https://www.zillow.com/homedetails/845-Sea-Ranch-Dr-Santa-Barbara-CA-93109/15896664_zpid'
url = 'https://www.homes.com/property/1803-fernald-point-ln-santa-barbara-ca/9wr35zkk9mxjq'
output_path = 'data/property.md'
asyncio.new_event_loop().run_until_complete(html_to_markdown(url, output_path))

# Read the markdown file
with open('data/property.md', 'r') as file:
    property_data = file.read()

agent = Agent(model=model, result_type=RealEstateListing, system_prompt=f'You are a real estate agent specialized in creating and parsing property listings in the US.')

# Run the agent
result = agent.run_sync(f"Can you extract the following information from the property listing? The raw data is {property_data}")
logfire.notice('Property markdown prompt LLM results: {result}', result = str(result.data))
logfire.info('Result type: {result}', result = type(result.data))

print("--------------------------------")

json_output = result.data.model_dump_json(indent=4)
print(json_output)