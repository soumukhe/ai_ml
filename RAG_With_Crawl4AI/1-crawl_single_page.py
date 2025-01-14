import asyncio
from crawl4ai import *

"""
The AsyncWebCrawler class is the core component for asynchronous web crawling in Crawl4AI. 

Here, the `arun` method of the crawler is called asynchronously with the URL “https://ai.pydantic.dev/” as an argument. 
The `await` keyword is used to pause the execution of the coroutine until the `arun` method completes its operation.

sitemap.xml: shows all the pages in the site
https://ai.pydantic.dev/sitemap.xml


Ethics for webscriaping:
robots.txt: shows all the pages that are not allowed to be scraped
https://cisco.com/robots.txt

"""


async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://ai.pydantic.dev/",
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())