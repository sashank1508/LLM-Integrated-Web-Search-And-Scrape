import aiohttp
import asyncio
import logging
import time
import re
import os
import requests
from typing import Optional, List, Tuple
from bs4 import BeautifulSoup
from urllib.parse import unquote, urlparse, parse_qs
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebScraperAI")

class WebScraperAI:
    def __init__(self):
        """Initialize the WebScraperAI with OpenAI client setup"""
        self.openai_api_key = self._get_openai_key()
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        
    def _get_openai_key(self) -> str:
        """Get OpenAI API key from environment variables"""
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is missing in environment variables.")
        return api_key

    async def scrape_content(self, path: str) -> str:
        """Scrape content from a URL using Jina AI reader service"""
        try:
            full_url = f"https://r.jina.ai/{path}"
            async with aiohttp.ClientSession() as session:
                async with session.get(full_url, timeout=10) as response:
                    response.raise_for_status()
                    return await response.text()
        except aiohttp.ClientError as e:
            logger.error(f"Error occurred while fetching {full_url}: {e}")
            return ""

    async def llm_query(self, query: str) -> str:
        """Send query to OpenAI LLM and get response"""
        try:
            chat_completion = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": query},
                ])
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise Exception(f"Failed to generate response: {e}")

    async def split_input_prompt(self, input_text: str) -> str:
        """Generate prompt to split input into URL and question"""
        instruction = """
        You are given an input string containing a URL and a question. Your task is to split the input into two parts: the URL and the question.
        The extracted question must retain full context and meaning from the original input. Ensure that no relevant information from the question is removed.
        The URL must always start with "https://". If the input does not include "https://" explicitly, prepend it to the extracted URL.
        If no URL is present in the input, provide the query to search for the URL.

        Return the result in the following format, with no additional text or markdown:

        URL: <extracted_url or query_to_search>
        Question: <extracted_question>

        For example:
        Input: "Visit https://example.com and find out What is the purpose of this website?"
        Output:
        URL: https://example.com
        Question: What is the purpose of this website?

        Input: "What does https://evergrowadvisors.com/ do?"
        Output:
        URL: https://evergrowadvisors.com
        Question: What does evergrowadvisors do?

        Input: "What does Evergrow Advisors do?"
        Output: URL: Evergrow Advisors
        Question: What does Evergrow Advisors do?

        Input: "is quicksell.co a product based company?"
        Output:
        URL: https://quicksell.co
        Question: Is quicksell a product based company?

        Strictly ensure the format matches the example provided, with the extracted URL and question on separate lines prefixed by "URL:" and "Question:".
        """
        return f"{instruction}\nInput: \"{input_text}\""

    async def extract_url_and_query(self, response: str) -> Tuple[str, str]:
        """Extract URL and query from LLM response"""
        match = re.search(r"URL: (.+)\nQuestion: (.+)", response)
        if match:
            base_url = match.group(1).strip()
            query = match.group(2).strip()
            return base_url, query
        else:
            raise ValueError("Response format does not match the expected pattern.")

    async def duckduckgo_search(self, query: str, num_results: int = 10) -> List[str]:
        """Search DuckDuckGo for URLs related to the query"""
        url = f"https://html.duckduckgo.com/html/?q={query}"
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("a", class_="result__url", limit=num_results)
            
            urls = []
            for result in results:
                raw_url = result["href"]
                parsed_url = urlparse(raw_url)
                query_params = parse_qs(parsed_url.query)
                if "uddg" in query_params:
                    decoded_url = unquote(query_params["uddg"][0])
                    urls.append(decoded_url)
            
            return urls
        except Exception as e:
            logger.error(f"Error during DuckDuckGo search: {e}")
            return []

    async def generate_analysis_prompt(self, scraped_content: str, query: str) -> str:
        """Generate prompt for analyzing scraped content"""
        prompt = f"""
        Based on the scraped content provided below, answer the query strictly following the format outlined. 

        Query: '{query}'
        Scraped Content: '{scraped_content}'
        
        Return the answer. If you don't find the answer in the scraped content, return the next URL to scrape 
        and go into it to find the answer based on the text of the hyperlink and text around the link.

        Instructions:
        1. Return the response in plain text only. Do not use any special formatting such as markdowns or bullet points.
        2. The response must strictly follow this format:
           Answer: <your_answer>
           Next URL: <next_url or None>
        3. If the answer is found in the scraped content, provide it under 'Answer' and set 'Next URL' to 'None'.
        4. If the answer is not found in the scraped content but there is a next URL to explore, set 'Answer' to 'Not Found' and provide the 'Next URL' to explore further.
        5. If the answer is not found in the scraped content and there is no URL to go next, set:
           Answer: Not Found
           Next URL: None
        6. The response must adhere strictly to the format without any deviation.
        """
        return prompt

    async def parse_analysis_response(self, response: str) -> Tuple[str, str]:
        """Parse the analysis response to extract answer and next URL"""
        match = re.search(r"Answer: (.+)\nNext URL: (.+)", response)
        if match:
            answer = match.group(1).strip()
            next_url = match.group(2).strip()
            return answer, next_url
        raise ValueError("Response format does not match the expected pattern.")

    def generate_final_prompt(self, collected_answers: List[str], query: str) -> str:
        """Generate final prompt to synthesize all collected answers"""
        formatted_answers = "; ".join(collected_answers)
        return f"Based On The Context Provided: {formatted_answers}, The Context Provided May Also Contain Some Irrelevant Information, So Provide The Accurate Answer For The Query: {query}"

    async def scrape_and_query(self, search_results: List[str], query: str, skip_urls: List[str] = None) -> Optional[str]:
        """Main scraping and querying logic"""
        if skip_urls is None:
            skip_urls = []
            
        attempt_count = 0
        visited_urls = set()
        collected_answers = []

        for current_url in search_results:
            while current_url:
                if current_url in visited_urls:
                    logger.warning(f"URL {current_url} has already been visited. Skipping...")
                    break

                if current_url in skip_urls:
                    logger.warning(f"URL {current_url} is in the skip list. Skipping...")
                    break

                attempt_count += 1
                logger.info("-" * 100)
                logger.info(f"Attempt {attempt_count}: Scraping content from URL: {current_url}")

                visited_urls.add(current_url)
                time.sleep(1)  # Rate limiting

                try:
                    scraped_content = await self.scrape_content(current_url)
                    if not scraped_content:
                        logger.error(f"Failed to scrape content from {current_url}. Trying next URL.")
                        break

                    logger.info("Successfully scraped content.")
                    logger.info("Generating LLM prompt...")

                    prompt = await self.generate_analysis_prompt(scraped_content, query)
                    
                    logger.info("Sending prompt to LLM...")
                    response = await self.llm_query(prompt)
                    answer, next_url = await self.parse_analysis_response(response)

                    logger.info(f"Response from LLM:\nAnswer: {answer}\nNext URL: {next_url}")

                    if answer != "Not Found":
                        logger.info("Answer found. Adding to collected answers.")
                        collected_answers.append(answer)

                    if next_url.lower() == "none":
                        logger.info("No further URLs suggested by LLM. Moving to next search result.")
                        break

                    if next_url == current_url:
                        logger.warning(f"Next URL is the same as current URL: {next_url}. Skipping to next search result.")
                        break

                    if answer == "Not Found" and next_url.lower() != "none":
                        logger.warning(f"Answer not found. Navigating to next URL: {next_url}")

                    current_url = next_url

                except Exception as e:
                    logger.error(f"Unexpected error occurred: {e}")
                    logger.info("Proceeding with collected answers so far.")
                    break

        # Generate final answer from collected answers
        if collected_answers:
            logger.info(f"Collected answers: {collected_answers}")
            final_prompt = self.generate_final_prompt(collected_answers, query)
            try:
                final_response = await self.llm_query(final_prompt)
                logger.info(f"Final response from LLM:\n{final_response}")
                return final_response
            except Exception as e:
                logger.error(f"Error generating final answer: {e}")
                return "Error generating final answer"

        logger.info("No answers found after attempting all search results.")
        return "No answer found after multiple attempts."

    async def process_query(self, input_text: str, skip_urls: List[str] = None) -> str:
        """Main method to process a natural language query"""
        if skip_urls is None:
            skip_urls = []
            
        try:
            # Split input into URL and query
            split_prompt = await self.split_input_prompt(input_text)
            split_response = await self.llm_query(split_prompt)
            
            base_url, query = await self.extract_url_and_query(split_response)
            
            # Get search results
            if not base_url.startswith("http"):
                logger.info("Attempting to find the URL via DuckDuckGo search...")
                search_results = await self.duckduckgo_search(base_url)
            else:
                search_results = [base_url] + await self.duckduckgo_search(base_url)

            logger.info(f"Base URL: {base_url}")
            logger.info(f"Query: {query}")
            logger.info(f"Found {len(search_results)} search results")
                    
            # Process the query
            final_answer = await self.scrape_and_query(search_results, query, skip_urls)
            return final_answer
            
        except ValueError as e:
            logger.error(f"Error during URL or query extraction: {e}")
            return f"Error during URL or query extraction: {e}"
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return f"An unexpected error occurred: {e}"


# Example usage
async def main():
    """Example usage of the WebScraperAI system"""
    scraper = WebScraperAI()
    
    # Example queries
    queries = [
            "Visit iitr.ac.in and find out about Department of Computer Science and engineering?",
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Processing query: {query}")
        print(f"{'='*80}")
        
        result = await scraper.process_query(query)
        print(f"\nFinal Answer: {result}")
        print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())