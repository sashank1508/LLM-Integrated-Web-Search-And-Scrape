import aiohttp
import asyncio
import logging
import time
import re
import os
import requests
from typing import Optional, List, Tuple, Set, Union
from bs4 import BeautifulSoup, Tag
from urllib.parse import unquote, urlparse, parse_qs, urljoin
from openai import AsyncOpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebScraperAI")

@dataclass
class ScrapingResult:
    """Data class to store scraping results"""
    url: str
    content: str
    success: bool
    error: Optional[str] = None
    depth: int = 0
    parent_url: Optional[str] = None

@dataclass
class AnalysisResult:
    """Data class to store analysis results"""
    answer: str
    next_url: Optional[str]
    confidence: float = 0.0
    source_url: str = ""

class EnhancedWebScraperAI:
    def __init__(self, max_depth: int = 5, max_concurrent_requests: int = 3, request_delay: float = 1.0):
        """
        Initialize the Enhanced WebScraperAI with optimization parameters
        
        Args:
            max_depth: Maximum depth to follow next URLs
            max_concurrent_requests: Maximum concurrent HTTP requests
            request_delay: Delay between requests in seconds
        """
        self.openai_api_key = self._get_openai_key()
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.max_depth = max_depth
        self.max_concurrent_requests = max_concurrent_requests
        self.request_delay = request_delay
        
        # Performance tracking
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.url_depth_map: dict = {}
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
    def _get_openai_key(self) -> str:
        """Get OpenAI API key from environment variables"""
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is missing in environment variables.")
        return api_key

    def _normalize_url(self, url: str, base_url: Optional[str] = None) -> Optional[str]:
        """Normalize URL and handle relative URLs"""
        if not url or url.lower() == 'none':
            return None
            
        url = url.strip()
        
        # Handle relative URLs
        if base_url and not url.startswith(('http://', 'https://')):
            url = urljoin(base_url, url)
        
        # Ensure https if no protocol
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        return url

    def _is_valid_url(self, url: Optional[str]) -> bool:
        """Check if URL is valid and not in failed URLs"""
        if not url or url.lower() == 'none':
            return False
            
        if url in self.failed_urls:
            return False
            
        # Basic URL validation
        parsed = urlparse(url)
        return bool(parsed.netloc and parsed.scheme in ['http', 'https'])

    def _detect_url_in_input(self, input_text: str) -> Optional[str]:
        """Detect if there's a direct URL in the input text"""
        # Enhanced URL detection patterns
        url_patterns = [
            r'https?://[^\s<>"]+',  # Standard HTTP/HTTPS URLs
            r'www\.[^\s<>"]+',      # www. URLs
            r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>"]*)?'  # Domain.com patterns
        ]
        
        for pattern in url_patterns:
            matches = re.findall(pattern, input_text)
            if matches:
                url = matches[0]
                # Clean up URL
                url = url.rstrip('.,!?;)')
                return self._normalize_url(url)
        return None

    async def scrape_content_with_retry(self, url: str, max_retries: int = 3) -> ScrapingResult:
        """Enhanced scraping with retry logic and error handling"""
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    await asyncio.sleep(self.request_delay)
                    
                    full_url = f"https://r.jina.ai/{url}"
                    timeout = aiohttp.ClientTimeout(total=15)
                    
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(full_url) as response:
                            response.raise_for_status()
                            content = await response.text()
                            
                            if len(content.strip()) < 100:  # Too short content
                                raise aiohttp.ClientError("Content too short")
                                
                            return ScrapingResult(
                                url=url,
                                content=content,
                                success=True,
                                depth=self.url_depth_map.get(url, 0)
                            )
                            
                except Exception as e:
                    logger.warning(f"Scraping attempt {attempt + 1} failed for {url}: {e}")
                    if attempt == max_retries - 1:
                        self.failed_urls.add(url)
                        return ScrapingResult(
                            url=url,
                            content="",
                            success=False,
                            error=str(e)
                        )
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # This should never be reached, but just in case
        return ScrapingResult(
            url=url,
            content="",
            success=False,
            error="Unexpected error in scraping"
        )

    async def llm_query_with_retry(self, query: str, max_retries: int = 3) -> str:
        """Enhanced LLM query with retry logic"""
        for attempt in range(max_retries):
            try:
                chat_completion = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": query},
                    ],
                    timeout=30
                )
                result = chat_completion.choices[0].message.content
                return result if result is not None else "No response generated"
            except Exception as e:
                logger.warning(f"LLM query attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to generate response after {max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)
        
        # This should never be reached, but just in case
        return "Failed to generate response"

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

    async def duckduckgo_search_enhanced(self, query: str, num_results: int = 15) -> List[str]:
        """Enhanced DuckDuckGo search with multiple fallback methods"""
        urls = []
        
        # Method 1: Try DuckDuckGo HTML
        try:
            logger.info(f"Searching DuckDuckGo for: {query}")
            search_url = f"https://html.duckduckgo.com/html/?q={query}&s=0"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            logger.info(f"DuckDuckGo response status: {response.status_code}")
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Try multiple selectors for different DuckDuckGo layouts
            selectors = [
                "a.result__url",
                "a[class*='result']",
                ".result__body a",
                ".results_links a",
                ".web-result a"
            ]
            
            results = []
            for selector in selectors:
                found_results = soup.select(selector)
                if found_results:
                    results = found_results[:num_results]
                    logger.info(f"Found {len(results)} results with selector: {selector}")
                    break
            
            # If no results with CSS selectors, try finding all links
            if not results:
                all_links = soup.find_all("a", href=True)
                logger.info(f"Fallback: Found {len(all_links)} total links")
                
                # Filter for result links
                results = []
                for link in all_links:
                    if isinstance(link, Tag):
                        href = link.get("href", "")
                        if href and ("uddg" in str(href) or str(href).startswith("http")):
                            results.append(link)
                            if len(results) >= num_results:
                                break
            
            # Process found results
            for result in results:
                try:
                    if isinstance(result, Tag):
                        raw_url = result.get("href", "")
                        
                        if not raw_url:
                            continue
                        
                        # Convert to string if it's not already
                        raw_url = str(raw_url)
                        
                        # Handle DuckDuckGo redirect URLs
                        if "uddg" in raw_url:
                            parsed_url = urlparse(raw_url)
                            query_params = parse_qs(parsed_url.query)
                            if "uddg" in query_params:
                                decoded_url = unquote(query_params["uddg"][0])
                            else:
                                continue
                        elif raw_url.startswith("http"):
                            decoded_url = raw_url
                        else:
                            continue
                        
                        # Clean and normalize URL
                        decoded_url = decoded_url.strip()
                        normalized_url = self._normalize_url(decoded_url)
                        
                        if self._is_valid_url(normalized_url) and normalized_url not in urls:
                            urls.append(normalized_url)
                            logger.info(f"Added URL: {normalized_url}")
                        
                except Exception as e:
                    logger.warning(f"Error processing search result: {e}")
                    continue
            
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
        
        # Method 2: Fallback to Google if DuckDuckGo fails
        if not urls:
            try:
                logger.info("Fallback: Trying Google search")
                urls = await self._google_search_fallback(query, num_results)
            except Exception as e:
                logger.warning(f"Google fallback failed: {e}")
        
        # Method 3: Manual URL construction for common sites
        if not urls:
            logger.info("Fallback: Constructing common URLs")
            urls = self._construct_common_urls(query)
        
        logger.info(f"Total URLs found: {len(urls)}")
        return urls[:num_results]
    
    async def _google_search_fallback(self, query: str, num_results: int = 10) -> List[str]:
        """Fallback Google search method"""
        try:
            from googlesearch import search
            logger.info("Using Google search fallback")
            results = search(query, num_results=num_results, lang="en")
            urls = []
            for url in results:
                if isinstance(url, str):
                    normalized_url = self._normalize_url(url)
                    if self._is_valid_url(normalized_url):
                        urls.append(normalized_url)
            return urls
        except ImportError:
            logger.warning("googlesearch library not available")
            return []
        except Exception as e:
            logger.warning(f"Google search fallback error: {e}")
            return []
    
    def _construct_common_urls(self, query: str) -> List[str]:
        """Construct URLs for common sites based on query"""
        urls = []
        query_clean = query.lower().replace(" ", "+")
        
        # Common news and information sites
        common_sites = [
            f"https://www.google.com/search?q={query_clean}",
            f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            f"https://www.reuters.com/search/news?blob={query_clean}",
            f"https://www.bbc.com/search?q={query_clean}",
            f"https://www.cnn.com/search?q={query_clean}"
        ]
        
        # For stock queries
        if "stock" in query.lower() or "price" in query.lower():
            ticker = query.split()[0].upper()  # Assume first word is ticker
            urls.extend([
                f"https://finance.yahoo.com/quote/{ticker}",
                f"https://www.marketwatch.com/investing/stock/{ticker}",
                f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
            ])
        
        # For company/organization queries
        if any(word in query.lower() for word in ["company", "organization", "corp", "inc"]):
            company_name = query.replace(" ", "")
            urls.extend([
                f"https://www.{company_name.lower()}.com",
                f"https://www.{company_name.lower()}.org",
                f"https://en.wikipedia.org/wiki/{company_name}"
            ])
        
        normalized_urls = []
        for url in common_sites:
            normalized_url = self._normalize_url(url)
            if normalized_url and self._is_valid_url(normalized_url):
                normalized_urls.append(normalized_url)
        return normalized_urls

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

    async def parse_analysis_response(self, response: str) -> AnalysisResult:
        """Parse the analysis response to extract answer and next URL"""
        match = re.search(r"Answer: (.+)\nNext URL: (.+)", response)
        if match:
            answer = match.group(1).strip()
            next_url = match.group(2).strip()
            
            # Normalize next URL
            if next_url and next_url.lower() != 'none':
                next_url = self._normalize_url(next_url)
            else:
                next_url = None
                
            return AnalysisResult(
                answer=answer,
                next_url=next_url,
                confidence=0.8 if answer != "Not Found" else 0.2
            )
        raise ValueError("Response format does not match the expected pattern.")

    def generate_final_prompt(self, collected_answers: List[str], query: str) -> str:
        """Generate final prompt to synthesize all collected answers"""
        formatted_answers = "; ".join(collected_answers)
        return f"Based On The Context Provided: {formatted_answers}, The Context Provided May Also Contain Some Irrelevant Information, So Provide The Accurate Answer For The Query: {query}"

    async def deep_scrape_single_url(self, start_url: str, query: str, max_depth: Optional[int] = None) -> List[str]:
        """Deep scrape a single URL following next URLs to specified depth"""
        if max_depth is None:
            max_depth = self.max_depth
            
        collected_answers = []
        current_url = start_url
        current_depth = 0
        
        logger.info(f"Starting deep scrape from: {start_url}")
        
        while current_url and current_depth < max_depth:
            if current_url in self.visited_urls:
                logger.info(f"URL already visited: {current_url}")
                break
                
            if not self._is_valid_url(current_url):
                logger.warning(f"Invalid URL: {current_url}")
                break
                
            self.visited_urls.add(current_url)
            self.url_depth_map[current_url] = current_depth
            
            logger.info(f"Scraping depth {current_depth}: {current_url}")
            
            # Scrape content
            scraping_result = await self.scrape_content_with_retry(current_url)
            
            if not scraping_result.success:
                logger.error(f"Failed to scrape {current_url}: {scraping_result.error}")
                break
                
            # Analyze content
            try:
                analysis_prompt = await self.generate_analysis_prompt(scraping_result.content, query)
                llm_response = await self.llm_query_with_retry(analysis_prompt)
                analysis_result = await self.parse_analysis_response(llm_response)
                
                logger.info(f"Analysis result - Answer: {analysis_result.answer[:100]}...")
                logger.info(f"Analysis result - Next URL: {analysis_result.next_url}")
                
                # Collect answer if found
                if analysis_result.answer != "Not Found":
                    collected_answers.append(analysis_result.answer)
                    
                # Check if we should continue
                if not analysis_result.next_url or analysis_result.next_url == current_url:
                    logger.info("No more URLs to explore or same URL returned")
                    break
                    
                current_url = analysis_result.next_url
                current_depth += 1
                
            except Exception as e:
                logger.error(f"Error analyzing content from {current_url}: {e}")
                break
                
        logger.info(f"Deep scrape completed. Found {len(collected_answers)} answers at depth {current_depth}")
        return collected_answers

    async def scrape_multiple_urls_parallel(self, urls: List[str], query: str) -> List[str]:
        """Scrape multiple URLs in parallel with controlled concurrency"""
        logger.info(f"Starting parallel scrape of {len(urls)} URLs")
        
        # Create tasks for each URL
        tasks = []
        for url in urls:
            if not self._is_valid_url(url) or url in self.visited_urls:
                continue
            task = asyncio.create_task(self.deep_scrape_single_url(url, query))
            tasks.append(task)
            
        # Execute tasks with progress tracking
        all_answers = []
        completed_tasks = 0
        
        for task in asyncio.as_completed(tasks):
            try:
                answers = await task
                all_answers.extend(answers)
                completed_tasks += 1
                logger.info(f"Completed {completed_tasks}/{len(tasks)} URL searches")
            except Exception as e:
                logger.error(f"Error in parallel scraping task: {e}")
                completed_tasks += 1
                
        logger.info(f"Parallel scraping completed. Total answers found: {len(all_answers)}")
        return all_answers

    async def process_query_enhanced(self, input_text: str, skip_urls: Optional[List[str]] = None, export_results: bool = False) -> Union[str, Tuple[str, Optional[str]]]:
        """Enhanced main method to process a natural language query"""
        if skip_urls is None:
            skip_urls = []
            
        # Reset state for new query
        self.visited_urls.clear()
        self.failed_urls.clear()
        self.url_depth_map.clear()
        
        # Add skip URLs to failed URLs to avoid them
        self.failed_urls.update(skip_urls)
        
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {input_text}")
            
            # Check if URL is directly detected in input
            direct_url = self._detect_url_in_input(input_text)
            
            if direct_url:
                logger.info(f"Direct URL detected: {direct_url}")
                # Extract query using LLM
                split_prompt = await self.split_input_prompt(input_text)
                split_response = await self.llm_query_with_retry(split_prompt)
                _, query = await self.extract_url_and_query(split_response)
                
                # Deep scrape the single URL
                collected_answers = await self.deep_scrape_single_url(direct_url, query)
                
            else:
                logger.info("No direct URL detected, using web search")
                # Split input into URL and query
                split_prompt = await self.split_input_prompt(input_text)
                split_response = await self.llm_query_with_retry(split_prompt)
                base_url, query = await self.extract_url_and_query(split_response)
                
                # Get search results
                search_results = await self.duckduckgo_search_enhanced(base_url, num_results=12)
                
                if not search_results:
                    result = "No search results found for the query."
                    return (result, None) if export_results else result
                    
                logger.info(f"Found {len(search_results)} search results")
                
                # Scrape all URLs in parallel
                collected_answers = await self.scrape_multiple_urls_parallel(search_results, query)
            
            # Generate final answer
            if collected_answers:
                logger.info(f"Generating final answer from {len(collected_answers)} collected answers")
                final_prompt = self.generate_final_prompt(collected_answers, query if 'query' in locals() else input_text)
                final_response = await self.llm_query_with_retry(final_prompt)
                
                processing_time = time.time() - start_time
                logger.info(f"Query processed successfully in {processing_time:.2f} seconds")
                logger.info(f"Visited {len(self.visited_urls)} URLs, Failed: {len(self.failed_urls)}")
                
                # Export results if requested
                if export_results:
                    stats = self.get_processing_stats()
                    filepath = self._export_to_file(input_text, final_response, stats, processing_time)
                    return final_response, filepath
                
                return final_response
            else:
                result = "No relevant information found after searching multiple sources."
                return (result, None) if export_results else result
                
        except Exception as e:
            logger.error(f"Error in process_query_enhanced: {e}")
            result = f"An error occurred while processing the query: {str(e)}"
            return (result, None) if export_results else result

    def get_processing_stats(self) -> dict:
        """Get processing statistics"""
        return {
            "visited_urls": len(self.visited_urls),
            "failed_urls": len(self.failed_urls),
            "max_depth_reached": max(self.url_depth_map.values()) if self.url_depth_map else 0,
            "url_depth_distribution": dict(sorted(self.url_depth_map.items(), key=lambda x: x[1]))
        }

    async def debug_search(self, query: str) -> dict:
        """Debug method to test search functionality"""
        logger.info(f"Debug: Testing search for query: {query}")
        
        debug_info = {
            "query": query,
            "search_url": f"https://html.duckduckgo.com/html/?q={query}&s=0",
            "results": [],
            "errors": []
        }
        
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={query}&s=0"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=15)
            debug_info["status_code"] = response.status_code
            debug_info["response_length"] = len(response.text)
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Check different selectors
            selectors = [
                "a.result__url",
                "a[class*='result']", 
                ".result__body a",
                ".results_links a"
            ]
            
            for selector in selectors:
                found = soup.select(selector)
                debug_info["results"].append({
                    "selector": selector,
                    "count": len(found),
                    "sample_hrefs": [str(a.get("href", ""))[:100] for a in found[:3] if isinstance(a, Tag)]
                })
            
            # Check all links
            all_links = soup.find_all("a", href=True)
            debug_info["total_links"] = len(all_links)
            debug_info["sample_links"] = [str(a.get("href", ""))[:100] for a in all_links[:10] if isinstance(a, Tag)]
            
        except Exception as e:
            debug_info["errors"].append(str(e))
            
        return debug_info

    def _generate_filename(self, query: str) -> str:
        """Generate a relevant filename from the query"""
        # Clean the query for filename
        clean_query = re.sub(r'[^\w\s-]', '', query)  # Remove special characters
        clean_query = re.sub(r'\s+', '_', clean_query)  # Replace spaces with underscores
        clean_query = clean_query.strip('_')  # Remove leading/trailing underscores
        
        # Limit length and add timestamp
        if len(clean_query) > 50:
            clean_query = clean_query[:50]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{clean_query}_{timestamp}.txt"
    
    def _create_answers_folder(self) -> Path:
        """Create the Answers folder if it doesn't exist"""
        answers_folder = Path("Answers")
        answers_folder.mkdir(exist_ok=True)
        return answers_folder
    
    def _export_to_file(self, query: str, answer: str, stats: dict, processing_time: float) -> str:
        """Export query results to a text file"""
        try:
            # Create folder and filename
            answers_folder = self._create_answers_folder()
            filename = self._generate_filename(query)
            filepath = answers_folder / filename
            
            # Prepare content
            content = self._format_export_content(query, answer, stats, processing_time)
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Results exported to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting to file: {e}")
            return f"Error: {e}"
    
    def _format_export_content(self, query: str, answer: str, stats: dict, processing_time: float) -> str:
        """Format the content for export"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"""
================================================================================
WEB SCRAPER AI - QUERY RESULTS
================================================================================

Timestamp: {timestamp}

QUESTION:
{query}

ANSWER:
{answer}

PROCESSING STATISTICS:
- Processing Time: {processing_time:.2f} seconds
- URLs Visited: {stats.get('visited_urls', 0)}
- URLs Failed: {stats.get('failed_urls', 0)}
- Max Depth Reached: {stats.get('max_depth_reached', 0)}
- Total URLs Processed: {stats.get('visited_urls', 0) + stats.get('failed_urls', 0)}

URL DEPTH DISTRIBUTION:
{self._format_url_depth_distribution(stats.get('url_depth_distribution', {}))}

VISITED URLS:
{self._format_visited_urls()}

FAILED URLS:
{self._format_failed_urls()}

================================================================================
Generated by Enhanced WebScraperAI v2.0
================================================================================
        """.strip()
        
        return content
    
    def _format_url_depth_distribution(self, depth_dist: dict) -> str:
        """Format the URL depth distribution for display"""
        if not depth_dist:
            return "No URLs processed"
        
        formatted = []
        depth_counts = {}
        
        # Count URLs at each depth
        for url, depth in depth_dist.items():
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        # Format the distribution
        for depth in sorted(depth_counts.keys()):
            count = depth_counts[depth]
            formatted.append(f"  Depth {depth}: {count} URL{'s' if count != 1 else ''}")
        
        return '\n'.join(formatted) if formatted else "No depth data available"
    
    def _format_visited_urls(self) -> str:
        """Format visited URLs for display"""
        if not self.visited_urls:
            return "No URLs visited"
        
        formatted = []
        for i, url in enumerate(sorted(self.visited_urls), 1):
            depth = self.url_depth_map.get(url, 0)
            formatted.append(f"  {i}. [Depth {depth}] {url}")
        
        return '\n'.join(formatted)
    
    def _format_failed_urls(self) -> str:
        """Format failed URLs for display"""
        if not self.failed_urls:
            return "No URLs failed"
        
        formatted = []
        for i, url in enumerate(sorted(self.failed_urls), 1):
            formatted.append(f"  {i}. {url}")
        
        return '\n'.join(formatted)


# Example usage with enhanced features and file export
async def main():
    """Example usage of the Enhanced WebScraperAI system with file export"""
    # Initialize with custom parameters
    scraper = EnhancedWebScraperAI(
        max_depth=6,  # Allow deeper exploration
        max_concurrent_requests=4,  # More parallel requests
        request_delay=0.5  # Faster processing
    )
    
    # Example queries
    queries = [
        # "What does https://openai.com do?",  # Direct URL - will deep scrape
        # "Tell me about https://github.com/microsoft/vscode features",  # Direct URL with specific question
        # "Tesla stock price today"  # No URL - will search multiple sources
        "Who is Debi Prosad Dogra?"
    ]
    
    exported_files = []
    
    for query in queries:
        print(f"\n{'='*100}")
        print(f"Processing query: {query}")
        print(f"{'='*100}")
        
        start_time = time.time()
        
        # Process query with file export
        result, filepath = await scraper.process_query_enhanced(query, export_results=True)
        processing_time = time.time() - start_time
        
        print(f"\nFinal Answer: {result}")
        
        # Get processing statistics
        stats = scraper.get_processing_stats()
        print(f"\nProcessing Statistics:")
        print(f"- Processing time: {processing_time:.2f} seconds")
        print(f"- URLs visited: {stats['visited_urls']}")
        print(f"- URLs failed: {stats['failed_urls']}")
        print(f"- Max depth reached: {stats['max_depth_reached']}")
        
        if filepath:
            print(f"- Results exported to: {filepath}")
            exported_files.append(filepath)
        
        print(f"{'='*100}")
    
    # Summary of exported files
    print(f"\n{'='*100}")
    print("EXPORT SUMMARY")
    print(f"{'='*100}")
    print(f"Total queries processed: {len(queries)}")
    print(f"Files exported: {len(exported_files)}")
    print("\nExported files:")
    for i, filepath in enumerate(exported_files, 1):
        print(f"  {i}. {filepath}")
    print(f"{'='*100}")


# Standalone function for single query processing
async def process_single_query(query: str, max_depth: int = 6, export: bool = True) -> Tuple[str, Optional[str]]:
    """Process a single query and optionally export results"""
    scraper = EnhancedWebScraperAI(max_depth=max_depth)
    
    print(f"Processing: {query}")
    result, filepath = await scraper.process_query_enhanced(query, export_results=export)
    
    if export and filepath:
        print(f"Results exported to: {filepath}")
    
    return result, filepath


if __name__ == "__main__":
    # You can either run the full demo or process a single query
    
    # Full demo
    asyncio.run(main())
    
    # Or process a single query
    # asyncio.run(process_single_query("What is the latest news about AI?"))