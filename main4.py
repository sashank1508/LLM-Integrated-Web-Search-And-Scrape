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
import concurrent.futures
from functools import partial
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
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

class ProductionWebScraperAI:
    def __init__(self, max_depth: int = 4, max_concurrent_requests: int = 3, request_delay: float = 0.8):
        """
        WebScraperAI - Optimized for reliability and rate limit compliance
        
        Args:
            max_depth: Maximum depth to follow next URLs
            max_concurrent_requests: Conservative concurrent requests to avoid rate limits
            request_delay: Longer delay to respect rate limits
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
        
        # Rate limiting awareness
        self.rate_limited_domains: Set[str] = set()
        self.request_count = 0
        self.last_request_time = 0
        
        # Store current URL for relative resolution
        self._current_base_url = None
        
        # Pre-compiled regex patterns
        self.url_patterns = [
            re.compile(r'https?://[^\s<>"]+'),
            re.compile(r'www\.[^\s<>"]+'),
            re.compile(r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>"]*)?')
        ]
        
        # Session for synchronous requests
        self.sync_session = requests.Session()
        self.sync_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
        
    def __del__(self):
        """Cleanup session"""
        if hasattr(self, 'sync_session'):
            self.sync_session.close()
        
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
        """Enhanced URL normalization with better relative URL handling"""
        if not url or url.lower() in ['none', 'null']:
            return None
            
        url = url.strip()
        
        # Use stored base URL if none provided
        if not base_url and hasattr(self, '_current_base_url'):
            base_url = self._current_base_url
        
        # Handle relative URLs more intelligently
        if base_url and not url.startswith(('http://', 'https://')):
            # If it's a relative URL, use urljoin properly
            if url.startswith('/'):
                # Absolute path relative to domain
                parsed_base = urlparse(base_url)
                url = f"{parsed_base.scheme}://{parsed_base.netloc}{url}"
            else:
                # Relative path
                url = urljoin(base_url, url)
        
        # Ensure https if no protocol
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        # Clean up URL
        url = url.replace(' ', '%20')
        
        # Remove fragments and clean up
        if '#' in url:
            url = url.split('#')[0]
        
        logger.debug(f"Normalized URL: {url}")
        return url

    def _is_valid_url(self, url: Optional[str]) -> bool:
        """Rate-limit aware URL validation"""
        if not url or url.lower() in ['none', 'null']:
            return False
            
        if url in self.failed_urls:
            logger.debug(f"URL in failed list: {url}")
            return False
        
        # Check if domain is rate limited
        try:
            domain = urlparse(url).netloc.lower()
            if domain in self.rate_limited_domains:
                logger.debug(f"Domain rate limited: {domain}")
                return False
        except:
            return False
            
        # Basic URL validation
        parsed = urlparse(url)
        is_valid = bool(parsed.netloc and parsed.scheme in ['http', 'https'])
        logger.debug(f"URL validation for {url}: {is_valid}")
        return is_valid

    def _detect_url_in_input(self, input_text: str) -> Optional[str]:
        """Optimized URL detection using pre-compiled patterns"""
        for pattern in self.url_patterns:
            matches = pattern.findall(input_text)
            if matches:
                url = matches[0]
                url = url.rstrip('.,!?;)')
                return self._normalize_url(url)
        return None

    async def _respect_rate_limits(self):
        """Intelligent rate limiting with adaptive delays"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Base delay
        min_delay = self.request_delay
        
        # Adaptive delay based on request count
        if self.request_count > 10:
            min_delay += 0.3  # Add extra delay after many requests
        if self.request_count > 20:
            min_delay += 0.5  # Even more delay
            
        # Add small random jitter to avoid thundering herd
        jitter = random.uniform(0, 0.2)
        total_delay = min_delay + jitter
        
        if time_since_last < total_delay:
            sleep_time = total_delay - time_since_last
            await asyncio.sleep(sleep_time)
            
        self.last_request_time = time.time()
        self.request_count += 1

    async def scrape_content_production(self, url: str, max_retries: int = 2) -> ScrapingResult:
        """Production-grade scraping with intelligent rate limiting and error handling"""
        async with self.semaphore:
            # Store current base URL for relative resolution
            self._current_base_url = url
            
            # Check if domain is rate limited
            try:
                domain = urlparse(url).netloc.lower()
                if domain in self.rate_limited_domains:
                    return ScrapingResult(url=url, content="", success=False, error="Domain rate limited")
            except:
                pass
            
            for attempt in range(max_retries):
                try:
                    # Respect rate limits
                    await self._respect_rate_limits()
                    
                    full_url = f"https://r.jina.ai/{url}"
                    
                    # Conservative timeout settings
                    timeout = aiohttp.ClientTimeout(total=15, connect=5)
                    
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(full_url) as response:
                            if response.status == 429:  # Rate limited
                                domain = urlparse(url).netloc.lower()
                                self.rate_limited_domains.add(domain)
                                logger.warning(f"Rate limited by {domain}, adding to blacklist")
                                
                                # Exponential backoff for rate limits
                                backoff_time = (2 ** attempt) * 2
                                await asyncio.sleep(backoff_time)
                                continue
                                
                            response.raise_for_status()
                            content = await response.text()
                            
                            # Content validation
                            if len(content.strip()) < 50:
                                raise aiohttp.ClientError("Content too short")
                                
                            logger.info(f"Successfully scraped {url}, content length: {len(content)}")
                            return ScrapingResult(
                                url=url,
                                content=content,
                                success=True,
                                depth=self.url_depth_map.get(url, 0)
                            )
                            
                except Exception as e:
                    error_msg = str(e)
                    
                    # Handle specific rate limiting errors
                    if "429" in error_msg or "Too Many Requests" in error_msg:
                        domain = urlparse(url).netloc.lower()
                        self.rate_limited_domains.add(domain)
                        logger.warning(f"Rate limited: {url}")
                        
                        # Longer backoff for rate limits
                        backoff_time = (2 ** attempt) * 3
                        await asyncio.sleep(backoff_time)
                        continue
                    
                    logger.warning(f"Scraping attempt {attempt + 1} failed for {url}: {e}")
                    if attempt == max_retries - 1:
                        self.failed_urls.add(url)
                        return ScrapingResult(url=url, content="", success=False, error=str(e))
                    
                    # Regular backoff for other errors
                    await asyncio.sleep(1)
        
        return ScrapingResult(url=url, content="", success=False, error="Unexpected error")

    async def llm_query_production(self, query: str, max_retries: int = 2) -> str:
        """Production LLM query with retry"""
        for attempt in range(max_retries):
            try:
                chat_completion = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": query},
                    ],
                    # timeout=25,
                    # max_tokens=2000
                )
                result = chat_completion.choices[0].message.content
                return result if result is not None else "No response generated"
            except Exception as e:
                logger.warning(f"LLM query attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to generate response after {max_retries} attempts: {e}")
                await asyncio.sleep(2)
        
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

    def duckduckgo_search_production(self, query: str, num_results: int = 15) -> List[str]:
        """Production DuckDuckGo search with better error handling"""
        urls = []
        
        try:
            logger.info(f"Searching DuckDuckGo for: {query}")
            search_url = f"https://html.duckduckgo.com/html/?q={query}&s=0"
            
            # Add random delay to avoid being blocked
            time.sleep(random.uniform(0.5, 1.5))
            
            response = self.sync_session.get(search_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Try multiple selectors
            selectors = ["a.result__url", "a[class*='result']", ".result__body a", ".results_links a"]
            
            results = []
            for selector in selectors:
                found_results = soup.select(selector)
                if found_results:
                    results = found_results[:num_results]
                    break
            
            # Fallback to all links if needed
            if not results:
                all_links = soup.find_all("a", href=True)
                results = [link for link in all_links if isinstance(link, Tag) and "uddg" in str(link.get("href", ""))][:num_results]
            
            # Process results
            for result in results:
                try:
                    if isinstance(result, Tag):
                        raw_url = str(result.get("href", ""))
                        
                        if not raw_url:
                            continue
                        
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
                        
                        normalized_url = self._normalize_url(decoded_url.strip())
                        
                        if self._is_valid_url(normalized_url) and normalized_url not in urls:
                            urls.append(normalized_url)
                        
                except Exception:
                    continue
            
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
        
        logger.info(f"Found {len(urls)} URLs in search")
        return urls[:num_results]

    async def duckduckgo_search_enhanced(self, query: str, num_results: int = 15) -> List[str]:
        """Async wrapper for production search"""
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            urls = await loop.run_in_executor(
                executor, 
                partial(self.duckduckgo_search_production, query, num_results)
            )
        
        return urls

    async def generate_analysis_prompt(self, scraped_content: str, query: str) -> str:
        """Generate prompt for analyzing scraped content with enhanced URL extraction"""
        # Conservative content truncation
        max_content_length = 100000
        if len(scraped_content) > max_content_length:
            scraped_content = scraped_content[:max_content_length] + "..."
        
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
        7. CRITICAL: When looking for next URLs, look for specific department links, faculty pages, or relevant section links. 
           For queries about department heads or specific information, look for departmental links like "Computer Science", "Engineering", "Faculty", etc.
           Provide the COMPLETE URL, including the full path if it's a relative URL.
        8. If you find multiple relevant links, choose the most specific one that directly relates to the query.
        9. Make sure to provide fully formed URLs, not partial paths.
        """
        return prompt

    async def parse_analysis_response(self, response: str) -> AnalysisResult:
        """Enhanced parsing with better URL extraction and validation"""
        logger.debug(f"Parsing LLM response: {response}")
        
        # More flexible regex to handle multiline responses
        match = re.search(r"Answer:\s*(.+?)\s*Next URL:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            next_url = match.group(2).strip()
            
            logger.debug(f"Extracted answer: {answer}")
            logger.debug(f"Extracted next URL: {next_url}")
            
            # Clean up the answer (remove any trailing newlines)
            answer = re.sub(r'\n+', ' ', answer).strip()
            
            # Normalize next URL with better validation
            if next_url and next_url.lower() not in ['none', 'null', 'n/a']:
                # Use the current base URL for relative resolution
                base_url = getattr(self, '_current_base_url', None)
                normalized_url = self._normalize_url(next_url, base_url)
                
                # Validate the normalized URL
                if self._is_valid_url(normalized_url):
                    next_url = normalized_url
                    logger.info(f"Successfully normalized next URL: {next_url}")
                else:
                    logger.warning(f"Invalid normalized URL: {normalized_url}")
                    next_url = None
            else:
                next_url = None
                
            return AnalysisResult(
                answer=answer,
                next_url=next_url,
                confidence=0.8 if answer != "Not Found" else 0.2
            )
        
        logger.error(f"Failed to parse response format: {response}")
        raise ValueError("Response format does not match the expected pattern.")

    def generate_final_prompt(self, collected_answers: List[str], query: str) -> str:
        """Generate final prompt to synthesize all collected answers"""
        formatted_answers = "; ".join(collected_answers)
        return f"Based On The Context Provided: {formatted_answers}, The Context Provided May Also Contain Some Irrelevant Information, So Provide The Accurate Answer For The Query: {query}"

    async def deep_scrape_single_url_production(self, start_url: str, query: str, max_depth: Optional[int] = None) -> List[str]:
        """Production deep scrape with enhanced URL tracking and validation"""
        if max_depth is None:
            max_depth = self.max_depth
            
        collected_answers = []
        current_url = start_url
        current_depth = 0
        
        logger.info(f"Production deep scrape from: {start_url}")
        
        while current_url and current_depth < max_depth:
            if current_url in self.visited_urls:
                logger.info(f"URL already visited: {current_url}")
                break
                
            if not self._is_valid_url(current_url):
                logger.warning(f"Invalid or rate-limited URL: {current_url}")
                break
                
            self.visited_urls.add(current_url)
            self.url_depth_map[current_url] = current_depth
            
            logger.info(f"Scraping depth {current_depth}: {current_url}")
            
            # Production scraping with rate limit handling
            scraping_result = await self.scrape_content_production(current_url)
            
            if not scraping_result.success:
                logger.error(f"Failed to scrape {current_url}: {scraping_result.error}")
                break
                
            # Analyze content
            try:
                analysis_prompt = await self.generate_analysis_prompt(scraping_result.content, query)
                llm_response = await self.llm_query_production(analysis_prompt)
                analysis_result = await self.parse_analysis_response(llm_response)
                
                logger.info(f"Analysis result - Answer: {analysis_result.answer[:100]}...")
                logger.info(f"Analysis result - Next URL: {analysis_result.next_url}")
                
                # Collect answer if found
                if analysis_result.answer != "Not Found":
                    collected_answers.append(analysis_result.answer)
                    
                # Enhanced URL validation to avoid loops
                if not analysis_result.next_url:
                    logger.info("No next URL provided")
                    break
                elif analysis_result.next_url == current_url:
                    logger.info("Same URL returned, stopping to avoid loop")
                    break
                elif analysis_result.next_url in self.visited_urls:
                    logger.info(f"Next URL already visited: {analysis_result.next_url}")
                    break
                else:
                    # Validate next URL before proceeding
                    if self._is_valid_url(analysis_result.next_url):
                        current_url = analysis_result.next_url
                        current_depth += 1
                        logger.info(f"Moving to next URL at depth {current_depth}: {current_url}")
                    else:
                        logger.warning(f"Next URL is invalid: {analysis_result.next_url}")
                        break
                
            except Exception as e:
                logger.error(f"Error analyzing content from {current_url}: {e}")
                break
                
        logger.info(f"Production deep scrape completed. Found {len(collected_answers)} answers at depth {current_depth}")
        return collected_answers

    async def scrape_multiple_urls_production(self, urls: List[str], query: str) -> List[str]:
        """Production parallel scraping with conservative concurrency"""
        logger.info(f"Production parallel scrape of {len(urls)} URLs")
        
        # Create tasks for parallel execution
        tasks = []
        for url in urls[:12]:  # Process reasonable number of URLs
            if not self._is_valid_url(url) or url in self.visited_urls:
                continue
            task = asyncio.create_task(self.deep_scrape_single_url_production(url, query))
            tasks.append(task)
            
        # Execute with conservative approach
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
                
        logger.info(f"Production parallel scraping completed. Total answers found: {len(all_answers)}")
        return all_answers

    async def process_query_production(self, input_text: str, skip_urls: Optional[List[str]] = None, export_results: bool = True) -> Union[str, Tuple[str, Optional[str]]]:
        """Production query processing - optimized for reliability"""
        if skip_urls is None:
            skip_urls = []
            
        # Reset state
        self.visited_urls.clear()
        self.failed_urls.clear()
        self.url_depth_map.clear()
        self.failed_urls.update(skip_urls)
        self.request_count = 0  # Reset request counter
        
        start_time = time.time()
        
        try:
            logger.info(f"Production processing: {input_text}")
            
            # URL detection
            direct_url = self._detect_url_in_input(input_text)
            
            if direct_url:
                logger.info(f"Direct URL detected: {direct_url}")
                split_prompt = await self.split_input_prompt(input_text)
                split_response = await self.llm_query_production(split_prompt)
                _, query = await self.extract_url_and_query(split_response)
                
                collected_answers = await self.deep_scrape_single_url_production(direct_url, query)
                
            else:
                logger.info("No direct URL, using production search")
                split_prompt = await self.split_input_prompt(input_text)
                split_response = await self.llm_query_production(split_prompt)
                base_url, query = await self.extract_url_and_query(split_response)
                
                # Production search
                search_results = await self.duckduckgo_search_enhanced(base_url, num_results=15)
                
                if not search_results:
                    result = "No search results found for the query."
                    return (result, None) if export_results else result
                    
                logger.info(f"Found {len(search_results)} search results")
                
                # Production parallel scraping
                collected_answers = await self.scrape_multiple_urls_production(search_results, query)
            
            # Generate final answer
            if collected_answers:
                logger.info(f"Generating final answer from {len(collected_answers)} collected answers")
                final_prompt = self.generate_final_prompt(collected_answers, query if 'query' in locals() else input_text)
                final_response = await self.llm_query_production(final_prompt)
                
                processing_time = time.time() - start_time
                logger.info(f"Production processing completed in {processing_time:.2f} seconds")
                logger.info(f"Visited {len(self.visited_urls)} URLs, Failed: {len(self.failed_urls)}")
                logger.info(f"Rate limited domains: {len(self.rate_limited_domains)}")
                
                # Export results
                if export_results:
                    stats = self.get_processing_stats()
                    filepath = self._export_to_file(input_text, final_response, stats, processing_time)
                    return final_response, filepath
                
                return final_response
            else:
                result = "No relevant information found after searching multiple sources."
                return (result, None) if export_results else result
                
        except Exception as e:
            logger.error(f"Error in production processing: {e}")
            result = f"An error occurred while processing the query: {str(e)}"
            return (result, None) if export_results else result

    def get_processing_stats(self) -> dict:
        """Get comprehensive processing statistics"""
        return {
            "visited_urls": len(self.visited_urls),
            "failed_urls": len(self.failed_urls),
            "rate_limited_domains": len(self.rate_limited_domains),
            "total_requests": self.request_count,
            "max_depth_reached": max(self.url_depth_map.values()) if self.url_depth_map else 0,
            "url_depth_distribution": dict(sorted(self.url_depth_map.items(), key=lambda x: x[1]))
        }

    def _generate_filename(self, query: str) -> str:
        """Generate a relevant filename from the query"""
        clean_query = re.sub(r'[^\w\s-]', '', query)
        clean_query = re.sub(r'\s+', '_', clean_query)
        clean_query = clean_query.strip('_')
        
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
            answers_folder = self._create_answers_folder()
            filename = self._generate_filename(query)
            filepath = answers_folder / filename
            
            content = self._format_export_content(query, answer, stats, processing_time)
            
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
- Rate Limited Domains: {stats.get('rate_limited_domains', 0)}
- Total HTTP Requests: {stats.get('total_requests', 0)}
- Max Depth Reached: {stats.get('max_depth_reached', 0)}
- Total URLs Processed: {stats.get('visited_urls', 0) + stats.get('failed_urls', 0)}

URL DEPTH DISTRIBUTION:
{self._format_url_depth_distribution(stats.get('url_depth_distribution', {}))}

VISITED URLS:
{self._format_visited_urls()}

FAILED URLS:
{self._format_failed_urls()}

RATE LIMITED DOMAINS:
{self._format_rate_limited_domains()}

================================================================================
Generated by Production WebScraperAI v1
================================================================================
        """.strip()
        
        return content
    
    def _format_url_depth_distribution(self, depth_dist: dict) -> str:
        """Format the URL depth distribution for display"""
        if not depth_dist:
            return "No URLs processed"
        
        formatted = []
        depth_counts = {}
        
        for url, depth in depth_dist.items():
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
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
    
    def _format_rate_limited_domains(self) -> str:
        """Format rate limited domains for display"""
        if not self.rate_limited_domains:
            return "No domains rate limited"
        
        formatted = []
        for i, domain in enumerate(sorted(self.rate_limited_domains), 1):
            formatted.append(f"  {i}. {domain}")
        
        return '\n'.join(formatted)


# Production demo - optimized for real-world reliability
async def production_demo():
    """Production demo - designed for reliability and rate limit compliance"""
    # Initialize with production settings
    scraper = ProductionWebScraperAI(
        max_depth=6,                # Increased depth to match old code
        max_concurrent_requests=3,  # Conservative to avoid rate limits
        request_delay=0.8           # Longer delay to respect servers
    )
    
    queries = [
        "Visit iitr.ac.in and find out who is head of the department of computer science along with his contact details",
    ]
    
    exported_files = []
    total_start = time.time()
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*100}")
        print(f"PRODUCTION QUERY {i}/{len(queries)}: {query}")
        print(f"{'='*100}")
        
        query_start = time.time()
        
        # Production processing with export
        result, filepath = await scraper.process_query_production(query, export_results=True)
        
        query_time = time.time() - query_start
        
        print(f"\nResult: {result}")
        print(f"Query processed in: {query_time:.2f} seconds")
        
        stats = scraper.get_processing_stats()
        print(f"URLs visited: {stats['visited_urls']}, Failed: {stats['failed_urls']}")
        print(f"Rate limited domains: {stats['rate_limited_domains']}")
        print(f"Total HTTP requests: {stats['total_requests']}")
        print(f"Max depth reached: {stats['max_depth_reached']}")
        
        if filepath:
            print(f"Results exported to: {filepath}")
            exported_files.append(filepath)
            
        # Add delay between queries to be extra polite
        if i < len(queries):
            print("Waiting 3 seconds before next query...")
            await asyncio.sleep(3)
    
    total_time = time.time() - total_start
    print(f"\n{'='*100}")
    print(f"TOTAL PROCESSING TIME: {total_time:.2f} seconds")
    print(f"AVERAGE TIME PER QUERY: {total_time/len(queries):.2f} seconds")
    print(f"TOTAL EXPORTED FILES: {len(exported_files)}")
    print(f"{'='*100}")
    
    # Summary of exported files
    print(f"\nEXPORTED FILES:")
    for i, filepath in enumerate(exported_files, 1):
        print(f"  {i}. {filepath}")


# Single production query function
async def production_query(query: str, export: bool = True) -> Tuple[str, Optional[str]]:
    """Process a single query with production-grade reliability"""
    scraper = ProductionWebScraperAI(
        max_depth=6,  # Match old code depth
        max_concurrent_requests=3,
        request_delay=0.8
    )
    
    start_time = time.time()
    result, filepath = await scraper.process_query_production(query, export_results=export)
    processing_time = time.time() - start_time
    
    print(f"Production query completed in {processing_time:.2f} seconds")
    if export and filepath:
        print(f"Results exported to: {filepath}")
    
    return result, filepath


# Speed-optimized version for testing (less conservative)
async def fast_production_query(query: str, export: bool = False) -> str:
    """Faster production query for testing - slightly more aggressive"""
    scraper = ProductionWebScraperAI(
        max_depth=6,  # Match old code depth
        max_concurrent_requests=4,  # Slightly higher
        request_delay=0.5  # Faster
    )
    
    start_time = time.time()
    result = await scraper.process_query_production(query, export_results=export)
    processing_time = time.time() - start_time
    
    print(f"Fast production query completed in {processing_time:.2f} seconds")
    return result


if __name__ == "__main__":
    # Choose your mode:
    
    # 1. Full production demo (recommended for real use)
    # asyncio.run(production_demo())
    
    # 2. Single production query
    result, filepath = asyncio.run(production_query("Weather In Guntur Now"))
    print(f"Result: {result}")
    
    # 3. Fast testing mode
    # result = asyncio.run(fast_production_query("Visit iitr.ac.in and find out who is head of the department of computer science along with his contact details"))
    # print(f"Result: {result}")