import aiohttp
import asyncio
import logging
import time
import re
import os
import hashlib
import json
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set
from urllib.parse import urljoin, urlparse, parse_qs, unquote
from datetime import datetime, timedelta
import random
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Try to import backoff, make it optional
try:
    import backoff
    HAS_BACKOFF = True
except ImportError:
    HAS_BACKOFF = False
    logger = logging.getLogger(__name__)
    logger.warning("backoff library not installed. Install with 'pip install backoff' for better retry handling.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedWebScraperAI")

@dataclass
class ScrapingResult:
    """Data class for scraping results"""
    url: str
    content: str
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ScrapingConfig:
    """Configuration for scraping behavior"""
    max_concurrent_requests: int = 5
    request_delay: float = 1.0
    max_retries: int = 3
    timeout: int = 15
    max_content_length: int = 1000000  # 1MB
    respect_robots_txt: bool = True
    use_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    user_agents: List[str] = None
    
    def __post_init__(self):
        if self.user_agents is None:
            self.user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            ]

class ContentAnalyzer:
    """Advanced content analysis using AI"""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        
    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract JSON from response text, handling various formats"""
        # Try direct JSON parsing first
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
            
        # Look for JSON block in markdown format
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
                
        # Look for JSON block without markdown
        json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
                
        return None
        
    async def extract_key_information(self, content: str, query: str) -> Dict:
        """Extract key information using AI with improved JSON handling"""
        # Limit content length for efficiency
        content_snippet = content[:5000] if len(content) > 5000 else content
        
        prompt = f"""
        Analyze the following content and extract information relevant to the query: "{query}"
        
        Content: {content_snippet}
        
        You must respond with valid JSON only. No additional text or explanation.
        
        {{
            "relevance_score": 0.8,
            "key_facts": ["example fact 1", "example fact 2"],
            "entities": ["entity1", "entity2"],
            "summary": "brief summary of the content",
            "confidence": 0.9,
            "recommended_next_urls": [],
            "answer_found": true,
            "answer": "direct answer if found, otherwise null"
        }}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=800,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert content analyzer. Always respond with valid JSON only. No markdown formatting or additional text."
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"Raw AI response: {response_text[:200]}...")
            
            # Extract JSON from response
            result = self._extract_json_from_response(response_text)
            
            if result is None:
                logger.warning(f"Could not parse JSON from response: {response_text[:100]}...")
                return self._get_default_analysis()
                
            # Validate required fields
            required_fields = ["relevance_score", "key_facts", "entities", "summary", "confidence", "answer_found"]
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Missing required field: {field}")
                    return self._get_default_analysis()
                    
            # Ensure proper types
            result["relevance_score"] = float(result.get("relevance_score", 0.0))
            result["confidence"] = float(result.get("confidence", 0.0))
            result["answer_found"] = bool(result.get("answer_found", False))
            
            if not isinstance(result.get("key_facts"), list):
                result["key_facts"] = []
            if not isinstance(result.get("entities"), list):
                result["entities"] = []
            if not isinstance(result.get("recommended_next_urls"), list):
                result["recommended_next_urls"] = []
                
            return result
            
        except Exception as e:
            logger.error(f"Error in content analysis: {e}")
            return self._get_default_analysis()
            
    def _get_default_analysis(self) -> Dict:
        """Return default analysis structure"""
        return {
            "relevance_score": 0.0,
            "key_facts": [],
            "entities": [],
            "summary": "",
            "confidence": 0.0,
            "recommended_next_urls": [],
            "answer_found": False,
            "answer": None
        }

class InMemoryCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self):
        self.cache: Dict[str, Tuple[any, datetime]] = {}
        
    def get(self, key: str, ttl: int = 3600) -> Optional[any]:
        """Get item from cache if not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=ttl):
                return value
            else:
                del self.cache[key]
        return None
        
    def set(self, key: str, value: any):
        """Set item in cache"""
        self.cache[key] = (value, datetime.now())
        
    def clear_expired(self, ttl: int = 3600):
        """Clear expired items"""
        current_time = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= timedelta(seconds=ttl)
        ]
        for key in expired_keys:
            del self.cache[key]

class EnhancedWebScraperAI:
    """Enhanced web scraper with AI capabilities and optimizations"""
    
    def __init__(self, config: ScrapingConfig = None):
        self.config = config or ScrapingConfig()
        self.openai_api_key = self._get_openai_key()
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.analyzer = ContentAnalyzer(self.client)
        self.cache = InMemoryCache()
        self.session_pool = None
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
    def _get_openai_key(self) -> str:
        """Get OpenAI API key from environment variables"""
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is missing in environment variables.")
        return api_key
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_requests * 2,
            limit_per_host=self.config.max_concurrent_requests,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session_pool = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': random.choice(self.config.user_agents)}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session_pool:
            await self.session_pool.close()
            
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.md5(url.encode()).hexdigest()
        
    def _retry_on_failure(self, func):
        """Decorator for retry logic - uses backoff if available, otherwise simple retry"""
        if HAS_BACKOFF:
            return backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=3)(func)
        else:
            # Simple retry wrapper
            async def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(3):
                    try:
                        return await func(*args, **kwargs)
                    except aiohttp.ClientError as e:
                        last_exception = e
                        if attempt < 2:  # Don't sleep on last attempt
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                raise last_exception
            return wrapper

    async def scrape_content_optimized(self, url: str, use_jina: bool = True) -> ScrapingResult:
        """Optimized content scraping with multiple methods"""
        cache_key = self._get_cache_key(url)
        
        # Check cache first
        if self.config.use_cache:
            cached_result = self.cache.get(cache_key, self.config.cache_ttl)
            if cached_result:
                logger.info(f"Cache hit for {url}")
                return cached_result
                
        async with self.semaphore:  # Limit concurrent requests
            # Apply retry logic
            scrape_func = self._retry_on_failure(self._scrape_with_retry)
            try:
                result = await scrape_func(url, use_jina)
                    
                # Cache successful results
                if result.success and self.config.use_cache:
                    self.cache.set(cache_key, result)
                    
                return result
                
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                return ScrapingResult(url=url, content="", success=False, error=str(e))
                
    async def _scrape_with_retry(self, url: str, use_jina: bool) -> ScrapingResult:
        """Internal scraping method with retry logic"""
        if use_jina:
            return await self._scrape_with_jina(url)
        else:
            return await self._scrape_direct(url)
                
    async def _scrape_with_jina(self, url: str) -> ScrapingResult:
        """Scrape using Jina AI reader service"""
        jina_url = f"https://r.jina.ai/{url}"
        
        try:
            async with self.session_pool.get(jina_url) as response:
                if response.status == 200:
                    content = await response.text()
                    if len(content) > self.config.max_content_length:
                        content = content[:self.config.max_content_length]
                        
                    return ScrapingResult(
                        url=url,
                        content=content,
                        success=True,
                        metadata={'method': 'jina', 'status': response.status}
                    )
                else:
                    return ScrapingResult(
                        url=url,
                        content="",
                        success=False,
                        error=f"HTTP {response.status}"
                    )
        except Exception as e:
            logger.error(f"Jina scraping failed for {url}: {e}")
            return ScrapingResult(url=url, content="", success=False, error=str(e))
                
    async def _scrape_direct(self, url: str) -> ScrapingResult:
        """Direct scraping as fallback"""
        headers = {
            'User-Agent': random.choice(self.config.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        try:
            async with self.session_pool.get(url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Parse with BeautifulSoup for better text extraction
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                        
                    text_content = soup.get_text()
                    
                    # Clean up text
                    lines = (line.strip() for line in text_content.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text_content = ' '.join(chunk for chunk in chunks if chunk)
                    
                    if len(text_content) > self.config.max_content_length:
                        text_content = text_content[:self.config.max_content_length]
                        
                    return ScrapingResult(
                        url=url,
                        content=text_content,
                        success=True,
                        metadata={'method': 'direct', 'status': response.status}
                    )
                else:
                    return ScrapingResult(
                        url=url,
                        content="",
                        success=False,
                        error=f"HTTP {response.status}"
                    )
        except Exception as e:
            logger.error(f"Direct scraping failed for {url}: {e}")
            return ScrapingResult(url=url, content="", success=False, error=str(e))
                
    async def enhanced_search(self, query: str, num_results: int = 10) -> List[str]:
        """Enhanced search with multiple engines and result filtering"""
        results = []
        
        # Try DuckDuckGo search
        try:
            duckduckgo_results = await self._duckduckgo_search(query, num_results)
            results.extend(duckduckgo_results)
            logger.info(f"DuckDuckGo returned {len(duckduckgo_results)} results")
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            
        # Fallback: Try a simple web search if no results
        if not results:
            try:
                fallback_results = await self._fallback_search(query, num_results)
                results.extend(fallback_results)
                logger.info(f"Fallback search returned {len(fallback_results)} results")
            except Exception as e:
                logger.warning(f"Fallback search failed: {e}")
                
        # Remove duplicates while preserving order
        unique_results = []
        seen = set()
        for url in results:
            if url not in seen and url.startswith('http'):
                unique_results.append(url)
                seen.add(url)
                
        return unique_results[:num_results]
        
    async def _fallback_search(self, query: str, num_results: int) -> List[str]:
        """Fallback search method"""
        # Simple search using a different approach
        search_url = f"https://www.google.com/search?q={query}"
        
        try:
            headers = {
                'User-Agent': random.choice(self.config.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            async with self.session_pool.get(search_url, headers=headers) as response:
                if response.status != 200:
                    return []
                    
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract URLs from search results
                urls = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith('/url?q='):
                        # Extract actual URL from Google's redirect
                        actual_url = href.split('url?q=')[1].split('&')[0]
                        actual_url = unquote(actual_url)
                        if actual_url.startswith('http'):
                            urls.append(actual_url)
                            
                return urls[:num_results]
                
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
        
    async def _duckduckgo_search(self, query: str, num_results: int) -> List[str]:
        """Optimized DuckDuckGo search with better error handling"""
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        
        try:
            async with self.session_pool.get(search_url) as response:
                if response.status != 200:
                    logger.warning(f"DuckDuckGo search returned status {response.status}")
                    return []
                    
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Try multiple selectors for better compatibility
                results = soup.find_all("a", class_="result__url", limit=num_results)
                if not results:
                    # Fallback selector
                    results = soup.find_all("a", href=True, limit=num_results)
                    results = [r for r in results if 'uddg' in r.get('href', '')]
                
                urls = []
                for result in results:
                    try:
                        raw_url = result.get("href", "")
                        if raw_url:
                            if raw_url.startswith('/l/?uddg='):
                                # Extract the actual URL from DuckDuckGo's redirect
                                decoded_url = unquote(raw_url.split('uddg=')[1].split('&')[0])
                                urls.append(decoded_url)
                            elif raw_url.startswith('http'):
                                urls.append(raw_url)
                    except Exception as e:
                        logger.warning(f"Error parsing search result: {e}")
                        continue
                        
                logger.info(f"DuckDuckGo search found {len(urls)} results")
                return urls
                
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
            
    async def intelligent_content_analysis(self, results: List[ScrapingResult], query: str) -> Dict:
        """Analyze multiple content sources intelligently"""
        analysis_tasks = []
        
        for result in results:
            if result.success and result.content:
                task = self.analyzer.extract_key_information(result.content, query)
                analysis_tasks.append(task)
                
        if not analysis_tasks:
            return {"answer": "No valid content found", "confidence": 0.0}
            
        # Process analyses concurrently
        analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Filter out exceptions and low-confidence results
        valid_analyses = [
            analysis for analysis in analyses 
            if isinstance(analysis, dict) and analysis.get("confidence", 0) > 0.2
        ]
        
        if not valid_analyses:
            return {"answer": "No reliable information found", "confidence": 0.0}
            
        # Find the best analysis or combine multiple high-confidence ones
        best_analysis = max(valid_analyses, key=lambda x: x.get("confidence", 0))
        
        if best_analysis.get("answer_found", False) and best_analysis.get("confidence", 0) > 0.6:
            return {
                "answer": best_analysis["answer"],
                "confidence": best_analysis["confidence"],
                "sources": len(valid_analyses),
                "key_facts": best_analysis.get("key_facts", [])
            }
            
        # If no direct answer, synthesize from multiple sources
        return await self._synthesize_from_multiple_sources(valid_analyses, query)
        
    async def _synthesize_from_multiple_sources(self, analyses: List[Dict], query: str) -> Dict:
        """Synthesize answer from multiple content analyses"""
        all_facts = []
        all_entities = []
        
        for analysis in analyses:
            all_facts.extend(analysis.get("key_facts", []))
            all_entities.extend(analysis.get("entities", []))
            
        # Limit facts and entities to avoid overwhelming the LLM
        all_facts = list(set(all_facts))[:20]  # Remove duplicates and limit
        all_entities = list(set(all_entities))[:15]
        
        # Use LLM to synthesize final answer
        synthesis_prompt = f"""
        Based on the following facts and entities extracted from multiple sources, 
        provide a comprehensive answer to the query: "{query}"
        
        Facts: {all_facts}
        Entities: {all_entities}
        
        Provide a clear, factual answer based on the available information.
        If the information is insufficient, clearly state what is missing.
        Keep the response concise and focused.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": "You are an expert analyst. Provide accurate, well-sourced answers."},
                    {"role": "user", "content": synthesis_prompt}
                ]
            )
            
            return {
                "answer": response.choices[0].message.content,
                "confidence": 0.7,  # Medium confidence for synthesized answers
                "sources": len(analyses),
                "synthesis": True
            }
            
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            return {"answer": "Error synthesizing information", "confidence": 0.0}
            
    async def process_query_enhanced(self, query: str, max_pages: int = 10) -> Dict:
        """Enhanced query processing with intelligent prioritization"""
        start_time = time.time()
        
        try:
            # Step 1: Parse query and get search results
            search_results = await self.enhanced_search(query, max_pages)
            
            if not search_results:
                return {
                    "answer": "No search results found",
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time,
                    "pages_scraped": 0,
                    "successful_scrapes": 0,
                    "search_results_count": 0
                }
                
            logger.info(f"Found {len(search_results)} search results")
            
            # Step 2: Scrape content concurrently with rate limiting
            scraping_tasks = []
            for url in search_results:
                task = self.scrape_content_optimized(url)
                scraping_tasks.append(task)
                
            # Process in batches to avoid overwhelming
            batch_size = self.config.max_concurrent_requests
            all_results = []
            
            for i in range(0, len(scraping_tasks), batch_size):
                batch = scraping_tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                
                # Filter out exceptions
                valid_results = [
                    result for result in batch_results 
                    if isinstance(result, ScrapingResult)
                ]
                all_results.extend(valid_results)
                
                # Rate limiting between batches
                if i + batch_size < len(scraping_tasks):
                    await asyncio.sleep(self.config.request_delay)
                    
            # Step 3: Analyze content and generate answer
            analysis_result = await self.intelligent_content_analysis(all_results, query)
            
            # Step 4: Add metadata
            successful_scrapes = sum(1 for r in all_results if r.success)
            
            return {
                **analysis_result,
                "processing_time": time.time() - start_time,
                "pages_scraped": len(all_results),
                "successful_scrapes": successful_scrapes,
                "search_results_count": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced query processing: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "pages_scraped": 0,
                "successful_scrapes": 0,
                "search_results_count": 0
            }

# Usage example with proper async context management
async def main():
    """Example usage of the enhanced system"""
    config = ScrapingConfig(
        max_concurrent_requests=10,
        request_delay=0.5,
        max_retries=3,
        use_cache=True,
        cache_ttl=3600
    )
    
    async with EnhancedWebScraperAI(config) as scraper:
        queries = [
            # "What are the latest developments in AI web scraping?",
            # "How is generative AI being used in web automation?",
            # "What are the current challenges in web scraping?"
            # "Who The Head Of Department Of Computer Science Department In IIT Roorkee?",
            "Who Is Omer Mullick?",
        ]
        
        for query in queries:
            print(f"\n{'='*80}")
            print(f"Processing query: {query}")
            print(f"{'='*80}")
            
            result = await scraper.process_query_enhanced(query, max_pages=10)
            
            print(f"\nAnswer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            print(f"Search results found: {result.get('search_results_count', 0)}")
            print(f"Pages scraped: {result.get('pages_scraped', 0)}")
            print(f"Successful scrapes: {result.get('successful_scrapes', 0)}")
            
            if result.get('key_facts'):
                print(f"Key facts found: {len(result['key_facts'])}")
                for fact in result['key_facts'][:3]:  # Show first 3 facts
                    print(f"  - {fact}")
            
            print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(main())