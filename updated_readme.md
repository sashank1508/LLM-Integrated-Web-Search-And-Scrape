# Enhanced WebScraperAI 🔍🤖

An intelligent, AI-powered web research system that autonomously navigates the internet to answer complex questions through deep content analysis and link following.

## 🌟 Features

### 🧠 **Intelligent Query Processing**

- **Natural Language Input**: Process queries in plain English with or without URLs
- **Smart URL Detection**: Automatically detects URLs in text or searches the web
- **AI-Powered Analysis**: Uses OpenAI GPT-4 to understand and analyze content
- **Context Preservation**: Maintains full context when splitting queries

### 🕷️ **Advanced Web Scraping**

- **Multi-Source Scraping**: Scrapes multiple URLs simultaneously with controlled concurrency
- **Deep Link Following**: Intelligently follows relevant links up to configurable depth
- **Robust Error Handling**: Implements retry logic with exponential backoff
- **Clean Content Extraction**: Uses Jina.ai reader for clean, readable content

### 🔍 **Intelligent Search**

- **Multi-Engine Search**: Primary DuckDuckGo with Google fallback
- **Adaptive URL Construction**: Builds relevant URLs for common sites and patterns
- **Result Filtering**: Validates and normalizes search results
- **Loop Prevention**: Tracks visited URLs to avoid infinite loops

### 📊 **Comprehensive Analytics**

- **Processing Statistics**: Detailed metrics on URLs visited, failed, and processing time
- **Depth Tracking**: Monitors how deep the scraper goes into link chains
- **Performance Monitoring**: Real-time logging and progress tracking
- **Export Capabilities**: Saves results with full audit trails

## 🚀 Quick Start

### Prerequisites

```bash
pip install aiohttp asyncio logging requests beautifulsoup4 openai python-dotenv pathlib
```

### Environment Setup

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Basic Usage

```python
import asyncio
from enhanced_webscraper import EnhancedWebScraperAI

async def main():
    # Initialize the scraper
    scraper = EnhancedWebScraperAI(
        max_depth=5,
        max_concurrent_requests=3,
        request_delay=1.0
    )
    
    # Process a query
    result = await scraper.process_query_enhanced("What does Tesla do?")
    print(result)

# Run the scraper
asyncio.run(main())
```

## 📖 Detailed Usage

### Initialization Parameters

```python
scraper = EnhancedWebScraperAI(
    max_depth=5,                    # Maximum depth to follow links
    max_concurrent_requests=3,      # Concurrent HTTP requests
    request_delay=1.0              # Delay between requests (seconds)
)
```

### Query Types

#### 1. **Direct URL Queries**

```python
# Scrape specific website with question
result = await scraper.process_query_enhanced(
    "What does https://openai.com do?"
)
```

#### 2. **Search-Based Queries**

```python
# Search the web for information
result = await scraper.process_query_enhanced(
    "Tesla stock price today"
)
```

#### 3. **Complex Research Queries**

```python
# Deep research with multiple sources
result = await scraper.process_query_enhanced(
    "What are the latest developments in quantum computing?"
)
```

### Export Results

```python
# Export results to file with detailed statistics
result, filepath = await scraper.process_query_enhanced(
    "Your query here",
    export_results=True
)

print(f"Results saved to: {filepath}")
```

### Skip Specific URLs

```python
# Avoid certain URLs during scraping
skip_urls = ["https://example-spam-site.com"]
result = await scraper.process_query_enhanced(
    "Your query",
    skip_urls=skip_urls
)
```

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    EnhancedWebScraperAI                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Input Parser  │  │  Search Engine  │  │   Scraper    │ │
│  │                 │  │                 │  │              │ │
│  │ • URL Detection │  │ • DuckDuckGo    │  │ • Jina.ai    │ │
│  │ • Query Split   │  │ • Google        │  │ • Retry      │ │
│  │ • Normalization │  │ • Fallbacks     │  │ • Parallel   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  AI Analyzer    │  │  Link Follower  │  │   Exporter   │ │
│  │                 │  │                 │  │              │ │
│  │ • Content Eval  │  │ • Depth Track   │  │ • File Save  │ │
│  │ • Next URL      │  │ • Loop Prevent  │  │ • Statistics │ │
│  │ • Synthesis     │  │ • Smart Follow  │  │ • Audit Log  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Query → Input Processing → Search/Direct URL → Parallel Scraping
     ↓
Content Analysis → Link Following → Deep Scraping → Answer Synthesis
     ↓
Final Answer + Export + Statistics
```

## 🔧 Configuration

### Advanced Configuration

```python
class Config:
    MAX_DEPTH = 6                      # Maximum link following depth
    MAX_CONCURRENT_REQUESTS = 4        # Parallel request limit
    REQUEST_DELAY = 0.5               # Delay between requests
    MAX_RETRIES = 3                   # Retry attempts per URL
    TIMEOUT = 15                      # Request timeout seconds
    SEARCH_RESULTS = 15               # Number of search results
    
scraper = EnhancedWebScraperAI(
    max_depth=Config.MAX_DEPTH,
    max_concurrent_requests=Config.MAX_CONCURRENT_REQUESTS,
    request_delay=Config.REQUEST_DELAY
)
```

### Custom Headers and User Agents

The scraper automatically uses appropriate headers for web requests:

```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive'
}
```

## 📊 Output Examples

### Console Output

```
================================================================================
Processing query: What does Tesla do?
================================================================================
[2024-01-15 10:30:15] - INFO - Direct URL detected: https://tesla.com
[2024-01-15 10:30:16] - INFO - Scraping depth 0: https://tesla.com
[2024-01-15 10:30:17] - INFO - Analysis result - Answer: Tesla is an electric vehicle...
[2024-01-15 10:30:17] - INFO - Next URL: https://tesla.com/about

Final Answer: Tesla, Inc. is an American electric vehicle and clean energy company...

Processing Statistics:
- Processing time: 12.45 seconds
- URLs visited: 3
- URLs failed: 0
- Max depth reached: 2
```

### Exported File Structure

```
================================================================================
WEB SCRAPER AI - QUERY RESULTS
================================================================================

Timestamp: 2024-01-15 10:30:29

QUESTION:
What does Tesla do?

ANSWER:
Tesla, Inc. is an American electric vehicle and clean energy company...

PROCESSING STATISTICS:
- Processing Time: 12.45 seconds
- URLs Visited: 3
- URLs Failed: 0
- Max Depth Reached: 2

URL DEPTH DISTRIBUTION:
  Depth 0: 1 URL
  Depth 1: 1 URL
  Depth 2: 1 URL

VISITED URLS:
  1. [Depth 0] https://tesla.com
  2. [Depth 1] https://tesla.com/about
  3. [Depth 2] https://tesla.com/mission

FAILED URLS:
No URLs failed
```

## 🛠️ API Reference

### Main Methods

#### `process_query_enhanced(input_text, skip_urls=None, export_results=False)`

Primary method for processing queries.

**Parameters:**

- `input_text` (str): The query or URL to process
- `skip_urls` (List[str], optional): URLs to avoid during scraping
- `export_results` (bool): Whether to export results to file

**Returns:**

- `str`: The final answer if export_results=False
- `Tuple[str, str]`: (answer, filepath) if export_results=True

#### `deep_scrape_single_url(start_url, query, max_depth=None)`

Deep scrape a single URL following links.

**Parameters:**

- `start_url` (str): Starting URL
- `query` (str): Question to answer
- `max_depth` (int, optional): Maximum depth to explore

**Returns:**

- `List[str]`: List of collected answers

#### `get_processing_stats()`

Get detailed processing statistics.

**Returns:**

- `dict`: Statistics including visited URLs, failed URLs, and depth distribution

### Data Classes

#### `ScrapingResult`

```python
@dataclass
class ScrapingResult:
    url: str
    content: str
    success: bool
    error: Optional[str] = None
    depth: int = 0
    parent_url: Optional[str] = None
```

#### `AnalysisResult`

```python
@dataclass
class AnalysisResult:
    answer: str
    next_url: Optional[str]
    confidence: float = 0.0
    source_url: str = ""
```

## 🚨 Error Handling

The scraper includes comprehensive error handling:

### Network Errors

- Automatic retry with exponential backoff
- Multiple search engine fallbacks
- Graceful degradation when sources fail

### Content Analysis Errors

- Fallback to alternative analysis methods
- Partial result aggregation
- Clear error reporting

### Rate Limiting

- Built-in request delays
- Concurrent request limiting
- Respectful scraping practices

## 🔍 Debugging

### Debug Search Functionality

```python
# Test search capabilities
debug_info = await scraper.debug_search("your search query")
print(debug_info)
```

### Enable Verbose Logging

```python
import logging
logging.getLogger("WebScraperAI").setLevel(logging.DEBUG)
```

## 📝 Use Cases

### 🏢 **Business Intelligence**

```python
# Research competitors
result = await scraper.process_query_enhanced(
    "What are the latest product releases from Apple?"
)
```

### 📊 **Market Research**

```python
# Analyze market trends
result = await scraper.process_query_enhanced(
    "Current trends in electric vehicle market 2024"
)
```

### 🎓 **Academic Research**

```python
# Research academic topics
result = await scraper.process_query_enhanced(
    "Latest research papers on machine learning in healthcare"
)
```

### 💼 **Due Diligence**

```python
# Company research
result = await scraper.process_query_enhanced(
    "Financial performance and leadership of https://company.com"
)
```

## ⚠️ Best Practices

### Rate Limiting

- Use appropriate delays between requests
- Respect robots.txt files
- Monitor server response times

### Content Quality

- Verify information from multiple sources
- Check source credibility
- Cross-reference important claims

### Resource Management

- Limit concurrent requests appropriately
- Set reasonable depth limits
- Monitor memory usage for large scraping jobs

## 🚫 Limitations

- **Rate Limits**: Some websites may block automated requests
- **JavaScript Content**: May not capture dynamically loaded content
- **Authentication**: Cannot access content behind login walls
- **Legal Compliance**: Users must ensure compliance with website terms of service

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

### Development Setup

```bash
git clone https://github.com/yourusername/enhanced-webscraper-ai.git
cd enhanced-webscraper-ai
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- Jina.ai for content extraction service
- BeautifulSoup for HTML parsing
- aiohttp for async HTTP requests

---

**Made with ❤️ for intelligent web research**
