import asyncio
from collections import deque
from typing import Dict, List, Set, Optional
from pydantic import BaseModel
from crawl4ai.models import CrawlResult
from crawl4ai import AsyncWebCrawler
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeGraph(BaseModel):
    """
    Represents a knowledge graph structure for storing extracted information.
    Attributes:
        entities (List[dict]): List of entities extracted from the content
        relationships (List[dict]): List of relationships between entities
    """
    entities: List[dict]
    relationships: List[dict]

class SimpleWebsiteScraper:
    """
    A web scraper that crawls websites and extracts content using LLM-based strategies.
    Attributes:
        crawler (AsyncWebCrawler): The async crawler instance used for web requests
        base_url (str): The starting URL for the crawl
        strategy (LLMExtractionStrategy): The strategy used for content extraction
    """

    def __init__(self, crawler: AsyncWebCrawler):
        """
        Initialize the scraper with a crawler instance.
        Args:
            crawler (AsyncWebCrawler): An instance of AsyncWebCrawler
        """
        self.crawler = crawler
        self.base_url: Optional[str] = None
        try:
            self.strategy = LLMExtractionStrategy(
                provider="ollama/qwen2.5",
                instruction="Extract meaningful content while filtering out navigation elements, headers, footers. Maintain proper context hierarchy of the documentation"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM strategy: {str(e)}")
            raise

    def is_valid_internal_link(self, link: str) -> bool:
        """
        Check if a link is a valid internal link within the base domain.
        Args:
            link (str): The URL to check
        Returns:
            bool: True if the link is valid and internal, False otherwise
        """
        try:
            if not link or link.startswith('#'):
                return False
            
            parsed_base = urlparse(self.base_url)
            parsed_link = urlparse(link)
            
            return (parsed_base.netloc == parsed_link.netloc and
                    parsed_link.path not in ['', '/'] and
                    parsed_link.path.startswith(parsed_base.path))
        except Exception as e:
            logger.warning(f"Error validating link {link}: {str(e)}")
            return False

    def normalize_url(self, url: str) -> str:
        """
        Normalize a URL by removing fragments and trailing slashes.
        Args:
            url (str): The URL to normalize
        Returns:
            str: The normalized URL
        Raises:
            ValueError: If the URL is invalid
        """
        try:
            parsed = urlparse(url)
            parsed = parsed._replace(fragment='')
            if parsed.path.endswith('/') and len(parsed.path) > 1:
                parsed = parsed._replace(path=parsed.path.rstrip('/'))
            return urlunparse(parsed)
        except Exception as e:
            logger.error(f"Failed to normalize URL {url}: {str(e)}")
            raise ValueError(f"Invalid URL format: {url}")

    def join_url(self, base: str, url: str) -> str:
        """
        Join a relative URL with a base URL while preserving the base path.
        Args:
            base (str): The base URL
            url (str): The URL to join
        Returns:
            str: The joined URL
        Raises:
            ValueError: If either URL is invalid
        """
        try:
            joined = urljoin(base, url)
            parsed_base = urlparse(self.base_url)
            parsed_joined = urlparse(joined)
            
            if not parsed_joined.path.startswith(parsed_base.path):
                new_path = parsed_base.path.rstrip('/') + '/' + parsed_joined.path.lstrip('/')
                parsed_joined = parsed_joined._replace(path=new_path)
            
            return urlunparse(parsed_joined)
        except Exception as e:
            logger.error(f"Failed to join URLs {base} and {url}: {str(e)}")
            raise ValueError(f"Invalid URL format: base={base}, url={url}")

    async def scrape(self, start_url: str, max_depth: int) -> Dict[str, CrawlResult]:
        """
        Scrape a website starting from a given URL up to a maximum depth.
        Args:
            start_url (str): The URL to start crawling from
            max_depth (int): Maximum depth to crawl
        Returns:
            Dict[str, CrawlResult]: Dictionary mapping URLs to their crawl results
        Raises:
            ValueError: If the start_url is invalid
            RuntimeError: If crawling fails
        """
        try:
            self.base_url = self.normalize_url(start_url)
        except ValueError as e:
            logger.error(f"Invalid start URL: {str(e)}")
            raise

        results: Dict[str, CrawlResult] = {}
        queue: deque = deque([(self.base_url, 0)])
        visited: Set[str] = set()

        while queue:
            try:
                current_url, current_depth = queue.popleft()
                
                if current_url in visited or current_depth > max_depth:
                    continue
                
                visited.add(current_url)
                logger.info(f"Crawling {current_url} at depth {current_depth}")
                
                result = await self.crawler.arun(current_url)
                
                if result.success:
                    results[current_url] = result
                    
                    if current_depth < max_depth:
                        internal_links = result.links.get('internal', [])
                        for link in internal_links:
                            try:
                                full_url = self.join_url(current_url, link['href'])
                                normalized_url = self.normalize_url(full_url)
                                if self.is_valid_internal_link(normalized_url) and normalized_url not in visited:
                                    queue.append((normalized_url, current_depth + 1))
                            except ValueError as e:
                                logger.warning(f"Skipping invalid URL: {str(e)}")
                                continue
                else:
                    logger.warning(f"Failed to crawl {current_url}: {result.error}")
                    
            except Exception as e:
                logger.error(f"Error processing URL {current_url}: {str(e)}")
                continue

        return results

class CustomCrawler:
    """
    Custom crawler implementation that saves results to a markdown file.
    """

    async def scrapper_tool(self, start_url: str, depth: int):
        """
        Scrape a website and save results to a markdown file.
        Args:
            start_url (str): The URL to start crawling from
            depth (int): Maximum depth to crawl
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If crawling or file writing fails
        """
        if depth < 0:
            raise ValueError("Depth must be non-negative")
            
        try:
            async with AsyncWebCrawler() as crawler:
                scraper = SimpleWebsiteScraper(crawler)
                results = await scraper.scrape(start_url, depth)

                output_file = Path("crawl_results.md")
                try:
                    with output_file.open("w", encoding="utf-8") as file:
                        file.write("# Crawl Results\n\n")
                        file.write(f"**Crawled {len(results)} pages starting from {start_url}**\n\n")
                        
                        for url, result in list(results.items())[:1]:  # Fixed the indexing
                            internal_links = len(result.links.get('internal', []))
                            external_links = len(result.links.get('external', []))
                            file.write(f"## {url}\n")
                            file.write(f"- **Internal Links**: {internal_links}\n")
                            file.write(f"- **External Links**: {external_links}\n\n")
                            
                            if result.markdown:
                                file.write(f"### Page Content\n{result.markdown}\n\n")
                    
                    logger.info(f"Crawl results saved to {output_file}")
                    
                except IOError as e:
                    logger.error(f"Failed to write results to file: {str(e)}")
                    raise RuntimeError(f"Failed to save results: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Crawler failed: {str(e)}")
            raise RuntimeError(f"Crawling failed: {str(e)}")

# if __name__ == "__main__":
#     try:
#         tool_tst = CustomCrawler()
#         start_url = "https://slack.com/intl/en-in/help"
#         depth = 1
        
#         asyncio.run(tool_tst.scrapper_tool(start_url, depth))
#     except Exception as e:
#         logger.error(f"Application failed: {str(e)}")
#         raise