import os
import json
import time
import pathlib
import asyncio
import aiofiles
from typing import Optional, Union, Dict, List, Any, Tuple
import regex as re
import faulthandler
from collections import OrderedDict

from .base import BaseTool, register_tool, strip_result_tags

faulthandler.enable()
DEBUG = False

class AsyncLRUCache:
    """Thread-safe LRU cache for async operations"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                # Check TTL
                if time.time() - self._timestamps[key] > self.ttl_seconds:
                    del self._cache[key]
                    del self._timestamps[key]
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: Any):
        async with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[key] = value
            self._timestamps[key] = time.time()


class WebContentExtractor:
    """
    Async web content extractor using Tavily API with proper session cleanup and caching.
    """

    def __init__(
        self,
        api_key: str,
        cache_file: Optional[str] = None,
        cache_size: int = 10000,
        cache_ttl: int = 3600
    ):
        """Initialize the content extractor with robust configuration."""
        # API configuration
        self._api_key = api_key
        
        # Async-safe caching
        self._memory_cache = AsyncLRUCache(cache_size, cache_ttl)
        self._setup_cache_file(cache_file)
        
        # Performance tracking
        self._extract_count = 0
    
    def _setup_cache_file(self, cache_file: Optional[str]) -> None:
        """Set up cache file path."""
        if cache_file is None:
            cache_dir = pathlib.Path.home() / ".verl_cache"
            cache_dir.mkdir(exist_ok=True)
            self._cache_file = cache_dir / f"tavily_extract_cache.jsonl"
        else:
            self._cache_file = pathlib.Path(cache_file)
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def _load_persistent_cache(self) -> None:
        """Load cache from file asynchronously."""
        if not self._cache_file.exists():
            return
            
        try:
            async with aiofiles.open(self._cache_file, "r", encoding="utf-8") as f:
                cache_entries = 0
                async for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            await self._memory_cache.set(item['query'], item['result'])
                            cache_entries += 1
                        except json.JSONDecodeError:
                            continue
                
                print(f"Loaded {cache_entries} cache entries from {self._cache_file}")
                
        except Exception as e:
            print(f"Failed to load cache: {e}")
    
    async def _append_to_persistent_cache(self, query: str, result: Union[str, Dict]) -> None:
        """Append to persistent cache asynchronously."""
        try:
            entry = {"query": query, "result": result, "timestamp": time.time()}
            
            async with aiofiles.open(self._cache_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"Cache write failed: {e}")
    
    async def _make_extract_request(self, urls: List[str], timeout: int) -> Dict:
        """
        Make extract request using Tavily API with proper async handling
        and retry logic for rate limits and timeouts.
        """
        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=self._api_key)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, lambda: client.extract(urls=urls))
                return response
            except ImportError:
                raise Exception("Tavily client not installed. Please install with: pip install tavily-python")
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "rate limit" in error_str or "429" in error_str or
                    "too many requests" in error_str or "quota" in error_str or
                    "excessive requests" in error_str or "rate" in error_str
                )
                is_timeout = "timeout" in error_str or "timed out" in error_str
                if (is_rate_limit or is_timeout) and attempt < max_retries:
                    delay = attempt + 1  # 1, 2, 3, 4, 5s
                    print(f"Tavily extract rate limit/timeout (attempt {attempt + 1}/{max_retries + 1}). Retrying after {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                if attempt >= max_retries:
                    raise Exception(f"Tavily extract API error after {max_retries + 1} attempts: {str(e)}")
                raise Exception(f"Tavily extract API error: {str(e)}")
    
    async def execute(self, urls: List[str], timeout: int = None) -> str:
        """
        Execute URL content extraction with comprehensive error handling and caching.
        """
        # Validate URLs
        if not urls:
            return "No URLs provided for extraction."
        
        # Create cache key from URLs
        cache_key = f"extract:{':'.join(sorted(urls))}"
        
        try:
            # Check memory cache first
            cached_result = await self._memory_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Make API request
            data = await self._make_extract_request(urls, timeout or 30)
            
            # Process results
            result = self._format_extract_results(data)
            
            # Cache results
            await self._cache_results(cache_key, result)
            
            return result
            
        except Exception as e:
            if DEBUG:
                raise e
            error_msg = f"URL extraction failed for {urls}: {str(e)}"
            return error_msg
    
    async def _cache_results(self, query: str, data: Union[str, Dict]) -> None:
        """Cache results in both memory and persistent storage."""
        try:
            # Memory cache
            await self._memory_cache.set(query, data)
            
            # Persistent cache
            cache_item = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            await self._append_to_persistent_cache(query, cache_item)
            
            self._extract_count += 1
            
        except Exception as e:
            print(f"Caching failed: {e}")
    
    def _format_extract_results(self, data: Dict) -> str:
        """Format extract results using only raw_content."""
        if 'results' not in data or not data['results']:
            return "No content extracted from URLs."
        
        results = []
        for idx, result in enumerate(data['results'], 1):
            url = result.get('url', 'Unknown URL')
            raw_content = result.get('raw_content', '').strip()
            
            if raw_content:
                formatted = f"**URL {idx}:** {url}\n**Content:**\n{raw_content}\n"
                results.append(formatted)
        
        # Handle failed results
        if 'failed_results' in data and data['failed_results']:
            failed_urls = [f"- {url}" for url in data['failed_results']]
            if results:
                results.append(f"\n**Failed to extract content from:**\n" + "\n".join(failed_urls))
            else:
                return f"Failed to extract content from all URLs:\n" + "\n".join(failed_urls)
        
        return "\n".join(results) if results else "No content could be extracted from the provided URLs."


@register_tool
class WebReaderTool(BaseTool):
    """
    Async web content extraction tool using Tavily API with proper cleanup.
    Only handles <web_read>url</web_read> tags.
    """
    
    tool_type = "web_reader"
    
    def __init__(
        self,
        num_workers=1,
        api_key: str = None,
        cache_file: Optional[str] = None,
        default_timeout: int = None,
        cache_size: int = 10000,
        cache_ttl: int = 3600
    ):
        """Initialize the web reader tool with production settings."""
        super().__init__(num_workers)
        
        # Validate API key
        if api_key is None:
            api_key = os.getenv('TAVILY_API_KEY')
            if api_key is None:
                raise ValueError(
                    "API key required: set TAVILY_API_KEY environment variable or pass api_key parameter"
                )
        
        # Initialize content extractor
        self.extractor = WebContentExtractor(
            api_key=api_key,
            cache_file=cache_file,
            cache_size=cache_size,
            cache_ttl=cache_ttl
        )
        
        self.default_timeout = default_timeout
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(16)  # Limit concurrent requests
    
    async def _ensure_initialized(self):
        """Ensure extractor is initialized (lazy initialization)."""
        if not self._initialized:
            async with self._init_lock:
                if not self._initialized:
                    await self.extractor._load_persistent_cache()
                    self._initialized = True
    
    def get_usage_inst(self):
        """Get usage instructions."""
        return "Extract content from URLs: <web_read>url</web_read>"
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """Parse action to extract URL from web_read tags with improved patterns."""
        patterns = [
            r"<web_read>(.*?)</web_read>",
            r"```\s*web_read\s*\n(.*?)\n```",
            r"web_read:\s*(.*?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, action, re.DOTALL | re.IGNORECASE)
            if matches:
                url = matches[0].strip()
                if url and self._is_valid_url(url):
                    return url, True
        
        return "", False
    
    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation."""
        import urllib.parse
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False

    async def aget_observations(
        self, 
        trajectory_ids: List[str], 
        actions: List[str], 
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[Union[str, dict]], List[bool], List[bool]]:
        """
        Process multiple web_read actions concurrently with proper error handling.
        """
        await self._ensure_initialized()
        
        async def process_single_action(trajectory_id, action, extra_field):
            async with self.semaphore:
                try:
                    return await self._conduct_action_async(trajectory_id, action, extra_field)
                except Exception as e:
                    return f"Web read error: {str(e)}", False, False
        
        # Create tasks for all actions
        tasks = [
            process_single_action(trajectory_id, action, extra_field)
            for trajectory_id, action, extra_field in zip(trajectory_ids, actions, extra_fields)
        ]
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
                
        # Unpack results and handle exceptions
        observations, dones, valids = [], [], []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if DEBUG:
                    raise result
                obs = f"Web read error: {str(result)}"
                done, valid = False, False
            else:
                obs, done, valid = result
            
            observations.append(obs)
            dones.append(done)
            valids.append(valid)
        
        # Cleanup environments
        self.maybe_cleanup_env(trajectory_ids, actions, extra_fields)
        
        return observations, dones, valids
    
    async def _conduct_action_async(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        """
        Conduct single web_read action asynchronously.
        """
        url, is_valid = self.parse_action(action)
        
        # Load environment
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = "Invalid web_read format. Use <web_read>url</web_read> format with a valid URL."
            done, valid = False, False
        else:
            # Get timeout from extra field
            timeout = extra_field.get('timeout', self.default_timeout)
            
            try:
                # Execute URL content extraction
                extract_results = await self.extractor.execute([url], timeout)
                observation = f"Content extracted from '{url}':\n\n{strip_result_tags(extract_results)}"
                done, valid = False, True
                
            except Exception as e:
                if DEBUG:
                    raise e
                observation = f"URL extraction failed: {str(e)}"
                done, valid = False, False
        
        # Update and save environment
        self.update_env(trajectory_id, env, action, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    
    def conduct_action(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        """
        Synchronous wrapper that properly handles async code.
        Creates a new event loop if needed to avoid conflicts.
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new thread to run async code
                import concurrent.futures
                import threading
                
                result = [None]
                exception = [None]
                
                def run_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result[0] = new_loop.run_until_complete(
                            self._conduct_action_async(trajectory_id, action, extra_field)
                        )
                    except Exception as e:
                        if DEBUG:
                            raise e
                        exception[0] = e
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_in_new_loop)
                thread.start()
                thread.join(timeout=60)  # 60 second timeout
                
                if exception[0]:
                    raise exception[0]
                if result[0] is None:
                    return "Web read timed out", False, False
                return result[0]
            else:
                # Use existing loop if not running
                return loop.run_until_complete(
                    self._conduct_action_async(trajectory_id, action, extra_field)
                )
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(self._conduct_action_async(trajectory_id, action, extra_field))
        except Exception as e:
            if DEBUG:
                raise e
            return f"Web read failed: {str(e)}", False, False


if __name__ == "__main__":
    # Test with same structure as web_text_to_text_search.py
    api_key = os.getenv('TAVILY_API_KEY')
    if not api_key:
        print("❌ Set TAVILY_API_KEY environment variable")
        exit(1)
    
    # Test basic web reader
    tool = WebReaderTool(api_key=api_key)
    
    # Test URL parsing only (safe test)
    url, valid = tool.parse_action("<web_read>https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval</web_read>")
    print(f"Parsed URL: '{url}', Valid: {valid}")
    
    # Test with invalid format
    # Test search execution
    observation, done, valid = tool.conduct_action("test", "<web_read>https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval</web_read>", {})
    print(f"Image search result - Valid: {valid}, Done: {done}")
    print(f"Preview: {observation[:2000]}...")
    print(len(observation.split()))

    print("✅ Test complete")


