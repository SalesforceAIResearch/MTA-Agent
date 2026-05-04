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

try:
    from .base import BaseTool, register_tool, strip_result_tags
except:
    from base import BaseTool, register_tool, strip_result_tags

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


class SerpApiImageSearchEngine:
    """
    Async SerpAPI Google Lens image search engine with proper session cleanup and caching.
    """

    def __init__(
        self,
        api_key: str,
        cache_file: Optional[str] = None,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        reverse_image_search_extracted_json: Optional[str] = None,
        reverse_image_search_inference_extracted_json: Optional[str] = None
    ):
        """Initialize the Google Lens image search engine with simplified configuration."""
        # API configuration
        self._api_key = api_key
        
        # Async-safe caching
        self._memory_cache = AsyncLRUCache(cache_size, cache_ttl)
        self._setup_cache_file(cache_file)
        
        # Reverse image search extracted JSON cache (rollout format)
        if reverse_image_search_extracted_json is None:
            _repo_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
            reverse_image_search_extracted_json = str(_repo_root / "reverse_image_search_extracted.json")
        self._reverse_image_search_extracted_json = pathlib.Path(reverse_image_search_extracted_json) if reverse_image_search_extracted_json else None
        self._reverse_image_search_cache = None
        self._reverse_image_search_cache_loaded = False
        self._reverse_image_search_cache_lock = asyncio.Lock()
        
        # Reverse image search inference extracted JSON (by_image_url_query_question -> <cache>, by_image_url_query_original -> no <cache>)
        if reverse_image_search_inference_extracted_json is None:
            _repo_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
            reverse_image_search_inference_extracted_json = str(_repo_root / "reverse_image_search_inference_extracted.json")
        self._reverse_image_search_inference_extracted_json = pathlib.Path(reverse_image_search_inference_extracted_json) if reverse_image_search_inference_extracted_json else None
        self._reverse_image_search_inference_by_image_url_query_question = None
        self._reverse_image_search_inference_by_image_url_query_original = None
        self._reverse_image_search_inference_cache_loaded = False
        self._reverse_image_search_inference_cache_lock = asyncio.Lock()
        
        # Performance tracking
        self._search_count = 0
    
    def _setup_cache_file(self, cache_file: Optional[str]) -> None:
        """Set up cache file path."""
        if cache_file is None:
            cache_dir = pathlib.Path.home() / ".verl_cache"
            cache_dir.mkdir(exist_ok=True)
            self._cache_file = cache_dir / f"serpapi_google_lens_cache.jsonl"
        else:
            self._cache_file = pathlib.Path(cache_file)
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def _load_persistent_cache(self) -> None:
        """Load cache from file asynchronously."""
        if not self._cache_file.exists():
            print(f"[Cache] Persistent cache file not found: {self._cache_file} (will create on first write)")
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
                print(f"[Cache] Loaded successfully: {self._cache_file} ({cache_entries} entries)")
        except Exception as e:
            print(f"[Cache] Load failed: {self._cache_file} - {e}")
    
    async def _load_reverse_image_search_extracted_cache(self) -> None:
        """Load reverse image search extracted JSON cache lazily."""
        if self._reverse_image_search_cache_loaded:
            return
        if self._reverse_image_search_extracted_json is None:
            print(f"[Cache] reverse_image_search_extracted not configured (path is None), skipping.")
            self._reverse_image_search_cache_loaded = True
            return
        async with self._reverse_image_search_cache_lock:
            if self._reverse_image_search_cache_loaded:
                return
            if not self._reverse_image_search_extracted_json.exists():
                print(f"[Cache] File not found (skipping): {self._reverse_image_search_extracted_json}")
                self._reverse_image_search_cache_loaded = True
                return
            try:
                loop = asyncio.get_event_loop()
                self._reverse_image_search_cache = await loop.run_in_executor(
                    None,
                    lambda: json.loads(self._reverse_image_search_extracted_json.read_text(encoding="utf-8"))
                )
                self._reverse_image_search_cache_loaded = True
                print(f"[Cache] Loaded successfully: {self._reverse_image_search_extracted_json} ({len(self._reverse_image_search_cache)} entries)")
            except Exception as e:
                print(f"[Cache] Load failed: {self._reverse_image_search_extracted_json} - {e}")
                self._reverse_image_search_cache = {}
                self._reverse_image_search_cache_loaded = True

    async def _load_reverse_image_search_inference_extracted_cache(self) -> None:
        """Load reverse image search inference extracted JSON (by_image_url_query_question -> with <cache>, by_image_url_query_original -> without <cache>)."""
        if self._reverse_image_search_inference_cache_loaded:
            return
        if self._reverse_image_search_inference_extracted_json is None:
            print(f"[Cache] reverse_image_search_inference_extracted not configured (path is None), skipping.")
            self._reverse_image_search_inference_cache_loaded = True
            return
        async with self._reverse_image_search_inference_cache_lock:
            if self._reverse_image_search_inference_cache_loaded:
                return
            if not self._reverse_image_search_inference_extracted_json.exists():
                print(f"[Cache] File not found (skipping): {self._reverse_image_search_inference_extracted_json}")
                self._reverse_image_search_inference_cache_loaded = True
                return
            try:
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None,
                    lambda: json.loads(self._reverse_image_search_inference_extracted_json.read_text(encoding="utf-8"))
                )
                self._reverse_image_search_inference_by_image_url_query_question = data.get("by_image_url_query_question") or {}
                self._reverse_image_search_inference_by_image_url_query_original = data.get("by_image_url_query_original") or {}
                self._reverse_image_search_inference_cache_loaded = True
                print(f"[Cache] Loaded successfully: {self._reverse_image_search_inference_extracted_json} (by_image_url_query_question={len(self._reverse_image_search_inference_by_image_url_query_question)}, by_image_url_query_original={len(self._reverse_image_search_inference_by_image_url_query_original)})")
            except Exception as e:
                print(f"[Cache] Load failed: {self._reverse_image_search_inference_extracted_json} - {e}")
                self._reverse_image_search_inference_by_image_url_query_question = {}
                self._reverse_image_search_inference_by_image_url_query_original = {}
                self._reverse_image_search_inference_cache_loaded = True
    
    async def _append_to_persistent_cache(self, query: str, result: Union[str, Dict]) -> None:
        """Append to persistent cache asynchronously."""
        try:
            entry = {"query": query, "result": result, "timestamp": time.time()}
            
            async with aiofiles.open(self._cache_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"Cache write failed: {e}")
    
    async def _make_reverse_image_search_request(self, image_url: str, timeout: int, query: Optional[str] = None) -> Dict:
        """
        Make reverse image search request using SerpAPI Google Lens with proper async handling
        and retry logic for rate limits and timeouts.
        """
        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                from serpapi import GoogleSearch
                params = {
                    "engine": "google_lens",
                    "url": image_url,
                    "api_key": self._api_key
                }
                loop = asyncio.get_event_loop()
                search = GoogleSearch(params)
                response = await loop.run_in_executor(None, lambda: search.get_dict())
                return response
            except ImportError:
                raise Exception("SerpAPI client not installed. Please install with: pip install google-search-results")
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
                    print(f"SerpAPI rate limit/timeout (attempt {attempt + 1}/{max_retries + 1}). Retrying after {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                if attempt >= max_retries:
                    raise Exception(f"SerpAPI Google Lens search error after {max_retries + 1} attempts: {str(e)}")
                raise Exception(f"SerpAPI Google Lens search error: {str(e)}")
    
    async def execute(self, image_url: str, timeout: int = None, query: Optional[str] = None, question: Optional[str] = None) -> str:
        """
        Execute Google Lens image search with comprehensive error handling and caching.
        
        Args:
            image_url: URL of the image to search
            timeout: Request timeout in seconds
            query: Optional text query (for display/context only, not used in API call)
            question: Optional research question for composite cache key
        """
        # Validate image URL
        image_url = image_url.strip()
        if not image_url:
            return "Empty image URL provided."
        
        if not self._is_valid_image_url(image_url):
            return "Invalid image URL format. Please provide a valid HTTP/HTTPS URL."
        
        # Normalize query and question to lowercase for cache key consistency
        query_normalized = query.strip().lower() if query and query.strip() else None
        question_normalized = question.strip().lower() if question and question.strip() else None
        
        try:
            # Create cache key that includes both image_url and query (normalized)
            cache_key = f"{image_url}|{query_normalized or ''}"
            
            # Check memory cache first
            cached_result = await self._memory_cache.get(cache_key)
            if cached_result is not None:
                return strip_result_tags(cached_result)
            
            # Check reverse_image_search_inference_extracted.json (by_image_url_query_question -> <cache>, by_image_url_query_original -> no <cache>)
            if not self._reverse_image_search_inference_cache_loaded:
                await self._load_reverse_image_search_inference_extracted_cache()
            if self._reverse_image_search_inference_by_image_url_query_question is not None and self._reverse_image_search_inference_by_image_url_query_original is not None:
                # by_image_url_query_question: key is image_url||query||question (or image_url when no query and no question)
                if question_normalized or query_normalized:
                    key_question = f"{image_url}||{query_normalized or ''}||{question_normalized or ''}"
                    if key_question in self._reverse_image_search_inference_by_image_url_query_question:
                        return f"<cache>{strip_result_tags(self._reverse_image_search_inference_by_image_url_query_question[key_question])}"
                if not question_normalized and not query_normalized:
                    if image_url in self._reverse_image_search_inference_by_image_url_query_question:
                        return f"<cache>{strip_result_tags(self._reverse_image_search_inference_by_image_url_query_question[image_url])}"
                # by_image_url_query_original: key is image_url||query||original or image_url||original
                key_original = f"{image_url}||{query_normalized or ''}||original" if query_normalized else f"{image_url}||original"
                if key_original in self._reverse_image_search_inference_by_image_url_query_original:
                    return strip_result_tags(self._reverse_image_search_inference_by_image_url_query_original[key_original])
            
            # Check reverse_image_search_extracted.json cache (rollout format)
            if not self._reverse_image_search_cache_loaded:
                await self._load_reverse_image_search_extracted_cache()
            if self._reverse_image_search_cache is not None:
                # Try composite keys with question first (if question is available)
                if question_normalized:
                    if query_normalized:
                        # Try: image_url||query||question
                        composite_key = f"{image_url}||{query_normalized}||{question_normalized}"
                        if composite_key in self._reverse_image_search_cache:
                            cached_value = self._reverse_image_search_cache[composite_key]
                            return f"<cache>{strip_result_tags(cached_value)}"
                    # Try: image_url||question
                    composite_key = f"{image_url}||{question_normalized}"
                    if composite_key in self._reverse_image_search_cache:
                        cached_value = self._reverse_image_search_cache[composite_key]
                        return f"<cache>{strip_result_tags(cached_value)}"
                
                # Fallback to keys without question
                if query_normalized:
                    # Try: image_url||query
                    cache_key_with_query = f"{image_url}||{query_normalized}"
                    if cache_key_with_query in self._reverse_image_search_cache:
                        cached_value = self._reverse_image_search_cache[cache_key_with_query]
                        return f"<cache>{strip_result_tags(cached_value)}"
                # Try: image_url (only if no query and no question)
                if not query_normalized and not question_normalized:
                    if image_url in self._reverse_image_search_cache:
                        cached_value = self._reverse_image_search_cache[image_url]
                        return f"<cache>{strip_result_tags(cached_value)}"
            
            # Make API request
            data = await self._make_reverse_image_search_request(image_url, timeout or 30, query)
            
            # Process results
            result = self._format_reverse_image_results(data, query)
            
            # Cache results
            await self._cache_results(cache_key, result)
            
            return result
            
        except Exception as e:
            if DEBUG:
                raise e
            error_msg = f"Google Lens image search failed for '{image_url}': {str(e)}"
            return error_msg
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Basic image URL validation."""
        import urllib.parse
        try:
            result = urllib.parse.urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False
            if result.scheme not in ['http', 'https']:
                return False
            # Check for common image extensions (optional, but helpful)
            common_image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']
            path_lower = result.path.lower()
            # Allow URLs without explicit extensions (many image URLs don't have them)
            return True
        except Exception:
            return False
    
    async def _cache_results(self, query: str, data: Union[str, Dict]) -> None:
        """Cache results in both memory and persistent storage."""
        try:
            # Memory cache
            await self._memory_cache.set(query, data)
            
            # Persistent cache
            cache_item = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            await self._append_to_persistent_cache(query, cache_item)
            
            self._search_count += 1
            
        except Exception as e:
            print(f"Caching failed: {e}")
    
    def _format_reverse_image_results(self, data: Dict, query: Optional[str] = None) -> str:
        """Format Google Lens search results with comprehensive information."""
        if not data:
            return "No Google Lens search results found."
        
        results = []
        
        # Add search query if provided
        if query:
            results.append(f"**Search Query:** {query}")
            results.append("")
        
        # Add visual matches (primary results from Google Lens)
        visual_matches = data.get('visual_matches', [])
        if visual_matches:
            results.append("**Visual Matches:**")
            for idx, match in enumerate(visual_matches[:10], 1):  # Limit to top 10
                position = match.get('position', idx)
                title = match.get('title', 'No title')
                link = match.get('link', '')
                source = match.get('source', '')
                thumbnail = match.get('thumbnail', '')
                
                formatted = f"\n**Match {position}:**\n"
                if title and title != 'No title':
                    formatted += f"Title: {title}\n"
                if source:
                    formatted += f"Source: {source}\n"
                if link:
                    formatted += f"Link: {link}\n"
                if thumbnail:
                    formatted += f"Thumbnail: {thumbnail}\n"
                
                results.append(formatted)
            results.append("")
        
        # Add knowledge graph if available
        knowledge_graph = data.get('knowledge_graph', {})
        if knowledge_graph:
            results.append("**Knowledge Graph Information:**")
            if 'title' in knowledge_graph:
                results.append(f"Title: {knowledge_graph['title']}")
            if 'description' in knowledge_graph:
                results.append(f"Description: {knowledge_graph['description']}")
            if 'source' in knowledge_graph:
                results.append(f"Source: {knowledge_graph['source']['name'] if isinstance(knowledge_graph['source'], dict) else knowledge_graph['source']}")
            if 'subtitle' in knowledge_graph:
                results.append(f"Subtitle: {knowledge_graph['subtitle']}")
            results.append("")
        
        # Add search information if available
        search_info = data.get('search_information', {})
        if search_info and 'total_results' in search_info:
            results.append("**Search Information:**")
            results.append(f"Total Results: {search_info['total_results']}")
            results.append("")
        
        # Add text results if available (related text information)
        text_results = data.get('text_results', [])
        if text_results:
            results.append("**Related Text Information:**")
            for idx, result in enumerate(text_results[:5], 1):  # Limit to top 5
                title = result.get('title', 'No title')
                link = result.get('link', '')
                snippet = result.get('snippet', '')
                
                formatted = f"\n**Result {idx}:**\n"
                if title and title != 'No title':
                    formatted += f"Title: {title}\n"
                if snippet:
                    formatted += f"Description: {snippet}\n"
                if link:
                    formatted += f"URL: {link}\n"
                
                results.append(formatted)
            results.append("")
        
        # Add inline images (similar images found)
        inline_images = data.get('inline_images', [])
        if inline_images:
            results.append("**Similar Images:**")
            for idx, image in enumerate(inline_images[:5], 1):  # Limit to top 5
                title = image.get('title', 'No title')
                link = image.get('link', '')
                source = image.get('source', '')
                thumbnail = image.get('thumbnail', '')
                
                formatted = f"\n**Image {idx}:**\n"
                if title and title != 'No title':
                    formatted += f"Title: {title}\n"
                if source:
                    formatted += f"Source: {source}\n"
                if link:
                    formatted += f"Link: {link}\n"
                if thumbnail:
                    formatted += f"Thumbnail: {thumbnail}\n"
                
                results.append(formatted)
        
        return "\n".join(results) if results else "No detailed information found for this image."


@register_tool
class WebImageToTextTool(BaseTool):
    """
    Async Google Lens image search tool using SerpAPI to find text information about images.
    """
    
    tool_type = "web_image_to_text"
    
    def __init__(
        self,
        num_workers=1,
        api_key: str = None,
        cache_file: Optional[str] = None,
        default_timeout: int = None,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        reverse_image_search_extracted_json: Optional[str] = None,
        reverse_image_search_inference_extracted_json: Optional[str] = None
    ):
        """Initialize the Google Lens image search tool with production settings."""
        super().__init__(num_workers)
        
        # Validate API key
        if api_key is None:
            api_key = os.getenv('SERPAPI_API_KEY')
            if api_key is None:
                raise ValueError(
                    "API key required: set SERPAPI_API_KEY environment variable or pass api_key parameter"
                )
        
        # Initialize search engine
        self.search_engine = SerpApiImageSearchEngine(
            api_key=api_key,
            cache_file=cache_file,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            reverse_image_search_extracted_json=reverse_image_search_extracted_json,
            reverse_image_search_inference_extracted_json=reverse_image_search_inference_extracted_json
        )
        
        self.default_timeout = default_timeout
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(16)  # Limit concurrent searches
    
    async def _ensure_initialized(self):
        """Ensure search engine is initialized (lazy initialization)."""
        if not self._initialized:
            async with self._init_lock:
                if not self._initialized:
                    await self.search_engine._load_persistent_cache()
                    self._initialized = True
    
    def get_usage_inst(self):
        """Get usage instructions."""
        return "Google Lens image search to find text information about images. Use <image_search_text>image_url||query</image_search_text> format. Query is optional (for display/context only)."
    
    def parse_action(self, action: str) -> Tuple[Union[str, Tuple[str, str]], bool]:
        """
        Parse action to extract image URL and optional query.
        
        Returns:
            Tuple of (parsed_content, is_valid) where:
            - parsed_content is either a string (image_url) or tuple (image_url, query)
            - is_valid is a boolean indicating if parsing succeeded
        """
        patterns = [
            r"<image_search_text>(.*?)</image_search_text>",
            r"```\s*image_search_text\s*\n(.*?)\n```",
            r"image_search_text:\s*(.*?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, action, re.DOTALL | re.IGNORECASE)
            if matches:
                content = matches[0].strip()
                if content:
                    # Check if content contains || separator for query
                    if "||" in content:
                        parts = content.split("||", 1)
                        image_url = parts[0].strip()
                        query = parts[1].strip()
                        return (image_url, query), True
                    else:
                        return content, True
        
        return "", False

    async def aget_observations(
        self, 
        trajectory_ids: List[str], 
        actions: List[str], 
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[Union[str, dict]], List[bool], List[bool]]:
        """
        Process multiple Google Lens image search actions concurrently with proper error handling.
        """
        await self._ensure_initialized()
        
        async def process_single_action(trajectory_id, action, extra_field):
            async with self.semaphore:
                try:
                    return await self._conduct_action_async(trajectory_id, action, extra_field)
                except Exception as e:
                    return f"Google Lens image search error: {str(e)}", False, False
        
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
                obs = f"Google Lens image search error: {str(result)}"
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
        Conduct single Google Lens image search action asynchronously.
        """
        parsed_content, is_valid = self.parse_action(action)
        
        # Load environment
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = "Invalid image search format. Use <image_search_text>image_url||query</image_search_text> format. Query is optional."
            done, valid = False, False
        else:
            # Get timeout from extra field
            timeout = extra_field.get('timeout', self.default_timeout)
            
            # Extract image URL and optional query
            if isinstance(parsed_content, tuple):
                image_url, query = parsed_content
            else:
                image_url = parsed_content
                query = None
            
            # If image_url is a local path, try to get the corresponding URL from extra_field
            if image_url and not (image_url.startswith("http://") or image_url.startswith("https://")):
                # It's a local path, try to find the corresponding URL
                image_urls = extra_field.get('image_urls', [])
                if image_urls:
                    from pathlib import Path
                    local_path = Path(image_url)
                    filename = local_path.name
                    
                    matched_url = None
                    
                    # Strategy 1: Try to match by filename pattern
                    # Local: mmsearch_plus_train_0_img_1.jpg -> URL: MMSearch-Plus_0_img_1.jpg
                    # Extract img_key (img_1, img_2, etc.) from local filename
                    if '_img_' in filename:
                        try:
                            # Extract img_key part (e.g., "img_1" from "mmsearch_plus_train_0_img_1.jpg")
                            img_key_match = re.search(r'_img_(\d+)', filename)
                            if img_key_match:
                                img_num = img_key_match.group(1)
                                # Look for URL with same img_key pattern
                                for url in image_urls:
                                    if url and f'_img_{img_num}' in url:
                                        matched_url = url
                                        break
                        except Exception:
                            pass
                    
                    # Strategy 2: Try to match by full filename (if URL contains the local filename)
                    if not matched_url:
                        for url in image_urls:
                            if url and filename in url:
                                matched_url = url
                                break
                    
                    # Strategy 3: Try to extract index from local path and match by position
                    # Format: mmsearch_plus_{split}_{idx}_{img_key}.jpg
                    if not matched_url and '_' in filename:
                        try:
                            parts = filename.split('_')
                            # Find the index part (usually the number before img_key)
                            for i, part in enumerate(parts):
                                if part.isdigit() and i + 1 < len(parts) and parts[i + 1].startswith('img'):
                                    # This is likely the image index
                                    img_idx = int(part)
                                    if 0 <= img_idx < len(image_urls):
                                        matched_url = image_urls[img_idx]
                                    break
                        except Exception:
                            pass
                    
                    # Strategy 4: If still no match, use first URL as fallback
                    if not matched_url and len(image_urls) > 0:
                        matched_url = image_urls[0]
                    
                    if matched_url:
                        image_url = matched_url
            
            try:
                # Execute Google Lens image search
                # Get question from extra field for composite cache key
                question = extra_field.get('question', None) if isinstance(extra_field, dict) else None
                
                # Execute search (pass question for composite cache key)
                search_results = await self.search_engine.execute(image_url, timeout, query, question=question)

                if search_results.startswith("<cache>") or search_results.startswith("<Google Lens image search results for>"):
                    observation = strip_result_tags(search_results)
                elif query and not search_results.startswith('Google Lens image search results'):
                    observation = f"Google Lens image search results for '{image_url}' with context '{query}':\n\n{strip_result_tags(search_results)}"
                elif not query and not search_results.startswith('Google Lens image search results'):
                    observation = f"Google Lens image search results for '{image_url}':\n\n{strip_result_tags(search_results)}"
                else:
                    observation = strip_result_tags(search_results)
                done, valid = False, True
                
            except Exception as e:
                if DEBUG:
                    raise e
                observation = f"Google Lens image search execution failed: {str(e)}"
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
                    return "Google Lens image search timed out", False, False
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
            return f"Google Lens image search failed: {str(e)}", False, False


if __name__ == "__main__":
    import os
    
    # Check API key
    api_key = os.getenv('SERPAPI_API_KEY')
    if not api_key:
        print("❌ Set SERPAPI_API_KEY environment variable")
        exit(1)
    
    # Test Google Lens image search
    tool = WebImageToTextTool(api_key=api_key)
    
    # Test query parsing without query
    parsed_content, valid = tool.parse_action("<image_search_text>https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/okvqa/images/mulberry_A-OKVQA_1_0.png</image_search_text>")
    print(f"Parsed (no query): '{parsed_content}', Valid: {valid}")
    
    # Test query parsing with query
    parsed_content, valid = tool.parse_action("<image_search_text>https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/FVQA/images/fvqa_train_4711.png||Hershey stock price April 7 2025</image_search_text>")
    print(f"Parsed (with query): '{parsed_content}', Valid: {valid}")

    
    # Test Google Lens image search execution with question
    image_url = "https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/Infoseek/images/409.png"
    question = "what is the street address of this building?"
    extra_field = {"question": question}
    observation, done, valid = tool.conduct_action("test2", f"<image_search_text>{image_url}</image_search_text>", extra_field)
    print('>>>', observation, '<<<')
    print("*" * 30)

    image_url = "https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/FVQA/images/fvqa_train_1115.png"
    query = "identify the university/campus in this image"
    question = "how many different actors' costumes are displayed alongside the woman in the blue-collared shirt?\n\noptions:\na. 1\nb. 2\nc. 3\nd. 4"
    extra_field = {"question": question}
    observation, done, valid = tool.conduct_action("test2",
                                                   f"<image_search_text>{image_url}||{query}</image_search_text>",
                                                   extra_field)
    print('>>>', observation, '<<<')

    image_url = "https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/FVQA/images/fvqa_train_1115.png"
    query = "identify the university/campus in this image"
    question = "in which city is the main campus of the organization associated with the campus in the image located?"
    extra_field = {"question": question}
    observation, done, valid = tool.conduct_action("test2", f"<image_search_text>{image_url}||{query}</image_search_text>", extra_field)
    print('>>>', observation, '<<<')

    
    print("✅ Test complete")
