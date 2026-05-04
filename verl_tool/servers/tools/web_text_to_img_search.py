import os
import json
import time
import pathlib
import asyncio
import aiofiles
from typing import Optional, Union, Dict, List, Any, Tuple
import regex as re
import faulthandler
import langid
from collections import OrderedDict

try:
    from .base import BaseTool, register_tool, strip_result_tags
    from .utils.deepsearch_utils import extract_relevant_info_tavily, extract_text_from_url, extract_snippet_with_context
    from .utils.web_agent_utils import generate_webpage_to_reasonchain, get_prev_reasoning_chain
    from .web_text_to_text_search import AsyncLRUCache
except:
    from base import BaseTool, register_tool, strip_result_tags
    from utils.deepsearch_utils import extract_relevant_info_tavily, extract_text_from_url, extract_snippet_with_context
    from utils.web_agent_utils import generate_webpage_to_reasonchain, get_prev_reasoning_chain
    from web_text_to_text_search import AsyncLRUCache

faulthandler.enable()
DEBUG = False


class TavilyImageSearchEngine:
    """
    Async Tavily search engine with proper session cleanup and caching.
    """

    def __init__(
        self,
        api_key: str,
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "us",
        language: str = "en",
        cache_file: Optional[str] = None,
        process_snippets: bool = False,
        summ_model_url: str = None,
        summ_model_path: str = None,
        max_doc_len: int = 3000,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        image_search_extracted_json: Optional[str] = None,
        image_search_inference_extracted_json: Optional[str] = None
    ):
        """Initialize the search engine with simplified configuration."""
        # API configuration
        self._api_key = api_key
        self._max_results = max_results
        self._result_length = result_length
        self._location = location
        self._language = language
        self.process_snippets = process_snippets
        self.summ_model_url = summ_model_url
        self.summ_model_path = summ_model_path
        self._max_doc_len = max_doc_len
        
        # Async-safe caching
        self._memory_cache = AsyncLRUCache(cache_size, cache_ttl)
        self._setup_cache_file(cache_file)
        
        # Image search extracted JSON cache (rollout format)
        if image_search_extracted_json is None:
            _repo_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
            image_search_extracted_json = str(_repo_root / "image_search_extracted.json")
        self._image_search_extracted_json = pathlib.Path(image_search_extracted_json) if image_search_extracted_json else None
        self._image_search_cache = None
        self._image_search_cache_loaded = False
        self._image_search_cache_lock = asyncio.Lock()
        
        # Image search inference extracted JSON (by_query_question -> <cache>, by_query_original -> no <cache>)
        if image_search_inference_extracted_json is None:
            _repo_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
            image_search_inference_extracted_json = str(_repo_root / "image_search_inference_extracted.json")
        self._image_search_inference_extracted_json = pathlib.Path(image_search_inference_extracted_json) if image_search_inference_extracted_json else None
        self._image_search_inference_by_query_question = None
        self._image_search_inference_by_query_original = None
        self._image_search_inference_cache_loaded = False
        self._image_search_inference_cache_lock = asyncio.Lock()
        
        # Performance tracking
        self._search_count = 0
    
    def _setup_cache_file(self, cache_file: Optional[str]) -> None:
        """Set up cache file path."""
        if cache_file is None:
            cache_dir = pathlib.Path.home() / ".verl_cache"
            cache_dir.mkdir(exist_ok=True)
            suffix = "with_summ" if self.process_snippets else "basic"
            self._cache_file = cache_dir / f"tavily_search_{suffix}_cache.jsonl"
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
    
    async def _load_image_search_extracted_cache(self) -> None:
        """Load image search extracted JSON cache lazily."""
        if self._image_search_cache_loaded:
            return
        if self._image_search_extracted_json is None:
            print(f"[Cache] image_search_extracted not configured (path is None), skipping.")
            self._image_search_cache_loaded = True
            return
        async with self._image_search_cache_lock:
            if self._image_search_cache_loaded:
                return
            if not self._image_search_extracted_json.exists():
                print(f"[Cache] File not found (skipping): {self._image_search_extracted_json}")
                self._image_search_cache_loaded = True
                return
            try:
                loop = asyncio.get_event_loop()
                self._image_search_cache = await loop.run_in_executor(
                    None,
                    lambda: json.loads(self._image_search_extracted_json.read_text(encoding="utf-8"))
                )
                self._image_search_cache_loaded = True
                print(f"[Cache] Loaded successfully: {self._image_search_extracted_json} ({len(self._image_search_cache)} entries)")
            except Exception as e:
                print(f"[Cache] Load failed: {self._image_search_extracted_json} - {e}")
                self._image_search_cache = {}
                self._image_search_cache_loaded = True

    async def _load_image_search_inference_extracted_cache(self) -> None:
        """Load image search inference extracted JSON (by_query_question -> with <cache>, by_query_original -> without <cache>)."""
        if self._image_search_inference_cache_loaded:
            return
        if self._image_search_inference_extracted_json is None:
            print(f"[Cache] image_search_inference_extracted not configured (path is None), skipping.")
            self._image_search_inference_cache_loaded = True
            return
        async with self._image_search_inference_cache_lock:
            if self._image_search_inference_cache_loaded:
                return
            if not self._image_search_inference_extracted_json.exists():
                print(f"[Cache] File not found (skipping): {self._image_search_inference_extracted_json}")
                self._image_search_inference_cache_loaded = True
                return
            try:
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None,
                    lambda: json.loads(self._image_search_inference_extracted_json.read_text(encoding="utf-8"))
                )
                self._image_search_inference_by_query_question = data.get("by_query_question") or {}
                self._image_search_inference_by_query_original = data.get("by_query_original") or {}
                self._image_search_inference_cache_loaded = True
                print(f"[Cache] Loaded successfully: {self._image_search_inference_extracted_json} (by_query_question={len(self._image_search_inference_by_query_question)}, by_query_original={len(self._image_search_inference_by_query_original)})")
            except Exception as e:
                print(f"[Cache] Load failed: {self._image_search_inference_extracted_json} - {e}")
                self._image_search_inference_by_query_question = {}
                self._image_search_inference_by_query_original = {}
                self._image_search_inference_cache_loaded = True
    
    async def _append_to_persistent_cache(self, query: str, result: Union[str, Dict]) -> None:
        """Append to persistent cache asynchronously."""
        try:
            entry = {"query": query, "result": result, "timestamp": time.time()}
            
            async with aiofiles.open(self._cache_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"Cache write failed: {e}")
    
    async def _detect_language(self, query: str) -> Tuple[str, str]:
        """Detect language for the query."""
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            lang_code = await loop.run_in_executor(
                None, lambda: langid.classify(query)[0]
            )
            
            if lang_code == 'zh':
                return "zh-cn", "cn"
            else:
                return self._language, self._location
                
        except Exception as e:
            print(f"Language detection failed: {e}")
            return self._language, self._location
    
    async def _make_search_request(self, query: str, timeout: int) -> Dict:
        """
        Make search request using Tavily API with proper async handling and retry logic for rate limits.
        """
        max_retries = 5
        base_delay = 1.0  # Base delay in seconds
        
        for attempt in range(max_retries + 1):
            try:
                # Import TavilyClient here to avoid import issues if not installed
                from tavily import TavilyClient
                
                # Create client
                client = TavilyClient(api_key=self._api_key)
                
                # Run the search in executor to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: client.search(
                        query=query,
                        max_results=min(self._max_results, 10),  # Tavily max is 20
                        include_images=True,
                        include_image_descriptions=True
                    )
                )
                
                return response
                
            except ImportError:
                raise Exception("Tavily client not installed. Please install with: pip install tavily-python")
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for rate limit errors
                is_rate_limit = (
                    "rate limit" in error_str or
                    "429" in error_str or
                    "too many requests" in error_str or
                    "quota" in error_str or 
                    "excessive requests" in error_str or 
                    "rate" in error_str
                )
                
                # Check for timeout errors
                is_timeout = (
                    "timeout" in error_str or
                    "timed out" in error_str
                )
                
                # Retry on rate limits or timeouts
                if (is_rate_limit or is_timeout) and attempt < max_retries:
                    # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s
                    delay = attempt + 1  # 1, 2, 3, 4, 5s
                    
                    print(f"Rate limit/timeout hit (attempt {attempt + 1}/{max_retries + 1}). Retrying after {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Last attempt or non-retryable error
                    if attempt >= max_retries:
                        raise Exception(f"Tavily API error after {max_retries + 1} attempts: {str(e)}")
                    else:
                        raise Exception(f"Tavily API error: {str(e)}")
    
    async def execute(self, query: str, timeout: int = None, prev_steps: Union[List[str], str] = None, question: Optional[str] = None) -> str:
        """
        Execute search with comprehensive error handling and caching.
        
        Args:
            query: Search query string
            timeout: Request timeout
            prev_steps: Previous steps for context
            question: Optional research question for composite cache key
        """
        # Validate and clean query
        query = query.strip().replace('"', '')
        if not query:
            return "Empty search query provided."
        
        if len(query) > 500:
            return "Search query too long (maximum 500 characters)."
        
        # Normalize query to lowercase for cache key consistency
        query_normalized = query.lower()
        
        # Normalize question if provided (strip and lowercase for consistency)
        question_normalized = question.strip().lower() if question and question.strip() else None
        
        try:
            # Check memory cache first (use normalized query)
            cached_result = await self._memory_cache.get(query_normalized)
            if cached_result is not None:
                if not self.process_snippets:
                    return strip_result_tags(cached_result)
                else:
                    data = json.loads(cached_result) if isinstance(cached_result, str) else cached_result
                    return await self._process_cached_data(query, data, prev_steps)
            
            # Check image_search_inference_extracted.json (by_query_question -> with <cache>, by_query_original -> without <cache>)
            if not self._image_search_inference_cache_loaded:
                await self._load_image_search_inference_extracted_cache()
            if self._image_search_inference_by_query_question is not None and self._image_search_inference_by_query_original is not None:
                if question_normalized:
                    composite_key = f"{query_normalized}||{question_normalized}"
                    if composite_key in self._image_search_inference_by_query_question:
                        return f"<cache>{strip_result_tags(self._image_search_inference_by_query_question[composite_key])}"
                key_original = f"{query_normalized}||original"
                if key_original in self._image_search_inference_by_query_original:
                    return strip_result_tags(self._image_search_inference_by_query_original[key_original])
            
            # Check image_search_extracted.json cache (rollout format)
            if not self._image_search_cache_loaded:
                await self._load_image_search_extracted_cache()
            if self._image_search_cache is not None:
                if question_normalized:
                    composite_key = f"{query_normalized}||{question_normalized}"
                    if composite_key in self._image_search_cache:
                        cached_value = self._image_search_cache[composite_key]
                        return f"<cache>{strip_result_tags(cached_value)}"
                if query_normalized in self._image_search_cache:
                    cached_value = self._image_search_cache[query_normalized]
                    return f"<cache>{strip_result_tags(cached_value)}"
            
            # Make API request (use original query for API call)
            data = await self._make_search_request(query, timeout or 30)
            
            # Process results (use original query for processing)
            result = await self._extract_and_format_results(query, data, prev_steps)
            
            # Cache results (use normalized query for cache key)
            await self._cache_results(query_normalized, data if self.process_snippets else result)
            
            return result
            
        except Exception as e:
            if DEBUG:
                raise e
            error_msg = f"Search failed for '{query}': {str(e)}"
            return error_msg
    
    async def _process_cached_data(self, query: str, data: Dict, prev_steps: Union[List[str], str] = None) -> str:
        """Process cached data for snippet processing mode."""
        return await self._extract_and_format_results(query, data, prev_steps)
    
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
    
    async def _extract_and_format_results(self, query: str, data: Dict, prev_steps: Union[List[str], str] = None) -> str:
        """Extract and format search results with async processing."""
        if 'images' not in data or not data['images']:
            return "No image results found."
        
        if not self.process_snippets:
            return await self._format_basic_results(data)
        else:
            return await self._process_snippets_async(query, data, prev_steps)
    
    async def _format_basic_results(self, data: Dict) -> str:
        """Format image results with title and description only."""
        results = []
        
        # Use images from the response instead of regular results
        images = data.get('images', [])
        if not images:
            return "No image results found."
        
        for idx, image in enumerate(images, 1):
            title = image.get('title', 'No title')
            title = title.strip() if title else 'No title'
            
            url = image.get('url', '')
            url = url.strip() if url else ''
            
            description = image.get('description')
            # Skip images without description
            if not description or not description.strip():
                continue
            
            description = description.strip()
            
            # Format image result
            formatted = f"**Image {len(results) + 1}**\n**Title:** {title}\n**URL:** {url}\n**Description:** {description}\n"
            results.append(formatted)
            
            # Stop if we've reached max_results
            if len(results) >= self._max_results:
                break

        return "\n".join(results) if results else "No image results found."
    
    async def _process_snippets_async(self, query: str, data: Dict, prev_steps: Union[List[str], str] = None) -> str:
        """Process snippets with full content extraction asynchronously."""
        max_doc_len = self._max_doc_len if self.summ_model_url else self._result_length
        do_summarization = self.summ_model_url is not None and self.summ_model_path is not None
        
        # Extract info in thread pool (CPU-bound)
        loop = asyncio.get_event_loop()
        extracted_info = await loop.run_in_executor(
            None, extract_relevant_info_tavily, data
        )
        
        # Process each URL concurrently
        processing_tasks = []
        for info in extracted_info:
            task = self._process_single_url(info, max_doc_len)
            processing_tasks.append(task)
        
        # Wait for all URL processing to complete
        processed_info = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Filter out exceptions and format results
        valid_info = []
        for i, result in enumerate(processed_info):
            if isinstance(result, Exception):
                print(f"URL processing failed: {result}")
                # Use original info without context
                valid_info.append(extracted_info[i])
            else:
                valid_info.append(result)
        
        # Format document
        formatted_document = ""
        for i, doc_info in enumerate(valid_info):
            formatted_document += f"**Web Page {i + 1}:**\n"
            formatted_document += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"

        if do_summarization and formatted_document:
            # Run summarization in thread pool
            summary = await loop.run_in_executor(
                None, self._run_summarization, query, formatted_document, prev_steps
            )
            return summary
        else:
            return formatted_document if formatted_document else "No relevant information found."
    
    async def _process_single_url(self, info: Dict, max_doc_len: int) -> Dict:
        """Process a single URL to extract context."""
        try:
            # Run URL extraction in thread pool
            loop = asyncio.get_event_loop()
            full_text = await loop.run_in_executor(
                None, lambda: extract_text_from_url(info['url'], use_jina=False)
            )
            
            if full_text and not full_text.startswith("Error"):
                success, context = extract_snippet_with_context(
                    full_text, info.get('content', info.get('snippet', '')), context_chars=max_doc_len
                )
                if success:
                    info['context'] = context
                else:
                    info['context'] = f"Could not extract context. First {max_doc_len} chars: {full_text[:max_doc_len]}"
            else:
                info['context'] = f"Failed to fetch content: {full_text or 'Unknown error'}"
                
        except Exception as e:
            info['context'] = f"Error processing URL: {str(e)}"
        
        return info
    
    def _run_summarization(self, query: str, formatted_document: str, prev_steps: Union[List[str], str] = None) -> str:
        """Run summarization in sync context (for thread pool execution)."""
        try:
            prev_reasoning_chain = get_prev_reasoning_chain(
                prev_steps, 
                begin_search_tag="<search>", 
                begin_search_result_tag="<result>"
            )
            return generate_webpage_to_reasonchain(
                prev_reasoning_chain,
                query,
                formatted_document,
                summ_model_url=self.summ_model_url,
                summ_model_path=self.summ_model_path
            )
        except Exception as e:
            if DEBUG:
                raise e
            print(f"Summarization failed: {e}")
            return formatted_document


@register_tool
class WebTextToImgSearchTool(BaseTool):
    """
    Async web text-to-image search tool using Tavily API with image descriptions.
    """
    
    tool_type = "web_text_to_img_search"
    
    def __init__(
        self,
        num_workers=1,
        api_key: str = None,
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "us",
        language: str = "en",
        cache_file: Optional[str] = None,
        default_timeout: int = None,
        process_snippets: bool = False,
        summ_model_url: str = None,
        summ_model_path: str = None,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        image_search_extracted_json: Optional[str] = None,
        image_search_inference_extracted_json: Optional[str] = None
    ):
        """Initialize the search tool with production settings."""
        super().__init__(num_workers)
        
        # Validate API key
        if api_key is None:
            api_key = os.getenv('TAVILY_API_KEY')
            if api_key is None:
                raise ValueError(
                    "API key required: set TAVILY_API_KEY environment variable or pass api_key parameter"
                )
        
        # Initialize search engine
        self.search_engine = TavilyImageSearchEngine(
            api_key=api_key,
            max_results=max_results,
            result_length=result_length,
            location=location,
            language=language,
            cache_file=cache_file,
            process_snippets=process_snippets,
            summ_model_url=summ_model_url,
            summ_model_path=summ_model_path,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            image_search_extracted_json=image_search_extracted_json,
            image_search_inference_extracted_json=image_search_inference_extracted_json
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
        return "Search for images using Tavily API and get their descriptions. Use <text_search_image>your query</text_search_image> format."
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """Parse action to extract search query with improved patterns."""
        patterns = [
            r"<text_search_image>(.*?)</text_search_image>",
            r"```\s*text_search_image\s*\n(.*?)\n```",
            r"text_search_image:\s*(.*?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, action, re.DOTALL | re.IGNORECASE)
            if matches:
                query = matches[0].strip()
                if query and len(query) <= 500:
                    return query, True
        
        return "", False

    async def aget_observations(
        self, 
        trajectory_ids: List[str], 
        actions: List[str], 
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[Union[str, dict]], List[bool], List[bool]]:
        """
        Process multiple search actions concurrently with proper error handling.
        """
        await self._ensure_initialized()
        
        async def process_single_action(trajectory_id, action, extra_field):
            async with self.semaphore:
                try:
                    return await self._conduct_action_async(trajectory_id, action, extra_field)
                except Exception as e:
                    return f"Search error: {str(e)}", False, False
        
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
                obs = f"Search error: {str(result)}"
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
        Conduct single search action asynchronously.
        """
        parsed_query, is_valid = self.parse_action(action)
        
        # Load environment
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = "Invalid search query. Use <text_search_image>your query</text_search_image> format."
            done, valid = False, False
        else:
            # Get timeout from extra field
            timeout = extra_field.get('timeout', self.default_timeout)
            
            # Extract previous actions for snippet processing
            prev_actions = []
            if self.search_engine.process_snippets and env.get('previous_obs'):
                prev_actions = [x.get('action') for x in env['previous_obs']]
            prev_actions += [action]
            
            try:
                # Get question from extra field for composite cache key
                question = extra_field.get('question', None) if isinstance(extra_field, dict) else None
                
                # Execute search (pass question for composite cache key)
                search_results = await self.search_engine.execute(parsed_query, timeout, prev_actions, question=question)
                if search_results.startswith("<cache>") or search_results.startswith("Search results for "):
                    observation = search_results
                else:
                    observation = f"Search results for '{parsed_query}':\n\n{strip_result_tags(search_results)}"
                done, valid = False, True
                
            except Exception as e:
                if DEBUG:
                    raise e
                observation = f"Search execution failed: {str(e)}"
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
                    return "Search timed out", False, False
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
            return f"Search failed: {str(e)}", False, False


if __name__ == "__main__":
    import os
    
    # Check API key
    api_key = os.getenv('TAVILY_API_KEY')
    if not api_key:
        print("❌ Set TAVILY_API_KEY environment variable")
        exit(1)
    
    # Test basic search
    tool = WebTextToImgSearchTool(api_key=api_key, max_results=2)
    
    # Test query parsing
    query, valid = tool.parse_action("<text_search_image>Python programming</text_search_image>")
    print(f"Parsed query: '{query}', Valid: {valid}")
    
    # Test search execution
    # Test image search with question
    query = "glossy floor coating being applied with squeegee"
    question = "what material is being used to coat the floor in the image?"
    extra_field = {"question": question}
    observation, done, valid = tool.conduct_action("test", f"<text_search_image>{query}</text_search_image>", extra_field)
    print('>>>', observation, '<<<')

    query = "campus alberta buildings with glass atrium and water feature"
    question = "what material is being used to coat the floor in the image?"
    extra_field = {"question": question}
    observation, done, valid = tool.conduct_action("test", f"<text_search_image>{query}</text_search_image>",
                                                   extra_field)
    print('>>>', observation, '<<<')

    query = "campus alberta buildings with glass atrium and water feature"
    question = "in which city is the main campus of the organization associated with the campus in the image located?"
    extra_field = {"question": question}
    observation, done, valid = tool.conduct_action("test", f"<text_search_image>{query}</text_search_image>",
                                                   extra_field)
    print('>>>', observation, '<<<')
    
    print("✅ Test complete")


