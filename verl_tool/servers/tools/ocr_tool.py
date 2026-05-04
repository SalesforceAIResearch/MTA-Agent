#!/usr/bin/env python3
"""OCR Tool - Extract text from images using GPT Vision."""

import os
import json
import time
import pathlib
import asyncio
import aiofiles
import openai
import base64
import mimetypes
from pathlib import Path
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

# OCR prompt for GPT Vision
OCR_PROMPT = """Extract ALL text visible in this image. Return ONLY the text content, no reasoning or analysis.

List each text item found:
- Signs, labels, titles
- Numbers (prices, dates, addresses, vehicle numbers)
- Brand names, logos
- Handwritten or printed text
- Text on screens, buttons

Output format (text only, no explanation or reasoning):
Bus number: 200
Street sign: Main Street
Store name: Coffee Shop
Price: $4.99

If no text visible, return: No text detected."""


def get_image_mime_type(file_path: str) -> str:
    """Get MIME type for image file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('image/'):
        return mime_type
    ext = Path(file_path).suffix.lower()
    mime_map = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
        '.gif': 'image/gif', '.bmp': 'image/bmp', '.tiff': 'image/tiff',
        '.tif': 'image/tiff', '.webp': 'image/webp'
    }
    return mime_map.get(ext, 'image/jpeg')


def encode_image_to_base64(file_path: str) -> str:
    """Encode image file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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
                if time.time() - self._timestamps[key] > self.ttl_seconds:
                    del self._cache[key]
                    del self._timestamps[key]
                    return None
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: Any):
        async with self._lock:
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            self._cache[key] = value
            self._timestamps[key] = time.time()


class GPTVisionOCREngine:
    """
    GPT Vision OCR engine for extracting text from images.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-5-mini",
        cache_file: Optional[str] = None,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        base_url: str = None,
        ocr_extracted_json: Optional[str] = None,
        ocr_inference_extracted_json: Optional[str] = None
    ):
        """Initialize the GPT Vision OCR engine."""
        self.x_api_key = api_key or os.getenv("X_API_KEY")
        if not self.x_api_key:
            raise ValueError("API key required: set X_API_KEY env var or pass api_key")
        
        self.model = model
        self.base_url = base_url or "https://gateway.salesforceresearch.ai/openai/process/v1/"
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key="dummy",
            default_headers={"X-Api-Key": self.x_api_key}
        )
        
        # Async-safe caching
        self._memory_cache = AsyncLRUCache(cache_size, cache_ttl)
        self._setup_cache_file(cache_file)
        
        # OCR extracted JSON cache (rollout format: flat key -> value)
        if ocr_extracted_json is None:
            _repo_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
            ocr_extracted_json = str(_repo_root / "ocr_extracted.json")
        self._ocr_extracted_json = pathlib.Path(ocr_extracted_json) if ocr_extracted_json else None
        self._ocr_cache = None
        self._ocr_cache_loaded = False
        self._ocr_cache_lock = asyncio.Lock()
        
        # OCR inference extracted JSON cache (by_url_question -> with <cache>, by_url -> without <cache>)
        if ocr_inference_extracted_json is None:
            # Default: same repo root as this file (verl_tool/servers/tools/ -> repo root)
            _repo_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
            ocr_inference_extracted_json = str(_repo_root / "ocr_inference_extracted.json")
        self._ocr_inference_extracted_json = pathlib.Path(ocr_inference_extracted_json) if ocr_inference_extracted_json else None
        self._ocr_inference_by_url_question = None
        self._ocr_inference_by_url = None
        self._ocr_inference_cache_loaded = False
        self._ocr_inference_cache_lock = asyncio.Lock()
        
        self._search_count = 0
    
    def _setup_cache_file(self, cache_file: Optional[str]) -> None:
        """Set up cache file path."""
        if cache_file is None:
            cache_dir = pathlib.Path.home() / ".verl_cache"
            cache_dir.mkdir(exist_ok=True)
            self._cache_file = cache_dir / "gpt_ocr_cache.jsonl"
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
    
    async def _load_ocr_extracted_cache(self) -> None:
        """Load OCR extracted JSON cache lazily."""
        if self._ocr_cache_loaded:
            return
        if self._ocr_extracted_json is None:
            print(f"[Cache] ocr_extracted not configured (path is None), skipping.")
            self._ocr_cache_loaded = True
            return
        
        async with self._ocr_cache_lock:
            if self._ocr_cache_loaded:
                return
            
            if not self._ocr_extracted_json.exists():
                print(f"[Cache] File not found (skipping): {self._ocr_extracted_json}")
                self._ocr_cache_loaded = True
                return
            
            try:
                # Load JSON file in executor to avoid blocking
                loop = asyncio.get_event_loop()
                self._ocr_cache = await loop.run_in_executor(
                    None,
                    lambda: json.loads(self._ocr_extracted_json.read_text(encoding="utf-8"))
                )
                self._ocr_cache_loaded = True
                print(f"[Cache] Loaded successfully: {self._ocr_extracted_json} ({len(self._ocr_cache)} entries)")
            except Exception as e:
                print(f"[Cache] Load failed: {self._ocr_extracted_json} - {e}")
                self._ocr_cache = {}
                self._ocr_cache_loaded = True
    
    async def _load_ocr_inference_extracted_cache(self) -> None:
        """Load OCR inference extracted JSON (by_url_question -> with <cache>, by_url -> without <cache>)."""
        if self._ocr_inference_cache_loaded:
            return
        if self._ocr_inference_extracted_json is None:
            print(f"[Cache] ocr_inference_extracted not configured (path is None), skipping.")
            self._ocr_inference_cache_loaded = True
            return
        async with self._ocr_inference_cache_lock:
            if self._ocr_inference_cache_loaded:
                return
            if not self._ocr_inference_extracted_json.exists():
                print(f"[Cache] File not found (skipping): {self._ocr_inference_extracted_json}")
                self._ocr_inference_cache_loaded = True
                return
            try:
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None,
                    lambda: json.loads(self._ocr_inference_extracted_json.read_text(encoding="utf-8"))
                )
                self._ocr_inference_by_url_question = data.get("by_url_question") or {}
                self._ocr_inference_by_url = data.get("by_url") or {}
                self._ocr_inference_cache_loaded = True
                print(f"[Cache] Loaded successfully: {self._ocr_inference_extracted_json} (by_url_question={len(self._ocr_inference_by_url_question)}, by_url={len(self._ocr_inference_by_url)})")
            except Exception as e:
                print(f"[Cache] Load failed: {self._ocr_inference_extracted_json} - {e}")
                self._ocr_inference_by_url_question = {}
                self._ocr_inference_by_url = {}
                self._ocr_inference_cache_loaded = True
    
    async def _append_to_persistent_cache(self, query: str, result: str) -> None:
        """Append to persistent cache asynchronously."""
        try:
            entry = {"query": query, "result": result, "timestamp": time.time()}
            async with aiofiles.open(self._cache_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Cache write failed: {e}")
    
    async def _make_vision_request(self, image_url: str = None, image_base64: str = None, 
                                    mime_type: str = "image/png", timeout: int = 60) -> str:
        """Make OCR request to GPT Vision API using OpenAI client."""
        try:
            loop = asyncio.get_event_loop()
            
            def sync_request():
                # Build image content
                if image_url:
                    image_content = {"type": "image_url", "image_url": {"url": image_url}}
                else:
                    image_content = {
                        "type": "image_url", 
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
                    }
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": OCR_PROMPT},
                                image_content
                            ]
                        }
                    ],
                    timeout=timeout
                )
                
                return response.choices[0].message.content
            
            return await loop.run_in_executor(None, sync_request)
            
        except Exception as e:
            raise Exception(f"GPT Vision OCR error: {str(e)}")
    
    async def execute(self, image_source: str, timeout: int = 60, question: Optional[str] = None) -> str:
        """
        Execute OCR on image using GPT Vision.
        
        Args:
            image_source: URL or file path of the image
            timeout: Request timeout in seconds
            question: Optional research question for composite cache key
        """
        image_source = image_source.strip()
        if not image_source:
            return "Empty image source provided."
        
        # Normalize question if provided (strip and lowercase for consistency)
        question_normalized = question.strip().lower() if question and question.strip() else None
        
        try:
            # Check memory cache first
            cached_result = await self._memory_cache.get(image_source)
            if cached_result is not None:
                print(f"[Cache] OCR memory cache hit: {image_source[:60]}...")
                return strip_result_tags(cached_result)
            
            # Check ocr_inference_extracted.json (by_url_question -> with <cache>, by_url -> without <cache>)
            if not self._ocr_inference_cache_loaded:
                await self._load_ocr_inference_extracted_cache()
            if self._ocr_inference_by_url_question is not None or self._ocr_inference_by_url is not None:
                if question_normalized and self._ocr_inference_by_url_question:
                    composite_key = f"{image_source}||{question_normalized}"
                    if composite_key in self._ocr_inference_by_url_question:
                        print(f"[Cache] OCR ocr_inference hit (by_url_question): {image_source[:60]}...")
                        return f"<cache>{strip_result_tags(self._ocr_inference_by_url_question[composite_key])}"
                if self._ocr_inference_by_url and image_source in self._ocr_inference_by_url:
                    print(f"[Cache] OCR ocr_inference hit (by_url): {image_source[:60]}...")
                    return strip_result_tags(self._ocr_inference_by_url[image_source])
            
            # Check ocr_extracted.json cache (rollout format); return with <cache>
            if not self._ocr_cache_loaded:
                await self._load_ocr_extracted_cache()
            if self._ocr_cache is not None:
                if question_normalized:
                    composite_key = f"{image_source}||{question_normalized}"
                    if composite_key in self._ocr_cache:
                        cached_value = self._ocr_cache[composite_key]
                        print(f"[Cache] OCR ocr_extracted hit (by_url_question): {image_source[:60]}...")
                        return f"<cache>{strip_result_tags(cached_value)}"
                if image_source in self._ocr_cache:
                    cached_value = self._ocr_cache[image_source]
                    print(f"[Cache] OCR ocr_extracted hit (by_url): {image_source[:60]}...")
                    return f"<cache>{strip_result_tags(cached_value)}"
            
            # Handle URL vs file path
            if image_source.startswith('http://') or image_source.startswith('https://'):
                # Send URL directly to GPT Vision
                result = await self._make_vision_request(image_url=image_source, timeout=timeout)
            elif os.path.exists(image_source):
                # Encode local file to base64
                mime_type = get_image_mime_type(image_source)
                image_base64 = encode_image_to_base64(image_source)
                result = await self._make_vision_request(
                    image_base64=image_base64, 
                    mime_type=mime_type, 
                    timeout=timeout
                )
            else:
                return f"Image source not found: {image_source}"
            
            # Cache results
            await self._cache_results(image_source, result)
            
            return result
            
        except Exception as e:
            if DEBUG:
                raise e
            return f"OCR failed for '{image_source}': {str(e)}"
    
    async def _cache_results(self, query: str, data: str) -> None:
        """Cache results in both memory and persistent storage."""
        try:
            await self._memory_cache.set(query, data)
            await self._append_to_persistent_cache(query, data)
            self._search_count += 1
        except Exception as e:
            print(f"Caching failed: {e}")


@register_tool
class OCRTool(BaseTool):
    """
    OCR tool using GPT Vision to extract and describe text from images.
    """
    
    tool_type = "ocr_tool"
    stop_tokens = ["</ocr_tool>"]
    
    def __init__(
        self,
        num_workers: int = 1,
        api_key: str = None,
        model: str = "gpt-5-nano",
        cache_file: Optional[str] = None,
        default_timeout: int = 60,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        base_url: str = None,
        ocr_extracted_json: Optional[str] = None,
        ocr_inference_extracted_json: Optional[str] = None
    ):
        """Initialize the OCR tool."""
        super().__init__(num_workers)
        
        self.ocr_engine = GPTVisionOCREngine(
            api_key=api_key,
            model=model,
            cache_file=cache_file,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            base_url=base_url,
            ocr_extracted_json=ocr_extracted_json,
            ocr_inference_extracted_json=ocr_inference_extracted_json
        )
        
        self.default_timeout = default_timeout
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(16)
    
    async def _ensure_initialized(self):
        """Ensure OCR engine is initialized."""
        if not self._initialized:
            async with self._init_lock:
                if not self._initialized:
                    await self.ocr_engine._load_persistent_cache()
                    self._initialized = True
    
    def get_usage_inst(self) -> str:
        """Get usage instructions."""
        return "OCR tool to extract text from images using GPT Vision. Use <ocr_tool>image_url_or_path</ocr_tool> format."
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """Parse action to extract image URL or path."""
        patterns = [
            r"<ocr_tool>(.*?)</ocr_tool>",
            r"```\s*ocr_tool\s*\n(.*?)\n```",
            r"ocr_tool:\s*(.*?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, action, re.DOTALL | re.IGNORECASE)
            if matches:
                content = matches[0].strip()
                if content:
                    return content, True
        
        return "", False
    
    async def aget_observations(
        self, 
        trajectory_ids: List[str], 
        actions: List[str], 
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[Union[str, dict]], List[bool], List[bool]]:
        """Process multiple OCR actions concurrently."""
        await self._ensure_initialized()
        
        async def process_single_action(trajectory_id, action, extra_field):
            async with self.semaphore:
                try:
                    return await self._conduct_action_async(trajectory_id, action, extra_field)
                except Exception as e:
                    return f"OCR error: {str(e)}", False, False
        
        tasks = [
            process_single_action(tid, act, ef)
            for tid, act, ef in zip(trajectory_ids, actions, extra_fields)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        observations, dones, valids = [], [], []
        for result in results:
            if isinstance(result, Exception):
                if DEBUG:
                    raise result
                obs, done, valid = f"OCR error: {str(result)}", False, False
            else:
                obs, done, valid = result
            observations.append(obs)
            dones.append(done)
            valids.append(valid)
        
        self.maybe_cleanup_env(trajectory_ids, actions, extra_fields)
        return observations, dones, valids
    
    async def _conduct_action_async(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        """Conduct single OCR action asynchronously."""
        image_source, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = "Invalid OCR format. Use <ocr_tool>image_url_or_path</ocr_tool> format."
            done, valid = False, False
        else:
            timeout = extra_field.get('timeout', self.default_timeout)
            
            try:
                # Get question from extra field for composite cache key
                question = extra_field.get('question', None) if isinstance(extra_field, dict) else None
                
                # Execute OCR (pass question for composite cache key)
                ocr_result = await self.ocr_engine.execute(image_source, timeout, question=question)
                
                if ocr_result.strip().startswith("<cache>"):
                    observation = ocr_result
                elif ocr_result.strip():
                    ocr_result = strip_result_tags(ocr_result)
                    if ocr_result.startswith("Text found in image:"):
                        observation = ocr_result
                    else:
                        observation = f"Text found in image:\n\n{ocr_result}"
                else:
                    observation = "No text detected in this image."
                
                done, valid = False, True
                
            except Exception as e:
                if DEBUG:
                    raise e
                observation = f"OCR failed: {str(e)}"
                done, valid = False, False
        
        observation = f"{observation}"
        self.update_env(trajectory_id, env, action, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    
    def conduct_action(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        """Synchronous wrapper for async conduct_action."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
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
                thread.join(timeout=120)
                
                if exception[0]:
                    raise exception[0]
                if result[0] is None:
                    return "OCR timed out", False, False
                return result[0]
            else:
                return loop.run_until_complete(
                    self._conduct_action_async(trajectory_id, action, extra_field)
                )
        except RuntimeError:
            return asyncio.run(self._conduct_action_async(trajectory_id, action, extra_field))
        except Exception as e:
            if DEBUG:
                raise e
            return f"OCR failed: {str(e)}", False, False


if __name__ == "__main__":
    # Check API key
    api_key = os.getenv('X_API_KEY')
    if not api_key:
        print("❌ Set X_API_KEY environment variable")
        exit(1)
    
    # Test OCR tool
    tool = OCRTool(api_key=api_key)
    
    test_image = "https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/News/images/cnn_b52db6ab20a49fb1167ef761047cc152.jpg"
    
    # Test query parsing
    parsed_content, valid = tool.parse_action(f"<ocr_tool>{test_image}</ocr_tool>")
    print(f"Parsed: '{parsed_content}', Valid: {valid}")
    
    # Test OCR execution with question
    test_image = "https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/FVQA/images/fvqa_train_1879.png"
    question = "what event does this map represent?"
    extra_field = {"question": question}
    observation, done, valid = tool.conduct_action("test1", f"<ocr_tool>{test_image}</ocr_tool>", extra_field)
    print('>>>', observation, '<<<')

    # Test OCR execution with question
    test_image = "https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/Infoseek/images/427.png"
    question = "who occupies this building???"
    extra_field = {"question": question}
    observation, done, valid = tool.conduct_action("test1", f"<ocr_tool>{test_image}</ocr_tool>", extra_field)
    print('>>>', observation, '<<<')
    
    # Test OCR execution with question
    test_image = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTc9APxkj0xClmrU3PpMZglHQkx446nQPG6lA&s"
    question = "who occupies this building?"
    extra_field = {"question": question}
    observation, done, valid = tool.conduct_action("test1", f"<ocr_tool>{test_image}</ocr_tool>", extra_field)
    print('>>>', observation, '<<<')
    
    print("\n✅ Test complete")

