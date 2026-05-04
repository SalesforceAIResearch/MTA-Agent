#!/usr/bin/env python3
"""
Multimodal Deep Research Test Cases - Base Class
Simulates actual deep research scenarios with multimodal data (text + images)
Base class for different model implementations (GPT, vLLM, etc.)
"""
import json
import requests
import fire
import logging
import time
import os
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import re
from abc import ABC, abstractmethod
from openai import OpenAI
import openai

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Try to load from project root .env file (4 levels up from this file)
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from {env_path}")
    else:
        # Try current directory
        load_dotenv()
        print("✓ Loaded environment variables from current directory .env (if exists)")
except ImportError:
    # python-dotenv not available, skip loading .env file
    print("⚠️  python-dotenv not available. Install with: pip install python-dotenv")

# Try importing vllm - it's optional and only needed for vllm model type
try:
    import vllm
except ImportError:
    vllm = None
    logger = logging.getLogger(__name__)
    logger.warning("vllm not installed - vLLM model type will not be available")

# Try importing Gemini - it's optional and only needed for gemini model type
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None
    logger = logging.getLogger(__name__)
    logger.warning("google-genai not installed - Gemini model type will not be available")

# Only set default headers if X_API_KEY is available
x_api_key = os.getenv("X_API_KEY")
if x_api_key:
    openai.default_headers = {"X-Api-Key": x_api_key}
openai.base_url = "https://gateway.salesforceresearch.ai/openai/process/v1/"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log X_API_KEY status after logging is configured
x_api_key_loaded = os.getenv("X_API_KEY")
if x_api_key_loaded:
    logger.info(f"✓ X_API_KEY loaded from environment (length: {len(x_api_key_loaded)} chars)")
else:
    logger.warning("⚠️  X_API_KEY not found in environment. Set it in .env file or environment variables.")

# Placeholder URL to hide actual image paths from LLM (avoids data leakage from URL)
IMAGE_URL_PLACEHOLDER = "http://image.png"


class MultimodalDeepResearchTesterBase(ABC):
    # Tool registry with descriptions and formats

        # "image_operations": {
    #     "description": "Crop and zoom into specific regions of an image for closer visual inspection",
    #     "sub_actions": {
    #         "zoom_in": {
    #             "params": {"bbox_2d": "[x1, y1, x2, y2] (normalized 0.0-1.0)"}, 
    #             "desc": "Crop to a specific region. bbox_2d=[left, top, right, bottom] in normalized coordinates (0.0=left/top edge, 1.0=right/bottom edge). Example: [0.0, 0.0, 1.0, 0.3] crops top 30% of image, [0.0, 0.7, 1.0, 1.0] crops bottom 30%"
    #         },
    #         "crop_image_normalized": {
    #             "params": {"bbox_2d": "[x1, y1, x2, y2] (normalized 0.0-1.0)"}, 
    #             "desc": "Same as zoom_in - crop to specific region using normalized coordinates"
    #         }
    #     },
    #     "format": "image_operations",
    #     "example": '<tool_call>{"name": "zoom_in", "arguments": {"target_image": 1, "bbox_2d": [0.0, 0.0, 1.0, 0.3]}}</tool_call>'
    # },

    TOOL_REGISTRY = {
        "web_text_to_text_search": {
            "description": "Search the web and get text-based search results with titles, URLs, and content snippets. Returns structured search results with summaries and relevant information from multiple web sources.",
            "parameters": {"query": "search terms"},
            "format": "web_search",
            "example": '<text_search_text>your search query</text_search_text>'
        },
        "web_text_to_img_search": {
            "description": "Search for images on the web and get detailed text descriptions of the images found. Returns image descriptions of what each image contains.",
            "parameters": {"query": "image search terms"},
            "format": "image_search",
            "example": '<text_search_image>your image search query</text_search_image>'
        },
        "web_url_reader": {
            "description": "Extract and read the full raw text content from any web page URL. Returns the complete text content of the webpage, including articles, blog posts, documentation, and other text-based content.",
            "parameters": {"url": "URL to extract content from"},
            "format": "content_extraction",
            "example": '<web_read>https://example.com</web_read>'
        },
        "web_image_to_text": {
            "description": "Google Lens image search to find text information about images. Uses SerpAPI Google Lens to identify objects, people, landmarks, and find visually similar images with detailed context. Returns visual matches with titles, sources, and links. Optionally provide a search query for display context.",
            "parameters": {"image_url": "URL of image to search", "query": "(optional) Text query for display/context only"},
            "format": "reverse_image_search",
            "example": '<image_search_text>https://example.com/image.jpg</image_search_text> or <image_search_text>https://example.com/image.jpg||famous painting</image_search_text>'
        },
        "ocr_tool": {
            "description": "Extract and describe ALL text content visible in an image. Returns a detailed list of text found including signs, labels, numbers, brand names, and any printed or handwritten text with context about where each text appears.",
            "parameters": {"image_url": "URL or path of image to extract text from"},
            "format": "ocr",
            "example": '<ocr_tool>https://example.com/image.jpg</ocr_tool>'
        },
        "ipython_code": {
            "description": "Execute Python code with IPython for data analysis and computation. Supports persistent state across executions and automatically displays the last expression value (no need for print()).",
            "parameters": {"code": "python code to execute"},
            "format": "code_execution",
            "example": '<python>\nx = 42\nx * 2  # Automatically displays result\n</python>'
        },
        "bash_terminal": {
            "description": "Execute bash commands for file operations and system tasks",
            "parameters": {"command": "bash command to execute"},
            "format": "bash_execution",
            "example": '<bash>\nls -la\n</bash>'
        },
        "finish": {
            "description": "Signal that research is complete and ready to provide final answer",
            "parameters": {},
            "format": "finish",
            "example": 'Set "should_stop": true in your response'
        }
    }
    
    def __init__(self, 
                 server_url: str = "http://localhost:4000/get_observation", 
                 output_dir: str = "simple_test_results_mm", 
                 input_dir: str = "data/mmsearch_plus_processed",
                 model_type: str = "gpt",  # "gpt" or "vllm"
                 model_config: Optional[Dict[str, Any]] = None,
                 enabled_tools: Optional[List[str]] = None,
                 difficulty_filter: Optional[str] = "",
                 image_url_prefix: Optional[str] = None,
                 prompt_tool: Optional[str] = None,
                 runs_per_question: int = 1,
                 run_start_number: int = 1,  # Index of first scenario to process (1-based)
                 file_name: Optional[str] = None):  # JSON file name to load scenarios from 
        self.server_url = server_url
        self.output_dir = Path(output_dir)
        self.input_dir = Path(input_dir)
        self.model_type = model_type
        self.model_config = model_config or {}
        self.difficulty_filter = difficulty_filter
        self.image_url_prefix = image_url_prefix
        self.prompt_tool = prompt_tool
        self.runs_per_question = runs_per_question
        self.run_start_number = run_start_number
        self.file_name = file_name or "qa_formatted.json"  # Default file name
        
        # Configure enabled tools (default to all tools)
        if enabled_tools is None:
            self.enabled_tools = ["web_text_to_text_search", "web_text_to_img_search", "web_url_reader", "web_image_to_text", "ocr_tool", "ipython_code", "bash_terminal", "finish"]
        else:
            self.enabled_tools = enabled_tools
        
        # Validate enabled tools
        invalid_tools = set(self.enabled_tools) - set(self.TOOL_REGISTRY.keys())
        if invalid_tools:
            raise ValueError(f"Invalid tools specified: {invalid_tools}. Available tools: {list(self.TOOL_REGISTRY.keys())}")
        
        logger.info(f"Enabled tools: {', '.join(self.enabled_tools)}")
        
        # Log tool capabilities for reference
        logger.info("Tool capabilities:")
        for tool_name in self.enabled_tools:
            if tool_name in self.TOOL_REGISTRY:
                tool_info = self.TOOL_REGISTRY[tool_name]
                logger.info(f"  - {tool_name}: {tool_info['description'][:80]}...")
                logger.info(f"    Example: {tool_info['example']}")
        
        # Initialize model-specific components (implemented by subclasses)
        self.initialize_model()

        self.output_dir.mkdir(exist_ok=True)

        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_dir = self.output_dir / f"deep_research_test_{timestamp}"
        self.test_dir.mkdir(exist_ok=True)

        # Create subdirectories for different outputs
        (self.test_dir / "trajectories").mkdir(exist_ok=True)
        (self.test_dir / "evaluations").mkdir(exist_ok=True)
        (self.test_dir / "logs").mkdir(exist_ok=True)
        (self.test_dir / "generated_actions").mkdir(exist_ok=True)

        logger.info(f"Test results will be saved to: {self.test_dir}")
        logger.info(f"Input scenarios will be loaded from: {self.input_dir}")
        logger.info(f"Using GPT model: {self.gpt_model}")
        logger.info(f"Runs per question: {self.runs_per_question}")
        if self.run_start_number > 1:
            logger.info(f"Starting from scenario index: {self.run_start_number} (will skip first {self.run_start_number - 1} scenarios)")
    
    def initialize_model(self):
        """Initialize the model based on model_type"""
        if self.model_type == "gpt":
            self._initialize_gpt_model()
        elif self.model_type == "vllm":
            self._initialize_vllm_model()
        elif self.model_type == "gemini":
            self._initialize_gemini_model()
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Use 'gpt', 'vllm', or 'gemini'")
    
    def _initialize_gpt_model(self):
        """Initialize OpenAI GPT model"""
        self.gpt_model = self.model_config.get("gpt_model", "gpt-4o")
        openai_api_key = self.model_config.get("openai_api_key")
        
        try:
            x_api_key = os.getenv("X_API_KEY")
            headers = {"X-Api-Key": x_api_key} if x_api_key else {}
            self.openai_client = openai.OpenAI(
                base_url="https://gateway.salesforceresearch.ai/openai/process/v1/",
                api_key="dummy",
                default_headers=headers
            )
        except:
            # Try to get from environment
            api_key = os.getenv('OPENAI_API_KEY') or openai_api_key
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                raise ValueError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass in model_config")
        
        logger.info(f"Using GPT model: {self.gpt_model}")
    
    def _initialize_vllm_model(self):
        """Initialize vLLM model"""
        # Check if vLLM is available
        if vllm is None:
            raise ImportError("vLLM is required for vllm model_type. Install with: pip install vllm")
        
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL is required for vllm model_type. Install with: pip install pillow")
        
        self.model_path = self.model_config.get("model_path", "/export/home/becky/Qwen3-VL/qwen-vl-finetune/output/2e5_ep3/")
        self.max_model_len = self.model_config.get("max_model_len", 10000)
        self.tensor_parallel_size = self.model_config.get("tensor_parallel_size", 1)
        self.gpu_id = self.model_config.get("gpu_id")
        
        logger.info(f"🔍 DEBUG: Loading vLLM model from: {self.model_path}")
        try:
            self.model, self.sampling_params, self.tokenizer = self._load_llm_sampling_params()
            logger.info(f"🔍 DEBUG: vLLM model loaded successfully")
            self._log_tokenizer_info()
        except Exception as e:
            logger.error(f"🔍 DEBUG: Failed to load vLLM model: {e}")
            raise ValueError(f"Failed to load vLLM model from {self.model_path}: {e}")
        
        # Set GPT model for summarization (even when using vLLM as main model)
        self.gpt_model = "gpt-4.1"  # Default GPT model for summarization
        
        # Also initialize OpenAI client for GPT-4.1 summarization
        try:
            x_api_key = os.getenv("X_API_KEY")
            headers = {"X-Api-Key": x_api_key} if x_api_key else {}
            self.openai_client = openai.OpenAI(
                base_url="https://gateway.salesforceresearch.ai/openai/process/v1/",
                api_key="dummy",
                default_headers=headers
            )
            logger.info("🔍 DEBUG: OpenAI client initialized for GPT-4.1 summarization")
        except Exception as e:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                logger.info("🔍 DEBUG: OpenAI client initialized with API key for GPT-4.1 summarization")
            else:
                logger.warning("🔍 DEBUG: Could not initialize OpenAI client for summarization. Will fall back to vLLM for summarization.")
                self.openai_client = None
    
    def _initialize_gemini_model(self):
        """Initialize Gemini model"""
        # Check if Gemini is available
        if genai is None:
            raise ImportError("google-genai is required for gemini model_type. Install with: pip install google-genai")
        
        self.gemini_model = self.model_config.get("gemini_model", "gemini-3-pro-preview")
        self.gemini_project = self.model_config.get("gemini_project", "salesforce-research-internal")
        self.gemini_location = self.model_config.get("gemini_location", "global")
        self.gemini_temperature = self.model_config.get("gemini_temperature", 1.0)
        self.gemini_top_p = self.model_config.get("gemini_top_p", 0.95)
        
        try:
            self.gemini_client = genai.Client(
                project=self.gemini_project,
                location=self.gemini_location
            )
            logger.info(f"✓ Gemini client initialized successfully")
            logger.info(f"  Project: {self.gemini_project}")
            logger.info(f"  Location: {self.gemini_location}")
            logger.info(f"  Model: {self.gemini_model}")
            logger.info(f"  Temperature: {self.gemini_temperature}")
            logger.info(f"  Top-p: {self.gemini_top_p}")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {e}")
        
        # Set GPT model for summarization (even when using Gemini as main model)
        self.gpt_model = "gpt-4.1"  # Default GPT model for summarization
        
        # Also initialize OpenAI client for GPT-4.1 summarization
        try:
            self.openai_client = openai.OpenAI(
                base_url="https://gateway.salesforceresearch.ai/openai/process/v1/",
                api_key="dummy",
                default_headers={"X-Api-Key": os.getenv("X_API_KEY")}
            )
            logger.info("🔍 DEBUG: OpenAI client initialized for GPT-4.1 summarization")
        except Exception as e:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                logger.info("🔍 DEBUG: OpenAI client initialized with API key for GPT-4.1 summarization")
            else:
                logger.warning("🔍 DEBUG: Could not initialize OpenAI client for summarization. Will fall back to Gemini for summarization.")
                self.openai_client = None
    
    def analyze_image_with_model(self, image_path: str, analysis_prompt: str = "Analyze this image and describe what you see in detail.") -> str:
        """Analyze an image using the configured model"""
        if self.model_type == "gpt":
            return self._analyze_image_with_gpt(image_path, analysis_prompt)
        elif self.model_type == "vllm":
            return self._analyze_image_with_vllm(image_path, analysis_prompt)
        elif self.model_type == "gemini":
            return self._analyze_image_with_gemini(image_path, analysis_prompt)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def generate_response_with_model(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.99) -> str:
        """Generate a response using the configured model"""
        if self.model_type == "gpt":
            return self._generate_response_with_gpt(messages, max_tokens, temperature)
        elif self.model_type == "vllm":
            return self._generate_response_with_vllm(messages, max_tokens, temperature)
        elif self.model_type == "gemini":
            return self._generate_response_with_gemini(messages, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def generate_response_with_gpt4_for_summarization(self, messages: List[Dict], max_tokens: int = 500, temperature: float = 0.1) -> str:
        """Generate a response using GPT-4.1 specifically for summarization"""
        logger.info(f"🔍 GPT4_SUMMARIZATION DEBUG: Starting GPT-4.1 summarization")
        logger.info(f"🔍 GPT4_SUMMARIZATION DEBUG: Main model type: {self.model_type}")
        logger.info(f"🔍 GPT4_SUMMARIZATION DEBUG: Has openai_client: {hasattr(self, 'openai_client')}")
        logger.info(f"🔍 GPT4_SUMMARIZATION DEBUG: openai_client is not None: {hasattr(self, 'openai_client') and self.openai_client is not None}")
        
        if hasattr(self, 'openai_client') and self.openai_client:
            logger.info(f"🔍 GPT4_SUMMARIZATION DEBUG: ✓ Using OpenAI GPT model for summarization")
            logger.info(f"🔍 GPT4_SUMMARIZATION DEBUG: Model: {self.gpt_model}, Max tokens: {max_tokens}, Temperature: {temperature}")
            
            try:
                # Use GPT-4.1 directly for summarization
                response = self.openai_client.chat.completions.create(
                    model="gpt-4.1",  # Always use GPT-4.1 for summarization
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature
                )
                result = response.choices[0].message.content
                logger.info(f"🔍 GPT4_SUMMARIZATION DEBUG: GPT API call successful")
                logger.info(f"🔍 GPT4_SUMMARIZATION DEBUG: Response length: {len(result)} chars")
                return result
                
            except Exception as e:
                logger.error(f"🔍 GPT4_SUMMARIZATION DEBUG: OpenAI API call failed: {e}")
                logger.error(f"🔍 GPT4_SUMMARIZATION DEBUG: Exception type: {type(e).__name__}")
                raise e  # Re-raise to be caught by the calling method
        else:
            logger.warning(f"🔍 GPT4_SUMMARIZATION DEBUG: ✗ No OpenAI client available, falling back to main model: {self.model_type}")
            # Fallback to the main model
            return self.generate_response_with_model(messages, max_tokens, temperature)
    
    def summarize_observation(self, observation: str, tool_name: str = None, question: str = None) -> str:
        """
        Summarize a long observation text to make it more concise for context.
        Uses different strategies based on the tool that generated the observation.
        
        Args:
            observation: The full observation text to summarize
            tool_name: The name of the tool that generated this observation
            question: The research question being answered (for context-aware summarization)
            
        Returns:
            A summarized version of the observation
        """
        obs_str = str(observation)
        
        # Debug logging
        logger.info(f"🔍 SUMMARIZATION DEBUG: Starting summarization")
        logger.info(f"🔍 SUMMARIZATION DEBUG: Observation length: {len(obs_str)} chars")
        logger.info(f"🔍 SUMMARIZATION DEBUG: Tool name: '{tool_name}'")
        logger.info(f"🔍 SUMMARIZATION DEBUG: Question: '{question}'")
        logger.info(f"🔍 SUMMARIZATION DEBUG: Has OpenAI client: {hasattr(self, 'openai_client') and self.openai_client is not None}")
        logger.info(f"🔍 SUMMARIZATION DEBUG: Model type: {getattr(self, 'model_type', 'unknown')}")
        logger.info(f"🔍 SUMMARIZATION DEBUG: GPT model: {getattr(self, 'gpt_model', 'unknown')}")
        
        # If short enough, return as-is
        # if len(obs_str) <= 1000:
        #     logger.info(f"🔍 SUMMARIZATION DEBUG: Observation is short ({len(obs_str)} ≤ 1000), returning as-is")
        #     return obs_str
        
        # Tool-specific summarization strategies
        logger.info(f"🔍 SUMMARIZATION DEBUG: Entering tool-specific logic")
        
        if tool_name == "web_search":
            logger.info(f"🔍 SUMMARIZATION DEBUG: Matched web_text_to_text_search tool")
            logger.info(f"🔍 SUMMARIZATION MODEL: Will use GPT-4.1 (via generate_response_with_gpt4_for_summarization)")
            # Use GPT-4.1 for intelligent web search summarization
            if question:
                logger.info(f"🔍 SUMMARIZATION DEBUG: Question provided, attempting GPT-4.1 summarization")
                try:
                    summarization_prompt = f"""You are summarizing a web search observation. Your goal is to capture ALL potentially useful information, not to filter based on perceived relevance.

Question asked: {question}

Tool used: Web Search

Instructions:
1. Start with: "Tool: web_search"
1. Then state: "Query: [the search query used]"
3. Then summarize: "Response: [comprehensive summary of what was found]"
4. For EACH useful page in the results, include:
   - Page title
   - Key information from the content (even if seemingly tangential - it might be useful)
   - Source URL
5. Keep the summary under 1,000 characters, but prioritize completeness over brevity.

Fetched information: {obs_str}

Output format (only the following three components):
Tool: web_search
Query: [search query]
Response: [comprehensive summary with all pages and their content]"""

                    messages = [{"role": "user", "content": summarization_prompt}]
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Calling GPT-4.1 for summarization")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Prompt length: {len(summarization_prompt)} chars")
                    
                    summary = self.generate_response_with_gpt4_for_summarization(messages, max_tokens=800, temperature=0.1)
                    
                    logger.info(f"🔍 SUMMARIZATION DEBUG: GPT-4.1 summarization successful")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Summary length: {len(summary)} chars")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Summary preview: {summary[:200]}...")
                    
                    return summary

                except Exception as e:
                    logger.error(f"🔍 SUMMARIZATION DEBUG: GPT-4.1 summarization failed: {e}")
                    logger.error(f"🔍 SUMMARIZATION DEBUG: Exception type: {type(e).__name__}")
                    logger.error(f"🔍 SUMMARIZATION DEBUG: Falling back to truncation")
                    return obs_str[:500]
            else:
                logger.info(f"🔍 SUMMARIZATION DEBUG: No question provided for web_text_to_text_search, using truncation")
                return obs_str[:500]
        
        elif tool_name == "image_search":
            # Use GPT-4.1 for intelligent image search summarization
            logger.info(f"🔍 SUMMARIZATION DEBUG: Matched image_search tool")
            logger.info(f"🔍 SUMMARIZATION MODEL: Will use GPT-4.1 (via generate_response_with_gpt4_for_summarization)")
            if question:
                logger.info(f"🔍 SUMMARIZATION DEBUG: Question provided, attempting GPT-4.1 summarization")
                try:
                    summarization_prompt = f"""You are summarizing an image search observation. Your goal is to capture ALL potentially useful information, not to filter based on perceived relevance.

Question asked: {question}

Tool used: Image Search

Instructions:
1. Start with: "Tool: image_search"
2. Then state: "Query: [the search query used]"
3. Then state: "Image URL: [the search image url]"
4. Then summarize: "Response: [comprehensive summary of all images found]"
5. For EACH useful image in the results, include:
   - Image URL
   - Title (if available)
   - Description/context
6. Keep the summary under 1,000 characters, but prioritize completeness over brevity.

Observation: {obs_str}

Output format  (only the following four components):
Tool: image_search
Query: [search query]
Image URL: [the search image url]
Response: [comprehensive summary with all images and their details]"""

                    messages = [{"role": "user", "content": summarization_prompt}]
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Calling GPT-4.1 for image search summarization")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Prompt length: {len(summarization_prompt)} chars")
                    
                    summary = self.generate_response_with_gpt4_for_summarization(messages, max_tokens=800, temperature=0.1)
                    
                    logger.info(f"🔍 SUMMARIZATION DEBUG: GPT-4.1 image search summarization successful")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Summary length: {len(summary)} chars")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Summary preview: {summary[:200]}...")
                    
                    return summary
                        
                except Exception as e:
                    logger.error(f"🔍 SUMMARIZATION DEBUG: GPT-4.1 image search summarization failed: {e}")
                    logger.error(f"🔍 SUMMARIZATION DEBUG: Exception type: {type(e).__name__}")
                    logger.error(f"🔍 SUMMARIZATION DEBUG: Falling back to truncation")
                    return obs_str[:500]
            else:
                logger.info(f"🔍 SUMMARIZATION DEBUG: No question provided for web_text_to_img_search, using truncation")
                return obs_str[:500]

        elif tool_name == "content_extraction":
            # Use GPT-4.1 for intelligent webpage content summarization
            logger.info(f"🔍 SUMMARIZATION DEBUG: Matched content_extraction tool")
            logger.info(f"🔍 SUMMARIZATION MODEL: Will use GPT-4.1 (via generate_response_with_gpt4_for_summarization)")
            if question:
                logger.info(f"🔍 SUMMARIZATION DEBUG: Question provided, attempting GPT-4.1 summarization")
                try:
                    summarization_prompt = f"""You are summarizing webpage content extraction. Your goal is to capture ALL potentially useful information, not to filter based on perceived relevance.

Question asked: {question}

Tool used: Content Extraction (Webpage Reader)

Instructions:
1. Start with: "Tool: content_extraction"
2. Then state: "Webpage URL: [the URL that was read]"
3. Then summarize: "Response: [comprehensive summary of the webpage content]"
4. Include possible useful key information from the webpage:
   - Main topics and headings
   - Facts, statistics, dates, names
   - Context and background information
5. Keep the summary under 1,000 characters, but prioritize completeness over brevity.

Web Page Content: {obs_str}

Output format (only the following three components):
Tool: content_extraction
Webpage URL: [URL]
Response: [comprehensive summary of all webpage content]"""

                    messages = [{"role": "user", "content": summarization_prompt}]
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Calling GPT-4.1 for content extraction summarization")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Prompt length: {len(summarization_prompt)} chars")
                    
                    summary = self.generate_response_with_gpt4_for_summarization(messages, max_tokens=800, temperature=0.1)
                    
                    logger.info(f"🔍 SUMMARIZATION DEBUG: GPT-4.1 content extraction summarization successful")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Summary length: {len(summary)} chars")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Summary preview: {summary[:200]}...")
                    
                    return summary

                except Exception as e:
                    logger.error(f"🔍 SUMMARIZATION DEBUG: GPT-4.1 content extraction summarization failed: {e}")
                    logger.error(f"🔍 SUMMARIZATION DEBUG: Exception type: {type(e).__name__}")
                    logger.error(f"🔍 SUMMARIZATION DEBUG: Falling back to truncation")
                    return obs_str[:500]
            else:
                logger.info(f"🔍 SUMMARIZATION DEBUG: No question provided for web_url_reader, using truncation")
                return obs_str[:500]
        
        elif tool_name == "reverse_image_search":
            # Use GPT-4.1 for intelligent Google Lens image search summarization
            logger.info(f"🔍 SUMMARIZATION DEBUG: Matched reverse_image_search tool (Google Lens)")
            logger.info(f"🔍 SUMMARIZATION MODEL: Will use GPT-4.1 (via generate_response_with_gpt4_for_summarization)")
            if question:
                logger.info(f"🔍 SUMMARIZATION DEBUG: Question provided, attempting GPT-4.1 summarization")
                try:
                    summarization_prompt = f"""You are summarizing a Google Lens image search observation. Your goal is to capture ALL potentially useful information, not to filter based on perceived relevance.

Question asked: {question}

Tool used: Google Lens Image Search

The observation contains Visual Matches with the following structure for each match:
- Match number (position)
- Title: Brief description/title of the match
- Source: Website/platform name (e.g., Wikipedia, ResearchGate, ScienceDirect)
- Link: Full URL to the source
- Thumbnail: URL of the thumbnail image

Instructions:
1. Start with: "Tool: reverse_image_search"
2. Then state: "Image URL: [the image URL that was searched]"
3. Then state: "Query:": [any context query if provided]"
4. Then summarize: "Response: [comprehensive summary of visual matches found]"
5. List the TOP 5-7 most relevant visual matches, including:
   - Match number and title
   - Source (keep it concise, e.g., "Wikipedia", "ResearchGate")
   - Brief note about what this match reveals about the image
   - Link (for reference)
6. Keep the summary under 1,000 characters, but prioritize completeness over brevity.

Fetched information: {obs_str}

Output format  (only the following four components):
Tool: reverse_image_search
Image URL: [image URL]
Query: [context query if provided, otherwise omit this line]
Response: Found X visual matches. Match 1: [title] from [source] - [brief insight] ([link]). Match 2: ... [Continue for top matches]"""

                    messages = [{"role": "user", "content": summarization_prompt}]
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Calling GPT-4.1 for Google Lens image search summarization")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Prompt length: {len(summarization_prompt)} chars")
                    
                    summary = self.generate_response_with_gpt4_for_summarization(messages, max_tokens=800, temperature=0.1)
                    
                    logger.info(f"🔍 SUMMARIZATION DEBUG: GPT-4.1 Google Lens image search summarization successful")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Summary length: {len(summary)} chars")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Summary preview: {summary[:200]}...")
                    
                    return summary

                except Exception as e:
                    logger.error(f"🔍 SUMMARIZATION DEBUG: GPT-4.1 Google Lens image search summarization failed: {e}")
                    logger.error(f"🔍 SUMMARIZATION DEBUG: Exception type: {type(e).__name__}")
                    logger.error(f"🔍 SUMMARIZATION DEBUG: Falling back to truncation")
                    return obs_str[:500]
            else:
                logger.info(f"🔍 SUMMARIZATION DEBUG: No question provided for web_image_to_text, using truncation")
                return obs_str[:500]
        
        elif tool_name == "code_execution":
            logger.info(f"🔍 SUMMARIZATION DEBUG: Matched ipython_code tool")
            logger.info(f"🔍 SUMMARIZATION MODEL: Will use GPT-4.1 (via generate_response_with_gpt4_for_summarization)")
            # Use GPT-4.1 for intelligent code execution summarization
            if question:
                logger.info(f"🔍 SUMMARIZATION DEBUG: Question provided, attempting GPT-4.1 summarization")
                try:
                    summarization_prompt = f"""You are summarizing a Python code execution observation. Your goal is to clearly show BOTH the code that was executed AND the output it produced.

Question asked: {question}

Tool used: IPython Code Execution

Instructions:
1. Start with: "Tool: code_execution"
2. Then state: "Code executed: [the Python code that was run]"
3. Then state: "Output: [the result/output from executing the code]"
4. If there was an error, clearly state: "Error: [error message]"
5. Keep the summary clear and concise, showing the calculation or computation that was performed.

Code execution observation: {obs_str}

Output format (only the following three components):
Tool: code_execution
Code executed: [Python code]
Output: [execution result or error]"""

                    messages = [{"role": "user", "content": summarization_prompt}]
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Calling GPT-4.1 for code execution summarization")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Prompt length: {len(summarization_prompt)} chars")
                    
                    summary = self.generate_response_with_gpt4_for_summarization(messages, max_tokens=500, temperature=0.1)
                    
                    logger.info(f"🔍 SUMMARIZATION DEBUG: GPT-4.1 code execution summarization successful")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Summary length: {len(summary)} chars")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Summary preview: {summary[:200]}...")
                    
                    return summary

                except Exception as e:
                    logger.error(f"🔍 SUMMARIZATION DEBUG: GPT-4.1 code execution summarization failed: {e}")
                    logger.error(f"🔍 SUMMARIZATION DEBUG: Exception type: {type(e).__name__}")
                    logger.error(f"🔍 SUMMARIZATION DEBUG: Falling back to truncation")
                    return f"Tool: code_execution\n{obs_str[:800]}"
            else:
                logger.info(f"🔍 SUMMARIZATION DEBUG: No question provided for ipython_code, using truncation")
                return f"Tool: code_execution\n{obs_str[:800]}"
        
        elif tool_name == "bash_terminal":
            logger.info(f"🔍 SUMMARIZATION DEBUG: Matched bash_terminal tool")
            logger.info(f"🔍 SUMMARIZATION DEBUG: Using simple truncation (TODO: implement bash summarization)")
            # TODO: Implement bash command result summarization
            # Strategy: Extract command output and status
            # Focus on: command success/failure, key output lines, file listings
            return obs_str[:1000]
        
        elif tool_name == "ocr":
            logger.info(f"🔍 SUMMARIZATION DEBUG: Matched ocr tool")
            # If short enough, return as-is
            if len(obs_str) <= 800:
                return f"Tool: ocr_tool\n\nOutput: {obs_str}"
            
            # Use GPT-4.1 for OCR summarization when observation is too long
            logger.info(f"🔍 SUMMARIZATION MODEL: Will use GPT-4.1 for OCR summarization (length: {len(obs_str)} > 800)")
            if question:
                try:
                    summarization_prompt = f"""You are summarizing OCR (text extraction) results from an image. Your goal is to list ALL text found concisely.

Question asked: {question}

Tool used: OCR (Text Extraction)

Instructions:
1. Start with: "Tool: ocr_tool"
2. Then state: "Output: [concise list of all text found + location]"
3. List each text item found in the image (signs, labels, numbers, names, etc.)
4. Keep it concise with short location.
5. Keep the summary under 600 characters.

OCR observation: {obs_str}

Output format (only the following two components):
Tool: ocr_tool
Output: [concise list of text found]"""

                    messages = [{"role": "user", "content": summarization_prompt}]
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Calling GPT-4.1 for OCR summarization")
                    
                    summary = self.generate_response_with_gpt4_for_summarization(messages, max_tokens=500, temperature=0.1)
                    
                    logger.info(f"🔍 SUMMARIZATION DEBUG: GPT-4.1 OCR summarization successful")
                    logger.info(f"🔍 SUMMARIZATION DEBUG: Summary length: {len(summary)} chars")
                    
                    return summary

                except Exception as e:
                    logger.error(f"🔍 SUMMARIZATION DEBUG: GPT-4.1 OCR summarization failed: {e}")
                    logger.error(f"🔍 SUMMARIZATION DEBUG: Falling back to truncation")
                    return f"Tool: ocr_tool\n\nOutput: {obs_str[:800]}"
            else:
                logger.info(f"🔍 SUMMARIZATION DEBUG: No question provided for ocr, using truncation")
                return f"Tool: ocr_tool\n\nOutput: {obs_str[:800]}"
        
        else:
            logger.info(f"🔍 SUMMARIZATION DEBUG: No specific handler for tool '{tool_name}', using generic fallback")
        
        logger.info(f"🔍 SUMMARIZATION DEBUG: Reached final fallback - returning first 500 characters")
        logger.debug(f"Summarizer fallback for tool '{tool_name}'. Returning first 500 characters of observation.")
        return obs_str[:1000]
    
    # ========== GPT Implementation Methods ==========
    
    def _analyze_image_with_gpt(self, image_path: str, analysis_prompt: str) -> str:
        """Use GPT vision API to analyze an image"""
        try:
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                return "Error: Could not encode image"

            response = self.openai_client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.99
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error analyzing image with GPT: {e}")
            return f"Error analyzing image: {str(e)}"
    
    def _generate_response_with_gpt(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.99) -> str:
        """Generate a response using OpenAI GPT"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.gpt_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response with GPT: {e}")
            return f"Error generating response: {str(e)}"
    
    # ========== vLLM Implementation Methods ==========
    
    def _analyze_image_with_vllm(self, image_path: str, analysis_prompt: str) -> str:
        """Use vLLM Qwen model to analyze an image"""
        try:
            response = self._call_qwen_vllm(text=analysis_prompt, images=[image_path], max_tokens=1000, temperature=0.1)
            return response.get('content', 'Error: No content in response')
        except Exception as e:
            logger.error(f"Error analyzing image with vLLM: {e}")
            return f"Error analyzing image: {str(e)}"
    
    def _save_base64_image_to_temp(self, base64_data: str) -> Optional[str]:
        """Save a base64 encoded image to a temporary file and return the path"""
        try:
            import tempfile
            from PIL import Image
            import io
            
            # Extract the base64 data (remove the data:image/...;base64, prefix if present)
            if ',' in base64_data:
                base64_data = base64_data.split(',', 1)[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_data)
            
            # Open image and save to temp file
            image = Image.open(io.BytesIO(image_bytes))
            
            # Create temp file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg', prefix='vllm_image_')
            os.close(temp_fd)  # Close the file descriptor
            
            # Save image
            image.save(temp_path, format='JPEG')
            
            logger.info(f"🔍 VLLM DEBUG: Saved base64 image to temp file: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"🔍 VLLM DEBUG: Error saving base64 image to temp file: {e}")
            return None
    
    def _generate_response_with_vllm(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.99) -> str:
        """Generate a response using vLLM Qwen model"""
        temp_image_paths = []  # Track temp files for cleanup
        try:
            # Extract text and images from messages
            text_parts = []
            image_paths = []
            
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                # Handle multimodal content (list with text and images)
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                text_parts.append(f"{role}: {item.get('text', '')}")
                            elif item.get('type') == 'image_url':
                                # Extract base64 image and convert to file path if needed
                                image_url = item.get('image_url', {}).get('url', '')
                                if image_url.startswith('data:image'):
                                    # Convert base64 to temp file
                                    temp_path = self._save_base64_image_to_temp(image_url)
                                    if temp_path:
                                        image_paths.append(temp_path)
                                        temp_image_paths.append(temp_path)
                                else:
                                    image_paths.append(image_url)
                        else:
                            text_parts.append(f"{role}: {item}")
                # Handle simple text content
                elif isinstance(content, str):
                    text_parts.append(f"{role}: {content}")
            
            text = "\n".join(text_parts)
            
            # Call vLLM with images if available
            if image_paths:
                response = self._call_qwen_vllm(text=text, images=image_paths, max_tokens=max_tokens, temperature=temperature)
            else:
                response = self._call_qwen_vllm(text=text, max_tokens=max_tokens, temperature=temperature)
            
            result = response.get('content', 'Error: No content in response')
            
            # Clean up temp files
            for temp_path in temp_image_paths:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        logger.info(f"🔍 VLLM DEBUG: Cleaned up temp image file: {temp_path}")
                except Exception as e:
                    logger.warning(f"🔍 VLLM DEBUG: Could not delete temp file {temp_path}: {e}")
            
            return result
        except Exception as e:
            logger.error(f"Error generating response with vLLM: {e}")
            # Clean up temp files on error
            for temp_path in temp_image_paths:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass
            return f"Error generating response: {str(e)}"
    
    # ========== vLLM Helper Methods ==========
    
    def _load_llm_sampling_params(self):
        """Load vLLM model, tokenizer, and sampling parameters"""
        import vllm
        
        # Set GPU device if specified
        if self.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            logger.info(f"🔍 DEBUG: Set CUDA_VISIBLE_DEVICES to GPU {self.gpu_id}")
        
        model = vllm.LLM(
            self.model_path,
            trust_remote_code=True,  # REQUIRED: To load the VL model's custom code
            max_model_len=self.max_model_len,
            enable_prefix_caching=True,
            load_format='safetensors',
            tensor_parallel_size=self.tensor_parallel_size
        )
        tokenizer = model.get_tokenizer()
        sampling_params = vllm.SamplingParams(
            temperature=0.85, 
            top_p=0.9, 
            max_tokens=2048,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
        )
        return model, sampling_params, tokenizer
    
    def _log_tokenizer_info(self):
        """Log tokenizer information for debugging multimodal issues"""
        logger.info(f"🔍 DEBUG: Tokenizer vocab size: {self.tokenizer.vocab_size}")
        logger.info(f"🔍 DEBUG: Tokenizer EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        logger.info(f"🔍 DEBUG: Tokenizer special tokens: {list(self.tokenizer.special_tokens_map.keys())}")
        
        # Test image token encoding
        test_tokens = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|im_start|>", "<|im_end|>"]
        for token in test_tokens:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                logger.info(f"🔍 DEBUG: Token '{token}' -> ID: {token_id}")
            except Exception as e:
                logger.warning(f"🔍 DEBUG: Could not encode token '{token}': {e}")
    
    def _call_qwen_vllm(self, text: str, images: Optional[List[str]] = None, max_tokens: int = 2048, temperature: float = 0.85) -> Dict[str, Any]:
        """Call Qwen VL model via vLLM with multimodal input"""
        try:
            # Prepare conversation format
            conversation = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # Add images if provided
            if images:
                for image_path in images:
                    if os.path.exists(image_path):
                        conversation[0]["content"].append({
                            "type": "image",
                            "image": image_path
                        })
                    else:
                        logger.warning(f"🔍 DEBUG: Image not found: {image_path}")
            
            # Add text content
            conversation[0]["content"].append({
                "type": "text", 
                "text": text
            })
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            logger.info(f"🔍 DEBUG: Generated prompt length: {len(prompt)} chars")
            logger.info(f"🔍 DEBUG: Generated prompt preview: {prompt[:200]}...")
            
            # Update sampling params for this call
            sampling_params = vllm.SamplingParams(
                temperature=temperature,
                top_p=0.9,
                max_tokens=max_tokens,
                stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")]
            )
            
            # Generate response
            outputs = self.model.generate([prompt], sampling_params)
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text
                logger.info(f"🔍 DEBUG: Generated response length: {len(generated_text)} chars")
                logger.info(f"🔍 DEBUG: Generated response preview: {generated_text[:200]}...")
                
                return {
                    "content": generated_text,
                    "prompt_tokens": len(self.tokenizer.encode(prompt)),
                    "completion_tokens": len(self.tokenizer.encode(generated_text)),
                    "total_tokens": len(self.tokenizer.encode(prompt)) + len(self.tokenizer.encode(generated_text))
                }
            else:
                logger.error("🔍 DEBUG: No output generated from vLLM")
                return {"content": "Error: No output generated", "error": "no_output"}
                
        except Exception as e:
            logger.error(f"🔍 DEBUG: Error calling Qwen vLLM: {e}")
            return {"content": f"Error: {str(e)}", "error": str(e)}
    
    def _analyze_image_with_gemini(self, image_path: str, analysis_prompt: str) -> str:
        """Use Gemini to analyze an image"""
        try:
            from pathlib import Path
            
            # Get the image path
            img_path = Path(image_path)
            
            # Determine mime type based on file extension
            extension = img_path.suffix.lower()
            mime_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_type_map.get(extension, 'image/jpeg')
            
            # Prepare multimodal content for Gemini using the new SDK format
            contents = [
                types.Part.from_bytes(
                    data=img_path.read_bytes(),
                    mime_type=mime_type
                ),
                analysis_prompt
            ]
            
            logger.info(f"🔍 GEMINI DEBUG: Analyzing image with Gemini")
            logger.info(f"🔍 GEMINI DEBUG: Image path: {image_path}")
            logger.info(f"🔍 GEMINI DEBUG: Prompt: {analysis_prompt[:100]}...")
            logger.info(f"🔍 GEMINI DEBUG: Temperature: {self.gemini_temperature}, Top-p: {self.gemini_top_p}")
            
            # Create config with temperature
            config = types.GenerateContentConfig(
                temperature=self.gemini_temperature,
                top_p=self.gemini_top_p,
            )
            
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config=config
            )
            
            result = response.text
            logger.info(f"🔍 GEMINI DEBUG: Response length: {len(result)} chars")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error analyzing image: {str(e)}"
    
    def _generate_response_with_gemini(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.99) -> str:
        """Generate a response using Gemini"""
        try:
            # Convert OpenAI-style messages to Gemini format
            contents = []
            
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                # Handle multimodal content (list with text and images)
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                contents.append(item.get('text', ''))
                            elif item.get('type') == 'image_url':
                                image_url = item.get('image_url', {}).get('url', '')
                                if image_url == "<cached_base64>":
                                    # Skip placeholder - image already processed
                                    continue
                                elif image_url.startswith('data:image'):
                                    # Extract base64 data
                                    if ',' in image_url:
                                        base64_data = image_url.split(',', 1)[1]
                                    else:
                                        base64_data = image_url
                                    
                                    # Decode base64 to bytes
                                    import base64
                                    image_bytes = base64.b64decode(base64_data)
                                    
                                    # Determine mime type from data URL
                                    mime_type = "image/jpeg"
                                    if image_url.startswith('data:image/png'):
                                        mime_type = "image/png"
                                    elif image_url.startswith('data:image/gif'):
                                        mime_type = "image/gif"
                                    elif image_url.startswith('data:image/webp'):
                                        mime_type = "image/webp"
                                    
                                    # Add image to contents using new SDK format
                                    contents.append(
                                        types.Part.from_bytes(
                                            data=image_bytes,
                                            mime_type=mime_type
                                        )
                                    )
                                else:
                                    # File path - read the file
                                    try:
                                        from pathlib import Path as PathLib
                                        img_file_path = PathLib(image_url)
                                        
                                        # Determine mime type based on file extension
                                        extension = img_file_path.suffix.lower()
                                        mime_type_map = {
                                            '.jpg': 'image/jpeg',
                                            '.jpeg': 'image/jpeg',
                                            '.png': 'image/png',
                                            '.gif': 'image/gif',
                                            '.webp': 'image/webp'
                                        }
                                        mime_type = mime_type_map.get(extension, 'image/jpeg')
                                        
                                        # Add image to contents using new SDK format
                                        contents.append(
                                            types.Part.from_bytes(
                                                data=img_file_path.read_bytes(),
                                                mime_type=mime_type
                                            )
                                        )
                                    except Exception as e:
                                        logger.error(f"🔍 GEMINI DEBUG: Error reading image file {image_url}: {e}")
                        else:
                            contents.append(str(item))
                # Handle simple text content
                elif isinstance(content, str):
                    contents.append(content)
            
            # Use the temperature parameter passed to the method, or fall back to configured value
            actual_temperature = temperature if temperature != 0.99 else self.gemini_temperature
            
            logger.info(f"🔍 GEMINI DEBUG: Generating response with Gemini")
            logger.info(f"🔍 GEMINI DEBUG: Number of content items: {len(contents)}")
            logger.info(f"🔍 GEMINI DEBUG: Model: {self.gemini_model}")
            logger.info(f"🔍 GEMINI DEBUG: Temperature: {actual_temperature}, Top-p: {self.gemini_top_p}")
            
            # Create config with temperature
            config = types.GenerateContentConfig(
                temperature=actual_temperature,
                top_p=self.gemini_top_p,
            )
            
            # Generate response
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config=config
            )
            
            result = response.text
            logger.info(f"🔍 GEMINI DEBUG: Response generated successfully")
            logger.info(f"🔍 GEMINI DEBUG: Response length: {len(result)} chars")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error generating response: {str(e)}"
    
    def load_multimodal_scenarios(self, file_name: str = "qa_formatted.json", max_scenarios: int = 3) -> List[Dict[str, Any]]:
        """Load multimodal research scenarios from JSON file with difficulty filtering"""
        file_path = self.input_dir / file_name

        if not file_path.exists():
            logger.error(f"Scenario file not found: {file_path}")
            return []

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            scenarios = data.get('scenarios', [])
            
            # Filter by difficulty if specified
            if self.difficulty_filter:
                original_count = len(scenarios)
                scenarios = [s for s in scenarios if s.get('difficulty', '').lower() == self.difficulty_filter.lower()]
                logger.info(f"Filtered scenarios by difficulty '{self.difficulty_filter}': {original_count} -> {len(scenarios)} scenarios")
            
            # Limit to max_scenarios for testing
            # try:
            #     scenarios = [s for s in scenarios if s["category"] == "search_required"]
            # except: pass

            if max_scenarios and len(scenarios) > max_scenarios:
                scenarios = scenarios[:max_scenarios]
                logger.info(
                    f"Loaded {len(scenarios)} multimodal scenarios from {file_name} (limited to {max_scenarios} for testing)")
            else:
                logger.info(f"Loaded {len(scenarios)} multimodal scenarios from {file_name}")

            # Add image_url if prefix is provided
            if self.image_url_prefix:
                for scenario in scenarios:
                    if "image" in scenario["question"] and type(scenario["question"]["image"]) == list:
                        scenario["question"]["image"] = scenario["question"]["image"][0]
                    if "question" in scenario and "image" in scenario["question"]:
                        image_path = scenario["question"]["image"].replace("./", "")
                        scenario["question"]["image_url"] = self.image_url_prefix + image_path

            return scenarios

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_name}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading {file_name}: {e}")
            return []

    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for GPT vision API"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return ""
    
    def _get_image_path_from_scenario(self, scenario: Dict[str, Any]) -> Optional[str]:
        """Safely extract image path from scenario, handling both list and string formats"""
        if not scenario or "question" not in scenario or "image" not in scenario["question"]:
            return None
        
        image = scenario["question"]["image"]
        
        # Handle list format (take first element)
        if isinstance(image, list):
            if len(image) > 0:
                image = image[0]
            else:
                return None
        
        # Handle string format
        if isinstance(image, str):
            # Remove leading "./" if present
            image_path = image.replace("./", "")
            return str(self.input_dir / "images" / image_path)
        
        return None
    
    def get_tool_descriptions(self) -> str:
        """Generate tool descriptions for enabled tools"""
        descriptions = []
        
        for tool_name in self.enabled_tools:
            if tool_name not in self.TOOL_REGISTRY:
                continue
                
            tool_info = self.TOOL_REGISTRY[tool_name]
            
            if tool_name == "image_operations":
                # Special handling for image_operations with sub-actions
                desc = f"- {tool_info['format']}: {tool_info['description']}\n"
                for sub_action, sub_info in tool_info['sub_actions'].items():
                    params_str = json.dumps(sub_info['params'])
                    desc += f"  * {sub_action}: {params_str} - {sub_info['desc']}\n"
                descriptions.append(desc.rstrip())
            elif tool_name == "finish":
                # Special handling for finish action
                descriptions.append(f"- {tool_info['description']}")
            else:
                # Standard tool format
                params_str = json.dumps(tool_info['parameters'])
                descriptions.append(f"- {tool_info['format']}: {params_str} - {tool_info['description']}")
        
        return "\n".join(descriptions)
    
    def analyze_image_with_gpt(self, image_path: str, analysis_prompt: str = "Analyze this image and describe what you see in detail.") -> str:
        """Wrapper method for backward compatibility. Calls the abstract method."""
        return self.analyze_image_with_model(image_path, analysis_prompt)

    def convert_gpt_action_to_tool_action(self, gpt_action: Dict[str, Any], scenario: Dict[str, Any] = None) -> str:
        """Convert GPT-generated action to tool server action format"""

        logger.info(f"Full gpt_action structure: {gpt_action}")
        
        # Extract action_type from the nested action structure
        action = gpt_action.get("action", gpt_action)
        action_type = gpt_action.get("action_type", "")
        action_params = gpt_action.get("action_parameters", {})
        
        logger.info(f"Converting action: type='{action_type}', params={action_params}")
        logger.info(f"Action type repr: {repr(action_type)}")
        logger.info(f"Action type == 'web_search': {action_type == 'web_search'}")
        
        # Check if the action type corresponds to an enabled tool
        tool_format_map = {info['format']: tool_name for tool_name, info in self.TOOL_REGISTRY.items()}
        
        # Map GPT action types to tool server actions
        if action_type == "web_search":
            # Check which web search tools are enabled
            if "web_text_to_text_search" in self.enabled_tools:
                query = action_params.get("query", "")
                result = f"<text_search_text>{query}</text_search_text>"
                logger.debug(f"Generated web_search action: {result}")
                return result
            else:
                result = f"<python>\nprint('Error: web_text_to_text_search tool is not enabled')\n</python>"
                logger.debug(f"Generated disabled tool error: {result}")
                return result

        elif action_type == "image_search":
            if "web_text_to_img_search" in self.enabled_tools:
                query = action_params.get("query", "")
                result = f"<text_search_image>{query}</text_search_image>"
                logger.debug(f"Generated image_search action: {result}")
                return result
            else:
                result = f"<python>\nprint('Error: web_text_to_img_search tool is not enabled')\n</python>"
                logger.debug(f"Generated disabled tool error: {result}")
                return result

        elif action_type == "content_extraction":
            if "web_url_reader" in self.enabled_tools:
                url = action_params.get("url", "")
                
                # Replace placeholder URL with real URL from scenario (if it's an image URL placeholder)
                # Note: content_extraction is typically for web URLs, but may receive image URL placeholder
                if url == IMAGE_URL_PLACEHOLDER and scenario and "question" in scenario:
                    # First try to get image_url from scenario (if image_url_prefix was set)
                    url = scenario["question"].get("image_url", url)
                    logger.debug(f"Tried to get image_url from scenario for content_extraction: {url}")
                    
                    # If still placeholder or empty, fall back to local image path
                    # (though content_extraction typically expects URLs, some tools may accept file paths)
                    if url == IMAGE_URL_PLACEHOLDER or not url:
                        image_path = self._get_image_path_from_scenario(scenario)
                        if image_path:
                            url = image_path
                            logger.debug(f"Fell back to local image path for content_extraction: {url}")
                        else:
                            logger.warning(f"Could not find image URL or path for content_extraction action")
                    else:
                        logger.debug(f"Using image_url from scenario for content_extraction: {url}")
                
                result = f"<web_read>{url}</web_read>"
                logger.debug(f"Generated content_extraction action: {result}")
                return result
            else:
                result = f"<python>\nprint('Error: web_url_reader tool is not enabled')\n</python>"
                logger.debug(f"Generated disabled tool error: {result}")
                return result

        elif action_type == "reverse_image_search":
            if "web_image_to_text" in self.enabled_tools:
                image_url = action_params.get("image_url", "")
                query = action_params.get("query", "")
                
                # Replace placeholder URL with real URL from scenario
                if image_url == IMAGE_URL_PLACEHOLDER and scenario and "question" in scenario:
                    # First try to get image_url from scenario (if image_url_prefix was set)
                    image_url = scenario["question"].get("image_url", image_url)
                    logger.debug(f"Tried to get image_url from scenario: {image_url}")
                    
                    # If still placeholder or empty, fall back to local image path
                    if image_url == IMAGE_URL_PLACEHOLDER or not image_url:
                        image_path = self._get_image_path_from_scenario(scenario)
                        if image_path:
                            image_url = image_path
                            logger.debug(f"Fell back to local image path: {image_url}")
                        else:
                            logger.warning(f"Could not find image URL or path for reverse_image_search action")
                    else:
                        logger.debug(f"Using image_url from scenario: {image_url}")
                
                # Format: image_url||query (query is optional)
                if query:
                    result = f"<image_search_text>{image_url}||{query}</image_search_text>"
                else:
                    result = f"<image_search_text>{image_url}</image_search_text>"
                
                logger.debug(f"Generated reverse_image_search action: {result}")
                return result
            else:
                result = f"<python>\\nprint('Error: web_image_to_text tool is not enabled')\\n</python>"
                logger.debug(f"Generated disabled tool error: {result}")
                return result

        elif action_type == "ocr":
            if "ocr_tool" in self.enabled_tools:
                image_url = action_params.get("image_url", "")
                
                # Replace placeholder URL with real URL from scenario
                if image_url == IMAGE_URL_PLACEHOLDER and scenario and "question" in scenario:
                    # First try to get image_url from scenario (if image_url_prefix was set)
                    image_url = scenario["question"].get("image_url", image_url)
                    logger.debug(f"Tried to get image_url from scenario: {image_url}")
                    
                    # If still placeholder or empty, fall back to local image path
                    if image_url == IMAGE_URL_PLACEHOLDER or not image_url:
                        image_path = self._get_image_path_from_scenario(scenario)
                        if image_path:
                            image_url = image_path
                            logger.debug(f"Fell back to local image path: {image_url}")
                        else:
                            logger.warning(f"Could not find image URL or path for OCR action")
                    else:
                        logger.debug(f"Using image_url from scenario: {image_url}")
                
                result = f"<ocr_tool>{image_url}</ocr_tool>"
                logger.debug(f"Generated ocr action: {result}")
                return result
            else:
                result = f"<python>\nprint('Error: ocr_tool is not enabled')\n</python>"
                logger.debug(f"Generated disabled tool error: {result}")
                return result

        elif action_type == "code_execution":
            if "ipython_code" not in self.enabled_tools:
                result = f"<python>\nprint('Error: ipython_code tool is not enabled')\n</python>"
                logger.debug(f"Generated disabled tool error: {result}")
                return result
            code = action_params.get("code", "")
            result = f"<python>\n{code}\n</python>"
            logger.debug(f"Generated code_execution action: {result[:100]}...")
            return result
        
        elif action_type == "bash_execution":
            if "bash_terminal" not in self.enabled_tools:
                result = f"<python>\nprint('Error: bash_terminal tool is not enabled')\n</python>"
                logger.debug(f"Generated disabled tool error: {result}")
                return result
            command = action_params.get("command", "")
            result = f"<bash>\n{command}\n</bash>"
            logger.debug(f"Generated bash_execution action: {result}")
            return result
        
        elif action_type == "image_operations":
            if "image_operations" not in self.enabled_tools:
                return f"<python>\nprint('Error: image_operations tool is not enabled')\n</python>"
            
            # Handle image operations (zoom, crop)
            # Supports: zoom_in, crop_image_normalized
            image_path = self._get_image_path_from_scenario(scenario)
            if image_path:
                
                # Get specific action from parameters (default to zoom_in)
                operation_action = action_params.get("action", "zoom_in")
                
                if operation_action in ["zoom_in", "crop_image_normalized"]:
                    # Zoom/crop action - requires bbox_2d
                    bbox_2d = action_params.get("bbox_2d", [0.0, 0.0, 1.0, 1.0])
                    tool_call = {
                        "name": operation_action,
                        "arguments": {
                            "bbox_2d": bbox_2d,
                            "target_image": 1
                        }
                    }
                    return f"<tool_call>{json.dumps(tool_call)}</tool_call>"
                
                else:
                    # Invalid action - return error
                    return f"<python>\nprint('Error: Invalid image_operations action: {operation_action}')\nprint('Valid actions: zoom_in, crop_image_normalized')\n</python>"
            else:
                return f"<python>\nprint('Error: No image available for operations')\n</python>"

        elif action_type == "file_operations":
            operation = action_params.get("operation", "")
            file_path = action_params.get("file_path", "")
            return f"<bash>\n{operation} {file_path}\n</bash>"

        elif action_type == "image_analysis":
            # Handle image analysis with GPT vision API
            image_path = self._get_image_path_from_scenario(scenario)
            if image_path:

                # Get specific query from action parameters
                query = action_params.get("query", "Analyze this image and describe what you see in detail.")
                analysis_prompt = f"Based on the research question: '{scenario['question']['text']}', {query}"

                # Use GPT to analyze the image
                analysis_result = self.analyze_image_with_gpt(image_path, analysis_prompt)

                # Return the analysis result in the correct python_code format
                escaped_result = analysis_result.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
                return f"<python>\nprint('Image Analysis Result:')\nprint('{escaped_result}')\n</python>"
            else:
                return f"<python>\nprint('Error: No image available for analysis')\n</python>"

        elif action_type == "database_query":
            query = action_params.get("query", "")
            return f"<python>\nprint('Database query: {query}')\nprint('Note: Database functionality not implemented in this test environment')\n</python>"
        
        else:
            # Unknown action type
            description = action.get("action_description", "Unknown action")
            result = f"<python>\nprint('Error: Unknown or unsupported action type: {action_type}')\nprint('Description: {description}')\n</python>"
            logger.debug(f"Generated unknown action error: {result}")
            return result
    
    def simulate_multimodal_research_trajectory(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a complete research trajectory using ReAct pattern
        
        Enhanced trajectory structure includes complete input/output traceability:
        
        For each step:
        - reasoning_step_details: GPT reasoning input/output
          - input_messages: Messages sent to GPT for reasoning
          - output_content: Raw GPT response content
          - step_type: "reasoning"
          - iteration: ReAct iteration number
        
        - tool_execution_details: Tool server input/output  
          - input_payload: Payload sent to tool server
          - output_response: Response from tool server
          - step_type: "tool_execution"
          - server_url: Tool server endpoint
          - processing_time_ms: Execution time
        
        For final answer:
        - final_answer_details: Final answer generation input/output
          - input_messages: Messages sent to GPT for final answer
          - output_content: Final answer content
          - step_type: "final_answer"
          - token_usage: Token usage statistics
        
        This provides complete end-to-end traceability for the entire ReAct process.
        """

        logger.info(f"Starting ReAct multimodal research trajectory for: {scenario['id']}")
        logger.info(f"Question: {scenario['question']['text']}")

        # Get model information for tracking
        if self.model_type == "gpt":
            model_info = {
                "model_type": "gpt",
                "model_name": self.gpt_model
            }
        elif self.model_type == "vllm":
            model_info = {
                "model_type": "vllm",
                "model_name": self.model_path
            }
        elif self.model_type == "gemini":
            model_info = {
                "model_type": "gemini",
                "model_name": self.gemini_model
            }
        else:
            model_info = {
                "model_type": self.model_type,
                "model_name": "unknown"
            }

        trajectory = {
            "scenario_id": scenario["id"],
            "question": scenario["question"],
            "ground_truth": scenario["ground_truth"],
            "model_info": model_info,  # Track which model was used
            "steps": [],  # Tool execution steps only
            "react_steps": [],  # ReAct reasoning steps
            "start_time": time.time(),
            "total_actions": 0,
            "react_iterations": 0
        }

        # ReAct loop: Reason -> Act -> Observe -> Repeat until conclusion
        max_iterations = 6  # Reduced from 8 to prevent excessive iterations
        current_iteration = 0
        consecutive_failures = 0
        max_consecutive_failures = 2

        while current_iteration < max_iterations:
            current_iteration += 1
            trajectory["react_iterations"] = current_iteration

            logger.info(f"\n--- ReAct Iteration {current_iteration} ---")

            # Step 1: Reasoning - Decide what to do next
            if current_iteration == 1:
                prompt_tool = self.prompt_tool
            else:
                prompt_tool = None

            reasoning_result = self._react_reasoning_step(trajectory, scenario, current_iteration, prompt_tool)

            # Always save ReAct step details, even if we're stopping
            react_step_entry = {
                "iteration": current_iteration,
                "type": "reasoning",
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            # Add reasoning step details
            if "react_step_details" in reasoning_result:
                react_step_entry["reasoning_step_details"] = reasoning_result["react_step_details"]

            # Check stopping criteria
            should_stop = reasoning_result.get("should_stop", False)
            confidence = reasoning_result.get("confidence", 0.5)

            # Stop if we have high confidence and sufficient information
            if should_stop and confidence > 0.7:
                logger.info(f"ReAct concluded after {current_iteration} iterations (confidence: {confidence:.2f})")
                react_step_entry["conclusion"] = f"High confidence conclusion (confidence: {confidence:.2f})"
                trajectory["react_steps"].append(react_step_entry)
                break

            # Stop if too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                logger.info(f"ReAct stopped due to {consecutive_failures} consecutive failures")
                react_step_entry["conclusion"] = f"Stopped due to {consecutive_failures} consecutive failures"
                trajectory["react_steps"].append(react_step_entry)
                break

            # Save the reasoning step to react_steps
            trajectory["react_steps"].append(react_step_entry)

            # Step 2: Acting - Execute the decided action
            action_result = self._react_action_step(reasoning_result["action"], scenario, current_iteration)
            
            # Add reasoning step details to the action result
            if "react_step_details" in reasoning_result:
                action_result["reasoning_step_details"] = reasoning_result["react_step_details"]
            
            trajectory["steps"].append(action_result)
            trajectory["total_actions"] += 1

            # Track consecutive failures
            if action_result["success"]:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            # Step 3: Observing - Process the result (already done in action step)
            logger.info(f"✓ Iteration {current_iteration} completed (success: {action_result['success']})")

            # Add delay between iterations
            time.sleep(0.5)  # Reduced delay

        trajectory["end_time"] = time.time()
        trajectory["duration"] = trajectory["end_time"] - trajectory["start_time"]

        # Generate final comprehensive answer
        final_answer, final_answer_tokens, final_answer_details = self._generate_final_answer(trajectory, scenario)
        trajectory["final_answer"] = final_answer
        trajectory["final_answer_tokens"] = final_answer_tokens
        trajectory["final_answer_details"] = final_answer_details

        return trajectory

    def _react_reasoning_step(self, trajectory: Dict[str, Any], scenario: Dict[str, Any], iteration: int, prompt_tool: str = None) -> Dict[
        str, Any]:
        """ReAct Reasoning: Analyze current state and decide next action"""

        # Collect all previous observations with success/failure status
        previous_observations = []
        previous_actions = []
        successful_steps = 0
        failed_steps = 0

        for step in trajectory["steps"]:
            # Extract action details
            gpt_action = step.get("gpt_action", {})
            action_type = gpt_action.get("action_type", "unknown")
            action_params = gpt_action.get("action_parameters", {})
            action_desc = gpt_action.get("action_description", "")
            
            # Format action parameters for display
            params_str = ", ".join([f"{k}={v}" for k, v in action_params.items()])
            
            if step["success"] and step["observation"]:
                # Extract tool name from the step's action
                tool_name = step.get("gpt_action", {}).get("action_type", None)
                # Extract question from scenario for context-aware summarization
                question = scenario.get("question", {}).get("text", "") if scenario else ""
                obs_summary = self.summarize_observation(step["observation"], tool_name, question)
                logger.info(f"🔍 SUMMARIZATION DEBUG: Obs summary: {obs_summary}")
                
                # Save the summary in the step for trajectory persistence
                step["observation_summary"] = obs_summary
                
                previous_observations.append(f"✓ Step {step['step_index'] + 1}: {obs_summary}")
                previous_actions.append(f"✓ Step {step['step_index'] + 1}: {action_type}({params_str}) - {action_desc}")
                successful_steps += 1
            else:
                error_msg = step.get('error', 'Unknown error')
                previous_observations.append(
                    f"✗ Step {step['step_index'] + 1}: Failed - {error_msg}")
                previous_actions.append(f"✗ Step {step['step_index'] + 1}: {action_type}({params_str}) - Failed: {error_msg}")
                failed_steps += 1
        
        # Get dynamic tool descriptions based on enabled tools
        tool_descriptions = self.get_tool_descriptions()
        
        # Enhanced reasoning prompt with better stopping criteria
        system_prompt = f"""You are an expert research assistant using the ReAct (Reasoning and Acting) pattern. 

Your task is to:
1. Analyze the current research state and what has been learned
2. Reason about what information is still needed to answer the question
3. Decide on the next action to take (or conclude if sufficient)
4. Ensure action parameters are properly formatted

IMPORTANT: You have access to both text and image information. The image is provided with each request and contains visual information relevant to the research question. Consider both textual and visual information when making decisions.

AVOID REDUNDANCY: Do NOT use the same tool consecutively with the same or highly similar query/parameters. If a previous step already searched for something, use different search terms, try a different tool, or move on to a different aspect of the research question.

Available actions and their required parameters:
{tool_descriptions}

STOPPING CRITERIA - Set "should_stop": true if:
- You have sufficient information to answer the research question comprehensively
- You have analyzed both the image and gathered relevant textual information
- You have tried 3+ different approaches and have good coverage
- You have identified the key facts, dates, names, and context needed
- Further searches would be redundant

Respond with a JSON object containing:
{{
    "reasoning": "Your analysis of current state and what's needed next",
    "action": {{
        "action_type": "action_type",
        "action_description": "What this action will do",
        "action_parameters": {{"param": "value"}},
        "expected_outcome": "What you expect to learn"
    }},
    "should_stop": false,
    "confidence": 0.8
}}

Set "should_stop": true only if you have sufficient information to answer the research question comprehensively."""

        # Use placeholder URL to avoid data leakage from actual URL paths
        user_prompt = f"""Research Question: {scenario['question']['text']}
The image url is {IMAGE_URL_PLACEHOLDER}.

Current Research State (Iteration {iteration}):
- Successful steps: {successful_steps}
- Failed steps: {failed_steps}

Previous Actions Taken:
{chr(10).join(previous_actions) if previous_actions else "No previous actions"}

Previous Findings:
{chr(10).join(previous_observations) if previous_observations else "No previous research steps"}

Based on the research question, previous actions, and current findings, what should be the next step? 
- If you have enough information to answer the question, set should_stop: true
- If you need more information, choose the most appropriate action with proper parameters
- Ensure action_parameters are correctly formatted for the chosen action_type
- AVOID repeating the same tool with the same query - diversify your research approach"""

        if prompt_tool:
            user_prompt += "\n" + prompt_tool

        # Prepare messages with image if available
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Add image to the user message if available
        image_path = self._get_image_path_from_scenario(scenario)
        if image_path:
            base64_image = self.encode_image_to_base64(image_path)

            if base64_image:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                })
            else:
                messages.append({"role": "user", "content": user_prompt})
        else:
            messages.append({"role": "user", "content": user_prompt})

        try:
            # Use the unified model interface (supports GPT, vLLM, and Gemini)
            if self.model_type == "gpt":
                logger.info(f"=" * 80)
                logger.info(f"🔍 REACT_STEP MODEL: Using GPT for ReAct reasoning (iteration {iteration})")
                logger.info(f"🔍 REACT_STEP MODEL: GPT model = {self.gpt_model}")
                logger.info(f"=" * 80)
                # For GPT, use the OpenAI client directly to get token usage
                response = self.openai_client.chat.completions.create(
                    model=self.gpt_model,
                    messages=messages,
                    # max_completion_tokens=1000
                )
                content = response.choices[0].message.content
                
                # Store token usage information
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            elif self.model_type == "vllm":
                logger.info(f"=" * 80)
                logger.info(f"🔍 REACT_STEP MODEL: Using vLLM for ReAct reasoning (iteration {iteration})")
                logger.info(f"🔍 REACT_STEP MODEL: vLLM model path = {self.model_path}")
                logger.info(f"=" * 80)
                # For vLLM, use the unified interface
                content = self.generate_response_with_model(messages, max_tokens=1000, temperature=0.99)
                logger.info(f"🔍 REASONING DEBUG: vLLM response length: {len(content)} chars")
                
                # Estimate token usage for vLLM (since it doesn't return usage)
                token_usage = {
                    "prompt_tokens": 0,  # vLLM doesn't provide token usage
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            elif self.model_type == "gemini":
                logger.info(f"=" * 80)
                logger.info(f"🔍 REACT_STEP MODEL: Using Gemini for ReAct reasoning (iteration {iteration})")
                logger.info(f"🔍 REACT_STEP MODEL: Gemini model = {self.gemini_model}")
                logger.info(f"=" * 80)
                # For Gemini, use the unified interface
                content = self.generate_response_with_model(messages, max_tokens=1000, temperature=0.99)
                logger.info(f"🔍 REASONING DEBUG: Gemini response length: {len(content)} chars")
                
                # Estimate token usage for Gemini (since it doesn't return usage in the same format)
                token_usage = {
                    "prompt_tokens": 0,  # Gemini doesn't provide token usage
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            else:
                # Fallback to generic interface
                content = self.generate_response_with_model(messages, max_tokens=1000, temperature=0.99)
                token_usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }

            # Extract JSON from response with better error handling
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                reasoning_data = json.loads(json_match.group(0))

                # Validate action parameters
                if "action" in reasoning_data and "action_parameters" in reasoning_data["action"]:
                    action_type = reasoning_data["action"]["action_type"]
                    params = reasoning_data["action"]["action_parameters"]

                    # Ensure required parameters exist
                    if action_type == "web_search" and "query" not in params:
                        params["query"] = "general information"
                    elif action_type == "image_operations":
                        # Set default action if not specified
                        if "action" not in params:
                            params["action"] = "zoom_in"
                            params["bbox_2d"] = [0.0, 0.0, 1.0, 1.0]
                    elif action_type == "image_analysis" and "query" not in params:
                        params["query"] = "Analyze this image for relevant information"
                
                logger.info(f"ReAct Reasoning: {reasoning_data.get('reasoning', 'No reasoning provided')}")
                logger.info(f"Action: {reasoning_data.get('action', {}).get('action_type', 'unknown')}")
                logger.info(f"Should stop: {reasoning_data.get('should_stop', False)}")

                # Add token usage to the reasoning data
                reasoning_data["token_usage"] = token_usage
                
                # Save input messages and output content for trajectory
                # Save base64 image once at trajectory level if not already saved
                if base64_image and "cached_base64" not in trajectory:
                    trajectory["cached_base64"] = base64_image
                
                # Replace base64 in messages with placeholder for storage
                messages_to_save = []
                for msg in messages:
                    if isinstance(msg.get("content"), list):
                        # This is a multimodal message with image
                        content_items = []
                        for item in msg["content"]:
                            if item.get("type") == "image_url":
                                # Replace actual base64 with placeholder
                                content_items.append({
                                    "type": "image_url",
                                    "image_url": {"url": "<cached_base64>"}
                                })
                            else:
                                content_items.append(item)
                        messages_to_save.append({
                            "role": msg["role"],
                            "content": content_items
                        })
                    else:
                        messages_to_save.append(msg)
                
                reasoning_data["react_step_details"] = {
                    "input_messages": messages_to_save,
                    "output_content": content,
                    "step_type": "reasoning",
                    "iteration": iteration
                }

                return reasoning_data
            else:
                logger.error(f"Could not extract JSON from reasoning response: {content[:200]}...")
                return self._get_fallback_action(scenario, iteration)

        except Exception as e:
            logger.error(f"Error in ReAct reasoning: {e}")
            return self._get_fallback_action(scenario, iteration)

    def _get_fallback_action(self, scenario: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Get a fallback action when reasoning fails"""
        if iteration == 1:
            return {
                "reasoning": "Fallback: Starting with OCR to extract text from image",
                "action": {
                    "action_type": "image_operations",
                    "action_description": "Zoom into the image for detailed visual inspection",
                    "action_parameters": {"action": "zoom_in", "bbox_2d": [0.0, 0.0, 1.0, 1.0]},
                    "expected_outcome": "Get a closer view of the image for visual analysis"
                },
                "should_stop": False,
                "confidence": 0.5
            }
        else:
            return {
                "reasoning": "Fallback: Searching for general information",
                "action": {
                    "action_type": "web_search",
                    "action_description": "Search for general information about the topic",
                    "action_parameters": {"query": "general information"},
                    "expected_outcome": "Find relevant information"
                },
                "should_stop": False,
                "confidence": 0.3
            }

    def _react_action_step(self, action: Dict[str, Any], scenario: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """ReAct Action: Execute the decided action"""

        logger.info(f"ReAct Action: {action['action_description']}")

        # Validate action before conversion
        if not self._validate_action(action):
            logger.error(f"Invalid action format: {action}")
            return {
                "step_index": iteration - 1,
                "gpt_action": action,
                "tool_action": "Invalid action",
                "processing_time_ms": 0,
                "response": None,
                "success": False,
                "error": "Invalid action format",
                "observation": None,
                "valid": False,
                "done": False,
                "react_iteration": iteration,
                "react_reasoning": action.get("reasoning", "")
            }

        # Convert action to tool action
        tool_action = self.convert_gpt_action_to_tool_action(action, scenario)
        logger.warning(f"Generated tool action: '{tool_action}'")
        logger.warning(f"Action type: {action.get('action_type', 'unknown')}")
        logger.warning(f"Action parameters: {action.get('action_parameters', {})}")

        # Validate tool action format
        if not self._validate_tool_action(tool_action):
            logger.error(f"Invalid tool action format: {tool_action}")
            return {
                "step_index": iteration - 1,
                "gpt_action": action,
                "tool_action": tool_action,
                "processing_time_ms": 0,
                "response": None,
                "success": False,
                "error": "Invalid tool action format",
                "observation": None,
                "valid": False,
                "done": False,
                "react_iteration": iteration,
                "react_reasoning": action.get("reasoning", "")
            }

        # Execute the action
        step_result = self._execute_research_step(tool_action, action, scenario["id"], iteration - 1, scenario)
        
        # Add ReAct-specific metadata
        step_result["react_iteration"] = iteration
        step_result["react_reasoning"] = action.get("reasoning", "")

        return step_result

    def _validate_action(self, action: Dict[str, Any]) -> bool:
        """Validate that action has required fields"""
        required_fields = ["action_type", "action_description", "action_parameters"]
        return all(field in action for field in required_fields)

    def _validate_tool_action(self, tool_action: str) -> bool:
        """Validate that tool action is properly formatted"""
        if not tool_action or not isinstance(tool_action, str):
            logger.warning(f"Tool action validation failed: not a string or empty. Type: {type(tool_action)}, Value: {tool_action}")
            return False
        
        # Check for proper XML-like tags or tool_call format
        valid_formats = [
            tool_action.startswith("<text_search_text>") and tool_action.endswith("</text_search_text>"),
            tool_action.startswith("<text_search_image>") and tool_action.endswith("</text_search_image>"),
            tool_action.startswith("<image_search_text>") and tool_action.endswith("</image_search_text>"),
            tool_action.startswith("<web_read>") and tool_action.endswith("</web_read>"),
            tool_action.startswith("<ocr_tool>") and tool_action.endswith("</ocr_tool>"),
            tool_action.startswith("<python>") and tool_action.endswith("</python>"),
            tool_action.startswith("<bash>") and tool_action.endswith("</bash>"),
            tool_action.startswith("<tool_call>") and tool_action.endswith("</tool_call>")
        ]

        is_valid = any(valid_formats)
        
        if not is_valid:
            logger.warning(f"Tool action validation failed. Action: '{tool_action[:100]}...' (truncated)")
            logger.warning(f"Checked formats: text_search_text, text_search_image, image_search_text, web_read, ocr_tool, python, bash, tool_call")
            # Check what format it actually has
            if "<" in tool_action and ">" in tool_action:
                start_tag = tool_action.split(">")[0] + ">"
                end_part = tool_action.split("<")[-1] if tool_action.count("<") > 1 else ""
                end_tag = "<" + end_part if end_part else "No end tag"
                logger.warning(f"Detected start tag: '{start_tag}', end tag: '{end_tag}'")
        
        return is_valid
    
    def _execute_research_step(self, tool_action: str, gpt_action: Dict[str, Any], scenario_id: str, step_index: int, scenario: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a single research step and return results"""

        trajectory_id = f"{scenario_id}_step_{step_index}"
        
        # Prepare extra_fields with images if this is an image_operations action
        extra_field = {
            "gpt_action": gpt_action  # Pass GPT action to all tools
        }
        
        if scenario and gpt_action.get("action_type") == "image_operations":
            image_path = self._get_image_path_from_scenario(scenario)
            if image_path:
                extra_field["images"] = [image_path]
                # Pass the current working directory as base_path for resolving relative paths
                extra_field["base_path"] = str(Path.cwd())
        
        payload = {
            "trajectory_ids": [trajectory_id],
            "actions": [tool_action],
            "finish": [False],
            "extra_fields": [extra_field]
        }

        try:
            start_time = time.time()
            response = requests.post(self.server_url, json=payload)
            response.raise_for_status()

            result = response.json()
            processing_time = (time.time() - start_time) * 1000

            step_result = {
                "step_index": step_index,
                "gpt_action": gpt_action,
                "tool_action": tool_action,
                "processing_time_ms": processing_time,
                "response": result,
                "success": True,
                "error": None,
                # Enhanced: Save tool execution input/output
                "tool_execution_details": {
                    "input_payload": payload,
                    "output_response": result,
                    "step_type": "tool_execution",
                    "server_url": self.server_url,
                    "processing_time_ms": processing_time
                }
            }

            # Extract observation
            if "observations" in result and len(result["observations"]) > 0:
                observation = result["observations"][0]
                step_result["observation"] = observation
                step_result["valid"] = result.get("valids", [False])[0]
                step_result["done"] = result.get("dones", [False])[0]
            else:
                step_result["observation"] = None
                step_result["valid"] = False
                step_result["done"] = False

            logger.info(f"✓ Step {step_index + 1} completed in {processing_time:.1f}ms")

        except Exception as e:
            logger.error(f"✗ Step {step_index + 1} failed: {str(e)}")
            step_result = {
                "step_index": step_index,
                "gpt_action": gpt_action,
                "tool_action": tool_action,
                "processing_time_ms": 0,
                "response": None,
                "success": False,
                "error": str(e),
                "observation": None,
                "valid": False,
                "done": False,
                # Enhanced: Save tool execution details even for errors
                "tool_execution_details": {
                    "input_payload": payload,
                    "output_response": None,
                    "step_type": "tool_execution",
                    "server_url": self.server_url,
                    "processing_time_ms": 0,
                    "error": str(e)
                }
            }

        return step_result

    def _generate_final_answer(self, trajectory: Dict[str, Any], scenario: Dict[str, Any]) -> Tuple[
        str, Dict[str, int]]:
        """Generate a comprehensive final answer based on all research findings"""

        # Collect all observations from successful steps
        # Prefer observation_summary (already summarized) over raw observation
        all_observations = []
        for step in trajectory["steps"]:
            if step["success"] and step["observation"]:
                # Use observation_summary if available (more concise, already processed)
                if "observation_summary" in step and step["observation_summary"]:
                    all_observations.append(step["observation_summary"])
                elif isinstance(step["observation"], str):
                    all_observations.append(step["observation"])
                elif isinstance(step["observation"], dict) and "obs" in step["observation"]:
                    all_observations.append(step["observation"]["obs"])

        # Combine all research findings
        combined_research = "\n\n".join(all_observations)

        # Create prompt for final answer generation
        system_prompt = """You are an expert research assistant. Based on the research findings provided, provide a direct answer to the research question.

Your answer should:
1. Concise reasoning step by step
2. Put your final concise answer inside the \\boxed{}
3. Consider both textual research findings and visual information from the image

Just provide the answer without additional formatting."""

        # Use placeholder URL to avoid data leakage from actual URL paths
        user_prompt = f"""Research Question: {scenario['question']['text']}
Image URL in this question is {IMAGE_URL_PLACEHOLDER}.

Research Findings:
{combined_research}""" + """

Answer using both text and image info. Give concise step-by-step reasoning within <reasoning> and </reasoning>, then put the final answer in \\boxed{...}."""

        # Prepare messages with image if available
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Initialize base64_image to None (may be set below if image is available)
        base64_image = None

        # Add image to the user message if available
        image_source = self._get_image_path_from_scenario(scenario)
        if image_source:
            base64_image = self.encode_image_to_base64(image_source)

            if base64_image:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                })
            else:
                messages.append({"role": "user", "content": user_prompt})
        else:
            messages.append({"role": "user", "content": user_prompt})

        try:
            # Use the unified model interface (supports GPT, vLLM, and Gemini)
            if self.model_type == "gpt":
                logger.info(f"=" * 80)
                logger.info(f"🔍 FINAL_ANSWER MODEL: Using GPT for final answer generation")
                logger.info(f"🔍 FINAL_ANSWER MODEL: GPT model = {self.gpt_model}")
                logger.info(f"=" * 80)
                # For GPT, use the OpenAI client directly to get token usage
                response = self.openai_client.chat.completions.create(
                    model=self.gpt_model,
                    messages=messages,
                    # max_completion_tokens=1000
                )
                final_answer = response.choices[0].message.content
                
                # Store token usage for final answer generation
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            elif self.model_type == "vllm":
                logger.info(f"=" * 80)
                logger.info(f"🔍 FINAL_ANSWER MODEL: Using vLLM for final answer generation")
                logger.info(f"🔍 FINAL_ANSWER MODEL: vLLM model path = {self.model_path}")
                logger.info(f"=" * 80)
                
                # For vLLM, check if input exceeds max_model_len - 200 and trim if needed
                max_input_tokens = self.max_model_len - 200
                
                # Estimate token count for current messages
                def estimate_token_count(msgs):
                    # Extract text from messages and tokenize
                    text_parts = []
                    for msg in msgs:
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            text_parts.append(content)
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                    combined_text = " ".join(text_parts)
                    # Use tokenizer to count tokens
                    tokens = self.tokenizer.encode(combined_text)
                    return len(tokens)
                
                # Function to rebuild user_prompt and messages with current observations
                def rebuild_prompt_and_messages(obs_list):
                    research_text = "\n\n".join(obs_list)
                    prompt = f"""Research Question: {scenario['question']['text']}
Image URL in this question is {IMAGE_URL_PLACEHOLDER}.

Research Findings:
{research_text}""" + """

Answer using both text and image info. Give concise step-by-step reasoning within <reasoning> and </reasoning>, then put the final answer in \\boxed{...}."""
                    
                    msgs = [{"role": "system", "content": system_prompt}]
                    image_source_for_rebuild = self._get_image_path_from_scenario(scenario)
                    if image_source_for_rebuild:
                        if base64_image:
                            msgs.append({
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                    }
                                ]
                            })
                        else:
                            msgs.append({"role": "user", "content": prompt})
                    else:
                        msgs.append({"role": "user", "content": prompt})
                    return prompt, msgs
                
                # Start with all observations and check token count
                token_count = estimate_token_count(messages)
                logger.info(f"🔍 FINAL_ANSWER vLLM: Initial token count = {token_count}, max allowed = {max_input_tokens}")
                
                # Progressively trim observations if exceeding max length
                current_observations = all_observations.copy()
                while token_count > max_input_tokens and len(current_observations) > 1:
                    logger.warning(f"🔍 FINAL_ANSWER vLLM: Token count {token_count} exceeds limit {max_input_tokens}, trimming observations")
                    # Pop the last observation
                    current_observations = current_observations[:-1]
                    # Rebuild user_prompt and messages with trimmed observations
                    user_prompt, messages = rebuild_prompt_and_messages(current_observations)
                    token_count = estimate_token_count(messages)
                    logger.info(f"🔍 FINAL_ANSWER vLLM: After trim - {len(current_observations)} observations, {token_count} tokens")
                
                if token_count > max_input_tokens:
                    logger.warning(f"🔍 FINAL_ANSWER vLLM: Still exceeds limit after trimming all but one observation. Proceeding anyway.")
                
                # Update all_observations and combined_research to reflect trimmed state
                all_observations = current_observations
                combined_research = "\n\n".join(all_observations)
                
                # For vLLM, use the unified interface with potentially trimmed messages
                final_answer = self.generate_response_with_model(messages, max_tokens=2000, temperature=0.99)
                logger.info(f"🔍 FINAL_ANSWER DEBUG: vLLM response length: {len(final_answer)} chars")
                
                # Estimate token usage for vLLM (since it doesn't return usage)
                token_usage = {
                    "prompt_tokens": token_count,  # Use our estimated count
                    "completion_tokens": 0,
                    "total_tokens": token_count
                }
            elif self.model_type == "gemini":
                logger.info(f"=" * 80)
                logger.info(f"🔍 FINAL_ANSWER MODEL: Using Gemini for final answer generation")
                logger.info(f"🔍 FINAL_ANSWER MODEL: Gemini model = {self.gemini_model}")
                logger.info(f"=" * 80)
                # For Gemini, use the unified interface
                final_answer = self.generate_response_with_model(messages, max_tokens=1000, temperature=0.99)
                logger.info(f"🔍 FINAL_ANSWER DEBUG: Gemini response length: {len(final_answer)} chars")
                
                # Estimate token usage for Gemini (since it doesn't return usage in the same format)
                token_usage = {
                    "prompt_tokens": 0,  # Gemini doesn't provide token usage
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            else:
                # Fallback to generic interface
                final_answer = self.generate_response_with_model(messages, max_tokens=1000, temperature=0.99)
                token_usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }

            logger.info(f"Generated final answer for scenario {scenario['id']} (tokens: {token_usage['total_tokens']})")
            
            # Return final answer, token usage, and step details
            # Save base64 image once at trajectory level if not already saved
            if base64_image and "cached_base64" not in trajectory:
                trajectory["cached_base64"] = base64_image
            
            # Replace base64 in messages with placeholder for storage
            messages_to_save = []
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    # This is a multimodal message with image
                    content_items = []
                    for item in msg["content"]:
                        if item.get("type") == "image_url":
                            # Replace actual base64 with placeholder
                            content_items.append({
                                "type": "image_url",
                                "image_url": {"url": "<cached_base64>"}
                            })
                        else:
                            content_items.append(item)
                    messages_to_save.append({
                        "role": msg["role"],
                        "content": content_items
                    })
                else:
                    messages_to_save.append(msg)
            
            final_answer_details = {
                "input_messages": messages_to_save,
                "output_content": final_answer,
                "step_type": "final_answer",
                "token_usage": token_usage
            }
            
            return final_answer, token_usage, final_answer_details

        except Exception as e:
            logger.error(f"Error generating final answer for scenario {scenario['id']}: {e}")
            error_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            error_details = {
                "input_messages": messages,
                "output_content": f"Error generating final answer: {str(e)}",
                "step_type": "final_answer",
                "token_usage": error_tokens,
                "error": str(e)
            }
            return f"Error generating final answer: {str(e)}", error_tokens, error_details

    def evaluate_multimodal_research_quality(self, trajectory: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[
        str, Any]:
        """Evaluate the quality of multimodal research based on multiple metrics"""

        evaluation = {
            "scenario_id": scenario["id"],
            "metrics": {},
            "analysis": {}
        }

        # 1. Success Rate
        successful_steps = sum(1 for step in trajectory["steps"] if step["success"])
        total_steps = len(trajectory["steps"])
        evaluation["metrics"]["success_rate"] = successful_steps / total_steps if total_steps > 0 else 0

        # 2. Tool Usage Accuracy
        valid_steps = sum(1 for step in trajectory["steps"] if step["success"] and step["valid"])
        evaluation["metrics"]["tool_accuracy"] = valid_steps / total_steps if total_steps > 0 else 0

        # 3. Response Time Analysis
        processing_times = [step["processing_time_ms"] for step in trajectory["steps"] if step["success"]]
        if processing_times:
            evaluation["metrics"]["avg_processing_time_ms"] = sum(processing_times) / len(processing_times)
            evaluation["metrics"]["max_processing_time_ms"] = max(processing_times)
            evaluation["metrics"]["min_processing_time_ms"] = min(processing_times)
        else:
            evaluation["metrics"]["avg_processing_time_ms"] = 0
            evaluation["metrics"]["max_processing_time_ms"] = 0
            evaluation["metrics"]["min_processing_time_ms"] = 0

        # 4. Content Quality Analysis
        all_observations = []
        for step in trajectory["steps"]:
            if step["success"] and step["observation"]:
                obs_text = str(step["observation"])
                all_observations.append(obs_text)

        # Check for relevance to the research question
        combined_observations = " ".join(all_observations).lower()
        question_text = scenario["question"]["text"].lower()

        # Simple relevance scoring based on keyword overlap
        question_words = set(re.findall(r'\b\w+\b', question_text))
        observation_words = set(re.findall(r'\b\w+\b', combined_observations))

        relevant_words = question_words.intersection(observation_words)
        relevance_score = len(relevant_words) / len(question_words) if question_words else 0

        evaluation["metrics"]["relevance_score"] = relevance_score
        evaluation["analysis"]["relevant_words"] = list(relevant_words)
        evaluation["analysis"]["question_words"] = list(question_words)

        # 5. Action Diversity
        action_types = [step["gpt_action"].get("action_type", "unknown") for step in trajectory["steps"]]
        unique_action_types = set(action_types)
        evaluation["metrics"]["action_diversity"] = len(unique_action_types) / len(action_types) if action_types else 0
        evaluation["analysis"]["action_types_used"] = list(unique_action_types)

        # 6. ReAct-specific metrics
        evaluation["metrics"]["react_iterations"] = trajectory.get("react_iterations", 0)
        evaluation["metrics"]["efficiency"] = 1.0 / trajectory.get("react_iterations", 1) if trajectory.get(
            "react_iterations", 0) > 0 else 0

        # 7. Overall Quality Score (including ReAct efficiency)
        quality_score = (
                evaluation["metrics"]["success_rate"] * 0.25 +
                evaluation["metrics"]["tool_accuracy"] * 0.2 +
                evaluation["metrics"]["relevance_score"] * 0.25 +
                evaluation["metrics"]["action_diversity"] * 0.15 +
                evaluation["metrics"]["efficiency"] * 0.15
        )
        evaluation["metrics"]["overall_quality_score"] = quality_score

        return evaluation

    def save_individual_result(self, trajectory: Dict[str, Any], evaluation: Dict[str, Any]):
        """Save individual trajectory and evaluation immediately after completion"""
        
        # Save individual trajectory
        trajectory_file = self.test_dir / "trajectories" / f"trajectory_{trajectory['scenario_id']}.json"
        with open(trajectory_file, 'w') as f:
            json.dump(trajectory, f, indent=2, default=str)
        
        # Save individual evaluation
        evaluation_file = self.test_dir / "evaluations" / f"evaluation_{evaluation['scenario_id']}.json"
        with open(evaluation_file, 'w') as f:
            json.dump(evaluation, f, indent=2, default=str)
        
        # Save generated actions if available
        if "gpt_generated_actions" in trajectory:
            actions_file = self.test_dir / "generated_actions" / f"actions_{trajectory['scenario_id']}.json"
            with open(actions_file, 'w') as f:
                json.dump(trajectory["gpt_generated_actions"], f, indent=2, default=str)
        
        logger.info(f"✓ Saved results for scenario: {trajectory['scenario_id']}")

    def save_results(self, trajectories: List[Dict[str, Any]], evaluations: List[Dict[str, Any]]):
        """Save final summary and aggregated results (individual results already saved)"""

        # Calculate tool usage statistics across all scenarios
        tool_usage_stats = self._calculate_tool_usage_stats(trajectories)

        # Calculate total token usage for main deep research model (excluding image analysis)
        total_tokens = self._calculate_total_tokens(trajectories)

        # Save summary report
        summary = {
            "test_timestamp": datetime.now().isoformat(),
            "total_scenarios": len(trajectories),
            "runs_per_question": self.runs_per_question,
            "run_start_number": self.run_start_number,
            "unique_questions": len(trajectories) // self.runs_per_question if self.runs_per_question > 0 else len(trajectories),
            "gpt_model": self.gpt_model,
            "overall_metrics": self._calculate_overall_metrics(evaluations),
            "tool_usage_stats": tool_usage_stats,
            "total_tokens_processed": total_tokens,
            "scenario_summaries": [
                {
                    "scenario_id": eval_data["scenario_id"],
                    "quality_score": eval_data["metrics"]["overall_quality_score"],
                    "success_rate": eval_data["metrics"]["success_rate"],
                    "relevance_score": eval_data["metrics"]["relevance_score"],
                    "action_diversity": eval_data["metrics"]["action_diversity"],
                    "react_iterations": eval_data["metrics"]["react_iterations"],
                    "efficiency": eval_data["metrics"]["efficiency"],
                    "question": trajectories[i].get("question", "Not available"),
                    "final_answer": trajectories[i].get("final_answer", "Not generated"),
                    "ground_truth": trajectories[i].get("ground_truth", "Not available")
                }
                for i, eval_data in enumerate(evaluations)
            ]
        }

        summary_file = self.test_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save detailed log
        log_file = self.test_dir / "logs" / "test_execution.log"
        with open(log_file, 'w') as f:
            f.write(f"Multimodal Deep Research Test Execution Log\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Server URL: {self.server_url}\n")
            f.write(f"GPT Model: {self.gpt_model}\n")
            f.write(f"Total Scenarios: {len(trajectories)}\n\n")

            for i, (trajectory, evaluation) in enumerate(zip(trajectories, evaluations)):
                f.write(f"=== Scenario {i + 1}: {trajectory['scenario_id']} ===\n")
                f.write(f"Question: {trajectory['question']['text']}\n")
                f.write(f"Duration: {trajectory['duration']:.2f}s\n")
                f.write(f"Quality Score: {evaluation['metrics']['overall_quality_score']:.3f}\n")
                f.write(f"Success Rate: {evaluation['metrics']['success_rate']:.3f}\n")
                f.write(f"Relevance Score: {evaluation['metrics']['relevance_score']:.3f}\n")
                f.write(f"Action Diversity: {evaluation['metrics']['action_diversity']:.3f}\n")
                f.write(f"ReAct Iterations: {evaluation['metrics']['react_iterations']}\n")
                f.write(f"Efficiency: {evaluation['metrics']['efficiency']:.3f}\n")
                f.write(f"Action Types Used: {evaluation['analysis']['action_types_used']}\n")

                # Add final answer if available
                if "final_answer" in trajectory:
                    f.write(f"\n--- Final Answer ---\n")
                    f.write(f"{trajectory['final_answer']}\n")

                f.write(f"\n")

        logger.debug(f"Results saved to: {self.test_dir}")
        logger.debug(f"Summary: {summary_file}")
        logger.debug(f"Log: {log_file}")

    def _calculate_overall_metrics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall metrics across all scenarios"""

        if not evaluations:
            return {}

        metrics = ["success_rate", "tool_accuracy", "relevance_score", "action_diversity", "react_iterations",
                   "efficiency", "overall_quality_score"]
        overall = {}

        for metric in metrics:
            values = [eval_data["metrics"][metric] for eval_data in evaluations]
            overall[f"avg_{metric}"] = sum(values) / len(values)
            overall[f"max_{metric}"] = max(values)
            overall[f"min_{metric}"] = min(values)

        return overall

    def _calculate_tool_usage_stats(self, trajectories: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate total tool usage statistics across all scenarios"""
        tool_counts = {}

        for trajectory in trajectories:
            for step in trajectory.get("steps", []):
                if step.get("success", False):
                    action_type = step.get("gpt_action", {}).get("action_type", "unknown")
                    tool_counts[action_type] = tool_counts.get(action_type, 0) + 1

        return tool_counts

    def _calculate_total_tokens(self, trajectories: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate total token usage for main deep research model (excluding image analysis)"""
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0

        for trajectory in trajectories:
            # Count tokens from ReAct reasoning steps
            for step in trajectory.get("steps", []):
                if step.get("success", False):
                    # Check if this is from the main research model (not image analysis)
                    action_type = step.get("gpt_action", {}).get("action_type", "")
                    
                    # Skip image operations and image analysis steps as they use a separate GPT vision API call
                    if action_type in ["image_operations", "image_analysis"]:
                        continue

                    # Extract token usage from response if available
                    response = step.get("response", {})
                    if isinstance(response, dict):
                        # Look for token usage in the response metadata
                        if "usage" in response:
                            usage = response["usage"]
                            total_input_tokens += usage.get("prompt_tokens", 0)
                            total_output_tokens += usage.get("completion_tokens", 0)
                            total_tokens += usage.get("total_tokens", 0)
                        elif "tokens" in response:
                            tokens = response["tokens"]
                            total_input_tokens += tokens.get("input", 0)
                            total_output_tokens += tokens.get("output", 0)
                            total_tokens += tokens.get("total", 0)

            # Count tokens from final answer generation (stored in trajectory metadata)
            if "final_answer_tokens" in trajectory:
                tokens = trajectory["final_answer_tokens"]
                total_input_tokens += tokens.get("prompt_tokens", 0)
                total_output_tokens += tokens.get("completion_tokens", 0)
                total_tokens += tokens.get("total_tokens", 0)

        return {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens
        }

    def run_multimodal_test(self, max_scenarios: Optional[int] = None, file_name: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive multimodal deep research test"""

        logger.info("=== Starting Multimodal Deep Research Test ===")

        # Use instance variable if file_name not provided
        if file_name is None:
            file_name = self.file_name
        
        # Load all scenarios first (we'll slice based on run_start_number and max_scenarios)
        all_scenarios = self.load_multimodal_scenarios(file_name=file_name, max_scenarios=None)
        logger.info(f"Loaded {len(all_scenarios)} total scenarios from {file_name}")

        if not all_scenarios:
            logger.error("No scenarios loaded. Check input files.")
            return {"error": "No scenarios loaded"}

        # Apply run_start_number and max_scenarios slicing
        start_index = self.run_start_number - 1  # Convert to 1-based to 0-based index
        
        if start_index >= len(all_scenarios):
            logger.error(f"run_start_number={self.run_start_number} is beyond the total scenarios ({len(all_scenarios)})")
            return {"error": f"run_start_number={self.run_start_number} exceeds total scenarios"}
        
        # Calculate end index based on max_scenarios
        if max_scenarios is not None:
            end_index = min(start_index + max_scenarios, len(all_scenarios))
        else:
            end_index = len(all_scenarios)
        
        # Slice scenarios
        scenarios = all_scenarios[start_index:end_index]
        
        # Log what we're processing
        if start_index > 0 or max_scenarios is not None:
            logger.info(f"Processing scenarios {self.run_start_number} to {start_index + len(scenarios)} ({len(scenarios)} scenarios)")
            if start_index > 0:
                logger.info(f"  Skipped first {start_index} scenarios")
            if max_scenarios is not None:
                logger.info(f"  Limited to {max_scenarios} scenarios")
        else:
            logger.info(f"Processing all {len(scenarios)} scenarios")

        # Execute all scenarios with multiple runs per question
        trajectories = []
        evaluations = []

        for i, scenario in enumerate(scenarios):
            # Calculate global scenario number (accounting for start_index)
            global_scenario_num = (self.run_start_number - 1) + i + 1 if self.run_start_number > 1 else i + 1
            logger.info(f"\n--- Executing Scenario {global_scenario_num} (local {i + 1}/{len(scenarios)}): {scenario['id']} ---")
            
            # Run each scenario multiple times
            for run_index in range(self.runs_per_question):
                if self.runs_per_question > 1:
                    logger.info(f"  Run {run_index + 1}/{self.runs_per_question}")
                
                # Create a unique scenario ID for this run
                run_scenario = scenario.copy()
                if self.runs_per_question > 1:
                    run_scenario['id'] = f"{scenario['id']}_run_{run_index + 1}"
                
                # Simulate research trajectory
                trajectory = self.simulate_multimodal_research_trajectory(run_scenario)
                trajectories.append(trajectory)

                # Evaluate research quality
                evaluation = self.evaluate_multimodal_research_quality(trajectory, run_scenario)
                evaluations.append(evaluation)

                # Save results immediately after each scenario completion
                self.save_individual_result(trajectory, evaluation)

                run_label = f"Run {run_index + 1}" if self.runs_per_question > 1 else "Result"
                logger.info(f"  {run_label} - Quality Score: {evaluation['metrics']['overall_quality_score']:.3f}")
                logger.info(f"  {run_label} - Success Rate: {evaluation['metrics']['success_rate']:.3f}")
                logger.info(f"  {run_label} - Relevance Score: {evaluation['metrics']['relevance_score']:.3f}")
                logger.info(f"  {run_label} - Action Diversity: {evaluation['metrics']['action_diversity']:.3f}")
                
                # Add a small delay between runs to avoid rate limiting
                if run_index < self.runs_per_question - 1:
                    time.sleep(1)

        # Save final summary and aggregated results (individual results already saved)
        self.save_results(trajectories, evaluations)

        # Calculate and return overall results
        overall_metrics = self._calculate_overall_metrics(evaluations)

        logger.debug("\n=== Test Complete ===")
        logger.debug(f"Overall Quality Score: {overall_metrics.get('avg_overall_quality_score', 0):.3f}")
        logger.debug(f"Average Success Rate: {overall_metrics.get('avg_success_rate', 0):.3f}")
        logger.debug(f"Average Relevance Score: {overall_metrics.get('avg_relevance_score', 0):.3f}")
        logger.debug(f"Average Action Diversity: {overall_metrics.get('avg_action_diversity', 0):.3f}")

        return {
            "trajectories": trajectories,
            "evaluations": evaluations,
            "overall_metrics": overall_metrics,
            "test_dir": str(self.test_dir)
        }

    def test_image_inclusion(self, scenario_id: str = "MMSearch-Plus_0") -> bool:
        """Test that images are properly included in reasoning and final answer steps"""
        try:
            # Load a test scenario
            scenarios = self.load_multimodal_scenarios(file_name=self.file_name)
            test_scenario = None
            for scenario in scenarios:
                if scenario["id"] == scenario_id:
                    test_scenario = scenario
                    break

            if not test_scenario:
                logger.error(f"Test scenario {scenario_id} not found")
                return False

            logger.info(f"Testing image inclusion with scenario: {scenario_id}")
            logger.info(f"Question: {test_scenario['question']['text']}")
            logger.info(f"Image: {test_scenario['question'].get('image', 'No image')}")

            # Test reasoning step with image
            trajectory = {
                "scenario_id": test_scenario["id"],
                "question": test_scenario["question"],
                "ground_truth": test_scenario["ground_truth"],
                "steps": [],
                "start_time": time.time(),
                "total_actions": 0,
                "react_iterations": 0
            }

            # Test reasoning step
            reasoning_result = self._react_reasoning_step(trajectory, test_scenario, 1)

            # Check if reasoning result contains image-related reasoning
            reasoning_text = reasoning_result.get("reasoning", "").lower()
            action_type = reasoning_result.get("action", {}).get("action_type", "")

            logger.info(f"Reasoning result: {reasoning_result.get('reasoning', '')[:200]}...")
            logger.info(f"Action type: {action_type}")

            # Test final answer generation
            trajectory["steps"] = [{
                "step_index": 0,
                "success": True,
                "observation": "Test observation for image inclusion test"
            }]

            final_answer, tokens, final_answer_details = self._generate_final_answer(trajectory, test_scenario)

            logger.info(f"Final answer generated: {len(final_answer)} characters")
            logger.info(f"Token usage: {tokens}")

            # Check if the system is working
            success = (
                    reasoning_result is not None and
                    "reasoning" in reasoning_result and
                    final_answer is not None and
                    len(final_answer) > 0
            )

            if success:
                logger.info("✓ Image inclusion test passed")
            else:
                logger.error("✗ Image inclusion test failed")

            return success

        except Exception as e:
            logger.error(f"Error in image inclusion test: {e}")
            return False


# Factory function for backward compatibility
def create_multimodal_tester(model_type: str = "gpt", **kwargs) -> MultimodalDeepResearchTesterBase:
    """
    Factory function to create a MultimodalDeepResearchTester with the specified model type.
    
    Args:
        model_type: "gpt", "vllm", or "gemini"
        **kwargs: Additional arguments passed to the constructor
        
    Returns:
        MultimodalDeepResearchTesterBase instance
    """
    if model_type == "gpt":
        # Extract GPT-specific config
        model_config = {
            "gpt_model": kwargs.pop("gpt_model", "gpt-4o"),
            "openai_api_key": kwargs.pop("openai_api_key", None)
        }
        return MultimodalDeepResearchTesterBase(model_type="gpt", model_config=model_config, **kwargs)
    
    elif model_type == "vllm":
        # Extract vLLM-specific config
        model_config = {
            "model_path": kwargs.pop("model_path", "/export/home/becky/Qwen3-VL/qwen-vl-finetune/output/2e5_ep3/"),
            "max_model_len": kwargs.pop("max_model_len", 10000),
            "tensor_parallel_size": kwargs.pop("tensor_parallel_size", 1),
            "gpu_id": kwargs.pop("gpu_id", None)
        }
        return MultimodalDeepResearchTesterBase(model_type="vllm", model_config=model_config, **kwargs)
    
    elif model_type == "gemini":
        # Extract Gemini-specific config
        model_config = {
            "gemini_model": kwargs.pop("gemini_model", "gemini-3-pro-preview"),
            "gemini_project": kwargs.pop("gemini_project", "salesforce-research-internal"),
            "gemini_location": kwargs.pop("gemini_location", "global"),
            "gemini_temperature": kwargs.pop("gemini_temperature", 1.0),
            "gemini_top_p": kwargs.pop("gemini_top_p", 0.95)
        }
        return MultimodalDeepResearchTesterBase(model_type="gemini", model_config=model_config, **kwargs)
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'gpt', 'vllm', or 'gemini'")

# Backward compatibility alias
class MultimodalDeepResearchTester(MultimodalDeepResearchTesterBase):
    """Backward compatibility wrapper - use create_multimodal_tester() instead"""
    
    def __init__(self, 
                 server_url: str = "http://localhost:4000/get_observation", 
                 output_dir: str = "simple_test_results_mm", 
                 input_dir: str = "data/mmsearch_plus_processed",
                 gpt_model: str = "gpt-4o",
                 openai_api_key: Optional[str] = None,
                 enabled_tools: Optional[List[str]] = None,
                 difficulty_filter: Optional[str] = "",
                 image_url_prefix: Optional[str] = None,
                 prompt_tool: Optional[str] = None,
                 runs_per_question: int = 1,
                 run_start_number: int = 1,
                 file_name: Optional[str] = None):
        
        # Use factory function to create GPT-based tester
        model_config = {
            "gpt_model": gpt_model,
            "openai_api_key": openai_api_key
        }
        
        # Call parent constructor with GPT configuration
        super().__init__(server_url=server_url, output_dir=output_dir, input_dir=input_dir,
                        model_type="gpt", model_config=model_config, enabled_tools=enabled_tools, 
                        difficulty_filter=difficulty_filter, image_url_prefix=image_url_prefix, 
                        prompt_tool=prompt_tool, runs_per_question=runs_per_question, 
                        run_start_number=run_start_number, file_name=file_name)


def main():
    """Main entry point for multimodal deep research testing

    Usage:
        # Using GPT (default):
        python test_mm_deep_research.py run --server_url=http://127.0.0.2:8002/get_observation
        python test_mm_deep_research.py run --output_dir=my_test_results_mm
        python test_mm_deep_research.py run --input_dir=data/mmsearch_plus_processed
        python test_mm_deep_research.py run --gpt_model=gpt-4o
        python test_mm_deep_research.py run --max_scenarios=5
        python test_mm_deep_research.py run --difficulty_filter=easy
        python test_mm_deep_research.py run --enabled_tools="web_text_to_text_search,web_text_to_img_search,web_url_reader,web_image_to_text,ipython_code"
        python test_mm_deep_research.py run --image_url_prefix="https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/MM-BrowseComp/images/"
        python test_mm_deep_research.py run --runs_per_question=3
        
        # Using Gemini (NEW):
        python test_mm_deep_research.py run_gemini --server_url=http://localhost:4000/get_observation
        python test_mm_deep_research.py run_gemini --gemini_model=gemini-3-pro-preview
        python test_mm_deep_research.py run_gemini --gemini_project=salesforce-research-internal
        python test_mm_deep_research.py run_gemini --gemini_temperature=1.2 --gemini_top_p=0.95
        python test_mm_deep_research.py run_gemini --max_scenarios=5
        
        # Start from scenario 300 and process 5 scenarios (300-304):
        python test_mm_deep_research.py run --run_start_number=300 --max_scenarios=5
        
        # Process scenarios 100-199 (100 scenarios starting from index 100):
        python test_mm_deep_research.py run --run_start_number=100 --max_scenarios=100
        
        python test_mm_deep_research.py run --file_name=custom_scenarios.json
        python test_mm_deep_research.py test_image --scenario_id=MMSearch-Plus_0
    """
    def run_test(server_url="http://localhost:4000/get_observation", 
                 output_dir="simple_test_results_mm", 
                 input_dir="data/mmsearch_plus_processed",
                 gpt_model="gpt-4o",
                 openai_api_key=None,
                 max_scenarios=3,
                 enabled_tools=None,
                 difficulty_filter="",
                 image_url_prefix=None,
                 prompt_tool=None,
                 runs_per_question=1,
                 run_start_number=1,
                 file_name="qa_formatted.json"):
        # Ensure numeric parameters are properly typed (Fire may pass them as strings)
        max_scenarios = int(max_scenarios) if max_scenarios is not None else None
        runs_per_question = int(runs_per_question)
        run_start_number = int(run_start_number)
        
        # Log the received parameters for debugging
        logger.info(f"run_test called with: max_scenarios={max_scenarios}, runs_per_question={runs_per_question}, run_start_number={run_start_number}")
        
        # Parse enabled_tools if provided as comma-separated string
        if enabled_tools and isinstance(enabled_tools, str):
            enabled_tools = [t.strip() for t in enabled_tools.split(',')]
        
        tester = MultimodalDeepResearchTester(
            server_url=server_url,
            output_dir=output_dir,
            input_dir=input_dir,
            gpt_model=gpt_model,
            openai_api_key=openai_api_key,
            enabled_tools=enabled_tools,
            difficulty_filter=difficulty_filter,
            image_url_prefix=image_url_prefix,
            prompt_tool=prompt_tool,
            runs_per_question=runs_per_question,
            run_start_number=run_start_number,
            file_name=file_name
        )
        result = tester.run_multimodal_test(max_scenarios=max_scenarios)
        
        # Don't return the result to prevent Fire from printing it
        print("Test completed successfully. Results saved to output directory.")
        return None
    
    def run_gemini(server_url="http://localhost:4000/get_observation", 
                   output_dir="simple_test_results_mm_gemini", 
                   input_dir="data/mmsearch_plus_processed",
                   gemini_model="gemini-3-pro-preview",
                   gemini_project="salesforce-research-internal",
                   gemini_location="global",
                   gemini_temperature=1.0,
                   gemini_top_p=0.95,
                   max_scenarios=3,
                   enabled_tools=None,
                   difficulty_filter="",
                   image_url_prefix=None,
                   prompt_tool=None,
                   runs_per_question=1,
                   run_start_number=1,
                   file_name="qa_formatted.json"):
        """Run test with Gemini model"""
        # Ensure numeric parameters are properly typed (Fire may pass them as strings)
        max_scenarios = int(max_scenarios) if max_scenarios is not None else None
        runs_per_question = int(runs_per_question)
        run_start_number = int(run_start_number)
        gemini_temperature = float(gemini_temperature)
        gemini_top_p = float(gemini_top_p)
        
        # Log the received parameters for debugging
        logger.info(f"run_gemini called with: gemini_model={gemini_model}, temperature={gemini_temperature}, top_p={gemini_top_p}, max_scenarios={max_scenarios}")
        
        # Parse enabled_tools if provided as comma-separated string
        if enabled_tools and isinstance(enabled_tools, str):
            enabled_tools = [t.strip() for t in enabled_tools.split(',')]
        
        # Create Gemini-based tester using factory function
        tester = create_multimodal_tester(
            model_type="gemini",
            gemini_model=gemini_model,
            gemini_project=gemini_project,
            gemini_location=gemini_location,
            gemini_temperature=gemini_temperature,
            gemini_top_p=gemini_top_p,
            server_url=server_url,
            output_dir=output_dir,
            input_dir=input_dir,
            enabled_tools=enabled_tools,
            difficulty_filter=difficulty_filter,
            image_url_prefix=image_url_prefix,
            prompt_tool=prompt_tool,
            runs_per_question=runs_per_question,
            run_start_number=run_start_number,
            file_name=file_name
        )
        result = tester.run_multimodal_test(max_scenarios=max_scenarios)
        
        # Don't return the result to prevent Fire from printing it
        print("Test completed successfully with Gemini. Results saved to output directory.")
        return None
    
    def test_image(server_url="http://localhost:4000/get_observation", 
                   output_dir="simple_test_results_mm", 
                   input_dir="data/mmsearch_plus_processed",
                   gpt_model="gpt-4o",
                   openai_api_key=None,
                   scenario_id="MMSearch-Plus_0",
                   enabled_tools=None,
                   difficulty_filter="",
                   file_name="qa_formatted.json"):
        # Parse enabled_tools if provided as comma-separated string
        if enabled_tools and isinstance(enabled_tools, str):
            enabled_tools = [t.strip() for t in enabled_tools.split(',')]
        
        tester = MultimodalDeepResearchTester(
            server_url=server_url,
            output_dir=output_dir,
            input_dir=input_dir,
            gpt_model=gpt_model,
            openai_api_key=openai_api_key,
            enabled_tools=enabled_tools,
            difficulty_filter=difficulty_filter,
            file_name=file_name
        )
        return tester.test_image_inclusion(scenario_id)
    
    fire.Fire({
        "run": run_test,
        "run_gemini": run_gemini,
        "test_image": test_image
    })


if __name__ == "__main__":
    main()