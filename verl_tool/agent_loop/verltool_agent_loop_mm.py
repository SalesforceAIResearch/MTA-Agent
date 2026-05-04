# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multimodal Agent Loop with GPT-4 Summarization and ReAct Reasoning.

This extends VerlToolAgentLoop with:
1. GPT-4 summarization of long observations
2. ReAct reasoning pattern support
3. Enhanced trajectory logging
"""
import logging
import os
import re
import json
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio
from uuid import uuid4

from openai import OpenAI

from .verltool_agent_loop import VerlToolAgentLoop, AgentActorConfig, MAX_OBS_LENGTH
from .agent_loop import register, AgentLoopOutput
from .vision_utils import decode_image_url, encode_image_url
from verl_tool.servers.tools.base import strip_result_tags
from verl_tool.utils.dataset.audio_utils import encode_audio_data
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Image URL placeholder constant
IMAGE_URL_PLACEHOLDER = "http://image.png"

# Load environment variables from .env file if it exists
# Try multiple common locations for .env file
try:
    from dotenv import load_dotenv
    
    # Try multiple potential .env file locations
    potential_env_paths = [
        Path(__file__).parent.parent.parent / '.env',  # verl-tool-lastest/.env
        Path(__file__).parent.parent.parent.parent / '.env',  # verl-tool-all/.env
        Path.cwd() / '.env',  # Current working directory
        Path.home() / '.env',  # Home directory
    ]
    
    env_loaded = False
    for env_path in potential_env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)  # Don't override existing env vars
            logger.info(f"Loaded environment variables from {env_path}")
            env_loaded = True
            break
    
    # If no .env file found in specific paths, try default load_dotenv() behavior
    # which searches current directory and parent directories
    if not env_loaded:
        result = load_dotenv(override=False)
        if result:
            logger.info("Loaded environment variables from .env file (auto-detected location)")
        else:
            logger.debug("No .env file found in common locations or current directory")
            
except ImportError:
    logger.warning("python-dotenv not installed. .env file loading is disabled. Install with: pip install python-dotenv")
except Exception as e:
    logger.warning(f"Failed to load .env file: {e}")

# Tool registry with descriptions and formats
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
        "description": "Extract and read the full raw text content from any web page URL. Returns the complete text content of the webpage, including articles, blog posts, documentation, and other text-based content. This tool is not for image URLs.",
        "parameters": {"url": "URL to extract content from, not for image URLs"},
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


@register("verltool_agent_mm")
class VerlToolAgentLoopMM(VerlToolAgentLoop):
    """
    Multimodal Agent Loop with GPT-4 Summarization and ReAct Reasoning.
    
    Extends VerlToolAgentLoop with:
    1. GPT-4 summarization of long observations (tool-specific strategies)
    2. ReAct reasoning pattern support
    3. Enhanced trajectory logging
    4. Question extraction and caching for context-aware summarization
    
    Configuration (via agent config):
    - gpt_summarization_model: str = "gpt-4.1" - GPT model for summarization
    - gpt_summarization_api_key: Optional[str] = None - API key (or use env vars)
    - enabled_tools: Optional[List[str]] = None - List of enabled tools
    - use_react_reasoning: bool = True - Enable ReAct reasoning
    - use_gpt_summarization: bool = True - Enable GPT-4 summarization (enabled by default)
    """
    
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        """
        Initialize class-level attributes with consistent chat template handling.
        
        This override ensures that:
        1. All observations (text, image, audio) get consistent role tags
        2. enable_mtrl is forced to True for consistent formatting
        3. mtrl_sep pattern is properly set up if not already configured
        """
        # Call parent init_class first to set up base configuration
        super().init_class(config, tokenizer, processor, **kwargs)
        
        # Force enable_mtrl to True for consistent role tagging across all observation types
        # This ensures text-only, image, and audio observations all get proper role tags
        cls.enable_mtrl = True
        cls.agent_config.enable_mtrl = True
        
        # Ensure mtrl_sep is properly configured if it wasn't already
        if cls.agent_config.mtrl_sep is None:
            messages = [{"role": "system", "content": "{obs}"}]
            cls.agent_config.mtrl_sep = cls.agent_config.turn_end_token + "\n" + cls.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            cls.agent_config.mtrl_sep = cls.agent_config.mtrl_sep.replace("system", cls.agent_config.mtrl_role)
            cls.mtrl_sep = cls.agent_config.mtrl_sep
        
        # Always ensure prefix/suffix IDs are set (even if mtrl_sep was pre-configured)
        # This is needed for consistent role tagging in _execute_action_and_collect_observation
        if not hasattr(cls, 'mtrl_sep_prefix_ids') or cls.mtrl_sep_prefix_ids is None:
            if '{obs}' in cls.agent_config.mtrl_sep:
                cls.mtrl_sep_prefix, cls.mtrl_sep_suffix = cls.agent_config.mtrl_sep.split("{obs}")
            else:
                # Fallback: assume the entire mtrl_sep is the prefix (no suffix)
                cls.mtrl_sep_prefix = cls.agent_config.mtrl_sep
                cls.mtrl_sep_suffix = ""
            cls.mtrl_sep_prefix_ids = cls.tokenizer.encode(cls.mtrl_sep_prefix)
            cls.mtrl_sep_suffix_ids = cls.tokenizer.encode(cls.mtrl_sep_suffix)
        
        # Create a suffix without assistant tag for observations
        # Observations are followed by another user message (react prompt), not an assistant response
        # So we should end with just <|im_end|> instead of <|im_end|>\n<|im_start|>assistant\n
        cls.mtrl_sep_suffix_no_assistant = "<|im_end|>\n"
        cls.mtrl_sep_suffix_no_assistant_ids = cls.tokenizer.encode(cls.mtrl_sep_suffix_no_assistant)
        
        logger.info(f"VerlToolAgentLoopMM.init_class: enable_mtrl={cls.enable_mtrl}, mtrl_role={cls.agent_config.mtrl_role}")
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get MM-specific config from agent config
        agent_config = self.agent_config
        self.use_react_reasoning = getattr(agent_config, 'use_react_reasoning', True)
        self.use_gpt_summarization = getattr(agent_config, 'use_gpt_summarization', True)  # Enabled by default
        self.gpt_summarization_model = getattr(agent_config, 'gpt_summarization_model', "gpt-4.1")
        gpt_summarization_api_key = getattr(agent_config, 'gpt_summarization_api_key', None)
        enabled_tools = getattr(agent_config, 'enabled_tools', None)
        
        # Configure enabled tools (default to all tools)
        if enabled_tools is None:
            self.enabled_tools = list(TOOL_REGISTRY.keys())
        else:
            self.enabled_tools = enabled_tools
        
        # Validate enabled tools
        invalid_tools = set(self.enabled_tools) - set(TOOL_REGISTRY.keys())
        if invalid_tools:
            logger.warning(f"Invalid tools specified: {invalid_tools}. Available tools: {list(TOOL_REGISTRY.keys())}")
            self.enabled_tools = [t for t in self.enabled_tools if t in TOOL_REGISTRY.keys()]
        
        # Initialize OpenAI client for GPT-4 summarization
        self.gpt_client = None
        if self.use_gpt_summarization:
            try:
                # Check for API key in order: parameter > X_API_KEY from env > OPENAI_API_KEY from env
                x_api_key = os.getenv("X_API_KEY")
                openai_api_key = os.getenv("OPENAI_API_KEY")
                
                # Debug logging to verify env vars are loaded
                logger.debug(f"GPT summarization config: use_gpt_summarization={self.use_gpt_summarization}")
                logger.debug(f"API key sources: gpt_summarization_api_key={'set' if gpt_summarization_api_key else 'None'}, "
                           f"X_API_KEY={'set' if x_api_key else 'None'}, "
                           f"OPENAI_API_KEY={'set' if openai_api_key else 'None'}")
                
                api_key = gpt_summarization_api_key or x_api_key or openai_api_key
                
                if api_key:
                    # Prefer X_API_KEY for the gateway, fallback to OPENAI_API_KEY
                    headers = {}
                    if x_api_key:
                        headers = {"X-Api-Key": x_api_key}
                    elif openai_api_key:
                        pass  # For standard OpenAI, use the api_key parameter
                    
                    self.gpt_client = OpenAI(
                        base_url="https://gateway.salesforceresearch.ai/openai/process/v1/",
                        api_key="dummy" if x_api_key else (openai_api_key or "dummy"),
                        default_headers=headers
                    )
                    logger.info(f"Initialized GPT client for summarization with model: {self.gpt_summarization_model}")
                else:
                    logger.warning("No API key found for GPT summarization. Summarization will be disabled.")
                    logger.warning("Set X_API_KEY or OPENAI_API_KEY in .env file or environment variables.")
                    self.use_gpt_summarization = False
            except Exception as e:
                logger.error(f"Failed to initialize GPT client: {e}")
                self.use_gpt_summarization = False
        
        # Store trajectory history for ReAct reasoning
        self.trajectory_history = {}  # key: traj_id, value: list of (action, observation, success) tuples
        self.trajectory_questions = {}  # key: traj_id, value: question string
        self.reasoning_trajectory = {}  # key: traj_id, value: list of step dicts
        self.trajectory_observations = {}  # key: traj_id, value: list of all observations (for final answer synthesis)
        self.trajectory_image_urls = {}  # key: traj_id, value: list of image URLs (for placeholder replacement)
        
        # Track if we're in a validation step (to save all trajectories during validation)
        self.is_validation_step = False
        
        # Track rollout trajectory counts per step (to limit to top 10)
        self.rollout_trajectory_counts = {}  # key: step_idx (str), value: count (int)
        
        logger.info(f"VerlToolAgentLoopMM initialized with ReAct reasoning: {self.use_react_reasoning}, GPT summarization: {self.use_gpt_summarization}")
        logger.info(f"Enabled tools: {', '.join(self.enabled_tools)}")
    
    def _log_full_trajectory_summary(self, traj_id: str, prompt_ids: List[int], stats_dict: dict, 
                                     response_ids: List[int] = None, response_mask: List[int] = None,
                                     original_prompt_ids: List[int] = None, is_validation: bool = False,
                                     global_step: Optional[int] = None):
        """
        Save the full trajectory text and metadata to a JSON file.
        Also saves the training data with masked tokens replaced by |mask|.
        Only saves trajectories during validation steps (saves both validation and rollout trajectories).
        No printing/logging during training - all data is saved to JSON files.
        
        Args:
            traj_id: Unique trajectory ID
            prompt_ids: Full prompt IDs (entire trajectory)
            stats_dict: Statistics dictionary
            response_ids: Response token IDs
            response_mask: Response mask
            original_prompt_ids: Original prompt IDs (before trajectory)
            is_validation: Whether this is a validation trajectory
            global_step: Global training step
        """
        # Only save trajectories during validation steps (saves both validation and rollout trajectories)
        if not self.is_validation_step:
            return
        
        try:
            # Decode the full prompt to see the entire conversation
            full_prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            
            # Prepare metadata
            metadata = {
                "traj_id": traj_id,
                "total_turns": stats_dict.get('num_turns', 0),
                "valid_actions": stats_dict.get('valid_action', 0),
                "is_finished": stats_dict.get('is_traj_finished', False),
                "total_action_tokens": sum(stats_dict.get('action_lengths', [])),
                "total_observation_tokens": sum(stats_dict.get('obs_lengths', [])),
                "is_validation": is_validation,
                "global_step": global_step
            }
            
            # Prepare trajectory data
            trajectory_data = {
                "traj_id": traj_id,
                "full_trajectory_text": full_prompt_text
            }
            
            # Prepare training data with masked tokens
            training_data = {
                "traj_id": traj_id,
                "has_response_data": response_ids is not None and response_mask is not None
            }
            
            if response_ids is not None and response_mask is not None:
                # Use original_prompt_ids if provided, otherwise use prompt_ids
                prompt_for_training = original_prompt_ids if original_prompt_ids is not None else prompt_ids
                training_data_text = self._reconstruct_training_data(
                    prompt_for_training, response_ids, response_mask
                )
                training_data.update({
                    "training_data_text": training_data_text,
                    "prompt_length": len(prompt_for_training),
                    "response_length": len(response_ids),
                    "response_mask_length": len(response_mask),
                    "masked_tokens_count": sum(1 for m in response_mask if m == 0),
                    "unmasked_tokens_count": sum(1 for m in response_mask if m == 1)
                })
            else:
                training_data["training_data_text"] = None
                training_data["error"] = "response_ids or response_mask is None"
            
            # Determine output directory and extract step index
            # Try to use rollout_data_dir from environment first
            rollout_dir = os.getenv("ROLLOUT_DATA_DIR")
            step_idx = global_step  # Start with global_step from kwargs
            
            # Debug logging
            if step_idx is None:
                logger.debug(f"global_step is None, will try to extract from environment variables or paths")
            
            if rollout_dir:
                # If rollout dir is set, save trajectories in the same run directory
                trajectories_dir = Path(rollout_dir).parent / "trajectories"
                # Try to extract step from rollout dir path (e.g., .../rollout/20.jsonl -> 20)
                if step_idx is None:
                    rollout_path = Path(rollout_dir)
                    if rollout_path.is_file():
                        # Extract step from filename (e.g., "20.jsonl" -> 20)
                        try:
                            step_idx = int(rollout_path.stem)
                        except ValueError:
                            pass
                    elif rollout_path.is_dir():
                        # If it's a directory, try to get step from parent or filename
                        parent_name = rollout_path.name
                        try:
                            step_idx = int(parent_name)
                        except ValueError:
                            pass
            else:
                # Try to infer from validation_data_dir
                validation_dir = os.getenv("VALIDATION_DATA_DIR")
                if validation_dir:
                    trajectories_dir = Path(validation_dir).parent / "trajectories"
                    # Try to extract step from validation dir path
                    if step_idx is None:
                        validation_path = Path(validation_dir)
                        if validation_path.is_file():
                            try:
                                step_idx = int(validation_path.stem)
                            except ValueError:
                                pass
                        elif validation_path.is_dir():
                            parent_name = validation_path.name
                            try:
                                step_idx = int(parent_name)
                            except ValueError:
                                pass
                else:
                    # Try to find verl_step_records in current directory or parent
                    current_dir = Path.cwd()
                    verl_records = None
                    # Check current dir and parents for verl_step_records
                    for parent in [current_dir] + list(current_dir.parents):
                        potential_records = parent / "verl_step_records"
                        if potential_records.exists() and potential_records.is_dir():
                            verl_records = potential_records
                            break
                    
                    if verl_records:
                        # Find the most recent run directory
                        run_dirs = sorted(verl_records.glob("*/rollout"), key=lambda p: p.stat().st_mtime, reverse=True)
                        if run_dirs:
                            trajectories_dir = run_dirs[0].parent / "trajectories"
                        else:
                            trajectories_dir = verl_records / "trajectories"
                    else:
                        # Fallback to default
                        output_base = os.getenv("VERL_STEP_RECORDS_DIR", "verl_step_records")
                        trajectories_dir = Path(output_base) / "trajectories"
            
            # Organize by step index, then by validation/rollout
            # Use extracted step_idx, or try to extract from existing trajectory directory structure
            if step_idx is None:
                # Try to extract step from existing trajectory directory structure
                # Look for numeric subdirectories in trajectories_dir
                if trajectories_dir.exists():
                    numeric_dirs = []
                    for subdir in trajectories_dir.iterdir():
                        if subdir.is_dir():
                            try:
                                step_num = int(subdir.name)
                                numeric_dirs.append((step_num, subdir.stat().st_mtime))
                            except ValueError:
                                pass
                    if numeric_dirs:
                        # Use the most recently modified numeric directory
                        numeric_dirs.sort(key=lambda x: x[1], reverse=True)
                        step_idx = numeric_dirs[0][0]
                        logger.info(f"Extracted step_idx={step_idx} from existing trajectory directory structure")
                    else:
                        step_idx = "unknown"
                        logger.warning(f"Could not determine step_idx, using 'unknown'. global_step={global_step}, rollout_dir={rollout_dir}, validation_dir={os.getenv('VALIDATION_DATA_DIR')}")
                else:
                    step_idx = "unknown"
                    logger.warning(f"Could not determine step_idx, using 'unknown'. trajectories_dir does not exist: {trajectories_dir}")
            else:
                logger.debug(f"Using step_idx={step_idx} from global_step parameter")
            step_dir = trajectories_dir / str(step_idx)
            
            # Create subdirectory for validation or rollout under the step directory
            if is_validation:
                trajectories_subdir = step_dir / "validation"
                # Always save all validation trajectories
                logger.debug(f"Saving validation trajectory {traj_id} for step {step_idx}")
            else:
                trajectories_subdir = step_dir / "rollout"
                
                # Limit rollout trajectories to top 10 per step
                step_idx_str = str(step_idx)
                if step_idx_str not in self.rollout_trajectory_counts:
                    self.rollout_trajectory_counts[step_idx_str] = 0
                
                # Check if we've already saved 10 rollout trajectories for this step
                if self.rollout_trajectory_counts[step_idx_str] >= 10:
                    logger.debug(f"Skipping rollout trajectory {traj_id} - already saved 10 rollout trajectories for step {step_idx}")
                    return
                
                # Increment counter
                self.rollout_trajectory_counts[step_idx_str] += 1
                logger.debug(f"Saving rollout trajectory {traj_id} ({self.rollout_trajectory_counts[step_idx_str]}/10) for step {step_idx}")
            
            trajectories_subdir.mkdir(parents=True, exist_ok=True)
            
            # Save to JSON file
            output_file = trajectories_subdir / f"{traj_id}.json"
            output_data = {
                "metadata": metadata,
                "trajectory": trajectory_data,
                "training_data": training_data
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Failed to save full trajectory summary for traj_id={traj_id}: {e}")
    
    def _reconstruct_training_data(self, prompt_ids: List[int], response_ids: List[int], 
                                   response_mask: List[int]) -> str:
        """
        Reconstruct the training data by replacing masked tokens (observations/synthesis prompt) 
        with |mask| token.
        
        Args:
            prompt_ids: Original prompt (system + user) - always shown
            response_ids: Response tokens (think/actions + observations + final answer)
            response_mask: Mask for response_ids (1 = train on, 0 = mask out)
            
        Returns:
            Decoded text with masked parts replaced by |mask|
        """
        # Decode the prompt (system + user) - this is always shown
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        
        # Ensure response_mask matches response_ids length
        if len(response_mask) < len(response_ids):
            # If mask is shorter, assume remaining tokens are masked
            response_mask = response_mask + [0] * (len(response_ids) - len(response_mask))
        elif len(response_mask) > len(response_ids):
            # If mask is longer, truncate it
            response_mask = response_mask[:len(response_ids)]
        
        # After padding/truncation, response_mask should match response_ids length
        assert len(response_mask) == len(response_ids), f"Mask length {len(response_mask)} != response_ids length {len(response_ids)}"
        
        # Reconstruct response with masked tokens replaced
        # Replace each masked token with |mask| (one mask per token)
        masked_response_parts = []
        for i in range(len(response_ids)):
            if response_mask[i] == 0:
                # This is a masked token (observation or synthesis prompt)
                # Replace with |mask| (one per token)
                masked_response_parts.append("<|mask|>")
            else:
                # This is a token to train on (think/action/final answer)
                # Decode this single token
                token_text = self.tokenizer.decode([response_ids[i]], skip_special_tokens=False)
                masked_response_parts.append(token_text)
        
        # Handle any remaining tokens if response_ids is longer
        if i < len(response_ids):
            masked_response_parts.append("<|mask|>")
        
        # Combine prompt and masked response
        response_text = "".join(masked_response_parts)
        training_data = prompt_text + response_text
        
        return training_data
    
    def get_tool_descriptions(self) -> str:
        """Generate tool descriptions for enabled tools."""
        descriptions = []
        
        for tool_name in self.enabled_tools:
            if tool_name not in TOOL_REGISTRY:
                continue
                
            tool_info = TOOL_REGISTRY[tool_name]
            
            if tool_name == "finish":
                descriptions.append(f"- {tool_info['description']}")
            else:
                params_str = json.dumps(tool_info['parameters'])
                descriptions.append(f"- {tool_info['format']}: {params_str} - {tool_info['description']}")
        
        return "\n".join(descriptions)
    
    def _extract_question_from_prompt(self, prompt_ids: List[int]) -> Optional[str]:
        """Extract research question from prompt token ids, including multiple-choice options if present."""
        try:
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            
            # Extract question from "Research Question:" format - include everything after it
            if 'Research Question:' in prompt_text:
                parts = prompt_text.split('Research Question:')
                if len(parts) > 1:
                    # Extract everything after "Research Question:" including Options if present
                    question = parts[1].strip()
                    # Stop at the next major section if present (e.g., "The image url is", "Based on the research question")
                    # Handle both single and double newlines before "The image url is"
                    if '\nThe image url is' in question:
                        question = question.split('\nThe image url is')[0].strip()
                    elif '\n\nThe image url is' in question:
                        question = question.split('\n\nThe image url is')[0].strip()
                    elif '\n\nBased on the research question' in question:
                        question = question.split('\n\nBased on the research question')[0].strip()
                    elif '\nBased on the research question' in question:
                        question = question.split('\nBased on the research question')[0].strip()
                    return question
            
            # Extract question from "Question:" format - include everything after it
            elif 'Question:' in prompt_text:
                parts = prompt_text.split('Question:')
                if len(parts) > 1:
                    question = parts[1].strip()
                    # Stop at the next major section if present
                    if '\nThe image url is' in question:
                        question = question.split('\nThe image url is')[0].strip()
                    elif '\n\nThe image url is' in question:
                        question = question.split('\n\nThe image url is')[0].strip()
                    elif '\n\nBased on the research question' in question:
                        question = question.split('\n\nBased on the research question')[0].strip()
                    elif '\nBased on the research question' in question:
                        question = question.split('\nBased on the research question')[0].strip()
                    return question
            
            # Extract question from "Based on the provided image" format - include everything including Options
            elif "Based on the provided image" in prompt_text or "Based on the image" in prompt_text:
                # Match everything after "Based on the provided image, " or "Based on the image, "
                # until we hit "The image url is" or "Based on the research question" or end of string
                question_match = re.search(
                    r'Based on (?:the provided image|the image)[,\s]+(.+?)(?:\n(?:The image url is|Based on the research question)|$)',
                    prompt_text,
                    re.IGNORECASE | re.DOTALL
                )
                if question_match:
                    question = question_match.group(1).strip()
                    return question
        except Exception as e:
            logger.debug(f"Failed to extract question from prompt: {e}")
        return None
    
    def _strip_tool_call_wrapper(self, action: str) -> str:
        """
        Strip <tool_call> wrapper from action if present.
        
        Handles cases where model outputs:
        - <tool_call><image_search_text>xxx</image_search_text></tool_call>
        - <tool_call><image_search_text>xxx</image_search_text>
        - <image_search_text>xxx</image_search_text> (no wrapper)
        - Leading/trailing whitespace around wrapper
        
        Returns the action with wrapper removed.
        """
        if not action:
            return action
        
        # Remove opening <tool_call> tag (handles leading whitespace and variations)
        # Matches: <tool_call>, <tool_call >, with optional leading whitespace
        action = re.sub(r'^\s*<tool_call\s*>\s*', '', action, flags=re.IGNORECASE)
        
        # Remove closing </tool_call> tag (handles trailing whitespace and variations)
        # Matches: </tool_call>, </tool_call >, with optional trailing whitespace
        action = re.sub(r'\s*</tool_call\s*>\s*$', '', action, flags=re.IGNORECASE)
        
        return action.strip()
    
    def _parse_json_response(self, gen_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON response from model that contains reasoning, action, should_stop, and confidence.
        
        Args:
            gen_text: Generated text that should contain JSON
            
        Returns:
            Parsed JSON dict or None if parsing fails
        """
        try:
            # Try to extract JSON from the response
            # JSON might be wrapped in markdown code blocks or just plain JSON
            json_match = re.search(r'\{.*\}', gen_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed
        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug(f"Failed to parse JSON from response: {e}. Text: {gen_text[:200]}...")
        return None
    
    def _convert_json_action_to_tool_call(self, action_dict: Dict[str, Any], extra_fields: Any = None) -> str:
        """
        Convert JSON action format to tool call format.
        
        Args:
            action_dict: Dict with action_type and action_parameters
            extra_fields: Extra fields that may contain image URLs for replacement
            
        Returns:
            Tool call string like <text_search_text>query</text_search_text>, or empty string for stop actions
        """
        action_type = action_dict.get("action_type", "")
        action_params = action_dict.get("action_parameters", {})
        
        # Handle stop-related action types (these should be handled by _parse_model_response, not converted to tool calls)
        if action_type.lower() in ["should_stop", "none", "finish", "stop"]:
            return ""  # Return empty string for stop actions
        
        # Normalize action_type: handle cases where model outputs tool tag names instead of action types
        action_type_aliases = {
            "image_search_text": "reverse_image_search",  # Model might output tag name instead of action type
            "text_search_text": "web_search",
            "text_search_image": "image_search",
            "web_read": "content_extraction",
            "ocr_tool": "ocr",
            "python": "code_execution",
            "bash": "bash_execution",
        }
        if action_type in action_type_aliases:
            logger.debug(f"Normalizing action_type '{action_type}' to '{action_type_aliases[action_type]}'")
            action_type = action_type_aliases[action_type]
        
        # Map action types to tool call formats
        if action_type == "web_search":
            query = action_params.get("query", "")
            return f"<text_search_text>{query}</text_search_text>"
        elif action_type == "image_search":
            query = action_params.get("query", "")
            return f"<text_search_image>{query}</text_search_image>"
        elif action_type == "content_extraction":
            url = action_params.get("url", "")
            # Replace placeholder if present
            if url == "http://image.png" and extra_fields:
                if isinstance(extra_fields, dict):
                    image_urls = extra_fields.get('image_urls', [])
                    if isinstance(image_urls, list) and len(image_urls) > 0:
                        url = image_urls[0]
            return f"<web_read>{url}</web_read>"
        elif action_type == "reverse_image_search":
            image_url = action_params.get("image_url", "")
            query = action_params.get("query", "")
            # Replace placeholder if present
            if image_url == "http://image.png" and extra_fields:
                if isinstance(extra_fields, dict):
                    image_urls = extra_fields.get('image_urls', [])
                    if isinstance(image_urls, list) and len(image_urls) > 0:
                        image_url = image_urls[0]
            if query:
                return f"<image_search_text>{image_url}||{query}</image_search_text>"
            else:
                return f"<image_search_text>{image_url}</image_search_text>"
        elif action_type == "ocr":
            image_url = action_params.get("image_url", "")
            # Replace placeholder if present
            if image_url == "http://image.png" and extra_fields:
                if isinstance(extra_fields, dict):
                    image_urls = extra_fields.get('image_urls', [])
                    if isinstance(image_urls, list) and len(image_urls) > 0:
                        image_url = image_urls[0]
            return f"<ocr_tool>{image_url}</ocr_tool>"
        elif action_type == "code_execution":
            code = action_params.get("code", "")
            return f"<python>\n{code}\n</python>"
        elif action_type == "bash_execution":
            command = action_params.get("command", "")
            return f"<bash>\n{command}\n</bash>"
        else:
            logger.warning(f"Unknown action_type: {action_type}")
            return ""
    
    def _extract_tool_name_from_action(self, action: str) -> Optional[str]:
        """Extract tool name from action string."""
        # Strip tool_call wrapper if present
        action = self._strip_tool_call_wrapper(action)
        
        if '<text_search_text>' in action:
            return 'web_search'
        elif '<text_search_image>' in action:
            return 'image_search'
        elif '<web_read>' in action:
            return 'content_extraction'
        elif '<image_search_text>' in action:
            return 'reverse_image_search'
        elif '<ocr_tool>' in action:
            return 'ocr'
        elif '<python>' in action:
            return 'code_execution'
        elif '<bash>' in action:
            return 'bash_terminal'
        return None
    
    def _extract_query_from_action(self, action: str) -> Optional[str]:
        """Extract query from action string."""
        import re
        # Strip tool_call wrapper if present
        action = self._strip_tool_call_wrapper(action)
        
        # For text_search_text: <text_search_text>query</text_search_text>
        text_search_match = re.search(r'<text_search_text>(.*?)</text_search_text>', action, re.DOTALL)
        if text_search_match:
            return text_search_match.group(1).strip()
        
        # For image_search_text: <image_search_text>URL||query</image_search_text>
        image_search_match = re.search(r'<image_search_text>(.*?)</image_search_text>', action, re.DOTALL)
        if image_search_match:
            content = image_search_match.group(1).strip()
            if "||" in content:
                parts = content.split("||", 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        return None
    
    def _extract_url_from_action(self, action: str) -> Optional[str]:
        """Extract URL from action string for content_extraction."""
        import re
        # Strip tool_call wrapper if present
        action = self._strip_tool_call_wrapper(action)
        
        # For web_read: <web_read>url</web_read>
        web_read_match = re.search(r'<web_read>(.*?)</web_read>', action, re.DOTALL)
        if web_read_match:
            return web_read_match.group(1).strip()
        
        return None
    
    def _format_observation(self, observation: str, tool_name: str = None, action: str = None) -> str:
        """
        Format observation with tool-specific format (without GPT-4 summarization).
        This is used for observations that don't need summarization but should still be formatted.
        
        Args:
            observation: The observation text to format
            tool_name: The name of the tool that generated this observation
            action: The action string to extract query/URL from
            
        Returns:
            A formatted version of the observation
        """
        obs_str = str(observation)
        
        # If no tool_name, try to extract from action
        if not tool_name and action:
            tool_name = self._extract_tool_name_from_action(action)
        
        if tool_name == "web_search":
            query = self._extract_query_from_action(action) if action else None
            if query:
                return f"Tool: web_search\nQuery: {query}\nResponse: {obs_str}"
            else:
                return f"Tool: web_search\nResponse: {obs_str}"
        
        elif tool_name == "image_search":
            query = self._extract_query_from_action(action) if action else None
            if query:
                return f"Tool: image_search\nQuery: {query}\nResponse: {obs_str}"
            else:
                return f"Tool: image_search\nResponse: {obs_str}"
        
        elif tool_name == "content_extraction":
            url = self._extract_url_from_action(action) if action else None
            if url:
                return f"Tool: content_extraction\nURL: {url}\nResponse: {obs_str}"
            else:
                return f"Tool: content_extraction\nResponse: {obs_str}"
        
        elif tool_name == "reverse_image_search":
            query = self._extract_query_from_action(action) if action else None
            if query:
                return f"Tool: reverse_image_search\nQuery: {query}\nResponse: {obs_str}"
            else:
                return f"Tool: reverse_image_search\nResponse: {obs_str}"
        
        elif tool_name == "code_execution":
            return f"Tool: code_execution\nCode: [executed code]\nOutput: {obs_str}"
        
        elif tool_name == "ocr":
            return f"Tool: ocr_tool\n\nResponse: {obs_str}"
        
        else:
            # Generic fallback - return as-is if tool_name is unknown or None
            return obs_str
    
    async def summarize_observation_with_gpt4(
        self, 
        observation: str, 
        tool_name: str = None, 
        question: str = None
    ) -> str:
        """
        Summarize a long observation text using GPT-4.
        
        Args:
            observation: The full observation text to summarize
            tool_name: The name of the tool that generated this observation
            question: The research question being answered (for context-aware summarization)
            
        Returns:
            A summarized version of the observation
        """
        obs_str = str(observation)
        obs_len = len(obs_str)
        
        if not self.use_gpt_summarization or not self.gpt_client:
            # Fallback: return truncated observation (limit to 1000 chars)
            logger.info(f"Summarization SKIPPED: use_gpt_summarization={self.use_gpt_summarization}, gpt_client={'available' if self.gpt_client else 'None'}. Returning truncated observation (len={min(obs_len, 1000)})")
            return obs_str[:1000] if obs_len > 1000 else obs_str
        
        logger.info(f"Summarization ATTEMPTING: tool_name={tool_name}, obs_len={obs_len}, has_question={question is not None}")
        
        # Tool-specific summarization strategies
        if tool_name == "web_search":
            if question:
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
                    response = self.gpt_client.chat.completions.create(
                        model=self.gpt_summarization_model,
                        messages=messages,
                        max_tokens=800,
                        temperature=0.1
                    )
                    summarized = response.choices[0].message.content
                    logger.info(f"Summarization SUCCESS: tool_name={tool_name}, original_len={obs_len}, summarized_len={len(summarized)}, reduction={obs_len - len(summarized)} chars")
                    return summarized
                except Exception as e:
                    logger.error(f"GPT-4 summarization FAILED: tool_name={tool_name}, error={e}. Returning truncated observation")
                    return obs_str[:1000]
            else:
                logger.info(f"Summarization SKIPPED: tool_name={tool_name}, no question provided. Returning truncated observation")
                return obs_str[:1000]
        
        elif tool_name == "image_search":
            if question:
                try:
                    summarization_prompt = f"""You are summarizing an image search observation. Your goal is to capture ALL potentially useful information, not to filter based on perceived relevance.

Question asked: {question}

Tool used: Image Search

Instructions:
1. Start with: "Tool: image_search"
2. Then state: "Query: [the search query used]"
3. Then summarize: "Response: [comprehensive summary of what was found]"
4. For EACH image result, include:
   - Image title/description
   - Key visual information
   - Source URL if available
5. Keep the summary under 1,000 characters, but prioritize completeness over brevity.

Fetched information: {obs_str}

Output format (only the following three components):
Tool: image_search
Query: [search query]
Response: [comprehensive summary with all images and their content]"""

                    messages = [{"role": "user", "content": summarization_prompt}]
                    response = self.gpt_client.chat.completions.create(
                        model=self.gpt_summarization_model,
                        messages=messages,
                        max_tokens=800,
                        temperature=0.1
                    )
                    summarized = response.choices[0].message.content
                    logger.info(f"Summarization SUCCESS: tool_name={tool_name}, original_len={obs_len}, summarized_len={len(summarized)}, reduction={obs_len - len(summarized)} chars")
                    return summarized
                except Exception as e:
                    logger.error(f"GPT-4 summarization FAILED: tool_name={tool_name}, error={e}. Returning truncated observation")
                    return obs_str[:1000]
            else:
                logger.info(f"Summarization SKIPPED: tool_name={tool_name}, no question provided. Returning truncated observation")
                return obs_str[:1000]
        
        elif tool_name == "content_extraction":
            if question:
                try:
                    summarization_prompt = f"""You are summarizing webpage content extraction. Your goal is to capture ALL potentially useful information, not to filter based on perceived relevance.

Question asked: {question}

Tool used: Content Extraction (Web Page Reader)

Instructions:
1. Start with: "Tool: content_extraction"
2. Then state: "URL: [the URL that was read]"
3. Then summarize: "Response: [comprehensive summary of the page content]"
4. Include:
   - Main topic/title of the page
   - Key facts, dates, names mentioned
   - Any relevant details that could help answer the question
5. Keep the summary under 1,000 characters, but prioritize completeness over brevity.

Fetched information: {obs_str}

Output format (only the following three components):
Tool: content_extraction
URL: [webpage URL]
Response: [comprehensive summary of the page content]"""

                    messages = [{"role": "user", "content": summarization_prompt}]
                    response = self.gpt_client.chat.completions.create(
                        model=self.gpt_summarization_model,
                        messages=messages,
                        max_tokens=800,
                        temperature=0.1
                    )
                    summarized = response.choices[0].message.content
                    logger.info(f"Summarization SUCCESS: tool_name={tool_name}, original_len={obs_len}, summarized_len={len(summarized)}, reduction={obs_len - len(summarized)} chars")
                    return summarized
                except Exception as e:
                    logger.error(f"GPT-4 summarization FAILED: tool_name={tool_name}, error={e}. Returning truncated observation")
                    return obs_str[:1000]
            else:
                logger.info(f"Summarization SKIPPED: tool_name={tool_name}, no question provided. Returning truncated observation")
                return obs_str[:1000]
        
        elif tool_name == "reverse_image_search":
            if question:
                try:
                    summarization_prompt = f"""You are summarizing a Google Lens image search observation. Your goal is to capture ALL potentially useful information, not to filter based on perceived relevance.

Question asked: {question}

Tool used: Reverse Image Search (Google Lens)

Instructions:
1. Start with: "Tool: reverse_image_search"
2. Then summarize: "Response: [comprehensive summary of what was found]"
3. Include:
   - Visual matches found
   - Text extracted from the image
   - Related images and their sources
   - Any identified objects, landmarks, or text
4. Keep the summary under 1,000 characters, but prioritize completeness over brevity.

Fetched information: {obs_str}

Output format (only the following two components):
Tool: reverse_image_search
Query: search query if provided, otherwise omit this line
Response: [comprehensive summary of the reverse image search results]"""

                    messages = [{"role": "user", "content": summarization_prompt}]
                    response = self.gpt_client.chat.completions.create(
                        model=self.gpt_summarization_model,
                        messages=messages,
                        max_tokens=800,
                        temperature=0.1
                    )
                    summarized = response.choices[0].message.content
                    logger.info(f"Summarization SUCCESS: tool_name={tool_name}, original_len={obs_len}, summarized_len={len(summarized)}, reduction={obs_len - len(summarized)} chars")
                    return summarized
                except Exception as e:
                    logger.error(f"GPT-4 summarization FAILED: tool_name={tool_name}, error={e}. Returning truncated observation")
                    return obs_str[:1000]
            else:
                logger.info(f"Summarization SKIPPED: tool_name={tool_name}, no question provided. Returning truncated observation")
                return obs_str[:1000]
        
        elif tool_name == "code_execution":
            if question:
                try:
                    summarization_prompt = f"""You are summarizing a Python code execution observation. Your goal is to clearly show BOTH the code that was executed AND the output it produced.

Question asked: {question}

Tool used: Python Code Execution

Instructions:
1. Start with: "Tool: code_execution"
2. Then show: "Code: [the Python code that was executed]"
3. Then show: "Output: [the output/result of the code execution]"
4. If there were errors, include the error message
5. Keep the summary concise but complete.

Fetched information: {obs_str}

Output format (only the following three components):
Tool: code_execution
Code: [executed code]
Output: [execution result or error]"""

                    messages = [{"role": "user", "content": summarization_prompt}]
                    response = self.gpt_client.chat.completions.create(
                        model=self.gpt_summarization_model,
                        messages=messages,
                        max_tokens=500,
                        temperature=0.1
                    )
                    summarized = response.choices[0].message.content
                    logger.info(f"Summarization SUCCESS: tool_name={tool_name}, original_len={obs_len}, summarized_len={len(summarized)}, reduction={obs_len - len(summarized)} chars")
                    return summarized
                except Exception as e:
                    logger.error(f"GPT-4 summarization FAILED: tool_name={tool_name}, error={e}. Returning truncated observation")
                    return f"Tool: code_execution\n{obs_str[:800]}"
            else:
                logger.info(f"Summarization SKIPPED: tool_name={tool_name}, no question provided. Returning truncated observation")
                return f"Tool: code_execution\n{obs_str[:800]}"
        
        elif tool_name == "ocr":
            # If short enough, return as-is
            if len(obs_str) <= 800:
                logger.info(f"Summarization SKIPPED: tool_name={tool_name}, observation already short (len={obs_len} <= 800). Returning as-is")
                return f"Tool: ocr_tool\n\nResponse: {obs_str}"
            
            # Use GPT-4.1 for OCR summarization when observation is too long
            if question:
                try:
                    summarization_prompt = f"""You are summarizing OCR (text extraction) results from an image. Your goal is to list ALL text found concisely.

Question asked: {question}

Tool used: OCR (Optical Character Recognition)

Instructions:
1. Start with: "Tool: ocr_tool"
2. Then list: "Extracted text: [all text found in the image]"
3. Preserve the structure if there are multiple text blocks
4. Keep the summary under 800 characters.

Fetched information: {obs_str}

Output format (only the following two components):
Tool: ocr_tool
Extracted text: [all text found]"""

                    messages = [{"role": "user", "content": summarization_prompt}]
                    response = self.gpt_client.chat.completions.create(
                        model=self.gpt_summarization_model,
                        messages=messages,
                        max_tokens=500,
                        temperature=0.1
                    )
                    summarized = response.choices[0].message.content
                    logger.info(f"Summarization SUCCESS: tool_name={tool_name}, original_len={obs_len}, summarized_len={len(summarized)}, reduction={obs_len - len(summarized)} chars")
                    return summarized
                except Exception as e:
                    logger.error(f"GPT-4 OCR summarization FAILED: tool_name={tool_name}, error={e}. Returning truncated observation")
                    return f"Tool: ocr_tool\n\nOutput: {obs_str[:800]}"
            else:
                logger.info(f"Summarization SKIPPED: tool_name={tool_name}, no question provided. Returning truncated observation")
                return f"Tool: ocr_tool\n\nOutput: {obs_str[:800]}"
        
        else:
            # Generic fallback
            logger.info(f"Summarization SKIPPED: tool_name={tool_name} (unknown tool type). Returning truncated observation (len={min(obs_len, 1000)})")
            return obs_str[:1000]
    
    def _replace_image_url_placeholder(self, action: str, traj_id: str = None, extra_fields: Any = None) -> str:
        """
        Replace "http://image.png" placeholder with real image URL in action string.
        
        Args:
            action: Action string that may contain "http://image.png" placeholder
            traj_id: Trajectory ID to look up stored image URLs
            extra_fields: Extra fields that may contain image_urls
            
        Returns:
            Action string with placeholder replaced by real URL
        """
        IMAGE_URL_PLACEHOLDER = "http://image.png"
        
        # If action doesn't contain placeholder, return as-is
        if IMAGE_URL_PLACEHOLDER not in action:
            return action
        
        # Try to get real image URL from multiple sources
        real_url = None
        
        # First, try trajectory_image_urls (stored from dataset)
        if traj_id and traj_id in self.trajectory_image_urls:
            image_urls = self.trajectory_image_urls[traj_id]
            if isinstance(image_urls, list) and len(image_urls) > 0:
                real_url = image_urls[0]
        
        # Second, try extra_fields
        if not real_url and extra_fields:
            if isinstance(extra_fields, dict):
                image_urls = extra_fields.get('image_urls', [])
                if isinstance(image_urls, list) and len(image_urls) > 0:
                    real_url = image_urls[0]
                # Also check for image_url (singular)
                if not real_url:
                    real_url = extra_fields.get('image_url')
        
        # If we have a real URL, replace the placeholder
        if real_url:
            action = action.replace(IMAGE_URL_PLACEHOLDER, real_url)
            logger.debug(f"Replaced image URL placeholder with real URL: {real_url[:50]}...")
        else:
            logger.warning(f"Image URL placeholder found in action but no real URL available. traj_id={traj_id}, Action: {action[:100]}...")
        
        return action
    
    async def interact_with_tool_server(
        self,
        traj_id: str,
        action: str,
        do_action: bool,
        extra_fields: Any = None,
        is_last_step: bool = False,
    ) -> Dict[str, Any]:
        """
        Interact with tool server and optionally summarize observations with GPT-4.
        
        Overrides parent method to add GPT-4 summarization capability.
        Also strips <tool_call> wrapper if present to ensure proper parsing.
        Also replaces "http://image.png" placeholder with real image URL.
        """
        # Strip tool_call wrapper if present before sending to tool server
        cleaned_action = self._strip_tool_call_wrapper(action)
        
        # Replace image URL placeholder with real URL before sending to tool server
        cleaned_action = self._replace_image_url_placeholder(cleaned_action, traj_id, extra_fields)
        
        # Call parent method to get tool interaction info
        tool_interact_info = await super().interact_with_tool_server(
            traj_id=traj_id,
            action=cleaned_action,
            do_action=do_action,
            extra_fields=extra_fields,
            is_last_step=is_last_step,
        )
        
        # Associate pending question with this traj_id if available
        if hasattr(self, '_pending_question') and self._pending_question:
            self.trajectory_questions[traj_id] = self._pending_question
            logger.debug(f"Associated question with traj {traj_id}: {self._pending_question[:100]}")
            delattr(self, '_pending_question')
        
        obs_text = tool_interact_info.get('obs', '')
        obs_len = len(obs_text) if obs_text else 0
        
        # Check if observation is from cache (starts with <cache>)
        if obs_text and obs_text.startswith("<cache>"):
            # Extract cached content (remove <cache> prefix)
            cached_content = obs_text[7:].strip()  # Remove "<cache>" prefix

            # if start with Tool:, formatted_obs = cached_content
            if cached_content.startswith("Tool: "):
                formatted_obs = cached_content
            else:
                # Extract tool name and query/URL from action
                tool_name = self._extract_tool_name_from_action(action)
                
                # Format based on tool type
                formatted_obs = f"Tool: {tool_name}\n"
                
                if tool_name == "content_extraction":
                    # For content_extraction, use URL instead of Query
                    url = self._extract_url_from_action(action)
                    if url:
                        formatted_obs += f"URL: {url}\n"
                    else:
                        # Log warning if URL extraction failed
                        logger.warning(f"Failed to extract URL from action for content_extraction: {action[:100]}")
                        formatted_obs += f"URL: [extraction failed]\n"
                    formatted_obs += f"Response: {cached_content}"
                    logger.debug(f"Using cached response for traj {traj_id}, tool={tool_name}, url={url[:50] if url else 'None'}")
                else:
                    # For other tools, use Query
                    query = self._extract_query_from_action(action)
                    if query:
                        formatted_obs += f"Query: {query}\n"
                    formatted_obs += f"Response: {cached_content}"
                    logger.debug(f"Using cached response for traj {traj_id}, tool={tool_name}, query={query[:50] if query else 'None'}")
            
            # Update tool_interact_info with formatted cached observation
            tool_interact_info['obs'] = formatted_obs[:MAX_OBS_LENGTH]
            tool_interact_info['was_cached'] = True
            tool_interact_info['was_summarized'] = False
        else:
            # Observation was not from cache (tool server returned fresh result or no observation)
            tool_interact_info['was_cached'] = False
            # Log summarization status for debugging
            if obs_text:
                logger.info(f"Summarization check for traj {traj_id}: use_gpt_summarization={self.use_gpt_summarization}, "
                           f"gpt_client={'available' if self.gpt_client else 'None'}, obs_len={obs_len}")
            
            if self.use_gpt_summarization and self.gpt_client:
                # Extract tool name from original action (before cleaning)
                tool_name = self._extract_tool_name_from_action(action)
                
                # Summarize if observation is long (threshold: 1000 chars)
                if obs_text and obs_len > 1000:
                    # Get question from trajectory cache (should be set by now)
                    question = self.trajectory_questions.get(traj_id)
                    
                    if not question:
                        # Try to extract from pending question as fallback
                        if hasattr(self, '_pending_question') and self._pending_question:
                            question = self._pending_question
                            self.trajectory_questions[traj_id] = question
                    
                    logger.debug(f"Summarizing observation for traj {traj_id}, tool={tool_name}, obs_length={len(obs_text)}, has_question={question is not None}")
                    
                    # Summarize observation
                    summarized_obs = await self.summarize_observation_with_gpt4(
                        observation=obs_text,
                        tool_name=tool_name,
                        question=question
                    )
                    
                    # Update tool_interact_info with summarized observation
                    tool_interact_info['obs'] = summarized_obs[:MAX_OBS_LENGTH]
                    tool_interact_info['original_obs'] = obs_text  # Save original observation for rollout trajectory
                    tool_interact_info['original_obs_length'] = len(obs_text)
                    tool_interact_info['summarized_obs_length'] = len(summarized_obs)
                    tool_interact_info['was_summarized'] = True
                elif obs_text:
                    # Format observation even if it's short (<= 1000 chars)
                    formatted_obs = self._format_observation(obs_text, tool_name=tool_name, action=action)
                    tool_interact_info['obs'] = formatted_obs[:MAX_OBS_LENGTH]
                    tool_interact_info['original_obs'] = obs_text  # Save original observation for rollout trajectory
                    tool_interact_info['original_obs_length'] = len(obs_text)
                    tool_interact_info['was_summarized'] = False
                    logger.debug(f"Formatted observation (no summarization) for traj {traj_id}: observation too short (len={obs_len} <= 1000)")
                else:
                    logger.debug(f"Summarization SKIPPED for traj {traj_id}: no observation text")
            else:
                # Format observation even if summarization is disabled
                if obs_text:
                    tool_name = self._extract_tool_name_from_action(action)
                    formatted_obs = self._format_observation(obs_text, tool_name=tool_name, action=action)
                    tool_interact_info['obs'] = formatted_obs[:MAX_OBS_LENGTH]
                    tool_interact_info['original_obs'] = obs_text  # Save original observation for rollout trajectory
                    tool_interact_info['original_obs_length'] = len(obs_text)
                    tool_interact_info['was_summarized'] = False
                    logger.debug(f"Formatted observation (summarization disabled) for traj {traj_id}: use_gpt_summarization={self.use_gpt_summarization}, gpt_client={'available' if self.gpt_client else 'None'}")
                else:
                    logger.debug(f"Summarization SKIPPED for traj {traj_id}: no observation text")
        
        # Log to trajectory history for ReAct reasoning
        if self.use_react_reasoning:
            if traj_id not in self.trajectory_history:
                self.trajectory_history[traj_id] = []
            
            # Use cleaned action for trajectory history
            self.trajectory_history[traj_id].append((
                cleaned_action,
                tool_interact_info.get('obs', ''),
                tool_interact_info.get('valid_action', False)
            ))
        
        # Track observations for final answer synthesis
        if traj_id not in self.trajectory_observations:
            self.trajectory_observations[traj_id] = []
        obs_text = tool_interact_info.get('obs', '')
        if obs_text:
            # Store the summarized observation if available, otherwise original
            self.trajectory_observations[traj_id].append(obs_text)
        
        return tool_interact_info
    
    def _create_final_answer_prompt(self, question: str) -> str:
        """
        Create a prompt that forces the model to synthesize all observations and provide a final answer.
        
        Args:
            question: The research question
            
        Returns:
            A prompt string to inject before final generation
        """
        prompt = f"""Based on the research findings provided, provide a direct answer to the research question.

Research Question: {question}

Your answer should:
1. Concise reasoning step by step
2. Put your final concise answer inside the \\boxed{{}}
3. Consider both textual research findings and visual information from the image

Just provide the answer without additional formatting."""
        
        return prompt
    
    async def _inject_synthesis_prompt_if_needed(
        self, 
        request_id: str, 
        is_last_step: bool,
        running_prompt_ids: List[int],
        available_length: int,
        response_mask: List[int],
        response_logprobs: List[float],
        prompt_ids: List[int],
        max_response_length: int
    ) -> Tuple[List[int], List[int], List[float], int]:
        """
        Inject synthesis prompt before last step generation if we have observations.
        
        Args:
            request_id: Trajectory ID
            is_last_step: Whether this is the last step
            running_prompt_ids: Current running prompt token IDs
            available_length: Available length for generation
            response_mask: Current response mask
            response_logprobs: Current response logprobs
            prompt_ids: Original prompt IDs (for length calculation)
            max_response_length: Maximum response length
            
        Returns:
            Updated (running_prompt_ids, response_mask, response_logprobs, available_length)
        """
        if not is_last_step or request_id not in self.trajectory_observations:
            return running_prompt_ids, response_mask, response_logprobs, available_length
        
        observations = self.trajectory_observations[request_id]
        question = self.trajectory_questions.get(request_id, "the question")
        
        if not observations or len(observations) == 0:
            return running_prompt_ids, response_mask, response_logprobs, available_length
        
        # Create synthesis prompt
        synthesis_prompt = self._create_final_answer_prompt(question)
        # Format with role tags for consistent chat template
        synthesis_prompt_ids = self._format_user_message_with_role_tags(synthesis_prompt)
        
        # Check if we have space for the synthesis prompt
        if len(synthesis_prompt_ids) >= available_length:
            logger.warning(f"Not enough space for synthesis prompt ({len(synthesis_prompt_ids)} tokens, available: {available_length})")
            return running_prompt_ids, response_mask, response_logprobs, available_length
        
        # Inject the synthesis prompt before final generation
        running_prompt_ids = running_prompt_ids.copy()
        running_prompt_ids.extend(synthesis_prompt_ids)
        
        if self.agent_config.mask_observations:
            response_mask.extend([0] * len(synthesis_prompt_ids))  # Don't train on injected prompt
        else:
            response_mask.extend([1] * len(synthesis_prompt_ids))
        response_logprobs.extend([0.0] * len(synthesis_prompt_ids))
        
        # Update available length (recalculate based on new running_prompt_ids length)
        available_length = max(max_response_length - len(running_prompt_ids) + len(prompt_ids), 0)
        
        logger.info(f"Injected synthesis prompt with {len(observations)} observations for traj_id={request_id}")
        
        return running_prompt_ids, response_mask, response_logprobs, available_length
    
    def _format_user_message_with_role_tags(self, content: str) -> List[int]:
        """
        Format a user message with proper role tags for consistent chat template.
        
        Args:
            content: The message content
            
        Returns:
            Token IDs for the formatted user message with role tags
            Format: <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
        """
        if self.enable_mtrl:
            # Format as a user message using apply_chat_template
            # Use add_generation_prompt=False to get: <|im_start|>user\n{content}<|im_end|>
            messages = [{"role": "user", "content": content}]
            formatted_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # Append assistant tag to prepare for the next response
            formatted_text = formatted_text + "\n<|im_start|>assistant\n"
            # Tokenize the formatted text
            formatted_token_ids = self.tokenizer.encode(formatted_text)
            return formatted_token_ids
        else:
            # Fallback: encode without role tags if mtrl is disabled
            return self.tokenizer.encode(content)
    
    def _insert_user_prompt_after_observation(
        self,
        request_id: str,
        running_prompt_ids: List[int],
        response_mask: List[int],
        response_logprobs: List[float],
        prompt_ids: List[int],
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        Insert user prompt (Research Question + JSON prompt) after observations.
        
        Args:
            request_id: Trajectory ID
            running_prompt_ids: Current running prompt token IDs
            response_mask: Current response mask
            response_logprobs: Current response logprobs
            prompt_ids: Original prompt IDs
            
        Returns:
            Updated (running_prompt_ids, response_mask, response_logprobs)
        """
        # Get question for the prompt
        question = self.trajectory_questions.get(request_id)
        if not question:
            # Try to extract from prompt if not cached
            question = self._extract_question_from_prompt(prompt_ids)
            if question:
                self.trajectory_questions[request_id] = question
        
        if question:
            # Get image URL for the prompt
            image_url_text = IMAGE_URL_PLACEHOLDER  # Default placeholder
            if request_id in self.trajectory_image_urls:
                image_urls = self.trajectory_image_urls[request_id]
                if isinstance(image_urls, list) and len(image_urls) > 0:
                    image_url_text = image_urls[0]
            
            # Build next user message: Research Question + JSON prompt (matching template format)
            next_user_message = f"Research Question: {question}\nThe image url is {image_url_text}.\n\nBased on the research question and above actions and findings, what should be the next step?\n- If you have enough information to answer the question, set should_stop: true\n- If you need more information, choose the most appropriate action\n- Ensure action_parameters are correctly formatted for the chosen action_type\n- AVOID repeating the same tool with the same query \n- diversify your research approach. \n\nRespond with a JSON object containing:\n{{\n    \"reasoning\": \"Your analysis of current state and what's needed next\",\n    \"action\": {{\n        \"action_type\": \"action_type\",\n        \"action_description\": \"What this action will do\",\n        \"action_parameters\": {{\"param\": \"value\"}},\n        \"expected_outcome\": \"What you expect to learn\"\n    }},\n    \"should_stop\": false,\n    \"confidence\": 0.8\n}}\n\nSet \"should_stop\": true only if you have sufficient information to answer the research question comprehensively."
            
            # Format with role tags for consistent chat template
            # The observation now ends with <|im_end|> (no assistant tag), so we can directly add the user message
            # which will start with <|im_start|>user\n and end with <|im_end|>\n<|im_start|>assistant\n
            next_user_token_ids = self._format_user_message_with_role_tags(next_user_message)
            
            running_prompt_ids.extend(next_user_token_ids)
            if self.agent_config.mask_observations:
                response_mask.extend([0] * len(next_user_token_ids))  # Don't train on the prompt
            else:
                response_mask.extend([1] * len(next_user_token_ids))
            response_logprobs.extend([0.0] * len(next_user_token_ids))
        
        return running_prompt_ids, response_mask, response_logprobs
    
    async def _generate_model_response(
        self,
        request_id: str,
        step: int,
        running_prompt_ids: List[int],
        running_image_data: Optional[List],
        running_audio_data: Optional[List],
        agent_sampling_params: Dict[str, Any],
        is_last_step: bool,
        metrics: Dict[str, Any],
        stats_dict: Dict[str, Any],
    ) -> Tuple[Any, List[int], List[float], str, str, str]:
        """
        Generate model response and handle cleaning/stats.
        
        Args:
            request_id: Trajectory ID
            step: Current step number
            running_prompt_ids: Current running prompt token IDs
            running_image_data: Current image data
            running_audio_data: Current audio data
            agent_sampling_params: Sampling parameters
            is_last_step: Whether this is the last step
            metrics: Metrics dictionary
            stats_dict: Stats dictionary
            
        Returns:
            (output, gen_ids, gen_logprobs, gen_text, finish_reason, stop_reason)
        """
        # On the last step, modify stop tokens to encourage final answer instead of tool calls
        original_stop = agent_sampling_params.get("stop", [])
        if is_last_step and self.action_stop_tokens:
            modified_stop = [s for s in original_stop if s not in self.action_stop_tokens]
            agent_sampling_params["stop"] = modified_stop
            logger.debug(f"Turn {step} (LAST): Removed action stop tokens to allow final answer generation")
        
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=running_prompt_ids,
                sampling_params=agent_sampling_params,
                image_data=running_image_data,
                audio_data=running_audio_data,
            )
        
        # Restore original stop tokens after generation
        if is_last_step:
            agent_sampling_params["stop"] = original_stop
            if output.text.strip() == "":
                logger.warning(f"Turn {step}: Generated empty response for traj_id={request_id}")
        
        gen_text = output.text
        
        # Always ensure proper formatting: <|im_start|>assistant\n{content}<|im_end|>
        # Remove any message tags if model accidentally generated them, then wrap with correct format
        # if "<|im_start|>" in gen_text or "<|im_end|>" in gen_text:
        #     logger.warning(
        #         f"Model generated message tags in output! Removing them and reformatting. "
        #         f"Traj {request_id}, Turn {step}. Text preview: {gen_text[:200]}..."
        #     )
        #     # Remove all message tags from the generated text
        #     gen_text = re.sub(r'<\|im_start\|>(user|assistant|system)?\s*\n?', '', gen_text, flags=re.IGNORECASE | re.MULTILINE)
        #     gen_text = re.sub(r'<\|im_end\|>', '', gen_text, flags=re.IGNORECASE | re.MULTILINE)
        #     gen_text = gen_text.strip()
        
        # Wrap with correct assistant format (always, whether tags were detected or not)
        # gen_text = f"<|im_start|>assistant\n{gen_text}<|im_end|>"
        
        # Re-tokenize the formatted text
        gen_ids = self.tokenizer.encode(gen_text)
        # Use zeros for logprobs since we can't map original logprobs to re-tokenized text
        gen_logprobs = [0.0] * len(gen_ids)
        
        finish_reason = output.finish_reason
        stop_reason = output.stop_reason
        if isinstance(stop_reason, int):
            stop_reason = self.tokenizer.decode([stop_reason])[0]
        
        return output, gen_ids, gen_logprobs, gen_text, finish_reason, stop_reason
    
    def _parse_model_response(
        self,
        gen_text: str,
        finish_reason: str,
        stop_reason: str,
        request_id: str,
        step: int,
    ) -> Dict[str, Any]:
        """
        Parse model response to extract action and stop information.
        
        Args:
            gen_text: Generated text
            finish_reason: Finish reason from output
            stop_reason: Stop reason from output
            request_id: Trajectory ID
            step: Current step number
            
        Returns:
            Dict with keys: action_text, should_stop, confidence, do_action, matched_stop_token
        """
        do_action = False
        action_text = ""
        should_stop = False
        confidence = 0.0
        matched_stop_token = None
        
        # Try to parse JSON response first (new format)
        json_response = self._parse_json_response(gen_text)
        if json_response:
            should_stop = json_response.get("should_stop", False)
            confidence = json_response.get("confidence", 0.0)
            action_dict = json_response.get("action", {})
            
            # Check if action_type indicates stopping (even if should_stop is not explicitly True)
            if action_dict and action_dict.get("action_type"):
                action_type = action_dict.get("action_type", "").lower()
                # If action_type is "should_stop" or "none", treat as stop signal
                # Only set should_stop = True if confidence is above threshold
                if action_type in ["should_stop", "none", "finish", "stop"]:
                    if confidence > 0.7:
                        should_stop = True
                        logger.info(f"Turn {step}: Action type '{action_type}' indicates stopping with confidence={confidence:.2f}. should_stop={should_stop}")
                    else:
                        logger.info(f"Turn {step}: Action type '{action_type}' indicates stopping but confidence={confidence:.2f} is too low. should_stop={should_stop}")
                else:
                    # If not stopping, convert JSON action to tool call format
                    # Prepare extra_fields for URL replacement
                    extra_fields_for_conversion = {}
                    if request_id in self.trajectory_image_urls:
                        extra_fields_for_conversion['image_urls'] = self.trajectory_image_urls[request_id]
                    
                    action_text = self._convert_json_action_to_tool_call(action_dict, extra_fields_for_conversion)
                    if action_text:
                        do_action = True
                        logger.info(f"Turn {step}: Parsed JSON action: {action_dict.get('action_type')}, should_stop={should_stop}, confidence={confidence:.2f}")
                    else:
                        logger.warning(f"Turn {step}: Failed to convert JSON action to tool call: {action_dict}")
            elif should_stop:
                # Explicit should_stop without action dict is also valid
                logger.info(f"Turn {step}: JSON response indicates should_stop=True, confidence={confidence:.2f}")
            else:
                logger.warning(f"Turn {step}: JSON response missing action or action_type: {json_response}")
        else:
            # Fallback to old format: check for action stop tokens
            if finish_reason == "stop" and stop_reason:
                for action_stop_token in self.action_stop_tokens:
                    if action_stop_token in stop_reason:
                        do_action = True
                        matched_stop_token = action_stop_token
                        action_text = (gen_text.split(action_stop_token)[0] + action_stop_token)
                        break
        
        return {
            "action_text": action_text,
            "should_stop": should_stop,
            "confidence": confidence,
            "do_action": do_action,
            "matched_stop_token": matched_stop_token,
            "json_response": json_response,
        }
    
    async def _check_should_stop(
        self,
        request_id: str,
        step: int,
        is_last_step: bool,
        use_tool: bool,
        finish_reason: str,
        stop_reason: str,
        gen_text: str,
        parsed_response: Dict[str, Any],
        running_prompt_ids: List[int],
        available_length: int,
        response_mask: List[int],
        response_logprobs: List[float],
        prompt_ids: List[int],
        max_response_length: int,
        stats_dict: Dict[str, Any],
    ) -> Tuple[bool, str, str, List[int], List[int], List[float]]:
        """
        Check if we should stop and determine stop reason.
        
        Args:
            request_id: Trajectory ID
            step: Current step number
            is_last_step: Whether this is the last step
            use_tool: Whether tools are enabled
            finish_reason: Finish reason from output
            stop_reason: Stop reason from output
            gen_text: Generated text
            parsed_response: Parsed response dict
            running_prompt_ids: Current running prompt token IDs
            available_length: Available length
            response_mask: Current response mask
            response_logprobs: Current response logprobs
            prompt_ids: Original prompt IDs
            max_response_length: Maximum response length
            stats_dict: Stats dictionary
            
        Returns:
            (should_stop, traj_stop_reason, stop_reason, running_prompt_ids, response_mask, response_logprobs)
        """
        should_stop = parsed_response.get("should_stop", False)
        confidence = parsed_response.get("confidence", 0.0)
        do_action = parsed_response.get("do_action", False)
        
        # Check if tools are disabled
        if not use_tool:
            stats_dict["is_traj_finished"] = True
            traj_stop_reason = f"no_tool-{finish_reason}"
            return True, traj_stop_reason, stop_reason, running_prompt_ids, response_mask, response_logprobs
        
        # Check JSON should_stop with confidence threshold
        # This covers both explicit stop action types and regular should_stop
        if should_stop and confidence > 0.7:
            logger.info(f"Turn {step}: JSON response indicates should_stop=true with confidence={confidence:.2f}. Adding final answer prompt.")
            stats_dict["is_traj_finished"] = True
            traj_stop_reason = f"json_should_stop_confidence_{confidence:.2f}"
            return True, traj_stop_reason, stop_reason, running_prompt_ids, response_mask, response_logprobs
        
        # Check if max_turns reached
        if is_last_step:
            logger.info(f"Turn {step} (LAST): Reached max_turns. Adding final answer prompt.")
            stats_dict["is_traj_finished"] = True
            traj_stop_reason = "max_turns_reached"
            return True, traj_stop_reason, stop_reason, running_prompt_ids, response_mask, response_logprobs
        
        # Inject synthesis prompt if needed (on last step) - do this BEFORE checking stop conditions
        # so it's available if we continue to the next generation
        if is_last_step:
            running_prompt_ids, response_mask, response_logprobs, available_length = await self._inject_synthesis_prompt_if_needed(
                request_id=request_id,
                is_last_step=is_last_step,
                running_prompt_ids=running_prompt_ids,
                available_length=available_length,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                prompt_ids=prompt_ids,
                max_response_length=max_response_length
            )
        
        # Check if no action was found and model chose to finish
        if not do_action:
            if finish_reason == "stop":
                if not stop_reason:
                    stats_dict["is_traj_finished"] = True
                    if gen_text.strip() == "":
                        traj_stop_reason = "model_chose_to_finish_with_empty_response"
                    else:
                        traj_stop_reason = "model_chose_to_finish"
                    return True, traj_stop_reason, stop_reason, running_prompt_ids, response_mask, response_logprobs
                else:
                    stats_dict["is_traj_finished"] = False
                    traj_stop_reason = "no_action_stop_token_found_in_stop_reason=" + stop_reason
                    return True, traj_stop_reason, stop_reason, running_prompt_ids, response_mask, response_logprobs
            elif finish_reason == "length":
                stats_dict["is_traj_finished"] = False
                traj_stop_reason = "max_response_length_reached"
                return True, traj_stop_reason, stop_reason, running_prompt_ids, response_mask, response_logprobs
            else:
                stats_dict["is_traj_finished"] = True
                traj_stop_reason = f"finish_reason_{finish_reason}"
                return True, traj_stop_reason, stop_reason, running_prompt_ids, response_mask, response_logprobs
        
        return False, "", stop_reason, running_prompt_ids, response_mask, response_logprobs
    
    async def _generate_final_answer(
        self,
        request_id: str,
        running_prompt_ids: List[int],
        running_image_data: Optional[List],
        running_audio_data: Optional[List],
        agent_sampling_params: Dict[str, Any],
        max_response_length: int,
        prompt_ids: List[int],
        max_action_length: int,
        response_mask: List[int],
        response_logprobs: List[float],
        metrics: Dict[str, Any],
        stats_dict: Dict[str, Any],
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        Generate final answer after stop condition is met.
        
        Args:
            request_id: Trajectory ID
            running_prompt_ids: Current running prompt token IDs
            running_image_data: Current image data
            running_audio_data: Current audio data
            agent_sampling_params: Sampling parameters
            max_response_length: Maximum response length
            prompt_ids: Original prompt IDs
            max_action_length: Maximum action length
            response_mask: Current response mask
            response_logprobs: Current response logprobs
            metrics: Metrics dictionary
            stats_dict: Stats dictionary
            
        Returns:
            Updated (running_prompt_ids, response_mask, response_logprobs)
        """
        # Add final answer user prompt (matching template format)
        question = self.trajectory_questions.get(request_id, "the question")
        final_answer_prompt = f"""Based on the research findings provided, provide a direct answer to the research question.

Research Question: {question}

Your answer should:
1. Concise reasoning step by step
2. Put your final concise answer inside the \\boxed{{}}
3. Consider both textual research findings and visual information from the image

Just provide the answer without additional formatting."""
        
        # Format with role tags for consistent chat template
        final_answer_token_ids = self._format_user_message_with_role_tags(final_answer_prompt)
        
        running_prompt_ids.extend(final_answer_token_ids)
        if self.agent_config.mask_observations:
            response_mask.extend([0] * len(final_answer_token_ids))  # Don't train on the prompt
        else:
            response_mask.extend([1] * len(final_answer_token_ids))
        response_logprobs.extend([0.0] * len(final_answer_token_ids))
        
        # Generate final answer
        available_length = max(max_response_length - len(running_prompt_ids) + len(prompt_ids), 0)
        max_tokens_for_final = min(max_action_length, available_length)
        if max_tokens_for_final > 0:
            agent_sampling_params["max_tokens"] = max_tokens_for_final
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=running_prompt_ids,
                    sampling_params=agent_sampling_params,
                    image_data=running_image_data,
                    audio_data=running_audio_data,
                )
            final_gen_ids = output.token_ids
            final_gen_logprobs = output.log_probs or [0.0] * len(final_gen_ids)
            
            running_prompt_ids.extend(final_gen_ids)
            response_mask.extend([1] * len(final_gen_ids))
            response_logprobs.extend(final_gen_logprobs)
            stats_dict["action_lengths"].append(len(final_gen_ids))
            stats_dict["action_logps"].append(np.mean(final_gen_logprobs) if len(final_gen_logprobs) > 0 else 0.0)
        
        return running_prompt_ids, response_mask, response_logprobs
    
    async def _execute_action_and_collect_observation(
        self,
        request_id: str,
        step: int,
        action_text: str,
        encoded_image_data: Optional[List],
        encoded_audio_data: Optional[List],
        is_last_step: bool,
        running_prompt_ids: List[int],
        running_image_data: Optional[List],
        running_audio_data: Optional[List],
        response_mask: List[int],
        response_logprobs: List[float],
        prompt_ids: List[int],
        max_obs_length: int,
        previous_length: int,
        metrics: Dict[str, Any],
        stats_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[int], bool, List[int], List[int], List[float], Optional[List], Optional[List]]:
        """
        Execute action and collect observation, then add next user prompt.
        
        Args:
            request_id: Trajectory ID
            step: Current step number
            action_text: Action text to execute
            encoded_image_data: Encoded image data
            encoded_audio_data: Encoded audio data
            is_last_step: Whether this is the last step
            running_prompt_ids: Current running prompt token IDs
            running_image_data: Current image data
            running_audio_data: Current audio data
            response_mask: Current response mask
            response_logprobs: Current response logprobs
            prompt_ids: Original prompt IDs
            max_obs_length: Maximum observation length
            previous_length: Previous length for retokenization
            metrics: Metrics dictionary
            stats_dict: Stats dictionary
            
        Returns:
            (tool_results, obs_token_ids, should_break)
        """
        # Prepare extra_fields for tool execution
        extra_fields = {}
        if encoded_image_data is not None:
            extra_fields["images"] = encoded_image_data
        if encoded_audio_data is not None:
            extra_fields["audio"] = encoded_audio_data
        
        # Add question to extra_fields for composite cache key
        question = self.trajectory_questions.get(request_id)
        if question:
            extra_fields["question"] = question
        
        # Execute tool action
        logger.info(f"Turn {step}: Executing action: {action_text[:100]}...")
        start = metrics.get("tool_calls", 0.0)
        with simple_timer("tool_calls", metrics):
            tool_results = await self.interact_with_tool_server(
                traj_id=request_id,
                action=action_text,
                do_action=True,
                extra_fields=extra_fields,
                is_last_step=is_last_step,
            )
        end = metrics.get("tool_calls", 0.0)
        tool_results['interact_time_ms'] = (end - start) * 1000.0
        
        # Process observations and format as user message
        obs_text = tool_results['obs']
        original_obs_length = len(obs_text)
        obs_text = obs_text.strip()
        # Strip any <result>/</result> from observation (do not add any)
        obs_text = strip_result_tags(obs_text)
        
        # Truncate to 1000 characters if summarization is disabled
        if not self.use_gpt_summarization and len(obs_text) > 1000:
            obs_text = obs_text[:1000]
            logger.debug(f"Observation truncated to 1000 characters (summarization disabled, original length was {original_obs_length})")
        
        # Wrap obs_text in <result> tags if not already present
        if not obs_text.startswith("<result>"):
            obs_text = f"<result>{obs_text}</result>"
        
        # Format as user message: <|im_start|>user\n<result>...</result><|im_end|>
        user_message_with_obs = obs_text#f"<|im_start|>user\n{obs_text}<|im_end|>"
        
        obs_token_ids = []
        if tool_results.get('image', None):
            images = [tool_results['image']] if not isinstance(tool_results['image'], list) else tool_results['image']
            decoded_images = [decode_image_url(img_url) for img_url in images]
            if running_image_data is None:
                running_image_data = []
            running_image_data.extend(decoded_images)
            num_image_tags = user_message_with_obs.count("<image>")
            if num_image_tags < len(decoded_images):
                user_message_with_obs += "<image>" * (len(decoded_images) - num_image_tags)
                num_image_tags = len(decoded_images)
            user_message_with_obs = user_message_with_obs.replace("<image>", self.qwen_image_placeholder, num_image_tags)
            obs_token_ids = self.processor(text=[user_message_with_obs], images=decoded_images, return_tensors="pt")["input_ids"].squeeze(0).tolist()
            if max_obs_length < len(obs_token_ids):
                if self.agent_config.truncate_obs_side == 'left':
                    truncation_index = max_obs_length
                    if obs_token_ids[max_obs_length] in self.non_truncate_token_ids:
                        truncation_index = max_obs_length - 1
                        while truncation_index > 0 and obs_token_ids[truncation_index] in self.non_truncate_token_ids:
                            truncation_index -= 1
                    obs_token_ids = obs_token_ids[:truncation_index]
                    obs_token_ids.extend(self.tokenizer.encode("...(truncated)"))
                else:
                    raise NotImplementedError(f"Only left truncation is supported for multimodal observations for now.")
        elif tool_results.get('audio', None):
            audios = [tool_results['audio']] if not isinstance(tool_results['audio'], list) else tool_results['audio']
            decoded_audios = [decode_image_url(audio_url) for audio_url in audios]
            if running_audio_data is None:
                running_audio_data = []
            running_audio_data.extend(decoded_audios)
            num_audio_tags = user_message_with_obs.count("<audio>")
            if num_audio_tags < len(decoded_audios):
                user_message_with_obs += "<audio>" * (len(decoded_audios) - num_audio_tags)
                num_audio_tags = len(decoded_audios)
            user_message_with_obs = user_message_with_obs.replace("<audio>", self.qwen_audio_placeholder, num_audio_tags)
            if self.enable_mtrl:
                user_message_with_obs = self.mtrl_sep.format(obs=user_message_with_obs)
            obs_token_ids = self.processor(text=[user_message_with_obs], audio=decoded_audios, return_tensors="pt")["input_ids"].squeeze(0).tolist()
            if max_obs_length < len(obs_token_ids):
                if self.agent_config.truncate_obs_side == 'left':
                    truncation_index = max_obs_length
                    if obs_token_ids[max_obs_length] in self.non_truncate_token_ids:
                        truncation_index = max_obs_length - 1
                        while truncation_index > 0 and obs_token_ids[truncation_index] in self.non_truncate_token_ids:
                            truncation_index -= 1
                    obs_token_ids = obs_token_ids[:truncation_index]
                    obs_token_ids.extend(self.tokenizer.encode("...(truncated)"))
                else:
                    raise NotImplementedError(f"Only left truncation is supported for multimodal observations for now.")
        else:
            # Text-only observation - encode the formatted user message
            obs_token_ids = self.tokenizer.encode(user_message_with_obs)
            if max_obs_length < len(obs_token_ids):
                if self.agent_config.truncate_obs_side == 'left':
                    obs_token_ids = obs_token_ids[-max_obs_length:]
                    obs_token_ids.extend(self.tokenizer.encode("...(truncated)"))
                elif self.agent_config.truncate_obs_side == 'right':
                    obs_token_ids = obs_token_ids[:max_obs_length]
                    obs_token_ids.extend(self.tokenizer.encode("(truncated)..."))
                elif self.agent_config.truncate_obs_side == 'middle':
                    half_len = max_obs_length // 2
                    obs_token_ids = obs_token_ids[:half_len] + self.tokenizer.encode("...(truncated)...") + obs_token_ids[-half_len:]
                else:
                    raise ValueError(f"Invalid truncate_obs_side: {self.agent_config.truncate_obs_side}")
        
        if self.enable_mtrl:
            # Use suffix without assistant tag since we're adding another user message (react prompt) right after
            # Observations should end with <|im_end|> not <|im_end|>\n<|im_start|>assistant\n
            obs_token_ids = self.mtrl_sep_prefix_ids + obs_token_ids + self.mtrl_sep_suffix_no_assistant_ids
        
        stats_dict["obs_lengths"].append(len(obs_token_ids))
        if 'reward' in tool_results and tool_results['reward'] is not None:
            stats_dict["rewards"].append(tool_results['reward'])
        stats_dict["valid_action"] += tool_results['valid_action'] if 'valid_action' in tool_results else 0
        stats_dict["tool_interact_info"].append(tool_results)
        
        # Track if cache was used in this trajectory
        if tool_results.get('was_cached', False):
            stats_dict["used_cache"] = True
        
        # Append observation as user message
        running_prompt_ids.extend(obs_token_ids)
        if self.agent_config.mask_observations:
            response_mask.extend([0] * len(obs_token_ids))
        else:
            response_mask.extend([1] * len(obs_token_ids))
        response_logprobs.extend([0.0] * len(obs_token_ids))
        
        # Add next user prompt (Research Question + JSON prompt)
        running_prompt_ids, response_mask, response_logprobs = self._insert_user_prompt_after_observation(
            request_id=request_id,
            running_prompt_ids=running_prompt_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            prompt_ids=prompt_ids,
        )
        
        # Handle retokenization if enabled
        if self.agent_config.retokenization:
            new_text = self.tokenizer.decode(running_prompt_ids[previous_length:], skip_special_tokens=False)
            new_ids = self.tokenizer.encode(new_text)
            if new_ids != running_prompt_ids[previous_length:]:
                stats_dict['retokenization_diff'].append(1)
            else:
                stats_dict['retokenization_diff'].append(0)
            running_prompt_ids = running_prompt_ids[:previous_length] + new_ids
            if len(response_mask) > (len(running_prompt_ids) - len(prompt_ids)):
                response_mask = response_mask[:len(running_prompt_ids) - len(prompt_ids)]
            else:
                response_mask.extend([response_mask[-1]] * (len(running_prompt_ids) - len(prompt_ids) - len(response_mask)))
            if len(response_logprobs) > (len(running_prompt_ids) - len(prompt_ids)):
                response_logprobs = response_logprobs[:len(running_prompt_ids) - len(prompt_ids)]
            else:
                response_logprobs.extend([0.0] * (len(running_prompt_ids) - len(prompt_ids) - len(response_logprobs)))
        
        # Check if tool signaled done
        should_break = tool_results.get('done', False)
        if should_break:
            stats_dict["is_traj_finished"] = True
        
        return tool_results, obs_token_ids, should_break, running_prompt_ids, response_mask, response_logprobs, running_image_data, running_audio_data
    
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Run agent loop with enhanced trajectory tracking and forced final answer on max turns.
        
        Overrides parent method to:
        1. Add question extraction and trajectory logging
        2. Force final answer generation when max_turns is reached by injecting synthesis prompt
        
        NOTE ON CODE DUPLICATION:
        This method duplicates most of the parent's run() method (~380 lines) because:
        - The parent's run() doesn't expose hooks to inject prompts before the last step generation
        - We need to inject a synthesis prompt right before the final generation (line ~761)
        - This is the only way to ensure the model receives all previous observations when max_turns is reached
        
        Future improvement: Refactor parent class to use a hook pattern (e.g., _before_last_step_generation())
        to allow subclasses to inject prompts without duplicating code.
        
        The key difference from parent is:
        - Line ~761: Calls _inject_synthesis_prompt_if_needed() before last step generation
        - Line ~625: Tracks observations in trajectory_observations for synthesis
        - Line ~997: Cleans up trajectory tracking dictionaries
        """
        # Track if we're in a validation step (set flag when we see validation, save all trajectories)
        is_validation = kwargs.get("validate", False)
        if is_validation:
            self.is_validation_step = True
        
        # Extract and cache question from prompt
        prompt_ids = list(kwargs.get("raw_prompt_ids", []))
        if prompt_ids:
            question = self._extract_question_from_prompt(prompt_ids)
            if question:
                self._pending_question = question
        
        # Get all the setup from parent class
        multi_modal_data = kwargs.get("multi_modal_data") or {}
        image_data = multi_modal_data.get("image")
        encoded_image_data = [encode_image_url(img) for img in image_data] if image_data is not None else None
        audio_data = multi_modal_data.get("audio")
        encoded_audio_data = [encode_audio_data(audio) for audio in audio_data] if audio_data is not None else None
        use_tool = kwargs.get("use_tool", self.agent_config.enable_agent)
        
        metrics = {}
        request_id = str(uuid4().hex)
        
        # Store image URLs for this trajectory if available in extra_info
        extra_info = kwargs.get("extra_info", {})
        if isinstance(extra_info, dict) and "image_urls" in extra_info:
            image_urls = extra_info["image_urls"]
            if isinstance(image_urls, list):
                self.trajectory_image_urls[request_id] = image_urls
        
        stats_dict = {
            "num_turns": 0,
            "empty_responses": 0,
            "valid_action": 0,
            "action_lengths": [],
            "action_logps": [],
            "obs_lengths": [],
            "rewards": [],
            "tool_interact_info": [],
            "is_traj_finished": False,
            "valid_traj": 1,
            "retokenization_diff": [],
            "used_cache": False  # Track if any cache was used in this trajectory
        }
        
        if image_data or audio_data:
            raw_prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            processor_kwargs = {"text": [raw_prompt_text], "return_tensors": "pt"}
            if image_data:
                processor_kwargs["images"] = image_data
            if audio_data:
                processor_kwargs["audio"] = audio_data
            model_inputs = self.processor(**processor_kwargs)
            prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
        
        max_turns = self.max_turns if not kwargs.get("validate", False) else self.val_max_turns
        max_response_length = self.train_max_response_length if not kwargs.get("validate", False) else self.val_max_response_length
        max_action_length = self.max_action_length or max_response_length
        max_obs_length = self.max_obs_length
        
        agent_sampling_params = sampling_params.copy()
        agent_sampling_params.update({
            "n": 1,
            "stop": self.action_stop_tokens,
            "include_stop_str_in_output": True,
            "logprobs": self.logprobs,
        })
        
        running_prompt_ids = prompt_ids.copy()
        running_image_data = image_data.copy() if image_data is not None else None
        running_audio_data = audio_data.copy() if audio_data is not None else None
        response_mask = []
        response_logprobs = []
        traj_stop_reason = ""

        logger.debug(f"Starting agent loop for traj_id={request_id} with use_tool={use_tool}, max_turns={max_turns}")
        
        # Main loop - refactored with clear step-by-step structure
        for step in range(max_turns + 1):
            previous_length = len(running_prompt_ids)
            available_length = max(max_response_length - len(running_prompt_ids) + len(prompt_ids), 0)
            max_tokens_for_this_turn = min(max_action_length, available_length)
            is_last_step = (step == max_turns)
            
            # Check if we have space for generation
            if max_tokens_for_this_turn <= 0:
                traj_stop_reason = "max_model_len_exceeded"
                break
            
            # Update max_tokens for this turn
            agent_sampling_params["max_tokens"] = max_tokens_for_this_turn
            logger.debug(f"Turn {step}: available_length={available_length}, max_tokens_for_this_turn={max_tokens_for_this_turn}, is_last_step={is_last_step}")
            
            # Step 1: Model gives response
            output, gen_ids, gen_logprobs, gen_text, finish_reason, stop_reason = await self._generate_model_response(
                request_id=request_id,
                step=step,
                running_prompt_ids=running_prompt_ids,
                running_image_data=running_image_data,
                running_audio_data=running_audio_data,
                agent_sampling_params=agent_sampling_params,
                is_last_step=is_last_step,
                metrics=metrics,
                stats_dict=stats_dict,
            )
            
            # Update running prompt with generated response
            running_prompt_ids.extend(gen_ids)
            response_mask.extend([1] * len(gen_ids))
            response_logprobs.extend(gen_logprobs)
            
            stats_dict["num_turns"] += 1
            stats_dict["action_lengths"].append(len(gen_ids))
            stats_dict["action_logps"].append(np.mean(gen_logprobs) if len(gen_logprobs) > 0 else 0.0)
            if gen_text.strip() == "":
                stats_dict["empty_responses"] += 1
            
            # Step 2: Parse response
            parsed_response = self._parse_model_response(
                gen_text=gen_text,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                request_id=request_id,
                step=step,
            )
            
            # Step 3: Decide if should stop
            should_stop, traj_stop_reason, stop_reason, running_prompt_ids, response_mask, response_logprobs = await self._check_should_stop(
                request_id=request_id,
                step=step,
                is_last_step=is_last_step,
                use_tool=use_tool,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                gen_text=gen_text,
                parsed_response=parsed_response,
                running_prompt_ids=running_prompt_ids,
                available_length=available_length,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                prompt_ids=prompt_ids,
                max_response_length=max_response_length,
                stats_dict=stats_dict,
            )
            
            # Recalculate available length after potential synthesis prompt injection
            available_length = max(max_response_length - len(running_prompt_ids) + len(prompt_ids), 0)
            
            # Step 4: Generate final answer if should stop
            if should_stop:
                running_prompt_ids, response_mask, response_logprobs = await self._generate_final_answer(
                    request_id=request_id,
                    running_prompt_ids=running_prompt_ids,
                    running_image_data=running_image_data,
                    running_audio_data=running_audio_data,
                    agent_sampling_params=agent_sampling_params,
                    max_response_length=max_response_length,
                    prompt_ids=prompt_ids,
                    max_action_length=max_action_length,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                    metrics=metrics,
                    stats_dict=stats_dict,
                )
                break
            
            # Step 5: Execute action and collect observation
            action_text = parsed_response.get("action_text", "")
            do_action = parsed_response.get("do_action", False)
            
            if do_action:
                tool_results, obs_token_ids, should_break, running_prompt_ids, response_mask, response_logprobs, running_image_data, running_audio_data = await self._execute_action_and_collect_observation(
                    request_id=request_id,
                    step=step,
                    action_text=action_text,
                    encoded_image_data=encoded_image_data,
                    encoded_audio_data=encoded_audio_data,
                    is_last_step=is_last_step,
                    running_prompt_ids=running_prompt_ids,
                    running_image_data=running_image_data,
                    running_audio_data=running_audio_data,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                    prompt_ids=prompt_ids,
                    max_obs_length=max_obs_length,
                    previous_length=previous_length,
                    metrics=metrics,
                    stats_dict=stats_dict,
                )
                
                if should_break:
                    traj_stop_reason = "tool_signaled_done"
                    break
            else:
                # No action found - finish the trajectory
                if finish_reason == "stop":
                    if not stop_reason:
                        stats_dict["is_traj_finished"] = True
                        if gen_text.strip() == "":
                            traj_stop_reason = "model_chose_to_finish_with_empty_response"
                        else:
                            traj_stop_reason = "model_chose_to_finish"
                    else:
                        stats_dict["is_traj_finished"] = False
                        traj_stop_reason = "no_action_stop_token_found_in_stop_reason=" + stop_reason
                elif finish_reason == "length":
                    stats_dict["is_traj_finished"] = False
                    traj_stop_reason = "max_response_length_reached"
                else:
                    stats_dict["is_traj_finished"] = True
                    traj_stop_reason = f"finish_reason_{finish_reason}"
                break
        
        logger.debug(f"Trajectory {request_id} finished after {step} turns. Stop reason: {traj_stop_reason}")
        response_ids = running_prompt_ids[len(prompt_ids):]
        assert len(response_ids) == len(response_mask), f"Response ids and mask length mismatch: {len(response_ids)} vs {len(response_mask)}"
        
        # Log full trajectory summary (only during validation steps - saves both val and rollout)
        is_validation = kwargs.get("validate", False)
        # Use running_prompt_ids for full trajectory, but original prompt_ids + response_ids for training data
        # Get global training step from kwargs (passed from agent_loop manager)
        global_step_value = kwargs.get("global_step", None)
        self._log_full_trajectory_summary(
            traj_id=request_id,
            prompt_ids=running_prompt_ids,  # Full trajectory for display
            stats_dict=stats_dict,
            response_ids=response_ids,  # Response part for training data reconstruction
            response_mask=response_mask,
            original_prompt_ids=prompt_ids,  # Original prompt for training data
            is_validation=is_validation,
            global_step=global_step_value
        )
        
        # Clean up trajectory tracking
        if request_id in self.trajectory_observations:
            del self.trajectory_observations[request_id]
        if request_id in self.trajectory_questions:
            del self.trajectory_questions[request_id]
        if request_id in self.trajectory_history:
            del self.trajectory_history[request_id]
        if request_id in self.trajectory_image_urls:
            del self.trajectory_image_urls[request_id]
        
        start = time.time()
        asyncio.create_task(self.close_traj_tool_threads(request_id=request_id))
        end = time.time()
        close_traj_time = end - start
        
        if self.agent_config.mask_overlong_loss and not stats_dict["is_traj_finished"]:
            response_mask = [0] * len(response_mask)
            stats_dict["valid_traj"] = 0
            logger.info(f"Masking the whole response for traj_id={request_id} due to overlong trajectory and not finished.")
        
        if self.agent_config.mask_void_traj:
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            has_answer = "\\boxed" in response_text or "final_answer(" in response_text
            if stats_dict["valid_action"] == 0 and not has_answer:
                response_mask = [0] * len(response_mask)
                stats_dict["valid_traj"] = 0
                logger.info(f"Masking the whole response for traj_id={request_id} due to void trajectory (no valid action or no final answer).")
        
        # Build metrics (same as parent)
        verl_tool_metrics = {
            "num_turns": stats_dict["num_turns"],
            "empty_responses": stats_dict["empty_responses"],
            "valid_action": stats_dict["valid_action"],
            "per_action_length": np.mean(stats_dict["action_lengths"]) if len(stats_dict["action_lengths"]) > 0 else 0,
            "per_obs_length": np.mean(stats_dict["obs_lengths"]) if len(stats_dict["obs_lengths"]) > 0 else 0,
            "per_action_logp": np.mean(stats_dict["action_logps"]) if len(stats_dict["action_logps"]) > 0 else 0,
            "per_reward_from_tool": np.mean(stats_dict["rewards"]) if len(stats_dict["rewards"]) > 0 else 0,
            "tool_call_success": np.mean([info.get("success", 1.0) for info in stats_dict["tool_interact_info"]]) if len(stats_dict["tool_interact_info"]) > 0 else 1.0,
            "traj_actions_length": sum(stats_dict["action_lengths"]),
            "traj_obs_length": sum(stats_dict["obs_lengths"]),
            "generated_length": sum(stats_dict["action_lengths"]),  # Use action_lengths instead of output variable
            "is_traj_finished": float(stats_dict["is_traj_finished"]),
            "valid_traj": float(stats_dict["valid_traj"]),
            "no_loss_on_traj": response_mask.count(1) == 0,
            "retokenization_diff": np.mean(stats_dict["retokenization_diff"]) if len(stats_dict["retokenization_diff"]) > 0 else 0,
            "tool_processing_time_sec": sum(info.get("processing_time_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "tool_queue_time_sec": sum(info.get("queue_time_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "client_queued_time_sec": sum(info.get("client_queued_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "client_conn_create_time_sec": sum(info.get("client_conn_create_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "client_request_send_time_sec": sum(info.get("client_request_send_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "response_read_time_sec": sum(info.get("response_read_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "response_size_mb": sum(info.get("response_size_mb", 0.0) for info in stats_dict["tool_interact_info"]),
            "response_await_time_sec": sum(info.get("time awaiting response_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "interact_time_time_sec": sum(info.get("interact_time_ms", 0.0) for info in stats_dict["tool_interact_info"]) / 1000.0,
            "session_request_time": sum(info.get("session_request_time", 0.0) for info in stats_dict["tool_interact_info"]),
            "X-Process-Time_sec": sum(info.get("X-Process-Time", 0.0) for info in stats_dict["tool_interact_info"]),
            "close_traj_time": close_traj_time,
            "used_cache": float(stats_dict.get("used_cache", False)),  # Whether any cache was used in this trajectory
        }
        
        if stats_dict["tool_interact_info"] and all("metrics" in info for info in stats_dict["tool_interact_info"]):
            tool_metric_keys = stats_dict["tool_interact_info"][0]["metrics"].keys()
            for key in tool_metric_keys:
                try:
                    verl_tool_metrics[f"tool_avg_{key}"] = np.mean([float(info["metrics"][key]) for info in stats_dict["tool_interact_info"] if key in info["metrics"]])
                except Exception as e:
                    logger.warning(f"Failed to compute mean for tool metric {key}: {e}")
        
        for i, logp in enumerate(stats_dict["action_logps"]):
            verl_tool_metrics[f"turn_{i+1}_action_logp"] = logp
        
        multi_modal_output = {}
        if running_image_data is not None:
            multi_modal_output["image"] = running_image_data
        if running_audio_data is not None:
            multi_modal_output["audio"] = running_audio_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length],
            multi_modal_data=multi_modal_output,
            num_turns=stats_dict["num_turns"],
            metrics=metrics,
            extra_fields={"tool_interact_info": stats_dict.get("tool_interact_info", []), "traj_stop_reason": traj_stop_reason, "verl_tool_metrics": verl_tool_metrics},
        )
        
        # Clean up pending question if it wasn't used
        if hasattr(self, '_pending_question'):
            delattr(self, '_pending_question')
        
        return output

