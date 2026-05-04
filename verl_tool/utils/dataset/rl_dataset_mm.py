import io
import base64
import numpy as np
import regex as re
import datasets
import logging
import json
from verl.utils.dataset.rl_dataset import RLHFDataset
from pathlib import Path
from typing import List, Optional, Tuple, Any
from copy import deepcopy
from collections import defaultdict

from verl_tool.utils.dataset.rl_dataset import nested_copy, RolloutMessagesMixin

# Import TOOL_REGISTRY to generate dynamic tool descriptions
try:
    from verl_tool.agent_loop.verltool_agent_loop_mm import TOOL_REGISTRY
except ImportError:
    # Fallback if import fails
    TOOL_REGISTRY = {}
    logger.warning("Could not import TOOL_REGISTRY from verltool_agent_loop_mm")

logger = logging.getLogger(__name__)


class VerlToolRLHFDatasetMM(RLHFDataset):
    """A multimodal dataset class for reinforcement learning tasks that supports both local image paths and image URLs.
    
    This class extends the base RLHFDataset class to provide support for:
    - Local image paths (for efficient processing)
    - Image URLs (for tools that require URLs, e.g., reverse image search)
    - Automatic fallback between local paths and URLs
    
    The dataset expects data with:
    - `images`: List of dicts with local paths, e.g., [{"image": "/path/to/image.jpg"}]
    - `image_urls`: Optional list of URLs, e.g., ["https://example.com/image.jpg"]
    
    If both are present, local paths are preferred for processing, but URLs are available
    for tools that need them.
    """
    
    _enhanced_prompt_logged = False  # Class-level flag to log enhanced prompt only once
    
    def _infer_enabled_tools_from_stop_tokens(self, action_stop_tokens: Optional[str] = None) -> List[str]:
        """
        Infer enabled tools from action_stop_tokens string.
        
        Args:
            action_stop_tokens: Comma-separated string of stop tokens like '</python>,</text_search_text>'
            
        Returns:
            List of enabled tool names
        """
        if not action_stop_tokens:
            return []
        
        # Map stop tokens to tool names
        stop_token_to_tool = {
            '</text_search_text>': 'web_text_to_text_search',
            '</text_search_image>': 'web_text_to_img_search',
            '</web_read>': 'web_url_reader',
            '</image_search_text>': 'web_image_to_text',
            '</ocr_tool>': 'ocr_tool',
            '</python>': 'ipython_code',
            '</bash>': 'bash_terminal',
        }
        
        enabled_tools = []
        stop_tokens = [token.strip() for token in action_stop_tokens.split(',')]
        for stop_token in stop_tokens:
            if stop_token in stop_token_to_tool:
                enabled_tools.append(stop_token_to_tool[stop_token])
        
        return enabled_tools
    
    def _generate_tool_descriptions(self, enabled_tools: Optional[List[str]] = None, action_stop_tokens: Optional[str] = None) -> str:
        """
        Generate tool descriptions in the format used in prompts.
        
        Args:
            enabled_tools: List of enabled tool names. If None, tries to infer from action_stop_tokens or uses all tools.
            action_stop_tokens: Optional comma-separated string of stop tokens to infer enabled tools.
            
        Returns:
            Formatted string with tool descriptions in action parameter format
        """
        if not TOOL_REGISTRY:
            return ""
        
        # Infer enabled tools from action_stop_tokens if not provided
        if enabled_tools is None:
            if action_stop_tokens:
                enabled_tools = self._infer_enabled_tools_from_stop_tokens(action_stop_tokens)
            else:
                # Default to all tools except 'finish' if not specified
                enabled_tools = [tool for tool in TOOL_REGISTRY.keys() if tool != "finish"]
        
        # Map tool registry names to action types (format field)
        tool_to_action_map = {
            "web_text_to_text_search": "web_search",
            "web_text_to_img_search": "image_search",
            "web_url_reader": "content_extraction",
            "web_image_to_text": "reverse_image_search",
            "ocr_tool": "ocr",
            "ipython_code": "code_execution",
            "bash_terminal": "bash_execution",
        }
        
        # Filter and collect tools
        available_tools = []
        for tool_name in enabled_tools:
            if tool_name in TOOL_REGISTRY and tool_name in tool_to_action_map:
                action_type = tool_to_action_map[tool_name]
                tool_info = TOOL_REGISTRY[tool_name]
                available_tools.append((action_type, tool_info))
        
        # Generate descriptions in the new format: - action_type: {params} - description
        descriptions = []
        for action_type, tool_info in available_tools:
            parameters = tool_info.get('parameters', {})
            description = tool_info.get('description', '')
            
            # Format parameters as JSON-like string
            import json
            params_str = json.dumps(parameters)
            
            # Format: - action_type: {params} - description
            tool_desc = f"- {action_type}: {params_str} - {description}"
            descriptions.append(tool_desc)
        
        if descriptions:
            return "\n".join(descriptions)
        return ""
    
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        
        # Get action_stop_tokens from config if available
        action_stop_tokens = None
        if hasattr(self, 'config') and self.config:
            # Try different possible config paths
            if hasattr(self.config, 'agent') and hasattr(self.config.agent, 'action_stop_tokens'):
                action_stop_tokens = self.config.agent.action_stop_tokens
            elif hasattr(self.config, 'action_stop_tokens'):
                action_stop_tokens = self.config.action_stop_tokens
            elif isinstance(self.config, dict):
                action_stop_tokens = self.config.get('agent', {}).get('action_stop_tokens') or self.config.get('action_stop_tokens')
        
        # Enhance system prompt in row_dict BEFORE building any messages
        # This ensures both _build_rollout_messages and the base class's _build_messages
        # will use the enhanced system prompt
        if self.prompt_key in row_dict:
            messages = row_dict[self.prompt_key]
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and message.get("role") == "system":
                        content = message.get("content", "")
                        if isinstance(content, str):
                            message["content"] = self._enhance_system_prompt_with_react(content, action_stop_tokens=action_stop_tokens)
                        elif isinstance(content, list):
                            # Enhance text elements in list format
                            for i, content_item in enumerate(content):
                                if isinstance(content_item, dict) and content_item.get("type") == "text":
                                    content_item["text"] = self._enhance_system_prompt_with_react(content_item["text"], action_stop_tokens=action_stop_tokens)
                                elif isinstance(content_item, str):
                                    message["content"][i] = {"type": "text", "text": self._enhance_system_prompt_with_react(content_item, action_stop_tokens=action_stop_tokens)}
        
        # Build rollout messages (will deepcopy the enhanced messages from row_dict)
        rollout_messages = self._build_rollout_messages(row_dict)
        
        # Clean up any debug calls that might have been stored
        row_dict.pop('_get_image_source_debug_calls', None)
        
        result = super().__getitem__(item)
        result['rollout_messages'] = rollout_messages
        
        # Store image URLs in extra_info so tools can access them
        if 'image_urls' in row_dict:
            result['image_urls'] = row_dict['image_urls']
            # Add image URLs to extra_info if it exists
            if 'extra_info' in result and isinstance(result['extra_info'], dict):
                result['extra_info']['image_urls'] = row_dict['image_urls']
            elif 'extra_info' not in result:
                result['extra_info'] = {'image_urls': row_dict['image_urls']}
        
        return result
    
    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    images = (
                        [process_image(image) for image in doc[image_key]] if image_key in doc else None # changed to get images from doc
                    )
                    videos = (
                        [process_video(video) for video in doc[video_key]] if video_key in doc else None # changed to get videos from doc
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe
    
    def _is_url(self, path: str) -> bool:
        """Check if a path is a URL."""
        if not isinstance(path, str):
            return False
        return (path.startswith("http://") or 
                path.startswith("https://") or 
                path.startswith("data:image") or 
                path.startswith("data:video"))
    
    def _get_image_source(self, example: dict, idx: int) -> Tuple[str, Optional[str]]:
        """Get image source (local path or URL) and optional URL.
        
        This method prioritizes local file paths from the 'images' field for model training.
        URLs from 'image_urls' are stored separately for tools (e.g., image_to_text search).
        
        Returns:
            tuple: (image_source, image_url) where:
                - image_source: The local file path to use for processing/encoding (for model training)
                - image_url: The URL if available (for tools only, not for training)
        """
        image_key = self.image_key
        image_urls_key = 'image_urls'
        
        # Get local path from images field - this is what we use for model training
        local_path = None
        if image_key in example:
            image_key_value = example[image_key]
            if image_key_value is not None:
                if isinstance(image_key_value, list) and idx < len(image_key_value):
                    img_dict = image_key_value[idx]
                    if isinstance(img_dict, dict) and "image" in img_dict:
                        local_path = img_dict["image"]
                    elif isinstance(img_dict, str):
                        local_path = img_dict
        
        # Get URL from image_urls field - this is only for tools, not for training
        url = None
        if image_urls_key in example:
            image_urls = example[image_urls_key]
            if isinstance(image_urls, list) and idx < len(image_urls):
                url = image_urls[idx]
        
        # Always prefer local path from images field for training
        if local_path:
            return local_path, url
        else:
            raise ValueError(f"No image source available at index {idx}.")
    
    def _build_messages(self, example: dict):
        """Override _build_messages to match base class behavior.
        
        The base class expects messages with {"type": "image"} items (without paths),
        and processes images separately from row_dict[self.image_key].
        We need to ensure images remain in example after this method is called.
        """
        messages: list = example.pop(self.prompt_key)
        
        # Get action_stop_tokens from config if available
        action_stop_tokens = None
        if hasattr(self, 'config') and self.config:
            # Try different possible config paths
            if hasattr(self.config, 'agent') and hasattr(self.config.agent, 'action_stop_tokens'):
                action_stop_tokens = self.config.agent.action_stop_tokens
            elif hasattr(self.config, 'action_stop_tokens'):
                action_stop_tokens = self.config.action_stop_tokens
            elif isinstance(self.config, dict):
                action_stop_tokens = self.config.get('agent', {}).get('action_stop_tokens') or self.config.get('action_stop_tokens')
        
        # Enhance system prompt in messages (in case it wasn't enhanced in __getitem__)
        # This ensures the enhanced system prompt is used for tokenization
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                content = message.get("content", "")
                if isinstance(content, str):
                    message["content"] = self._enhance_system_prompt_with_react(content, action_stop_tokens=action_stop_tokens)
                elif isinstance(content, list):
                    # Enhance text elements in list format
                    for i, content_item in enumerate(content):
                        if isinstance(content_item, dict) and content_item.get("type") == "text":
                            content_item["text"] = self._enhance_system_prompt_with_react(content_item["text"], action_stop_tokens=action_stop_tokens)
                        elif isinstance(content_item, str):
                            message["content"][i] = {"type": "text", "text": self._enhance_system_prompt_with_react(content_item, action_stop_tokens=action_stop_tokens)}

        # Collect image URLs to add to user messages
        image_urls_to_add = []
        if 'image_urls' in example:
            image_urls = example['image_urls']
            if isinstance(image_urls, list):
                image_urls_to_add = [url for url in image_urls if url and isinstance(url, str)]
        
        if self.image_key in example or self.video_key in example:
            segment_idx = defaultdict(int)
            for message in messages:
                content = message.get("content", "")
                # Only process string content (list content is already in the right format)
                if isinstance(content, str):
                    content_list = []
                    segments = re.split("(<image>|<video>)", content)
                    segments = [item for item in segments if item != ""]
                    for segment in segments:
                        if segment == "<image>":
                            # Base class expects {"type": "image"} without path
                            # Images will be processed separately from row_dict[self.image_key]
                            content_list.append({"type": "image"})
                            segment_idx[segment] += 1
                        elif segment == "<video>":
                            content_list.append({"type": "video"})
                            segment_idx[segment] += 1
                        else:
                            content_list.append({"type": "text", "text": segment})
                    message["content"] = content_list
        
        # Extract question text once from the first user message (research question is typically in the initial prompt)
        question_text = None
        image_url = "http://image.png"  # Default placeholder
        if image_urls_to_add and len(image_urls_to_add) > 0:
            image_url = image_urls_to_add[0]
        
        # Find the first user message and extract the question
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content", "")
                is_list_content = isinstance(content, list)
                
                # Extract text content from message
                text_content = ""
                if is_list_content:
                    # Find text elements
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "") + "\n"
                        elif isinstance(item, str):
                            text_content += item + "\n"
                elif isinstance(content, str):
                    text_content = content
                
                # Extract question text if present - support multiple formats
                # For multiple choice questions, include the Options section as well
                if "Research Question:" in text_content:
                    parts = text_content.split("Research Question:")
                    if len(parts) > 1:
                        # Extract everything after "Research Question:" including Options if present
                        question_text = parts[1].strip()
                        break
                elif "Based on the provided image" in text_content or "Based on the image" in text_content:
                    # Extract question from "Based on the provided image, ..." format
                    # Include everything including Options section if present
                    question_match = re.search(r'Based on (?:the provided image|the image)[,\s]+(.+)', text_content, re.IGNORECASE | re.DOTALL)
                    if question_match:
                        question_text = question_match.group(1).strip()
                        break
                elif "research question" in text_content.lower():
                    # Try to extract question from various formats
                    # Include everything after "research question:" including Options if present
                    question_match = re.search(r'research question[:\s]+(.+)', text_content, re.IGNORECASE | re.DOTALL)
                    if question_match:
                        question_text = question_match.group(1).strip()
                        break
                
                # If no question extracted but we have substantial text, use the entire text content
                # This handles cases where the question doesn't match any of the above patterns
                if not question_text and text_content.strip():
                    # Use the entire text content (may include Options for multiple choice)
                    question_text = text_content.strip()
                    break
        
        # Enhance user messages with JSON format requirements if we found a research question
        # and add image URLs
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content", "")
                is_list_content = isinstance(content, list)
                
                # Extract text content from message
                text_content = ""
                if is_list_content:
                    # Find text elements
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "") + "\n"
                        elif isinstance(item, str):
                            text_content += item + "\n"
                elif isinstance(content, str):
                    text_content = content
                
                # Check if JSON prompt is already present to avoid duplication
                if "Based on the research question and above actions and findings" in text_content:
                    continue
                
                # If no question text found, skip enhancement
                if not question_text:
                    continue
                
                # Build enhanced user prompt with ReAct JSON format
                enhanced_text_content = self._build_user_prompt_with_react(question_text, image_url)
                
                # Update the user message content
                if is_list_content:
                    # Find the last text element and update/append
                    last_text_idx = None
                    for idx in range(len(content) - 1, -1, -1):
                        if isinstance(content[idx], dict) and content[idx].get("type") == "text":
                            last_text_idx = idx
                            break
                    
                    if last_text_idx is not None:
                        # Update the last text element with enhanced content
                        content[last_text_idx]["text"] = enhanced_text_content
                    else:
                        # Add new text element with full prompt
                        content.append({"type": "text", "text": enhanced_text_content})
                else:
                    # Convert string to list format and add JSON prompt
                    message["content"] = [
                        {"type": "text", "text": enhanced_text_content}
                    ]

        return messages
    
    def _build_user_prompt_with_react(self, question_text: str, image_url: str) -> str:
        """
        Build enhanced user prompt with ReAct JSON format requirements.
        
        Args:
            question_text: The research question text
            image_url: Image URL to include in the prompt
            
        Returns:
            Enhanced text content string with Research Question format and JSON prompt
        """
        # Build the prompt with Research Question and image URL
        image_url_line = f"The image url is {image_url}."
        prompt = f"Research Question: {question_text}\n{image_url_line}"
        
        # Add JSON prompt suffix
        json_prompt_suffix = f"""

Based on the research question and above actions and findings, what should be the next step?
- If you have enough information to answer the question, set should_stop: true
- If you need more information, choose the most appropriate action
- Ensure action_parameters are correctly formatted for the chosen action_type
- AVOID repeating the same tool with the same query 
- diversify your research approach. 

Respond with a JSON object containing:
{{{{
    "reasoning": "Your analysis of current state and what's needed next",
    "action": {{{{
        "action_type": "action_type",
        "action_description": "What this action will do",
        "action_parameters": {{{{"param": "value"}}}},
        "expected_outcome": "What you expect to learn"
    }}}},
    "should_stop": false,
    "confidence": 0.8
}}}}

Set "should_stop": true only if you have sufficient information to answer the research question comprehensively."""
        
        return prompt + json_prompt_suffix
    
    @staticmethod
    def parse_json_response(response_text: str) -> Optional[dict]:
        """
        Parse JSON response from model that contains reasoning, action, should_stop, and confidence.
        
        This method extracts a JSON object from the model's text response, which may contain
        additional text before or after the JSON.
        
        Args:
            response_text: Raw text response from the model
            
        Returns:
            Parsed JSON dict with keys: reasoning, action, should_stop, confidence
            Returns None if parsing fails
        """
        if not response_text:
            return None
        
        try:
            # Try to find JSON object in the response
            # Look for the first { that starts a JSON object
            json_start = response_text.find('{')
            if json_start == -1:
                logger.debug("No JSON object found in response")
                return None
            
            # Find the matching closing brace
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end == -1:
                logger.debug("Could not find matching closing brace for JSON object")
                return None
            
            # Extract and parse JSON
            json_str = response_text[json_start:json_end]
            parsed = json.loads(json_str)
            
            # Validate required fields
            if not isinstance(parsed, dict):
                logger.debug("Parsed JSON is not a dictionary")
                return None
            
            # Ensure action is properly structured
            if "action" in parsed and isinstance(parsed["action"], dict):
                action = parsed["action"]
                if "action_type" not in action:
                    logger.debug("Action missing 'action_type' field")
                    return None
                if "action_parameters" not in action:
                    logger.debug("Action missing 'action_parameters' field")
                    action["action_parameters"] = {}
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON from response: {e}. Text: {response_text[:200]}...")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error parsing JSON: {e}")
            return None
    
    def _enhance_system_prompt_with_react(self, system_content: str, enabled_tools: Optional[List[str]] = None, action_stop_tokens: Optional[str] = None) -> str:
        """
        Enhance system prompt with new ReAct instructions that use JSON format for reasoning and actions.
        This makes it clear that the model should respond with JSON containing reasoning, action, should_stop, and confidence.
        Also dynamically injects tool descriptions based on enabled tools.
        
        Args:
            system_content: Original system prompt content
            enabled_tools: List of enabled tool names. If None, tries to infer from action_stop_tokens or uses all tools.
            action_stop_tokens: Optional comma-separated string of stop tokens to infer enabled tools.
        """
        # Check if the prompt already has enhanced instructions (avoid double enhancement)
        if "You are an expert research assistant using the ReAct" in system_content or "Available actions and their required parameters" in system_content:
            # Even if already enhanced, we should update tool descriptions if they exist
            # Check if we need to replace tool descriptions
            if "Available actions and their required parameters" in system_content:
                # Generate new tool descriptions
                tool_descriptions = self._generate_tool_descriptions(enabled_tools, action_stop_tokens)
                if tool_descriptions:
                    # Replace existing tools section
                    import re
                    # Pattern to match the "Available actions" section
                    pattern = r'(Available actions and their required parameters:\n)(.*?)(\n\nSTOPPING CRITERIA)'
                    replacement = f"Available actions and their required parameters:\n{tool_descriptions}\n\nSTOPPING CRITERIA"
                    if re.search(pattern, system_content, re.DOTALL):
                        system_content = re.sub(pattern, replacement, system_content, flags=re.DOTALL)
            return system_content
        
        # Generate dynamic tool descriptions
        tool_descriptions = self._generate_tool_descriptions(enabled_tools, action_stop_tokens)
        
        # Create new enhanced ReAct instructions with JSON format
        enhanced_instructions = f"""You are an expert research assistant using the ReAct (Reasoning and Acting) pattern. 

Your task is to:
1. Analyze the current research state and what has been learned
2. Reason about what information is still needed to answer the question
3. Decide on the next action to take (or conclude if sufficient)
4. Ensure action parameters are properly formatted

IMPORTANT: You have access to both text and image information. The image is provided with each request and contains visual information relevant to the research question.

AVOID REDUNDANCY: Do NOT use the same tool consecutively with the same or highly similar query/parameters.

Available actions and their required parameters:
{tool_descriptions}

STOPPING CRITERIA - Set "should_stop": true if:
- You have sufficient information to answer the research question comprehensively
- You have analyzed both the image and gathered relevant textual information
- You have tried 3+ different approaches and have good coverage
- Further searches would be redundant"""
        
        # Replace the entire original system prompt with enhanced ReAct instructions
        # The old system prompt contains:
        # - "You are an expert multimodal research assistant..." introduction
        # - "# Tools" section (outdated tool descriptions)
        # - "# Instructions" section (old instructions)
        # All of this is replaced by the new ReAct instructions above
        # Return only the enhanced ReAct instructions
        enhanced_system = enhanced_instructions
        
        # Log the enhanced prompt once per dataset (to avoid excessive logging)
        if not VerlToolRLHFDatasetMM._enhanced_prompt_logged:
            logger.info("=" * 80)
            logger.info("ENHANCED SYSTEM PROMPT WITH REACT INSTRUCTIONS")
            logger.info("=" * 80)
            logger.info(f"\n{enhanced_system}")
            logger.info("=" * 80)
            VerlToolRLHFDatasetMM._enhanced_prompt_logged = True
        
        return enhanced_system
    
    def _build_rollout_messages(self, example: dict):
        # Deepcopy messages (which should already be enhanced from __getitem__)
        messages = deepcopy(example[self.prompt_key])
        
        # Note: System prompt enhancement is now done in __getitem__ before this method is called
        # The deepcopy above will include the enhanced system prompt

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                # If content is already a list, skip processing (it's already in the right format)
                if isinstance(content, list):
                    continue
                # Only process string content
                if not isinstance(content, str):
                    continue
                    
                content_list = []
                try:
                    segments = re.split("(<image>|<video>)", content)
                except Exception as e:
                    raise ValueError(f"Error splitting content: {content}") from e
                segments = [item for item in segments if item != ""]
                segment_idx = defaultdict(int)
                for segment in segments:
                    if segment == "<image>":
                        # Get image source (prefer local, fallback to URL)
                        image_source, image_url = self._get_image_source(example, segment_idx[segment])
                        content_list.append({
                            "type": "image", 
                            "image": image_source,
                            "image_url": image_url  # Store URL separately for tools
                        })
                        segment_idx[segment] += 1
                    elif segment == "<video>":
                        # For videos, use the same logic
                        if self.video_key in example and segment_idx[segment] < len(example[self.video_key]):
                            video_dict = example[self.video_key][segment_idx[segment]]
                            if isinstance(video_dict, dict) and "video" in video_dict:
                                video_source = video_dict["video"]
                            elif isinstance(video_dict, str):
                                video_source = video_dict
                            else:
                                raise ValueError(f"Invalid video format at index {segment_idx[segment]}")
                        else:
                            raise ValueError(f"No video available at index {segment_idx[segment]}")
                        content_list.append({"type": "video", "video": video_source})
                        segment_idx[segment] += 1
                    else:
                        # Preserve the enhanced text (including system prompt enhancements)
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list
        
        # Ensure all system messages are in list format (even if example has no images)
        # This ensures the enhanced system prompt is preserved in a consistent format
        for message in messages:
            if message.get("role") == "system" and isinstance(message.get("content"), str):
                # Convert string system message to list format to preserve enhanced text
                message["content"] = [{"type": "text", "text": message["content"]}]

        if self.processor is not None:
            # multi-modal inputs
            from verl_tool.agent_loop.vision_utils import encode_image_url, encode_video_url
            for i, message in enumerate(messages):
                if isinstance(message['content'], list):
                    # Collect image URLs to add to text content
                    image_urls_to_add = []
                    
                    for j in range(len(message['content'])):
                        content = message['content'][j]
                        if content['type'] == 'image':
                            # image_source should be the local path from 'images' field (for model training)
                            image_source = content['image']
                            # image_url is from 'image_urls' field (for tools only, e.g., image_to_text search)
                            image_url = content.get('image_url')
                            
                            # Verify local path exists (only check for local paths, not URLs)
                            if not self._is_url(image_source):
                                assert Path(image_source).exists(), f"Image file {image_source} does not exist."
                            
                            # Encode the local path for model training
                            # This will load the image from the local file path and encode it to base64
                            message['content'][j] = {
                                "type": "image_url",
                                "image_url": {
                                    "url": encode_image_url(image_source),  # Uses local path for encoding
                                }
                            }
                            # Store the URL separately for tools (e.g., image_to_text search)
                            # This URL is NOT used for model training, only for tools
                            if image_url:
                                message['content'][j]['image_url']['original_url'] = image_url
                                # Collect URL to add to text content later (as plain text, not encoded)
                                image_urls_to_add.append(image_url)
                                
                        elif content['type'] == 'video':
                            video_source = content['video']
                            # Only check file existence for local paths, not URLs
                            if not self._is_url(video_source):
                                assert Path(video_source).exists(), f"Video file {video_source} does not exist."
                            message['content'][j] = {
                                "type": "video_url",
                                "video_url": {
                                    "url": encode_video_url(video_source),
                                }
                            }
                        elif content['type'] == 'text':
                            message['content'][j] = {
                                "type": "text",
                                "text": content['text']
                            }
                        else:
                            raise ValueError(f"Unknown content element type: {content['type']}")
                    
                    # Add image URLs to text content so the model can see them (as plain text, not encoded)
                    # Deduplicate URLs to avoid adding the same URL multiple times
                    # Append to the last text element, or add a new text element at the end
                    if image_urls_to_add:
                        # Remove duplicates while preserving order
                        unique_urls = []
                        seen = set()
                        for url in image_urls_to_add:
                            if url and url not in seen:
                                unique_urls.append(url)
                                seen.add(url)
                        
                        url_text_parts = []
                        for i, url in enumerate(unique_urls, 1):
                            url_text_parts.append(f"Image URL {i}: {url}")
                        url_text = "\n\n" + "Available image URLs for reverse image search:\n" + "\n".join(url_text_parts) + "\n\nWhen using <image_search_text>, use one of these actual URLs instead of a placeholder."
                        
                        # Find the last text element in the message
                        last_text_idx = None
                        for idx in range(len(message['content']) - 1, -1, -1):
                            if message['content'][idx].get('type') == 'text':
                                last_text_idx = idx
                                break
                        
                        if last_text_idx is not None:
                            # Append to existing text element
                            message['content'][last_text_idx]['text'] += url_text
                        else:
                            # Add a new text element at the end
                            message['content'].append({
                                "type": "text",
                                "text": url_text
                            })
                            
                elif isinstance(message['content'], str):
                    message['content'] = [{"type": "text", "text": message['content']}]
                else:
                    raise ValueError(f"Unknown content type: {type(message['content'])}")
                    
        return RolloutMessagesMixin(messages)

