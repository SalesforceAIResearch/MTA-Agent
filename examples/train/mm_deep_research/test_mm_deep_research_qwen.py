#!/usr/bin/env python3
"""
Multimodal Deep Research Test Cases
Simulates actual deep research scenarios with multimodal data (text + images)
Uses Qwen2.5-VL model for tool use generation and executes via agent tools
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultimodalDeepResearchTester:
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
            "description": "Reverse image search to find text information about images. Uses SerpAPI to search for similar images and web pages containing the image, returning detailed text descriptions and context.",
            "parameters": {"image_url": "URL of image to search"},
            "format": "reverse_image_search",
            "example": '<image_search_text>https://example.com/image.jpg</image_search_text>'
        },
        "python_code": {
            "description": "Execute Python code for data analysis and computation",
            "parameters": {"code": "python code to execute"},
            "format": "code_execution",
            "example": '<python>\nprint("Hello, World!")\n</python>'
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
                 qwen_model: str = "qwen2-5-vl",
                 api_key: Optional[str] = None,
                 enabled_tools: Optional[List[str]] = None,
                 difficulty_filter: Optional[str] = "",
                 image_url_prefix: Optional[str] = None,
                 prompt_tool: Optional[str] = None,
                 runs_per_question: int = 1,
                 run_start_number: int = 1): 
        self.server_url = server_url
        self.output_dir = Path(output_dir)
        self.input_dir = Path(input_dir)
        self.qwen_model = qwen_model
        self.api_key = api_key or os.getenv("X_API_KEY")
        self.qwen_api_url = f"https://gateway.salesforceresearch.ai/{qwen_model}/process"
        self.difficulty_filter = difficulty_filter
        self.image_url_prefix = image_url_prefix
        self.prompt_tool = prompt_tool
        self.runs_per_question = runs_per_question
        self.run_start_number = run_start_number
        
        # Configure enabled tools (default to all tools)
        if enabled_tools is None:
            self.enabled_tools = ["web_text_to_text_search", "web_text_to_img_search", "web_url_reader", "web_image_to_text", "python_code", "bash_terminal", "finish"]
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
        
        # Validate API key
        if not self.api_key:
            raise ValueError(
                "API key not provided. Set X_API_KEY environment variable or pass api_key parameter.")
        
        logger.info(f"🔍 DEBUG: API key configured (length: {len(self.api_key)} chars)")
        logger.info(f"🔍 DEBUG: Qwen API URL: {self.qwen_api_url}")
        logger.info(f"🔍 DEBUG: Qwen model: {self.qwen_model}")

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
        logger.info(f"Using Qwen model: {self.qwen_model}")
        logger.info(f"Qwen API URL: {self.qwen_api_url}")
        logger.info(f"Runs per question: {self.runs_per_question}")
        if self.runs_per_question > 1:
            logger.info(f"Run numbering starts from: {self.run_start_number}")
    
    def load_multimodal_scenarios(self, filename: str = "qa_formatted.json", max_scenarios: int = 3) -> List[Dict[str, Any]]:
        """Load multimodal research scenarios from JSON file with difficulty filtering"""
        file_path = self.input_dir / filename

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
                    f"Loaded {len(scenarios)} multimodal scenarios from {filename} (limited to {max_scenarios} for testing)")
            else:
                logger.info(f"Loaded {len(scenarios)} multimodal scenarios from {filename}")

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
            logger.error(f"Invalid JSON in {filename}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []

    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for Qwen vision API"""
        try:
            logger.info(f"🔍 DEBUG: Encoding image: {image_path}")
            logger.info(f"🔍 DEBUG: Image path exists: {os.path.exists(image_path)}")
            
            if not os.path.exists(image_path):
                logger.error(f"🔍 DEBUG: Image file not found: {image_path}")
                return ""
            
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                logger.info(f"🔍 DEBUG: Image file size: {len(image_data)} bytes")
                
                encoded = base64.b64encode(image_data).decode('utf-8')
                logger.info(f"🔍 DEBUG: Base64 encoded length: {len(encoded)} chars")
                return encoded
        except Exception as e:
            logger.error(f"🔍 DEBUG: Error encoding image {image_path}: {e}")
            import traceback
            logger.error(f"🔍 DEBUG: Traceback: {traceback.format_exc()}")
            return ""
    
    def call_qwen_api(self, text: str, images: List[str] = None, max_tokens: int = 1000, temperature: float = 0.99) -> Dict[str, Any]:
        """Call Qwen2.5-VL API directly using HTTP requests"""
        try:
            # Prepare the payload
            payload = {
                "videos": [],
                "images": images or [],
                "text": text
            }
            
            # Set headers
            headers = {
                "accept": "application/json",
                "X-Api-Key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Debug logging
            logger.info(f"🔍 DEBUG: Making API call to {self.qwen_api_url}")
            logger.info(f"🔍 DEBUG: API Key (first 10 chars): {self.api_key[:10]}..." if self.api_key else "🔍 DEBUG: No API key")
            logger.info(f"🔍 DEBUG: Text length: {len(text)} characters")
            logger.info(f"🔍 DEBUG: Number of images: {len(images) if images else 0}")
            logger.info(f"🔍 DEBUG: Text preview: {text[:200]}...")
            if images:
                logger.info(f"🔍 DEBUG: First image base64 length: {len(images[0])} characters")
                logger.info(f"🔍 DEBUG: First image base64 preview: {images[0][:50]}...")
            
            # Make the request
            logger.info(f"🔍 DEBUG: Sending POST request...")
            response = requests.post(self.qwen_api_url, json=payload, headers=headers)
            
            logger.info(f"🔍 DEBUG: Response status code: {response.status_code}")
            logger.info(f"🔍 DEBUG: Response headers: {dict(response.headers)}")
            
            # Check if request was successful
            if response.status_code != 200:
                logger.error(f"🔍 DEBUG: HTTP Error {response.status_code}")
                logger.error(f"🔍 DEBUG: Response text: {response.text}")
                response.raise_for_status()
            
            # Try to parse JSON response
            try:
                result = response.json()
            except ValueError as e:  # requests uses ValueError for JSON decode errors
                logger.error(f"🔍 DEBUG: Failed to parse JSON response: {e}")
                logger.error(f"🔍 DEBUG: Raw response text: {response.text}")
                # If JSON parsing fails, treat the raw text as the response
                result = response.text
            logger.info(f"🔍 DEBUG: Raw API response: {result}")
            logger.info(f"🔍 DEBUG: Response type: {type(result)}")
            
            # Handle different response formats
            if isinstance(result, str):
                # API returned a string directly
                response_text = result
                logger.info(f"🔍 DEBUG: API returned string directly")
            elif isinstance(result, dict):
                # API returned a JSON object, extract the response field
                logger.info(f"🔍 DEBUG: Available dict keys: {list(result.keys())}")
                response_text = result.get("response", result.get("text", result.get("content", result.get("output", ""))))
                logger.info(f"🔍 DEBUG: API returned dict, extracted from field")
                
                # If still empty, try to get any string value from the dict
                if not response_text:
                    for key, value in result.items():
                        if isinstance(value, str) and len(value) > 0:
                            response_text = value
                            logger.info(f"🔍 DEBUG: Found response in key '{key}': {value[:100]}...")
                            break
            else:
                # Unexpected format, convert to string
                response_text = str(result)
                logger.info(f"🔍 DEBUG: Unexpected response format, converted to string")
            
            logger.info(f"🔍 DEBUG: Final response text: {response_text}")
            logger.info(f"🔍 DEBUG: Response text length: {len(response_text)}")
            
            # Handle empty responses
            if not response_text:
                logger.warning(f"🔍 DEBUG: Empty response received from API")
                response_text = "Empty response from API"
            
            # Create a response format similar to OpenAI's structure for compatibility
            formatted_response = {
                "choices": [{"message": {"content": response_text}}],
                "usage": {
                    "prompt_tokens": len(text.split()) * 1.3,  # Rough estimation
                    "completion_tokens": len(response_text.split()) * 1.3,  # Rough estimation
                    "total_tokens": (len(text.split()) + len(response_text.split())) * 1.3
                }
            }
            
            logger.info(f"🔍 DEBUG: Formatted response created successfully")
            return formatted_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"🔍 DEBUG: Request exception: {e}")
            logger.error(f"🔍 DEBUG: Request exception type: {type(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"🔍 DEBUG: Error response status: {e.response.status_code}")
                logger.error(f"🔍 DEBUG: Error response text: {e.response.text}")
            # Return a fallback response
            return {
                "choices": [{"message": {"content": f"Request error calling Qwen API: {str(e)}"}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        except json.JSONDecodeError as e:
            logger.error(f"🔍 DEBUG: JSON decode error: {e}")
            logger.error(f"🔍 DEBUG: Response text that failed to parse: {response.text if 'response' in locals() else 'No response'}")
            return {
                "choices": [{"message": {"content": f"JSON decode error calling Qwen API: {str(e)}"}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        except Exception as e:
            logger.error(f"🔍 DEBUG: Unexpected error calling Qwen API: {e}")
            logger.error(f"🔍 DEBUG: Error type: {type(e)}")
            import traceback
            logger.error(f"🔍 DEBUG: Traceback: {traceback.format_exc()}")
            # Return a fallback response
            return {
                "choices": [{"message": {"content": f"Error calling Qwen API: {str(e)}"}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
    
    
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
    
    def analyze_image_with_qwen(self, image_path: str, analysis_prompt: str = "Analyze this image and describe what you see in detail.") -> str:
        """Use Qwen2.5-VL API to analyze an image"""
        try:
            # Encode image to base64
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                return "Error: Could not encode image"

            # Call Qwen API with image and text
            response = self.call_qwen_api(
                text=analysis_prompt,
                images=[base64_image],
                max_tokens=1000,
                temperature=0.99
            )

            # Extract content from the formatted response
            if "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["message"]["content"]
            else:
                logger.error(f"🔍 DEBUG: Unexpected response format in image analysis: {response}")
                return str(response)

        except Exception as e:
            logger.error(f"Error analyzing image with Qwen: {e}")
            return f"Error analyzing image: {str(e)}"

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
                result = f"<image_search_text>{image_url}</image_search_text>"
                logger.debug(f"Generated reverse_image_search action: {result}")
                return result
            else:
                result = f"<python>\\nprint('Error: web_image_to_text tool is not enabled')\\n</python>"
                logger.debug(f"Generated disabled tool error: {result}")
                return result

        elif action_type == "code_execution":
            if "python_code" not in self.enabled_tools:
                result = f"<python>\nprint('Error: python_code tool is not enabled')\n</python>"
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
            if scenario and "question" in scenario and "image" in scenario["question"]:
                image_path = str(self.input_dir / "images" / scenario["question"]["image"].replace("./", ""))
                
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
            if scenario and "question" in scenario and "image" in scenario["question"]:
                image_path = str(self.input_dir / "images" / scenario["question"]["image"].replace("./", ""))

                # Get specific query from action parameters
                query = action_params.get("query", "Analyze this image and describe what you see in detail.")
                analysis_prompt = f"Based on the research question: '{scenario['question']['text']}', {query}"

                # Use Qwen to analyze the image
                analysis_result = self.analyze_image_with_qwen(image_path, analysis_prompt)

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

        trajectory = {
            "scenario_id": scenario["id"],
            "question": scenario["question"],
            "ground_truth": scenario["ground_truth"],
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
        successful_steps = 0
        failed_steps = 0

        for step in trajectory["steps"]:
            if step["success"] and step["observation"]:
                obs_text = str(step["observation"])[:500]  # Limit length
                previous_observations.append(f"✓ Step {step['step_index'] + 1}: {obs_text}")
                successful_steps += 1
            else:
                previous_observations.append(
                    f"✗ Step {step['step_index'] + 1}: Failed - {step.get('error', 'Unknown error')}")
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

        user_prompt = f"""Research Question: {scenario['question']['text']}
The image url is {scenario['question']['image_url']}.

Current Research State (Iteration {iteration}):
- Successful steps: {successful_steps}
- Failed steps: {failed_steps}
- Previous findings:
{chr(10).join(previous_observations) if previous_observations else "No previous research steps"}

Based on the research question and current findings, what should be the next step? 
- If you have enough information to answer the question, set should_stop: true
- If you need more information, choose the most appropriate action with proper parameters
- Ensure action_parameters are correctly formatted for the chosen action_type"""

        if prompt_tool:
            user_prompt += "\n" + prompt_tool

        # Prepare the full prompt text
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Prepare images if available
        images = []
        if scenario and "question" in scenario and "image" in scenario["question"]:
            image_path = str(self.input_dir / "images" / scenario["question"]["image"].replace("./", ""))
            base64_image = self.encode_image_to_base64(image_path)
            if base64_image:
                images.append(base64_image)

        try:
            logger.info(f"🔍 DEBUG: About to call Qwen API for reasoning step {iteration}")
            logger.info(f"🔍 DEBUG: Full prompt length: {len(full_prompt)} chars")
            logger.info(f"🔍 DEBUG: Images count: {len(images)}")
            
            response = self.call_qwen_api(
                text=full_prompt,
                images=images,
                max_tokens=1200,
                temperature=0.99
            )

            logger.info(f"🔍 DEBUG: Received response from Qwen API")
            logger.info(f"🔍 DEBUG: Response structure: {list(response.keys())}")
            
            # Extract content from the formatted response
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                logger.info(f"🔍 DEBUG: Extracted content from choices structure")
            else:
                logger.error(f"🔍 DEBUG: Unexpected response format: {response}")
                content = str(response)
                
            logger.info(f"🔍 DEBUG: Extracted content length: {len(content)} chars")
            logger.info(f"🔍 DEBUG: Content preview: {content[:300]}...")

            # Store token usage information
            token_usage = {
                "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": response.get("usage", {}).get("total_tokens", 0)
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
                reasoning_data["react_step_details"] = {
                    "input_text": full_prompt,
                    "input_images": images,
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
            tool_action.startswith("<python>") and tool_action.endswith("</python>"),
            tool_action.startswith("<bash>") and tool_action.endswith("</bash>"),
            tool_action.startswith("<tool_call>") and tool_action.endswith("</tool_call>")
        ]

        is_valid = any(valid_formats)
        
        if not is_valid:
            logger.warning(f"Tool action validation failed. Action: '{tool_action[:100]}...' (truncated)")
            logger.warning(f"Checked formats: text_search_text, text_search_image, image_search_text, web_read, python, bash, tool_call")
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
        extra_field = {}
        if scenario and gpt_action.get("action_type") == "image_operations":
            if "question" in scenario and "image" in scenario["question"]:
                image_path = str(self.input_dir / "images" / scenario["question"]["image"].replace("./", ""))
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
        all_observations = []
        for step in trajectory["steps"]:
            if step["success"] and step["observation"]:
                if isinstance(step["observation"], str):
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

        user_prompt = f"""Research Question: {scenario['question']['text']}
Image URL in this question is {scenario['question']['image_url']}.

Research Findings:
{combined_research}""" + """

Answer using both text and image info. Give concise step-by-step reasoning, then final answer in \\boxed{...}."""

        # Prepare the full prompt text
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Prepare images if available
        images = []
        if scenario and "question" in scenario and "image" in scenario["question"]:
            image_source = str(self.input_dir / "images" / scenario["question"]["image"].replace("./", ""))
            base64_image = self.encode_image_to_base64(image_source)
            if base64_image:
                images.append(base64_image)

        try:
            response = self.call_qwen_api(
                text=full_prompt,
                images=images,
                max_tokens=2000,
                temperature=0.99
            )

            # Extract final answer from the formatted response
            if "choices" in response and len(response["choices"]) > 0:
                final_answer = response["choices"][0]["message"]["content"]
                logger.info(f"🔍 DEBUG: Extracted final answer from choices structure")
            else:
                logger.error(f"🔍 DEBUG: Unexpected response format in final answer: {response}")
                final_answer = str(response)

            # Store token usage for final answer generation
            token_usage = {
                "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": response.get("usage", {}).get("total_tokens", 0)
            }

            logger.info(f"Generated final answer for scenario {scenario['id']} (tokens: {token_usage['total_tokens']})")
            
            # Return final answer, token usage, and step details
            final_answer_details = {
                "input_text": full_prompt,
                "input_images": images,
                "output_content": final_answer,
                "step_type": "final_answer",
                "token_usage": token_usage
            }
            
            return final_answer, token_usage, final_answer_details

        except Exception as e:
            logger.error(f"Error generating final answer for scenario {scenario['id']}: {e}")
            error_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            error_details = {
                "input_text": full_prompt,
                "input_images": images,
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
            "qwen_model": self.qwen_model,
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
            f.write(f"Qwen Model: {self.qwen_model}\n")
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

    def run_multimodal_test(self, max_scenarios: Optional[int] = None) -> Dict[str, Any]:
        """Run comprehensive multimodal deep research test"""

        logger.info("=== Starting Multimodal Deep Research Test ===")
        logger.info(f"🔍 DEBUG: Test configuration:")
        logger.info(f"🔍 DEBUG: - Server URL: {self.server_url}")
        logger.info(f"🔍 DEBUG: - Input dir: {self.input_dir}")
        logger.info(f"🔍 DEBUG: - Output dir: {self.output_dir}")
        logger.info(f"🔍 DEBUG: - Qwen model: {self.qwen_model}")
        logger.info(f"🔍 DEBUG: - API URL: {self.qwen_api_url}")
        logger.info(f"🔍 DEBUG: - Max scenarios: {max_scenarios}")

        # Load multimodal scenarios
        scenarios = self.load_multimodal_scenarios(max_scenarios=max_scenarios)
        logger.info(f"Loaded {len(scenarios)} multimodal scenarios")
        logger.info(f"🔍 DEBUG: Scenarios loaded successfully: {len(scenarios) > 0}")
        
        # Debug: Show all scenario IDs
        if scenarios:
            logger.info("Scenario IDs to be processed:")
            for i, scenario in enumerate(scenarios):
                logger.info(f"  {i+1}. {scenario['id']}")

        if not scenarios:
            logger.error("No scenarios loaded. Check input files.")
            return {"error": "No scenarios loaded"}

        # Execute all scenarios with multiple runs per question
        trajectories = []
        evaluations = []

        for i, scenario in enumerate(scenarios):
            logger.info(f"\n--- Executing Scenario {i + 1}/{len(scenarios)}: {scenario['id']} ---")
            logger.info(f"Question preview: {scenario['question']['text'][:100]}...")
            
            # Run each scenario multiple times
            for run_index in range(self.runs_per_question):
                run_num = self.run_start_number + run_index
                if self.runs_per_question > 1:
                    logger.info(f"  Run {run_num} ({run_index + 1}/{self.runs_per_question})")
                
                # Create a unique scenario ID for this run
                run_scenario = scenario.copy()
                if self.runs_per_question > 1:
                    run_scenario['id'] = f"{scenario['id']}_run_{run_num}"
                
                # Simulate research trajectory
                trajectory = self.simulate_multimodal_research_trajectory(run_scenario)
                trajectories.append(trajectory)

                # Evaluate research quality
                evaluation = self.evaluate_multimodal_research_quality(trajectory, run_scenario)
                evaluations.append(evaluation)

                # Save results immediately after each scenario completion
                self.save_individual_result(trajectory, evaluation)

                logger.info(f"  Run {run_num} - Quality Score: {evaluation['metrics']['overall_quality_score']:.3f}")
                logger.info(f"  Run {run_num} - Success Rate: {evaluation['metrics']['success_rate']:.3f}")
                logger.info(f"  Run {run_num} - Relevance Score: {evaluation['metrics']['relevance_score']:.3f}")
                logger.info(f"  Run {run_num} - Action Diversity: {evaluation['metrics']['action_diversity']:.3f}")
                
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

    def test_api_connection(self) -> bool:
        """Test basic API connection with a simple request"""
        try:
            logger.info("🔍 DEBUG: Testing API connection...")
            
            # Simple test request
            test_response = self.call_qwen_api(
                text="Hello, this is a test message. Please respond with 'API connection successful'.",
                images=[],
                max_tokens=100,
                temperature=0.1
            )
            
            logger.info(f"🔍 DEBUG: Test response received")
            logger.info(f"🔍 DEBUG: Test response content: {test_response}")
            
            content = test_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info(f"🔍 DEBUG: Test content: {content}")
            
            success = len(content) > 0 and "error" not in content.lower()
            logger.info(f"🔍 DEBUG: API connection test {'PASSED' if success else 'FAILED'}")
            
            return success
            
        except Exception as e:
            logger.error(f"🔍 DEBUG: API connection test failed: {e}")
            return False

    def test_image_inclusion(self, scenario_id: str = "MMSearch-Plus_0") -> bool:
        """Test that images are properly included in reasoning and final answer steps"""
        try:
            # Load a test scenario
            scenarios = self.load_multimodal_scenarios()
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


def main():
    """Main entry point for multimodal deep research testing

    Usage:
        python test_mm_deep_research_qwen.py run --server_url=http://localhost:4000/get_observation
        python test_mm_deep_research_qwen.py run --output_dir=my_test_results_mm
        python test_mm_deep_research_qwen.py run --input_dir=data/mmsearch_plus_processed
        python test_mm_deep_research_qwen.py run --qwen_model=qwen2-5-vl
        python test_mm_deep_research_qwen.py run --api_key=your_api_key
        python test_mm_deep_research_qwen.py run --max_scenarios=5
        python test_mm_deep_research_qwen.py run --difficulty_filter=easy
        python test_mm_deep_research_qwen.py run --enabled_tools="web_text_to_text_search,web_text_to_img_search,web_url_reader,web_image_to_text,python_code"
        python test_mm_deep_research_qwen.py run --image_url_prefix="https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/MM-BrowseComp/images/"
        python test_mm_deep_research_qwen.py run --runs_per_question=3
        python test_mm_deep_research_qwen.py run --runs_per_question=3 --run_start_number=5
        python test_mm_deep_research_qwen.py test_image --scenario_id=MMSearch-Plus_0
        python test_mm_deep_research_qwen.py test_api --api_key=your_api_key
    """
    def test_api(qwen_model="qwen2-5-vl", api_key=None):
        """Test API connection"""
        tester = MultimodalDeepResearchTester(
            qwen_model=qwen_model, 
            api_key=api_key,
            output_dir="temp_test_api"
        )
        return tester.test_api_connection()
    
    def run_test(server_url="http://localhost:4000/get_observation", 
                 output_dir="simple_test_results_mm", 
                 input_dir="data/mmsearch_plus_processed",
                 qwen_model="qwen2-5-vl",
                 api_key=None,
                 max_scenarios=3,
                 enabled_tools=None,
                 difficulty_filter="",
                 image_url_prefix=None,
                 prompt_tool=None,
                 runs_per_question=1,
                 run_start_number=1):
        # Parse enabled_tools if provided as comma-separated string
        if enabled_tools and isinstance(enabled_tools, str):
            enabled_tools = [t.strip() for t in enabled_tools.split(',')]
        
        tester = MultimodalDeepResearchTester(
            server_url, output_dir, input_dir, qwen_model, api_key, enabled_tools, difficulty_filter, image_url_prefix, prompt_tool, runs_per_question, run_start_number
        )
        result = tester.run_multimodal_test(max_scenarios)
        
        # Don't return the result to prevent Fire from printing it
        print("Test completed successfully. Results saved to output directory.")
        return None
    
    def test_image(server_url="http://localhost:4000/get_observation", 
                   output_dir="simple_test_results_mm", 
                   input_dir="data/mmsearch_plus_processed",
                   qwen_model="qwen2-5-vl",
                   api_key=None,
                   scenario_id="MMSearch-Plus_0",
                   enabled_tools=None,
                   difficulty_filter=""):
        # Parse enabled_tools if provided as comma-separated string
        if enabled_tools and isinstance(enabled_tools, str):
            enabled_tools = [t.strip() for t in enabled_tools.split(',')]
        
        tester = MultimodalDeepResearchTester(
            server_url, output_dir, input_dir, qwen_model, api_key, enabled_tools, difficulty_filter
        )
        return tester.test_image_inclusion(scenario_id)
    
    fire.Fire({
        "run": run_test,
        "test_image": test_image,
        "test_api": test_api
    })


if __name__ == "__main__":
    main()