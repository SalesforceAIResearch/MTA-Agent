#!/usr/bin/env python
"""
Test script for web_text_to_text_search tool.

Run the server first, e.g.:
    python -m verl_tool.servers.serve \
        --tool_type web_text_to_text_search \
        --host=127.0.0.2 \
        --port=8002

Then execute:
    python -m verl_tool.servers.tests.test_web_text_to_text_search search \
        --url=http://127.0.0.2:8002/get_observation
"""

import json
import requests
import fire
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_search(
    url: str = "http://127.0.0.2:8002/get_observation",
    trajectory_id: str = "test-search-001",
):
    """Test web text-to-text search tool"""
    
    print("--- Testing 1: Basic search query ---")
    action = """<text_search_text>Python programming language</text_search_text>"""
    print(_send_test_request(url, trajectory_id, action, "Basic Search"))
    
    print("--- Testing 2: Code block format ---")
    action = """```text_search_text
machine learning algorithms
```"""
    print(_send_test_request(url, trajectory_id, action, "Code Block Format"))
    
    print("--- Testing 3: Alternative format ---")
    action = """text_search_text: artificial intelligence"""
    print(_send_test_request(url, trajectory_id, action, "Alternative Format"))
    
    print("--- Testing 4: Complex query ---")
    action = """<text_search_text>What is the latest version of PyTorch and its new features?</text_search_text>"""
    print(_send_test_request(url, trajectory_id, action, "Complex Query"))
    
    print("--- Testing 5: News search ---")
    action = """<text_search_text>latest news about quantum computing 2024</text_search_text>"""
    print(_send_test_request(url, trajectory_id, action, "News Search"))
    
    print("--- Testing 6: Technical search ---")
    action = """```text_search_text
How to implement async/await in Python
```"""
    print(_send_test_request(url, trajectory_id, action, "Technical Search"))
    
    print("--- Testing 7: Short query ---")
    action = """<text_search_text>Python</text_search_text>"""
    print(_send_test_request(url, trajectory_id, action, "Short Query"))
    
    # print("--- Testing 8: Invalid format (should fail gracefully) ---")
    # action = """This is not a valid search format"""
    # print(_send_test_request(url, trajectory_id, action, "Invalid Format"))
    
    return True

def _send_test_request(url, trajectory_id, action, test_name):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name}...")
    print(f"Sending action: {action[:100]}...")
    
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [{}]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Response received for {test_name}")
        
        # Print observation
        if "observations" in result and len(result["observations"]) > 0:
            observation = result["observations"][0]
            # Truncate long observations for readability
            obs_str = str(observation)
            if len(obs_str) > 500:
                obs_str = obs_str[:500] + "... (truncated)"
            logger.info(f"\n--- {test_name} Result ---\n{obs_str}\n")
        else:
            logger.error(f"No observation found in response for {test_name}")
        
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": str(e)}

def main():
    """Main entry point for the test script
    Run with:
        python -m verl_tool.servers.tests.test_web_text_to_text_search search --url=http://127.0.0.2:8002/get_observation
    """
    fire.Fire({
        "search": test_search,
    })

if __name__ == "__main__":
    main()

