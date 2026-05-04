#!/usr/bin/env python
"""
Test script for web_image_to_text tool.

Run the server first, e.g.:
    python -m verl_tool.servers.serve \
        --tool_type web_image_to_text \
        --host=127.0.0.2 \
        --port=8002

Then execute:
    python -m verl_tool.servers.tests.test_web_image_to_text search \
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
    trajectory_id: str = "test-image-to-text-001",
):
    """Test web image-to-text search tool"""
    
    # Sample image URLs for testing
    sample_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Python_logo_and_wordmark.svg/1920px-Python_logo_and_wordmark.svg.png"
    sample_image_url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/272px-Google_2015_logo.svg.png"
    
    print("--- Testing 1: Basic image search with URL only ---")
    action = f"""<image_search_text>{sample_image_url}</image_search_text>"""
    print(_send_test_request(url, trajectory_id, action, "Basic Image Search"))
    
    print("--- Testing 2: Image search with URL and query ---")
    action = f"""<image_search_text>{sample_image_url}||Python programming logo</image_search_text>"""
    print(_send_test_request(url, trajectory_id, action, "Image Search with Query"))
    
    print("--- Testing 3: Code block format with query ---")
    action = f"""```image_search_text
{sample_image_url2}||Google logo
```"""
    print(_send_test_request(url, trajectory_id, action, "Code Block Format"))
    
    print("--- Testing 4: Alternative format ---")
    action = f"""image_search_text: {sample_image_url}||What is this image?"""
    print(_send_test_request(url, trajectory_id, action, "Alternative Format"))
    
    print("--- Testing 5: Image search with descriptive query ---")
    action = f"""<image_search_text>{sample_image_url}||Identify the programming language logo in this image</image_search_text>"""
    print(_send_test_request(url, trajectory_id, action, "Descriptive Query"))
    
    print("--- Testing 6: Image URL without query ---")
    action = f"""<image_search_text>{sample_image_url2}</image_search_text>"""
    print(_send_test_request(url, trajectory_id, action, "URL Without Query"))
    
    # print("--- Testing 7: Invalid format (should fail gracefully) ---")
    # action = """This is not a valid image search format"""
    # print(_send_test_request(url, trajectory_id, action, "Invalid Format"))
    
    # print("--- Testing 8: Invalid URL (should fail gracefully) ---")
    # action = """<image_search_text>not-a-valid-url</image_search_text>"""
    # print(_send_test_request(url, trajectory_id, action, "Invalid URL"))
    
    # print("--- Testing 9: Non-existent image URL (should handle gracefully) ---")
    # action = """<image_search_text>https://this-image-does-not-exist-12345.com/image.jpg||test query</image_search_text>"""
    # print(_send_test_request(url, trajectory_id, action, "Non-existent Image URL"))
    
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
        python -m verl_tool.servers.tests.test_web_image_to_text search --url=http://127.0.0.2:8002/get_observation
    """
    fire.Fire({
        "search": test_search,
    })

if __name__ == "__main__":
    main()

