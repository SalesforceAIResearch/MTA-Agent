#!/usr/bin/env python
"""
Test script for web_url_reader tool.

Run the server first, e.g.:
    python -m verl_tool.servers.serve \
        --tool_type web_url_reader \
        --host=127.0.0.2 \
        --port=8002

Then execute:
    python -m verl_tool.servers.tests.test_web_url_reader read \
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

def test_read(
    url: str = "http://127.0.0.2:8002/get_observation",
    trajectory_id: str = "test-url-reader-001",
):
    """Test web URL reader tool"""
    
    print("--- Testing 1: Basic URL read ---")
    action = """<web_read>https://www.python.org</web_read>"""
    print(_send_test_request(url, trajectory_id, action, "Basic URL Read"))
    
    print("--- Testing 2: Code block format ---")
    action = """```web_read
https://en.wikipedia.org/wiki/Python_(programming_language)
```"""
    print(_send_test_request(url, trajectory_id, action, "Code Block Format"))
    
    print("--- Testing 3: Alternative format ---")
    action = """web_read: https://github.com"""
    print(_send_test_request(url, trajectory_id, action, "Alternative Format"))
    
    print("--- Testing 4: News article URL ---")
    action = """<web_read>https://www.bbc.com/news</web_read>"""
    print(_send_test_request(url, trajectory_id, action, "News Article URL"))
    
    print("--- Testing 5: Wikipedia article ---")
    action = """<web_read>https://en.wikipedia.org/wiki/Machine_learning</web_read>"""
    print(_send_test_request(url, trajectory_id, action, "Wikipedia Article"))
    
    print("--- Testing 6: Image URL (should handle gracefully) ---")
    action = """<web_read>https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Python_logo_and_wordmark.svg/1920px-Python_logo_and_wordmark.svg.png</web_read>"""
    print(_send_test_request(url, trajectory_id, action, "Image URL"))
    
    # print("--- Testing 7: Invalid URL format (should fail gracefully) ---")
    # action = """This is not a valid URL format"""
    # print(_send_test_request(url, trajectory_id, action, "Invalid Format"))
    
    # print("--- Testing 8: Invalid URL (should fail gracefully) ---")
    # action = """<web_read>not-a-valid-url</web_read>"""
    # print(_send_test_request(url, trajectory_id, action, "Invalid URL"))
    
    # print("--- Testing 9: Non-existent URL (should handle gracefully) ---")
    # action = """<web_read>https://this-url-does-not-exist-12345.com</web_read>"""
    # print(_send_test_request(url, trajectory_id, action, "Non-existent URL"))
    
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
        python -m verl_tool.servers.tests.test_web_url_reader read --url=http://127.0.0.2:8002/get_observation
    """
    fire.Fire({
        "read": test_read,
    })

if __name__ == "__main__":
    main()

