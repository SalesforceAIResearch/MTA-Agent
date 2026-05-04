#!/usr/bin/env python
"""
Test script for ipython_code tool.

Run the server first, e.g.:
    python -m verl_tool.servers.serve \
        --tool_type ipython_code \
        --host=127.0.0.2 \
        --port=8002

Then execute:
    python -m verl_tool.servers.tests.test_ipython_code python \
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

def test_python(
    url: str = "http://127.0.0.2:8002/get_observation",
    trajectory_id: str = "test-ipython-001",
):
    """Test IPython code execution tool"""
    
    print("--- Testing 1: Basic Python code execution ---")
    action = """<python>print("Hello from Python!")</python>"""
    print(_send_test_request(url, trajectory_id, action, "Basic Python"))
    
    print("--- Testing 2: Code block format ---")
    action = """```python
import math
result = math.sqrt(16)
print(f"Square root of 16 is {result}")
```"""
    print(_send_test_request(url, trajectory_id, action, "Code Block Format"))
    
    print("--- Testing 3: Variable persistence across calls ---")
    action1 = """<python>x = 10</python>"""
    print(_send_test_request(url, trajectory_id, action1, "Set Variable"))
    
    action2 = """<python>print(f"x is {x}")</python>"""
    print(_send_test_request(url, trajectory_id, action2, "Use Variable"))
    
    print("--- Testing 4: List operations ---")
    action = """```python
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(f"Original: {numbers}")
print(f"Squared: {squared}")
```"""
    print(_send_test_request(url, trajectory_id, action, "List Operations"))
    
    print("--- Testing 5: Dictionary operations ---")
    action = """<python>
data = {"name": "Python", "version": "3.9", "type": "programming language"}
for key, value in data.items():
    print(f"{key}: {value}")
</python>"""
    print(_send_test_request(url, trajectory_id, action, "Dictionary Operations"))
    
    print("--- Testing 6: Function definition and call ---")
    action = """```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(7)
print(f"Fibonacci(7) = {result}")
```"""
    print(_send_test_request(url, trajectory_id, action, "Function Definition"))
    
    print("--- Testing 7: Error handling ---")
    action = """<python>
print("This should work")
result = 1 / 0  # This will cause an error
print("This should not execute")
</python>"""
    print(_send_test_request(url, trajectory_id, action, "Error Handling"))
    
    print("--- Testing 8: Import libraries ---")
    action = """```python
import datetime
now = datetime.datetime.now()
print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
```"""
    print(_send_test_request(url, trajectory_id, action, "Import Libraries"))
    
    print("--- Testing 9: Tool call format ---")
    action = """<tool_call>
import json
data = {"test": "value", "number": 42}
json_str = json.dumps(data)
print(json_str)
</tool_call>"""
    print(_send_test_request(url, trajectory_id, action, "Tool Call Format"))
    
    print("--- Testing 10: Complex calculation ---")
    action = """<python>
import math
# Calculate area of a circle
radius = 5
area = math.pi * radius ** 2
circumference = 2 * math.pi * radius
print(f"Circle with radius {radius}:")
print(f"  Area: {area:.2f}")
print(f"  Circumference: {circumference:.2f}")
</python>"""
    print(_send_test_request(url, trajectory_id, action, "Complex Calculation"))
    
    # print("--- Testing 11: Invalid format (should fail gracefully) ---")
    # action = """This is not valid Python code"""
    # print(_send_test_request(url, trajectory_id, action, "Invalid Format"))
    
    return True

def _send_test_request(url, trajectory_id, action, test_name):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name}...")
    print(f"Sending action: {action[:150]}...")
    
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
        python -m verl_tool.servers.tests.test_ipython_code python --url=http://127.0.0.2:8002/get_observation
    """
    fire.Fire({
        "python": test_python,
    })

if __name__ == "__main__":
    main()

