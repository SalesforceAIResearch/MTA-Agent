#!/usr/bin/env python3
"""
Extract reverse_image_search entries from rollout JSONL files.
Extracts image_url and response text for each reverse_image_search call.
"""

import json
import re
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


def is_valid_response(response: str) -> bool:
    """
    Check if a response is valid (not an error message).
    Returns False if the response indicates an API error or failure.
    """
    if not response or not isinstance(response, str):
        return False
    
    response_lower = response.lower().strip()
    
    # Check for empty or very short responses (likely errors)
    if len(response_lower) < 10:
        return False
    
    # Check for common error patterns
    error_patterns = [
        "google lens image search failed",
        "google lens image search execution failed",
        "serpapi google lens search error",
        "search failed",
        "api error",
        "execution failed",
        "empty image url provided",
        "invalid image url format",
        "error:",
        "exception:",
        "failed:",
        "timeout",
        "timed out",
        "rate limit",
        "quota exceeded",
        "invalid",
        "not found",
        "no results found",
        "no google lens search results found",
        "no detailed information found",
        "no detailed information",
    ]
    
    # Check if response starts with or contains error patterns
    for pattern in error_patterns:
        if pattern in response_lower:
            # Always ignore "No detailed information" regardless of length
            if pattern == "no detailed information":
                return False
            # Allow "No results found" if it's part of a longer valid response
            if pattern == "no results found" and len(response_lower) > 50:
                continue
            if pattern == "no google lens search results found" and len(response_lower) > 50:
                continue
            if pattern == "no detailed information found" and len(response_lower) > 50:
                continue
            return False
    
    # Check if response looks like an error message (starts with error indicators)
    error_starters = [
        "error",
        "failed",
        "exception",
        "invalid",
        "empty",
    ]
    
    first_words = response_lower.split()[:3]
    for starter in error_starters:
        if any(word.startswith(starter) for word in first_words):
            return False
    
    # Response appears valid
    return True


def extract_question_from_input(input_data: list) -> Optional[str]:
    """Extract Research Question from input array."""
    if not input_data or not isinstance(input_data, list):
        return None
    
    # Join the input array to search through it
    full_text = "\n".join([str(item) for item in input_data])
    
    # Extract Research Question
    # Pattern: "Research Question: <question>"
    question_match = re.search(r'Research Question:\s*(.+?)(?:\n(?:The image url is|Based on the research question)|$)', full_text, re.DOTALL)
    if question_match:
        question = question_match.group(1).strip()
        # Clean up any trailing newlines or whitespace
        question = question.strip()
        return question
    
    return None


def extract_reverse_image_search_from_line(line: str, data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Extract reverse_image_search entries from a single JSONL line.
    Returns a list of dictionaries with image_url, query, question, and response text.
    """
    results = []
    
    try:
        if data is None:
            data = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return results
    
    # Extract question from input field
    input_data = data.get("input", [])
    question = extract_question_from_input(input_data)
    
    # Check output field for reverse_image_search action
    if "output" in data and isinstance(data["output"], list):
        # Join output array to reconstruct the full text
        output_text = "".join(data["output"])
        
        # Find all JSON objects in output that contain reverse_image_search
        # Look for pattern: "action_type": "reverse_image_search"
        pattern = r'"action_type"\s*:\s*"reverse_image_search"'
        matches = list(re.finditer(pattern, output_text))
        
        for match in matches:
            # Extract the JSON object containing this action
            start_pos = match.start()
            # Find the start of the JSON object (look backwards for opening brace)
            json_start = output_text.rfind("{", 0, start_pos)
            if json_start == -1:
                continue
            
            # Find the end of the JSON object
            brace_count = 0
            json_end = json_start
            for i in range(json_start, min(json_start + 5000, len(output_text))):  # Limit search range
                if output_text[i] == "{":
                    brace_count += 1
                elif output_text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end > json_start:
                try:
                    json_str = output_text[json_start:json_end]
                    action_data = json.loads(json_str)
                    
                    if isinstance(action_data, dict) and "action" in action_data:
                        action = action_data["action"]
                        if isinstance(action, dict) and action.get("action_type") == "reverse_image_search":
                            action_params = action.get("action_parameters", {})
                            image_url = action_params.get("image_url")
                            
                            # Try to extract query from action_params or action string
                            query = action_params.get("query") or action_params.get("text_query")
                            cache_key = None
                            
                            # Also check if action has a string representation with || separator
                            # Look for <image_search_text>URL||query</image_search_text> pattern in output_text
                            if image_url:
                                # Search for the action string pattern near this action
                                action_str_pattern = rf'<image_search_text>(.*?{re.escape(image_url)}.*?)</image_search_text>'
                                action_str_match = re.search(action_str_pattern, output_text, re.DOTALL)
                                if action_str_match:
                                    content = action_str_match.group(1).strip()
                                    if "||" in content:
                                        parts = content.split("||", 1)
                                        if parts[0].strip() == image_url:
                                            query = parts[1].strip()
                            
                            # Build cache_key with question: image_url||query||question or image_url||question
                            cache_key = None
                            if image_url:
                                if query:
                                    if question:
                                        cache_key = f"{image_url}||{query.lower()}||{question.lower()}"
                                    else:
                                        cache_key = f"{image_url}||{query.lower()}"
                                else:
                                    if question:
                                        cache_key = f"{image_url}||{question.lower()}"
                                    else:
                                        cache_key = image_url
                            
                            if cache_key:
                                # Find the response text that follows this action
                                response_text = None
                                
                                # First, check tool_interact_info for response
                                if "tool_interact_info" in data and isinstance(data["tool_interact_info"], list):
                                    for tool_info in data["tool_interact_info"]:
                                        if isinstance(tool_info, dict) and "obs" in tool_info:
                                            obs_list = tool_info.get("obs", [])
                                            for obs_item in obs_list:
                                                if isinstance(obs_item, str) and "Tool: reverse_image_search" in obs_item:
                                                    # Extract response text - look for "Response:" followed by text
                                                    if "Response:" in obs_item:
                                                        parts = obs_item.split("Response:", 1)
                                                        if len(parts) > 1:
                                                            response_text = parts[1].strip()
                                                            break
                                
                                # If not found in tool_interact_info, look in output
                                if not response_text:
                                    # Find response after the action in output
                                    # The response might be split across multiple array elements
                                    # Pattern: <result>Tool: reverse_image_search ... Response: ...</result>
                                    response_pattern = r'<result>Tool:\s*reverse_image_search[^R]*Response:\s*(.+?)</result>'
                                    response_match = re.search(response_pattern, output_text, re.DOTALL)
                                    if response_match:
                                        response_text = response_match.group(1).strip()
                                    else:
                                        # Try a more flexible pattern - look for the result tag and extract everything until </result>
                                        result_start = output_text.find('<result>Tool: reverse_image_search')
                                        if result_start != -1:
                                            result_end = output_text.find('</result>', result_start)
                                            if result_end != -1:
                                                result_content = output_text[result_start:result_end]
                                                # Extract response part
                                                if 'Response:' in result_content:
                                                    response_text = result_content.split('Response:', 1)[1].strip()
                                
                                result = {
                                    "image_url": image_url,
                                    "query": query.lower() if query else None,
                                    "question": question.lower() if question else None,
                                    "cache_key": cache_key,
                                    "response": response_text,
                                    "input_id": data.get("input_id"),
                                    "step": data.get("step"),
                                    "score": data.get("score"),
                                    "accuracy": data.get("accuracy")
                                }
                                results.append(result)
                except (json.JSONDecodeError, KeyError, AttributeError) as e:
                    continue
    
    # Check tool_interact_info - this is the primary source
    if "tool_interact_info" in data and isinstance(data["tool_interact_info"], list):
        for tool_info in data["tool_interact_info"]:
            if isinstance(tool_info, dict):
                obs_list = tool_info.get("obs", [])
                action_str = tool_info.get("action", "")
                
                # Check if action contains <image_search_text> tag
                has_reverse_search = False
                if isinstance(action_str, str) and "<image_search_text>" in action_str:
                    has_reverse_search = True
                
                if has_reverse_search:
                    # Extract image_url and query from action field
                    # Action might be in format: <image_search_text>URL</image_search_text> or <image_search_text>URL||query</image_search_text>
                    image_url = None
                    query = None
                    cache_key = None
                    
                    if isinstance(action_str, str):
                        # Extract content between tags
                        content_match = re.search(r'<image_search_text>(.*?)</image_search_text>', action_str, re.DOTALL)
                        if content_match:
                            content = content_match.group(1).strip()
                            # Check if it contains || separator
                            if "||" in content:
                                parts = content.split("||", 1)
                                image_url = parts[0].strip()
                                query = parts[1].strip()
                            else:
                                # Just URL, no query
                                image_url = content.strip()
                    
                    # Build cache_key with question: image_url||query||question or image_url||question
                    if image_url:
                        if query:
                            if question:
                                cache_key = f"{image_url}||{query.lower()}||{question.lower()}"
                            else:
                                cache_key = f"{image_url}||{query.lower()}"
                        else:
                            if question:
                                cache_key = f"{image_url}||{question.lower()}"
                            else:
                                cache_key = image_url
                    
                    # Extract response from obs - join all obs items and extract everything
                    # Join all obs items together, filtering out empty strings
                    full_obs_text = "\n".join([str(item) for item in obs_list if isinstance(item, str) and str(item).strip()])
                    
                    # Remove <result> and </result> tags
                    full_obs_text = full_obs_text.replace('<result>', '').replace('</result>', '')
                    
                    # Look for "Response:" pattern and extract everything after it
                    response_text = None
                    if "Response:" in full_obs_text:
                        parts = full_obs_text.split("Response:", 1)
                        if len(parts) > 1:
                            # Get everything after "Response:"
                            response_text = parts[1].strip()
                    else:
                        # If "Response:" not found, use the full obs text
                        response_text = full_obs_text.strip()
                    
                    # Only add if we have image_url (cache_key)
                    if cache_key:
                        # Check if we already have this cache_key
                        existing = any(r.get("cache_key") == cache_key for r in results)
                        if not existing:
                                result = {
                                    "image_url": image_url,
                                    "query": query.lower() if query else None,
                                    "question": question.lower() if question else None,
                                    "cache_key": cache_key,
                                    "response": response_text,
                                    "input_id": data.get("input_id"),
                                    "step": data.get("step"),
                                    "score": data.get("score"),
                                    "accuracy": data.get("accuracy")
                                }
                                results.append(result)
    
    return results


def process_rollout_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a single rollout JSONL file and extract all reverse_image_search entries.
    Handles both single-line and multi-line JSON objects.
    """
    all_results = []
    
    print(f"Processing file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse multiple JSON objects from the file
        # They are separated by complete JSON objects
        decoder = json.JSONDecoder()
        idx = 0
        line_num = 1
        entry_count = 0
        
        while idx < len(content):
            # Skip whitespace
            while idx < len(content) and content[idx].isspace():
                if content[idx] == '\n':
                    line_num += 1
                idx += 1
            
            if idx >= len(content):
                break
            
            try:
                # Try to decode a JSON object starting at idx
                obj, end_idx = decoder.raw_decode(content, idx)
                # Pass the original JSON string for response extraction
                json_str = content[idx:end_idx]
                results = extract_reverse_image_search_from_line(json_str, obj)
                if results:
                    all_results.extend(results)
                    entry_count += len(results)
                    print(f"  Found {len(results)} reverse_image_search entry/entries (entry #{entry_count})")
                idx = end_idx
            except (json.JSONDecodeError, ValueError) as e:
                # If we can't parse, skip to next potential JSON start
                # Look for next '{' that might start a new object
                next_brace = content.find('{', idx + 1)
                if next_brace == -1:
                    break
                idx = next_brace
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
    
    return all_results


def find_rollout_files(directory: str) -> List[str]:
    """
    Find all JSONL files in rollout directories.
    """
    rollout_files = []
    base_path = Path(directory)
    
    # Look for rollout directories
    for rollout_dir in base_path.rglob("rollout"):
        if rollout_dir.is_dir():
            for jsonl_file in rollout_dir.glob("*.jsonl"):
                rollout_files.append(str(jsonl_file))
    
    # Also check if the directory itself contains JSONL files
    for jsonl_file in base_path.rglob("*.jsonl"):
        if "rollout" in str(jsonl_file):
            if str(jsonl_file) not in rollout_files:
                rollout_files.append(str(jsonl_file))
    
    return sorted(rollout_files)


def main():
    """
    Main function to extract reverse_image_search entries from rollout files.
    """
    parser = argparse.ArgumentParser(
        description="Extract reverse_image_search entries from rollout JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default verl_step_records directory
  python extract_reverse_image_search.py
  
  # Process a single folder
  python extract_reverse_image_search.py -d verl_step_records
  
  # Process multiple folders
  python extract_reverse_image_search.py -d folder1 folder2 folder3
  
  # Specify output file
  python extract_reverse_image_search.py -d verl_step_records -o output.json
  
  # Verbose mode
  python extract_reverse_image_search.py -d verl_step_records -v
        """
    )
    
    parser.add_argument(
        '-d', '--directories',
        nargs='+',
        default=['verl_step_records'],
        help='One or more directories to search for rollout files (default: verl_step_records)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='reverse_image_search_extracted.json',
        help='Output JSON file path (default: reverse_image_search_extracted.json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output for each file processed'
    )
    
    args = parser.parse_args()
    
    # Find all rollout files from all specified directories
    all_rollout_files = []
    for base_dir in args.directories:
        if not os.path.exists(base_dir):
            print(f"Warning: Directory '{base_dir}' does not exist, skipping...")
            continue
        
        rollout_files = find_rollout_files(base_dir)
        if rollout_files:
            all_rollout_files.extend(rollout_files)
            if args.verbose:
                print(f"Found {len(rollout_files)} rollout file(s) in {base_dir}")
        else:
            if args.verbose:
                print(f"No rollout JSONL files found in {base_dir}")
    
    if not all_rollout_files:
        print(f"No rollout JSONL files found in any of the specified directories: {args.directories}")
        return
    
    print(f"Total: Found {len(all_rollout_files)} rollout file(s) across {len(args.directories)} directory/ies")
    print()
    
    # Process all files
    all_results = []
    for file_path in all_rollout_files:
        results = process_rollout_file(file_path)
        all_results.extend(results)
        if not args.verbose and results:
            # Only show summary if not verbose
            pass
    
    print()
    print(f"Total reverse_image_search entries found: {len(all_results)}")
    
    # Convert to simple dictionary format: cache_key -> response
    # cache_key is either "image_url" or "image_url||query"
    # Only save entries with valid responses (skip errors and None responses)
    result_dict = {}
    skipped_count = 0
    for entry in all_results:
        cache_key = entry.get('cache_key')
        response = entry.get('response')
        if cache_key:
            # Skip if no response
            if not response:
                skipped_count += 1
                if args.verbose:
                    print(f"  Skipping entry with no response for {cache_key[:80]}...")
                continue
            
            # Validate response - skip if it's an error
            if not is_valid_response(response):
                skipped_count += 1
                if args.verbose:
                    print(f"  Skipping invalid response for {cache_key[:80]}...: {response[:100]}")
                continue
            
            # Only save valid responses
            # If multiple entries have the same cache_key, keep the first valid one
            if cache_key not in result_dict:
                result_dict[cache_key] = response
    
    # Save to JSON file as a simple dictionary
    output_file = args.output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary
    if result_dict:
        print("\nSummary:")
        print(f"  Total unique cache keys with valid responses: {len(result_dict)}")
        if skipped_count > 0:
            print(f"  Skipped entries (no response or errors): {skipped_count}")
        
        # Show first few entries
        print("\nFirst 3 entries:")
        for i, (cache_key, response) in enumerate(list(result_dict.items())[:3], 1):
            print(f"\n  Entry {i}:")
            print(f"    Cache Key: {cache_key}")
            print(f"    Response: {response[:100]}..." if response else "    Response: None")


if __name__ == "__main__":
    main()
