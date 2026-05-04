#!/usr/bin/env python3
"""
Extract image_search entries from rollout JSONL files.
Extracts query and response text for each image_search call.
Output format: {query: response}
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
        "search failed",
        "tavily api error",
        "api error",
        "execution failed",
        "empty search query provided",
        "search query too long",
        "invalid search query",
        "error:",
        "exception:",
        "failed:",
        "timeout",
        "timed out",
        "rate limit",
        "quota exceeded",
        "invalid",
        "not found",
        "no search results found",
        "no image results found",
        "no image results",
    ]
    
    # Check if response starts with or contains error patterns
    for pattern in error_patterns:
        if pattern in response_lower:
            # Always ignore "No image results" regardless of length
            if pattern == "no image results":
                return False
            # Allow "No search results found" if it's part of a longer valid response
            if pattern == "no search results found" and len(response_lower) > 50:
                continue
            if pattern == "no image results found" and len(response_lower) > 50:
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


def extract_image_search_from_line(line: str, data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Extract image_search entries from a single JSONL line.
    Returns a list of dictionaries with query, question, and response text.
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
    
    # Check tool_interact_info - this is the primary source
    if "tool_interact_info" in data and isinstance(data["tool_interact_info"], list):
        for tool_info in data["tool_interact_info"]:
            if isinstance(tool_info, dict):
                obs_list = tool_info.get("obs", [])
                action_str = tool_info.get("action", "")
                
                # Check if action contains <text_search_image> tag
                has_image_search = False
                if isinstance(action_str, str) and "<text_search_image>" in action_str:
                    has_image_search = True
                
                if has_image_search:
                    # Extract query from action field
                    query = None
                    response_text = None
                    
                    if isinstance(action_str, str):
                        # Extract content between tags
                        content_match = re.search(r'<text_search_image>(.*?)</text_search_image>', action_str, re.DOTALL)
                        if content_match:
                            query = content_match.group(1).strip().lower()
                    
                    # Extract response from obs - join all obs items and extract everything
                    # Join all obs items together, filtering out empty strings
                    full_obs_text = "\n".join([str(item) for item in obs_list if isinstance(item, str) and str(item).strip()])
                    
                    # Remove <result> and </result> tags
                    full_obs_text = full_obs_text.replace('<result>', '').replace('</result>', '')
                    
                    # Look for "Response:" pattern and extract everything after it
                    if "Response:" in full_obs_text:
                        parts = full_obs_text.split("Response:", 1)
                        if len(parts) > 1:
                            # Get everything after "Response:"
                            response_text = parts[1].strip()
                    else:
                        # If "Response:" not found, use the full obs text
                        response_text = full_obs_text.strip()
                    
                    # Only add if we have both query and response
                    if query:
                        # Build cache_key with question: query||question or just query
                        if question:
                            cache_key = f"{query}||{question.lower()}"
                        else:
                            cache_key = query
                        
                        result = {
                            "query": query,
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
    Process a single rollout JSONL file and extract all image_search entries.
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
                results = extract_image_search_from_line(json_str, obj)
                if results:
                    all_results.extend(results)
                    entry_count += len(results)
                    print(f"  Found {len(results)} image_search entry/entries (entry #{entry_count})")
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
    Main function to extract image_search entries from rollout files.
    """
    parser = argparse.ArgumentParser(
        description="Extract image_search entries from rollout JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default verl_step_records directory
  python extract_image_search.py
  
  # Process a single folder
  python extract_image_search.py -d verl_step_records
  
  # Process multiple folders
  python extract_image_search.py -d folder1 folder2 folder3
  
  # Specify output file
  python extract_image_search.py -d verl_step_records -o output.json
  
  # Verbose mode
  python extract_image_search.py -d verl_step_records -v
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
        default='image_search_extracted_new.json',
        help='Output JSON file path (default: image_search_extracted.json)'
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
    print(f"Total image_search entries found: {len(all_results)}")
    
    # Convert to simple dictionary format: cache_key -> response
    # cache_key format: "query||question" or "query"
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
                    print(f"  Skipping entry with no response for cache_key: {cache_key[:80]}...")
                continue
            
            # Validate response - skip if it's an error
            if not is_valid_response(response):
                skipped_count += 1
                if args.verbose:
                    print(f"  Skipping invalid response for cache_key {cache_key[:80]}...: {response[:100]}")
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
