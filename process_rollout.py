#!/usr/bin/env python3
"""
Process rollout JSONL file(s) to:
1. Extract Research Question and image URL pairs
2. Combine trajectories for same pair across all folders
3. Only consider pairs with >= min_trajectories (default 8) for exclusion
4. Output pairs with avg accuracy > threshold (default 0.75) and accuracy == 0

Supports:
- Single file processing
- Folder processing (all .jsonl files)
- Multiple folders -> one combined JSON output (trajectories combined across folders)
- Minimum trajectory threshold to filter out pairs with insufficient data

Usage examples:
  # Single file
  python process_rollout.py rollout/1.jsonl -o output.json
  
  # Single folder
  python process_rollout.py rollout/ -o rollout_summary.json
  
  # Multiple folders -> one JSON (combines all trajectories)
  python process_rollout.py folder1/rollout folder2/rollout folder3/rollout -o combined_summary.json
  
  # With minimum trajectory threshold (only pairs with >= 8 trajectories)
  python process_rollout.py folder1/rollout folder2/rollout -o summary.json -m 8
"""

import json
import re
import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import hashlib


def extract_dataset_name(image_url: str) -> str:
    """
    Extract dataset name from image URL.
    
    Example:
        "https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/News/images/..."
        -> "News"
    
    Args:
        image_url: Image URL
        
    Returns:
        Dataset name or "Unknown" if not found
    """
    # Pattern: .../mm-deepsearch-sfr-2025/<DATASET>/images/...
    match = re.search(r'/mm-deepsearch-sfr-2025/([^/]+)/', image_url)
    if match:
        return match.group(1)
    
    # Try alternative patterns
    match = re.search(r'/([A-Z][a-zA-Z]+)/images/', image_url)
    if match:
        return match.group(1)
    
    return "Unknown"


def generate_id_from_pair(question: str, image_url: str) -> str:
    """
    Generate a unique ID from question and image URL.
    
    Args:
        question: Research question text
        image_url: Image URL
        
    Returns:
        A unique identifier string
    """
    # Create a hash from question and image URL
    combined = f"{question}|{image_url}"
    hash_obj = hashlib.md5(combined.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:12]
    
    # Create a readable ID
    # Extract filename from URL if possible, otherwise use hash
    if '/' in image_url:
        filename = image_url.split('/')[-1]
        # Remove extension
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        # Clean filename for ID
        filename = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)[:30]
        return f"hard_{filename}_{hash_hex}"
    else:
        return f"hard_{hash_hex}"


def save_hard_examples_format(zero_accuracy_pairs: List[Dict], output_file: str):
    """
    Save zero accuracy pairs in the qa_formatted_test.json format, grouped by dataset.
    Each dataset gets its own file.
    
    Args:
        zero_accuracy_pairs: List of zero accuracy pair dictionaries
        output_file: Base path for output JSON files (will be modified per dataset)
    """
    # Group pairs by dataset
    dataset_groups = defaultdict(list)
    
    for pair in zero_accuracy_pairs:
        dataset_name = extract_dataset_name(pair['image_url'])
        dataset_groups[dataset_name].append(pair)
    
    # Determine base output path
    output_path = Path(output_file)
    base_dir = output_path.parent
    base_stem = output_path.stem
    
    # Remove "_hard_examples" from stem if present to get clean base name
    if base_stem.endswith('_hard_examples'):
        base_stem = base_stem[:-len('_hard_examples')]
    
    # Save each dataset to its own file
    dataset_stats = {}
    
    for dataset_name, pairs in sorted(dataset_groups.items()):
        scenarios = []
        
        for pair in pairs:
            # Get ground truth (use first one if available)
            ground_truth = ""
            if pair.get('ground_truths') and len(pair['ground_truths']) > 0:
                # If ground_truths is a list, get first element
                gt = pair['ground_truths'][0]
                if isinstance(gt, list):
                    ground_truth = gt[0] if len(gt) > 0 else ""
                else:
                    ground_truth = str(gt)
            
            # Generate ID
            scenario_id = generate_id_from_pair(pair['research_question'], pair['image_url'])
            
            # Create scenario object
            scenario = {
                "id": scenario_id,
                "question": {
                    "text": pair['research_question'],
                    "image": pair['image_url']
                },
                "ground_truth": ground_truth,
                "answer": ground_truth,
                "acceptable_answers": ground_truth,
                "question_type": "Visual Question Answering",
                "category": "Information Seeking"
            }
            
            scenarios.append(scenario)
        
        # Create output structure
        output_data = {
            "scenarios": scenarios
        }
        
        # Create dataset-specific filename
        dataset_output_file = base_dir / f"{dataset_name}_hard_examples.json"
        
        # Write to file
        dataset_output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dataset_output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Store stats
        dataset_stats[dataset_name] = {
            'file': str(dataset_output_file),
            'count': len(scenarios),
            'unique_pairs': len(pairs)
        }
    
    # Print summary
    print(f"\n{'='*80}")
    print("HARD EXAMPLES BY DATASET")
    print(f"{'='*80}")
    total_examples = 0
    total_pairs = 0
    
    for dataset_name in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset_name]
        total_examples += stats['count']
        total_pairs += stats['unique_pairs']
        print(f"\n{dataset_name}:")
        print(f"  File: {stats['file']}")
        print(f"  Total scenarios: {stats['count']}")
        print(f"  Unique question/image pairs: {stats['unique_pairs']}")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(dataset_stats)} dataset(s), {total_examples} total scenarios, {total_pairs} unique pairs")
    print(f"{'='*80}")


def extract_question_and_image_url(input_data: list) -> tuple:
    """
    Extract Research Question and image URL from the input array.
    
    Args:
        input_data: List of strings containing the conversation
        
    Returns:
        Tuple of (research_question, image_url) or (None, None) if not found
    """
    research_question = None
    image_url = None
    
    # Join the input array to search through it
    full_text = "\n".join(input_data)
    
    # Extract Research Question
    # Pattern: "Research Question: <question>"
    question_match = re.search(r'Research Question:\s*(.+?)(?:\n|$)', full_text)
    if question_match:
        research_question = question_match.group(1).strip()
    
    # Extract image URL
    # Pattern: "The image url is <url>."
    url_match = re.search(r'The image url is\s+(https?://[^\s.]+(?:\.[^\s.]+)+)', full_text)
    if url_match:
        image_url = url_match.group(1).strip()
    
    return research_question, image_url


def parse_multi_json_file(file_path: str):
    """
    Parse a file containing multiple pretty-printed JSON objects.
    Handles both standard JSONL (one JSON per line) and pretty-printed format
    where objects are separated by blank lines.
    
    Args:
        file_path: Path to the file
        
    Yields:
        Parsed JSON objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Try standard JSONL first (one JSON per line)
    if lines:
        first_line = lines[0].strip()
        if first_line and first_line.startswith('{') and first_line.endswith('}'):
            try:
                json.loads(first_line)
                # It's standard JSONL
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            pass
                return
            except json.JSONDecodeError:
                pass
    
    # Handle pretty-printed JSON objects
    # Each object starts with { on its own line (possibly after blank line)
    # and ends with } on its own line (before blank line or EOF)
    current_obj_lines = []
    brace_count = 0
    in_string = False
    escape_next = False
    
    for line in lines:
        stripped = line.strip()
        
        # Start of new object
        if not current_obj_lines and stripped == '{':
            current_obj_lines = [line]
            brace_count = 1
            continue
        
        if current_obj_lines:
            current_obj_lines.append(line)
            
            # Track braces (but not inside strings)
            for i, char in enumerate(stripped):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
            
            # End of object
            if brace_count == 0:
                obj_str = ''.join(current_obj_lines)
                try:
                    yield json.loads(obj_str)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON object: {e}")
                current_obj_lines = []
                in_string = False
                escape_next = False


def process_rollout_file(input_file: str, output_file: str = None, threshold: float = 0.75, min_trajectories: int = 1, hard_examples_output: str = None, save_all_zero_accuracy: bool = False):
    """
    Process the rollout JSONL file and output high-accuracy pairs.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output JSON file (optional, defaults to input_file_high_accuracy.json)
        threshold: Accuracy threshold (default 6/8 = 0.75)
        save_all_zero_accuracy: If True, save all zero accuracy pairs regardless of min_trajectories
    """
    # Dictionary to store results grouped by (question, image_url)
    # Key: (research_question, image_url)
    # Value: list of accuracy scores
    pair_accuracies = defaultdict(list)
    
    # Also store additional info for each pair
    pair_info = {}
    
    print(f"Processing file: {input_file}")
    
    total_entries = 0
    entries_with_pairs = 0
    
    for entry in parse_multi_json_file(input_file):
        # Skip non-dictionary entries
        if not isinstance(entry, dict):
            continue
        
        # Skip entries that don't have the expected structure
        if 'input' not in entry or 'accuracy' not in entry:
            continue
            
        total_entries += 1
        
        # Extract question and image URL
        input_data = entry.get('input', [])
        research_question, image_url = extract_question_and_image_url(input_data)
        
        if research_question and image_url:
            entries_with_pairs += 1
            pair_key = (research_question, image_url)
            
            # Get accuracy (can be 0.0 or 1.0 typically)
            accuracy = entry.get('accuracy', 0.0)
            pair_accuracies[pair_key].append(accuracy)
            
            # Store additional info (ground truth, etc.)
            if pair_key not in pair_info:
                pair_info[pair_key] = {
                    'gts': entry.get('gts', []),
                    'input_ids': []
                }
            pair_info[pair_key]['input_ids'].append(entry.get('input_id', ''))
    
    print(f"\nTotal entries processed: {total_entries}")
    print(f"Entries with valid question/image pairs: {entries_with_pairs}")
    print(f"Unique question/image pairs: {len(pair_accuracies)}")
    
    # Calculate average accuracy for each pair and filter
    high_accuracy_pairs = []
    zero_accuracy_pairs = []
    skipped_insufficient = 0
    
    for (question, image_url), accuracies in pair_accuracies.items():
        num_rollouts = len(accuracies)
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        num_correct = sum(1 for a in accuracies if a > 0)
        
        pair_result = {
            'research_question': question,
            'image_url': image_url,
            'avg_accuracy': avg_accuracy,
            'num_rollouts': num_rollouts,
            'num_correct': num_correct,
            'accuracy_ratio': f"{num_correct}/{num_rollouts}",
            'individual_accuracies': accuracies,
            'ground_truths': pair_info[(question, image_url)]['gts'],
            'input_ids': pair_info[(question, image_url)]['input_ids']
        }
        
        # For high accuracy pairs, still apply min_trajectories filter
        if avg_accuracy > threshold:
            if num_rollouts >= min_trajectories:
                high_accuracy_pairs.append(pair_result)
            else:
                skipped_insufficient += 1
        # For zero accuracy pairs, check if we should save all or filter by min_trajectories
        elif avg_accuracy == 0:
            if save_all_zero_accuracy or num_rollouts >= min_trajectories:
                zero_accuracy_pairs.append(pair_result)
            else:
                skipped_insufficient += 1
    
    if skipped_insufficient > 0:
        print(f"Skipped {skipped_insufficient} pairs with < {min_trajectories} trajectories")
    
    # Sort by average accuracy descending
    high_accuracy_pairs.sort(key=lambda x: x['avg_accuracy'], reverse=True)
    zero_accuracy_pairs.sort(key=lambda x: x['research_question'])
    
    print(f"\nPairs with avg accuracy > {threshold} ({threshold*100:.0f}%): {len(high_accuracy_pairs)}")
    print(f"Pairs with avg accuracy == 0: {len(zero_accuracy_pairs)}")
    
    # Determine output file path
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_high_accuracy.json"
    
    # Write output
    output_data = {
        'metadata': {
            'source_file': str(input_file),
            'threshold': threshold,
            'total_entries': total_entries,
            'unique_pairs': len(pair_accuracies),
            'high_accuracy_pairs_count': len(high_accuracy_pairs),
            'zero_accuracy_pairs_count': len(zero_accuracy_pairs)
        },
        'high_accuracy_pairs': high_accuracy_pairs,
        'zero_accuracy_pairs': zero_accuracy_pairs
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput written to: {output_file}")
    
    # Save hard examples in separate file if any exist
    hard_examples_with_enough_traj = [p for p in zero_accuracy_pairs if p['num_rollouts'] >= min_trajectories]
    if hard_examples_with_enough_traj:
        if hard_examples_output:
            hard_output_file = hard_examples_output
        else:
            output_path = Path(output_file)
            hard_output_file = output_path.parent / f"{output_path.stem}_hard_examples.json"
        save_hard_examples_format(hard_examples_with_enough_traj, str(hard_output_file))
    
    # Print summary of high accuracy pairs
    if high_accuracy_pairs:
        print(f"\n{'='*80}")
        print("HIGH ACCURACY PAIRS SUMMARY")
        print(f"{'='*80}")
        for i, pair in enumerate(high_accuracy_pairs[:10], 1):  # Show top 10
            print(f"\n{i}. Question: {pair['research_question'][:80]}...")
            print(f"   Image URL: {pair['image_url']}")
            print(f"   Accuracy: {pair['accuracy_ratio']} ({pair['avg_accuracy']:.2%})")
            print(f"   Ground Truth: {pair['ground_truths']}")
        
        if len(high_accuracy_pairs) > 10:
            print(f"\n... and {len(high_accuracy_pairs) - 10} more pairs")
    
    return output_data


def process_rollout_folder(
    input_folder: str, 
    output_dest: str, 
    threshold: float = 0.75,
    combine: bool = True,
    min_trajectories: int = 1,
    hard_examples_output: str = None,
    save_all_zero_accuracy: bool = False
) -> Dict[str, Any]:
    """
    Process all JSONL files in a folder and output high-accuracy pairs.
    
    Args:
        input_folder: Path to the folder containing JSONL files
        output_dest: Path to the output destination (file or folder)
        threshold: Accuracy threshold (default 6/8 = 0.75)
        combine: If True, combine all results into one file. If False, create separate files.
        
    Returns:
        Combined output data or dict of per-file outputs
    """
    input_path = Path(input_folder)
    output_path = Path(output_dest)
    
    # Find all JSONL files
    jsonl_files = list(input_path.rglob('*.jsonl'))
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_folder}")
        return {}
    
    print(f"Found {len(jsonl_files)} JSONL file(s) in {input_folder}")
    print("=" * 80)
    
    # Dictionary to aggregate results across all files
    # Key: (research_question, image_url)
    # Value: dict with accuracies, gts, input_ids, source_files
    all_pair_data = defaultdict(lambda: {
        'accuracies': [],
        'gts': [],
        'input_ids': [],
        'source_files': []
    })
    
    total_entries_all = 0
    
    for jsonl_file in sorted(jsonl_files):
        print(f"\nProcessing: {jsonl_file.name}")
        
        entry_count = 0
        for entry in parse_multi_json_file(str(jsonl_file)):
            if not isinstance(entry, dict):
                continue
            if 'input' not in entry or 'accuracy' not in entry:
                continue
            
            entry_count += 1
            input_data = entry.get('input', [])
            research_question, image_url = extract_question_and_image_url(input_data)
            
            if research_question and image_url:
                pair_key = (research_question, image_url)
                accuracy = entry.get('accuracy', 0.0)
                
                all_pair_data[pair_key]['accuracies'].append(accuracy)
                all_pair_data[pair_key]['input_ids'].append(entry.get('input_id', ''))
                all_pair_data[pair_key]['source_files'].append(str(jsonl_file))
                
                # Store ground truths (only need to do once per pair)
                if not all_pair_data[pair_key]['gts']:
                    all_pair_data[pair_key]['gts'] = entry.get('gts', [])
        
        total_entries_all += entry_count
        print(f"  -> {entry_count} entries")
    
    print("\n" + "=" * 80)
    print(f"Total entries across all files: {total_entries_all}")
    print(f"Unique question/image pairs: {len(all_pair_data)}")
    
    # Calculate average accuracy for each pair and filter
    high_accuracy_pairs = []
    zero_accuracy_pairs = []
    skipped_insufficient = 0
    
    for (question, image_url), data in all_pair_data.items():
        accuracies = data['accuracies']
        num_rollouts = len(accuracies)
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        num_correct = sum(1 for a in accuracies if a > 0)
        
        pair_result = {
            'research_question': question,
            'image_url': image_url,
            'avg_accuracy': avg_accuracy,
            'num_rollouts': num_rollouts,
            'num_correct': num_correct,
            'accuracy_ratio': f"{num_correct}/{num_rollouts}",
            'individual_accuracies': accuracies,
            'ground_truths': data['gts'],
            'input_ids': data['input_ids'],
            'source_files': list(set(data['source_files']))  # Unique source files
        }
        
        # For high accuracy pairs, still apply min_trajectories filter
        if avg_accuracy > threshold:
            if num_rollouts >= min_trajectories:
                high_accuracy_pairs.append(pair_result)
            else:
                skipped_insufficient += 1
        # For zero accuracy pairs, check if we should save all or filter by min_trajectories
        elif avg_accuracy == 0:
            if save_all_zero_accuracy or num_rollouts >= min_trajectories:
                zero_accuracy_pairs.append(pair_result)
            else:
                skipped_insufficient += 1
    
    # Sort by average accuracy descending
    high_accuracy_pairs.sort(key=lambda x: x['avg_accuracy'], reverse=True)
    zero_accuracy_pairs.sort(key=lambda x: x['research_question'])
    
    if skipped_insufficient > 0:
        print(f"Skipped {skipped_insufficient} pairs with < {min_trajectories} trajectories")
    print(f"Pairs with avg accuracy > {threshold} ({threshold*100:.0f}%): {len(high_accuracy_pairs)}")
    print(f"Pairs with avg accuracy == 0: {len(zero_accuracy_pairs)}")
    
    # Prepare output
    output_data = {
        'metadata': {
            'source_folder': str(input_folder),
            'num_source_files': len(jsonl_files),
            'source_files': [str(f) for f in sorted(jsonl_files)],
            'threshold': threshold,
            'total_entries': total_entries_all,
            'unique_pairs': len(all_pair_data),
            'high_accuracy_pairs_count': len(high_accuracy_pairs),
            'zero_accuracy_pairs_count': len(zero_accuracy_pairs)
        },
        'high_accuracy_pairs': high_accuracy_pairs,
        'zero_accuracy_pairs': zero_accuracy_pairs
    }
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If output_dest is a directory, create a default filename
    if output_path.is_dir() or str(output_dest).endswith('/'):
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / "high_accuracy_pairs.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput written to: {output_file}")
    
    # Save hard examples in separate file if any exist
    # If save_all_zero_accuracy is True, save all zero accuracy pairs; otherwise only those with enough trajectories
    if save_all_zero_accuracy:
        hard_examples_with_enough_traj = zero_accuracy_pairs
    else:
        hard_examples_with_enough_traj = [p for p in zero_accuracy_pairs if p['num_rollouts'] >= min_trajectories]
    if hard_examples_with_enough_traj:
        if hard_examples_output:
            hard_output_file = hard_examples_output
        else:
            hard_output_file = output_path.parent / f"{output_path.stem}_hard_examples.json"
        save_hard_examples_format(hard_examples_with_enough_traj, str(hard_output_file))
    
    # Print summary
    if high_accuracy_pairs:
        print(f"\n{'='*80}")
        print("HIGH ACCURACY PAIRS SUMMARY")
        print(f"{'='*80}")
        for i, pair in enumerate(high_accuracy_pairs[:10], 1):
            print(f"\n{i}. Question: {pair['research_question'][:80]}...")
            print(f"   Image URL: {pair['image_url']}")
            print(f"   Accuracy: {pair['accuracy_ratio']} ({pair['avg_accuracy']:.2%})")
            print(f"   Ground Truth: {pair['ground_truths']}")
            print(f"   From {len(pair['source_files'])} file(s)")
        
        if len(high_accuracy_pairs) > 10:
            print(f"\n... and {len(high_accuracy_pairs) - 10} more pairs")
    
    return output_data


def process_multiple_folders(
    input_folders: List[str],
    output_file: str,
    threshold: float = 0.75,
    min_trajectories: int = 8,
    hard_examples_output: str = None,
    save_all_zero_accuracy: bool = False
) -> Dict[str, Any]:
    """
    Process multiple folders and combine results into one JSON file.
    
    Args:
        input_folders: List of paths to folders containing JSONL files
        output_file: Path to the output JSON file
        threshold: Accuracy threshold (default 6/8 = 0.75)
        
    Returns:
        Combined output data
    """
    print(f"Processing {len(input_folders)} folders...")
    print("=" * 80)
    
    # Dictionary to aggregate results across all folders
    all_pair_data = defaultdict(lambda: {
        'accuracies': [],
        'gts': [],
        'input_ids': [],
        'source_files': []
    })
    
    total_entries_all = 0
    all_source_files = []
    
    for folder in input_folders:
        input_path = Path(folder)
        if not input_path.exists():
            print(f"Warning: Folder not found: {folder}, skipping...")
            continue
            
        jsonl_files = list(input_path.rglob('*.jsonl'))
        if not jsonl_files:
            print(f"Warning: No JSONL files in {folder}, skipping...")
            continue
        
        print(f"\nFolder: {folder}")
        print(f"  Found {len(jsonl_files)} JSONL file(s)")
        
        for jsonl_file in sorted(jsonl_files):
            all_source_files.append(str(jsonl_file))
            entry_count = 0
            
            for entry in parse_multi_json_file(str(jsonl_file)):
                if not isinstance(entry, dict):
                    continue
                if 'input' not in entry or 'accuracy' not in entry:
                    continue
                
                entry_count += 1
                input_data = entry.get('input', [])
                research_question, image_url = extract_question_and_image_url(input_data)
                
                if research_question and image_url:
                    pair_key = (research_question, image_url)
                    accuracy = entry.get('accuracy', 0.0)
                    
                    all_pair_data[pair_key]['accuracies'].append(accuracy)
                    all_pair_data[pair_key]['input_ids'].append(entry.get('input_id', ''))
                    all_pair_data[pair_key]['source_files'].append(str(jsonl_file))
                    
                    if not all_pair_data[pair_key]['gts']:
                        all_pair_data[pair_key]['gts'] = entry.get('gts', [])
            
            total_entries_all += entry_count
        
        print(f"  Processed entries so far: {total_entries_all}")
    
    print("\n" + "=" * 80)
    print(f"Total entries across all folders: {total_entries_all}")
    print(f"Unique question/image pairs: {len(all_pair_data)}")
    print(f"Minimum trajectories required: {min_trajectories}")
    
    # Calculate average accuracy for each pair and filter
    high_accuracy_pairs = []
    zero_accuracy_pairs = []
    skipped_insufficient = 0
    
    for (question, image_url), data in all_pair_data.items():
        accuracies = data['accuracies']
        num_rollouts = len(accuracies)
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        num_correct = sum(1 for a in accuracies if a > 0)
        
        pair_result = {
            'research_question': question,
            'image_url': image_url,
            'avg_accuracy': avg_accuracy,
            'num_rollouts': num_rollouts,
            'num_correct': num_correct,
            'accuracy_ratio': f"{num_correct}/{num_rollouts}",
            'individual_accuracies': accuracies,
            'ground_truths': data['gts'],
            'input_ids': data['input_ids'],
            'source_files': list(set(data['source_files']))
        }
        
        # For high accuracy pairs, still apply min_trajectories filter
        if avg_accuracy > threshold:
            if num_rollouts >= min_trajectories:
                high_accuracy_pairs.append(pair_result)
            else:
                skipped_insufficient += 1
        # For zero accuracy pairs, check if we should save all or filter by min_trajectories
        elif avg_accuracy == 0:
            if save_all_zero_accuracy or num_rollouts >= min_trajectories:
                zero_accuracy_pairs.append(pair_result)
            else:
                skipped_insufficient += 1
    
    # Sort
    high_accuracy_pairs.sort(key=lambda x: x['avg_accuracy'], reverse=True)
    zero_accuracy_pairs.sort(key=lambda x: x['research_question'])
    
    if skipped_insufficient > 0:
        print(f"Skipped {skipped_insufficient} pairs with < {min_trajectories} trajectories")

    print(f"Pairs with >= {min_trajectories} trajectories: {len(all_pair_data) - skipped_insufficient}")
    print(f"Pairs with avg accuracy > {threshold} ({threshold*100:.0f}%): {len(high_accuracy_pairs)}")
    print(f"Pairs with avg accuracy == 0: {len(zero_accuracy_pairs)}")
    
    # Prepare output
    output_data = {
        'metadata': {
            'source_folders': input_folders,
            'num_source_folders': len(input_folders),
            'num_source_files': len(all_source_files),
            'source_files': sorted(all_source_files),
            'threshold': threshold,
            'min_trajectories': min_trajectories,
            'total_entries': total_entries_all,
            'unique_pairs': len(all_pair_data),
            'pairs_with_enough_trajectories': len(all_pair_data) - skipped_insufficient,
            'high_accuracy_pairs_count': len(high_accuracy_pairs),
            'zero_accuracy_pairs_count': len(zero_accuracy_pairs)
        },
        'high_accuracy_pairs': high_accuracy_pairs,
        'zero_accuracy_pairs': zero_accuracy_pairs
    }
    
    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput written to: {output_file}")
    
    # Save hard examples in separate file if any exist
    # If save_all_zero_accuracy is True, save all zero accuracy pairs; otherwise only those with enough trajectories
    if save_all_zero_accuracy:
        hard_examples_with_enough_traj = zero_accuracy_pairs
    else:
        hard_examples_with_enough_traj = [p for p in zero_accuracy_pairs if p['num_rollouts'] >= min_trajectories]
    if hard_examples_with_enough_traj:
        if hard_examples_output:
            hard_output_file = hard_examples_output
        else:
            hard_output_file = output_path.parent / f"{output_path.stem}_hard_examples.json"
        save_hard_examples_format(hard_examples_with_enough_traj, str(hard_output_file))
    
    # Print summary
    if high_accuracy_pairs:
        print(f"\n{'='*80}")
        print("HIGH ACCURACY PAIRS SUMMARY")
        print(f"{'='*80}")
        for i, pair in enumerate(high_accuracy_pairs[:10], 1):
            print(f"\n{i}. Question: {pair['research_question'][:80]}...")
            print(f"   Image URL: {pair['image_url']}")
            print(f"   Accuracy: {pair['accuracy_ratio']} ({pair['avg_accuracy']:.2%})")
            print(f"   Ground Truth: {pair['ground_truths']}")
            print(f"   From {len(pair['source_files'])} file(s)")
        
        if len(high_accuracy_pairs) > 10:
            print(f"\n... and {len(high_accuracy_pairs) - 10} more pairs")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='Process rollout JSONL file(s) to find high-accuracy question/image pairs'
    )
    parser.add_argument(
        'input_path',
        type=str,
        nargs='+',
        help='Path(s) to input JSONL file or folder(s) containing JSONL files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output JSON file (default: <input>_high_accuracy.json or rollout_summary.json for multiple inputs)'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.95,
        help='Accuracy threshold (default: 0.75 = 6/8)'
    )
    parser.add_argument(
        '-m', '--min-trajectories',
        type=int,
        default=8,
        help='Minimum number of trajectories required for a pair to be considered (default: 8)'
    )
    parser.add_argument(
        '--hard-examples-output',
        type=str,
        default=None,
        help='Path to output hard examples JSON file (default: <output>_hard_examples.json)'
    )
    parser.add_argument(
        '--save-all-zero-accuracy',
        action='store_true',
        help='Save all zero accuracy pairs regardless of min_trajectories (only filter out high accuracy pairs)'
    )
    
    args = parser.parse_args()
    
    # Handle multiple input paths
    if len(args.input_path) > 1:
        # Multiple inputs - combine into one output
        if args.output is None:
            args.output = "rollout_summary.json"
        process_multiple_folders(args.input_path, args.output, args.threshold, args.min_trajectories, args.hard_examples_output, args.save_all_zero_accuracy)
    else:
        # Single input
        input_path = Path(args.input_path[0])
        
        if input_path.is_dir():
            # Process folder
            if args.output is None:
                output_dest = input_path / "rollout_summary.json"
            else:
                output_dest = args.output
            process_rollout_folder(str(input_path), str(output_dest), args.threshold, True, args.min_trajectories, args.hard_examples_output, args.save_all_zero_accuracy)
        elif input_path.is_file():
            # Process single file
            process_rollout_file(str(input_path), args.output, args.threshold, args.min_trajectories, args.hard_examples_output, args.save_all_zero_accuracy)
        else:
            print(f"Error: {args.input_path[0]} is not a valid file or directory")
            exit(1)


if __name__ == '__main__':
    main()
    """
python process_rollout.py /export/home/becky/verl-tool-mm-deepsearch/verl_step_records/multimodal_deepsearch-fsdp-agent-_export_share_beckypeng_models_qwen3-vl-8b-instruct-grpo-n8-b16-t1.0-lr1e-6-mm_deepsearch_multi-mmresearch-8b-poc-20260106-202324/rollout /export/home/becky/verl-tool-mm-deepsearch/verl_step_records/multimodal_deepsearch-fsdp-agent-_export_share_beckypeng_models_qwen3-vl-8b-instruct-grpo-n4-b16-t1.0-lr2e-6-mm_deepsearch_multi-mmresearch-8b-poc-20260106-055615/rollout /export/home/becky/verl-tool-mm-deepsearch/verl_step_records/can/rollout_1 /export/home/becky/verl-tool-mm-deepsearch/verl_step_records/can/rollout_2 /export/home/becky/verl-tool-mm-deepsearch/verl_step_records/can/rollout_3 /export/home/becky/verl-tool-mm-deepsearch/verl_step_records/can/rollout_2 /export/home/becky/verl-tool-mm-deepsearch/verl_step_records/can/rollout_4 /export/home/becky/verl-tool-mm-deepsearch/verl_step_records/can/rollout_5 /export/home/becky/verl-tool-mm-deepsearch/verl_step_records/multimodal_deepsearch-fsdp-agent-_export_share_beckypeng_models_qwen3-vl-8b-instruct-grpo-n8-b64-t1.0-lr2e-6-mm_deepsearch_multi-mmresearch-8b-dapo-poc-20260202-205002/rollout /export/home/becky/verl-tool-mm-deepsearch/verl_step_records/multimodal_deepsearch-fsdp-agent-_export_share_beckypeng_models_qwen3-vl-8b-instruct-grpo-n8-b64-t1.0-lr2e-6-mm_deepsearch_multi-mmresearch-8b-dapo-baseline-20260214-031026/rollout -o /export/home/becky/verl-tool-mm-deepsearch/verl_step_records/multimodal_deepsearch-fsdp-agent-_export_share_beckypeng_models_qwen3-vl-8b-instruct-grpo-n8-b64-t1.0-lr2e-6-mm_deepsearch_multi-mmresearch-8b-dapo-poc-20260202-205002/rollout_summary.json
    """
