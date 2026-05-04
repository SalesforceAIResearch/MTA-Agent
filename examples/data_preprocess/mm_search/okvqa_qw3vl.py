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
Preprocess the OKVQA dataset to parquet format for model training (Qwen3VL version)
"""
import fire
import json
import datasets
import regex as re
import shutil
from pathlib import Path
from PIL import Image
from collections import defaultdict
from copy import deepcopy
import random

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


def extract_image_filename(image_url_or_path: str) -> str:
    """Extract the image filename from a URL or path."""
    if not image_url_or_path:
        return ''
    # Get the last part after '/' and remove any query params
    filename = image_url_or_path.split('/')[-1].split('?')[0]
    return filename.lower().strip()


def load_exclusion_list(exclusion_file: str, dataset_name: str) -> set:
    """
    Load exclusion list from rollout_summary.json and return a set of (question, image_filename) tuples to exclude.
    Excludes both high_accuracy_pairs (>threshold) and zero_accuracy_pairs (==0).
    Uses both question AND image to handle same questions with different images.
    """
    if not exclusion_file:
        return set()
    exclusion_path = Path(exclusion_file)
    if not exclusion_path.exists():
        print(f"Warning: Exclusion file not found: {exclusion_file}")
        return set()
    with open(exclusion_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    exclusion_set = set()
    high_acc_count = 0
    zero_acc_count = 0
    
    # Load high accuracy pairs for exclusion
    for entry in data.get('high_accuracy_pairs', []):
        image_url = entry.get('image_url', '')
        question = entry.get('research_question', '')
        
        # Extract dataset from URL: .../mm-deepsearch-sfr-2025/{dataset}/images/...
        if f'/{dataset_name}/' in image_url or f'/{dataset_name.lower()}/' in image_url.lower():
            if question:
                image_filename = extract_image_filename(image_url)
                exclusion_set.add((question.lower().strip(), image_filename))
                high_acc_count += 1
    
    # # Load zero accuracy pairs for exclusion
    # for entry in data.get('zero_accuracy_pairs', []):
    #     image_url = entry.get('image_url', '')
    #     question = entry.get('research_question', '')
    #
    #     if f'/{dataset_name}/' in image_url or f'/{dataset_name.lower()}/' in image_url.lower():
    #         if question:
    #             image_filename = extract_image_filename(image_url)
    #             exclusion_set.add((question.lower().strip(), image_filename))
    #             zero_acc_count += 1
    
    print(f"Loaded {len(exclusion_set)} exclusions for dataset '{dataset_name}' (high_acc: {high_acc_count}, zero_acc: {zero_acc_count})")
    return exclusion_set


def load_questions_from_parquet(parquet_path: str) -> set:
    """
    Load (question, image_filename) tuples from an existing parquet file to use for exclusion.
    This allows reusing val/test sets and only regenerating train.
    """
    if not parquet_path:
        return set()
    
    parquet_file = Path(parquet_path)
    if not parquet_file.exists():
        print(f"Warning: Parquet file not found: {parquet_path}")
        return set()
    
    import pyarrow.parquet as pq
    
    exclusions = set()
    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        # Extract questions and images from the parquet
        for idx, row in df.iterrows():
            question = None
            image_filename = None
            
            # Extract question from prompt (use direct column access for pandas Series)
            try:
                prompt = row['prompt'] if 'prompt' in df.columns else None
                if prompt is not None:
                    for msg in prompt:
                        if isinstance(msg, dict) and msg.get('role') == 'user':
                            content = msg.get('content', '')
                            if '<image>' in content:
                                question = content.split('<image>')[-1].strip()
                            else:
                                question = content.strip()
                            break
            except Exception:
                pass
            
            # Extract image filename from image_urls or images
            try:
                image_urls = row['image_urls'] if 'image_urls' in df.columns else None
                if image_urls is not None and len(image_urls) > 0:
                    image_filename = extract_image_filename(image_urls[0])
                else:
                    images = row['images'] if 'images' in df.columns else None
                    if images is not None and len(images) > 0:
                        if isinstance(images[0], dict):
                            image_filename = extract_image_filename(images[0].get('image', ''))
                        else:
                            image_filename = extract_image_filename(str(images[0]))
            except Exception:
                pass
            
            if question and image_filename:
                exclusions.add((question.lower().strip(), image_filename))
        
        print(f"Loaded {len(exclusions)} (question, image) pairs from {parquet_path}")
    except Exception as e:
        print(f"Error loading parquet file {parquet_path}: {e}")
    
    return exclusions


def should_exclude(question_text: str, image_path: str, exclusion_set: set) -> bool:
    """Check if a scenario should be excluded based on question AND image."""
    if not exclusion_set:
        return False
    image_filename = extract_image_filename(image_path)
    return (question_text.lower().strip(), image_filename) in exclusion_set


system_prompt = """You are an expert multimodal research assistant that can analyze images, search the web, and execute code to answer complex research questions.

# Tools

You have access to the following tools to help answer questions:

1. **Web Search (Text)**: Search the web and get text-based search results
   - Format: `<text_search_text>your search query</text_search_text>`
   - Returns structured search results with titles, URLs, and content snippets

2. **Web Search (Images)**: Search for images on the web and get text descriptions
   - Format: `<text_search_image>your image search query</text_search_image>`
   - Returns image descriptions of what each image contains

3. **Web Content Extraction**: Extract and read full text content from web pages
   - Format: `<web_read>https://example.com</web_read>`
   - Returns the complete text content of the webpage

4. **Reverse Image Search**: Find text information about images using reverse image search
   - Format: `<image_search_text>https://example.com/image.jpg</image_search_text>`
   - Returns detailed text descriptions and context about the image

5. **Python Code Execution**: Execute Python code for data analysis and computation
   - Format: `<python>\nprint("Hello, World!")\n</python>`
   - Returns the output of the code execution

6. **Bash Terminal**: Execute bash commands for file operations and system tasks
   - Format: `<bash>\nls -la\n</bash>`
   - Returns the command output

# Instructions

- Analyze the given image and question carefully
- Use the appropriate tools to gather information
- Combine information from multiple sources (visual analysis, web search, code execution)
- Provide comprehensive, well-structured answers
- Use `<think>` tags for your reasoning process
- Put your final answer in `\\boxed{answer}` format"""

def extract_ground_truth(ground_truth, question_text=None):
    """
    Extract ground truth answers from various formats.
    
    Args:
        ground_truth: Can be a string, list, or dict with answer information
        question_text: Optional question text to extract option text from
    
    Returns a list of acceptable answers.
    """
    if not ground_truth:
        return []
    
    answers = []
    
    # Handle different ground truth formats
    if isinstance(ground_truth, str):
        # String format - could be single answer or comma-separated
        if ',' in ground_truth:
            answers = [ans.strip() for ans in ground_truth.split(',') if ans.strip()]
        else:
            answers = [ground_truth.strip()] if ground_truth.strip() else []
    elif isinstance(ground_truth, list):
        # List format
        answers = [str(ans).strip() for ans in ground_truth if ans]
    elif isinstance(ground_truth, dict):
        # Dict format - try common keys
        for key in ['answer', 'answers', 'ground_truth', 'correct_answer']:
            if key in ground_truth:
                val = ground_truth[key]
                if isinstance(val, list):
                    answers.extend([str(a).strip() for a in val if a])
                else:
                    answers.append(str(val).strip())
                break
    else:
        # Try to convert to string
        answers = [str(ground_truth).strip()]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_answers = []
    for ans in answers:
        ans_lower = ans.lower().strip()
        if ans_lower and ans_lower not in seen:
            seen.add(ans_lower)
            unique_answers.append(ans.strip())
    
    return unique_answers if unique_answers else []

def get_mm_content_len(processor, example):
    """
    Calculate multimodal content length for Qwen3VL models.
    
    Qwen3VL uses image_patch_size=16 and process_vision_info returns
    (images, videos, video_kwargs) instead of (image_inputs, _).
    """
    messages = deepcopy(example['prompt'])
    for message in messages:
        content = message["content"]
        content_list = []
        segments = re.split("(<image>)", content)
        segments = [item for item in segments if item != ""]
        segment_idx = defaultdict(int)
        for segment in segments:
            if segment == "<image>":
                content_list.append({"type": "image", "image": example['images'][segment_idx[segment]]["image"]})
                segment_idx[segment] += 1
            else:
                content_list.append({"type": "text", "text": segment})

        message["content"] = content_list
    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    
    # Qwen3VL: process_vision_info returns (images, videos, video_kwargs) with image_patch_size=16
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,  # Qwen3VL uses patch size 16
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    inputs = processor(
        text=[raw_prompt],
        images=images,  # Use images directly (not image_inputs)
        padding=True,
        return_tensors="pt",
    )
    return inputs.input_ids.shape[1]

def process_scenarios(
    scenarios,
    split_name,
    images_dir,
    output_image_dir,
    image_sep,
    image_url_prefix,
    filter_len,
    processor,
    start_idx=0,
    exclusion_set=None,
    exclusion_log=None,
):
    """
    Process scenarios into dataset format for a specific split.
    
    Returns:
        List of processed data items
    """
    processed_data = []
    excluded_count = 0
    
    print(f"Processing {len(scenarios)} scenarios for {split_name} split...")
    for idx, scenario in enumerate(scenarios):
        try:
            # Extract fields - handle different data formats
            data_id = scenario.get('id') or scenario.get('qid') or scenario.get('question_id') or scenario.get('image_id') or f'okvqa_{start_idx + idx}'
            
            # Extract question text - handle nested question object
            question_obj = scenario.get('question', {})
            if isinstance(question_obj, dict):
                question_text = question_obj.get('text', '') or question_obj.get('question', '')
            else:
                question_text = question_obj or scenario.get('question_text') or scenario.get('text', '')
            
            if not question_text:
                print(f"Warning: Skipping scenario {start_idx + idx} - no question text found")
                continue
            
            # Extract image path - handle nested question object and different formats
            # OKVQA often uses COCO image IDs, so we might need to construct the path
            image_id = scenario.get('image_id') or scenario.get('imageId') or scenario.get('img_id')
            image_path_rel = None
            
            # First check nested question object
            if isinstance(question_obj, dict):
                image_path_rel = question_obj.get('image') or question_obj.get('image_path') or question_obj.get('image_file')
            
            # Then check top-level scenario
            if not image_path_rel:
                image_path_rel = scenario.get('image') or scenario.get('image_path') or scenario.get('image_file') or scenario.get('img')
            
            # If we have image_id but no path, construct it (COCO format: COCO_val2014_000000123456.jpg)
            if image_id and not image_path_rel:
                image_path_rel = f"COCO_{split_name}2014_{str(image_id).zfill(12)}.jpg"
            
            if not image_path_rel:
                # Debug: print available keys to help diagnose the issue
                available_keys = list(scenario.keys())
                question_keys = list(question_obj.keys()) if isinstance(question_obj, dict) else []
                print(f"Warning: Skipping scenario {start_idx + idx} - no image path found. Available keys: {available_keys}, question keys: {question_keys}")
                continue
            
            # Check exclusion list before processing (based on question AND image)
            if exclusion_set and should_exclude(question_text, str(image_path_rel), exclusion_set):
                excluded_count += 1
                if exclusion_log is not None:
                    exclusion_log.append({
                        'split': split_name,
                        'index': start_idx + idx,
                        'question': question_text,
                        'image': str(image_path_rel) if image_path_rel else None,
                    })
                continue
            
            # Remove leading "./" if present
            image_filename = str(image_path_rel).lstrip('./')
            image_path_local = images_dir / image_filename
            
            if not image_path_local.exists():
                print(f"Warning: Skipping scenario {start_idx + idx} - image not found: {image_path_local}")
                continue
            
            # Load and verify image
            try:
                img = Image.open(image_path_local)
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            except Exception as e:
                print(f"Warning: Skipping scenario {start_idx + idx} - could not load image {image_path_local}: {e}")
                continue
            
            # Move image to output directory (instead of copying)
            img_path_output = output_image_dir / image_filename
            # Create parent directories if needed
            img_path_output.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the image file
            if img_path_output.exists():
                print(f"Warning: Image already exists at {img_path_output}, skipping move")
            else:
                shutil.move(str(image_path_local), str(img_path_output))
            
            image_path_absolute = img_path_output.absolute().as_posix()
            
            # Create URL path for output using original filename
            image_url = image_url_prefix.rstrip('/') + '/' + image_filename
            
            # Extract ground truth - OKVQA often has multiple answers
            ground_truth_raw = scenario.get('answer') or scenario.get('answers') or scenario.get('ground_truth') or scenario.get('correct_answer')
            if isinstance(ground_truth_raw, list) and len(ground_truth_raw) > 0:
                # If it's a list of dicts with 'answer' key (common in OKVQA)
                if isinstance(ground_truth_raw[0], dict):
                    ground_truth_raw = [ans.get('answer', '') for ans in ground_truth_raw if ans.get('answer')]
            
            ground_truth = extract_ground_truth(ground_truth_raw, question_text)
            if not ground_truth:
                print(f"Warning: Skipping scenario {start_idx + idx} - could not extract ground truth")
                continue
            
            # Build multimodal content
            mm_content = image_sep + question_text
            
            # Create data structure
            data_item = {
                "data_source": "okvqa",
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": mm_content,
                    }
                ],
                "images": [{"image": image_path_absolute}],
                'image_urls': [image_url],  # Store URL separately, will be used for reverse image search
                "ability": "visual_reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    'split': split_name,
                    'index': start_idx + idx,
                    'qid': data_id,
                    'is_video': False,
                    'num_images': 1,
                }
            }
            
            # Calculate content length if filter is enabled
            if filter_len and filter_len > 0 and processor:
                try:
                    data_for_len = deepcopy(data_item)
                    data_for_len['images'] = [{"image": image_path_absolute}]
                    mm_content_len = get_mm_content_len(processor, data_for_len)
                    data_item['extra_info']['mm_content_len'] = mm_content_len
                except Exception as e:
                    print(f"Warning: Could not calculate content length for scenario {start_idx + idx}: {e}")
                    data_item['extra_info']['mm_content_len'] = None
            
            processed_data.append(data_item)
            
        except Exception as e:
            print(f"Warning: Skipping scenario {start_idx + idx} due to error: {e}")
            continue
    
    if excluded_count > 0:
        print(f"Excluded {excluded_count} scenarios due to exclusion list")
    
    return processed_data

# python -c 'import json; print(len(json.load(open("/export/share/beckypeng/mm_dr/data/okvqa/qa_formatted_train.json"))["scenarios"]))'
def main(
    qa_json_path: str = '/export/share/beckypeng/mm_dr/data/okvqa/qa_formatted_train.json',
    images_dir: str = '/export/share/beckypeng/mm_dr/data/okvqa/images',
    split: str = None,  # 'train', 'val', 'test', or None/'all' to process all splits
    local_dir: str = '/export/home/becky/verl-tool-mm-deepsearch/data/mm_deepsearch/okvqa_qw3vl',
    image_sep: str = "<image>",
    filter_len: int = None,
    image_url_prefix: str = 'https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/okvqa/images/',
    max_examples: int = None,
    train_size: int = None,  # None = use all remaining after val/test
    val_size: int = 40,
    test_size: int = 60,
    seed: int = 42,
    model_name: str = 'Qwen/Qwen3-VL-8B-Instruct',
    exclusion_file: str = None,  # Path to rollout_summary.json to filter out high-accuracy pairs
    max_output: int = None,  # Maximum number of examples to output per split (applied after all filtering)
    val_parquet: str = "/export/home/becky/verl-tool-mm-deepsearch/data/mm_deepsearch/okvqa_qw3vl/val.parquet",  # Path to existing val.parquet - questions will be excluded from train
    test_parquet: str = "/export/home/becky/verl-tool-mm-deepsearch/data/mm_deepsearch/okvqa_qw3vl/test.parquet",  # Path to existing test.parquet - questions will be excluded from train
):
    """
    Preprocess OKVQA dataset to parquet format compatible with Qwen3VL model training.
    
    Args:
        qa_json_path: Path to qa_formatted.json file
        images_dir: Directory containing images
        split: Dataset split to process ('train', 'val', 'test', or None/'all' to process all splits)
        local_dir: Local directory to save processed data
        image_sep: Image separator token (default: "<image>")
        filter_len: Optional filter for maximum content length
        image_url_prefix: URL prefix for image paths
        max_examples: Optional limit on number of examples to process (useful for testing)
        train_size: Number of examples for training split (if None, uses all available)
        val_size: Number of examples for validation split (if None, uses all available)
        test_size: Number of examples for test split (if None, uses all available)
        seed: Random seed for splitting (default: 42)
        model_name: Qwen3VL model name (default: 'Qwen/Qwen3-VL-8B-Instruct')
        exclusion_file: Path to exclusion_list.json or rollout_summary.json to filter out high-accuracy pairs (optional)
    """
    # Load exclusion list if provided
    exclusion_set = load_exclusion_list(exclusion_file, 'okvqa') if exclusion_file else set()
    
    # Load questions from existing val/test parquet files to exclude from train
    val_questions = load_questions_from_parquet(val_parquet) if val_parquet else set()
    test_questions = load_questions_from_parquet(test_parquet) if test_parquet else set()
    
    # Combine all exclusions
    if val_questions or test_questions:
        if not exclusion_set:
            exclusion_set = set()
        exclusion_set = exclusion_set | val_questions | test_questions
        print(f"Total exclusions after adding val/test parquet: {len(exclusion_set)}")
        
        # If val/test parquet provided, only process train
        if split is None or split.lower() == 'all':
            print("Val/test parquet provided - will only generate train split")
            split = 'train'
    
    # Convert empty set to None for compatibility
    if exclusion_set is not None and len(exclusion_set) == 0:
        exclusion_set = None
    
    processor = AutoProcessor.from_pretrained(model_name) if filter_len else None
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    qa_json_path = Path(qa_json_path)
    images_dir = Path(images_dir)
    
    if not qa_json_path.exists():
        raise FileNotFoundError(f"QA JSON file not found: {qa_json_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Load JSON data
    print("Loading JSON data...")
    with open(qa_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        scenarios = data
    elif isinstance(data, dict):
        scenarios = data.get('scenarios', data.get('data', data.get('questions', data.get('annotations', []))))
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")
    
    print(f"Loaded {len(scenarios)} scenarios from JSON")
    
    # Limit scenarios if max_examples is specified
    if max_examples is not None and max_examples > 0:
        original_size = len(scenarios)
        scenarios = scenarios[:min(max_examples, len(scenarios))]
        print(f"Limited to {len(scenarios)} examples (from {original_size})")
    
    # Check if scenarios have split field
    has_split_field = any('split' in scenario or 'set' in scenario for scenario in scenarios)
    
    # Determine which splits to process
    if split is None or split.lower() == 'all':
        if has_split_field:
            # Use existing splits from data
            splits_to_process = ['train', 'val', 'test']
        else:
            # Create splits automatically
            splits_to_process = ['train', 'val', 'test']
    else:
        splits_to_process = [split]
    
    # Split scenarios if data doesn't have split field
    if not has_split_field:
        total_available = len(scenarios)
        
        # If train_size is None, use all remaining after val/test
        if train_size is None:
            train_size = max(0, total_available - val_size - test_size)
        
        print(f"Data does not have split field. Creating splits: train={train_size}, val={val_size}, test={test_size}")
        total_requested = train_size + val_size + test_size
        
        if total_available < total_requested:
            print(f"Warning: Only {total_available} scenarios available, but {total_requested} requested.")
            print(f"Adjusting sizes proportionally...")
            # Adjust sizes proportionally, keeping val/test ratio
            val_test_total = val_size + test_size
            if val_test_total >= total_available:
                # Not enough for val+test, split proportionally
                val_size = int(total_available * val_size / val_test_total)
                test_size = total_available - val_size
                train_size = 0
            else:
                train_size = total_available - val_size - test_size
            print(f"Adjusted sizes: train={train_size}, val={val_size}, test={test_size}")
        
        # Shuffle scenarios with seed for reproducibility
        random.seed(seed)
        scenarios_shuffled = scenarios.copy()
        random.shuffle(scenarios_shuffled)
        
        # Split scenarios: val and test first, train gets the rest
        val_scenarios = scenarios_shuffled[:val_size]
        test_scenarios = scenarios_shuffled[val_size:val_size + test_size]
        train_scenarios = scenarios_shuffled[val_size + test_size:val_size + test_size + train_size]
        
        split_scenarios = {
            'train': train_scenarios,
            'val': val_scenarios,
            'test': test_scenarios,
        }
        
        print(f"Split data: train={len(train_scenarios)}, val={len(val_scenarios)}, test={len(test_scenarios)}")
    else:
        print("Data has split field. Using existing splits from data.")
        # Group scenarios by split
        split_scenarios = defaultdict(list)
        for scenario in scenarios:
            scenario_split = scenario.get('split') or scenario.get('set', 'train')
            split_scenarios[scenario_split].append(scenario)
        
        print(f"Found splits in data: {dict((k, len(v)) for k, v in split_scenarios.items())}")
    
    # Create images directory for processed images
    output_image_dir = local_dir / 'images'
    output_image_dir.mkdir(parents=True, exist_ok=True)
    
    # Track exclusions for logging
    exclusion_log = []
    
    # Track final dataset sizes
    final_sizes = {}
    
    # Process each split
    for split_name in splits_to_process:
        if split_name not in split_scenarios:
            print(f"Warning: Split '{split_name}' not found. Skipping.")
            continue
        
        scenarios_for_split = split_scenarios[split_name]
        if not scenarios_for_split:
            print(f"Warning: No scenarios for split '{split_name}'. Skipping.")
            continue
        
        # Process scenarios for this split
        processed_data = process_scenarios(
            scenarios_for_split,
            split_name,
            images_dir,
            output_image_dir,
            image_sep,
            image_url_prefix,
            filter_len,
            processor,
            start_idx=0,
            exclusion_set=exclusion_set,
            exclusion_log=exclusion_log,
        )
        
        print(f"Processed {len(processed_data)} scenarios for {split_name} split")
        
        # Filter by content length if specified
        if filter_len and filter_len > 0:
            original_count = len(processed_data)
            processed_data = [
                item for item in processed_data 
                if item['extra_info'].get('mm_content_len') is not None 
                and item['extra_info']['mm_content_len'] <= filter_len
            ]
            filtered_count = original_count - len(processed_data)
            if filtered_count > 0:
                print(f"Filtered {filtered_count}/{original_count} examples due to content length > {filter_len} for {split_name} split")
        
        # Limit output size if max_output is specified
        if max_output and max_output > 0 and len(processed_data) > max_output:
            original_count = len(processed_data)
            processed_data = processed_data[:max_output]
            print(f"Limited {split_name} split to {max_output} examples (from {original_count})")
        
        if not processed_data:
            print(f"Warning: No valid data processed for {split_name} split. Skipping.")
            continue
        
        # Convert to dataset
        print(f"Converting {split_name} split to dataset format...")
        dataset = datasets.Dataset.from_list(processed_data)
        
        # Print sample
        print(f"Processed {len(dataset)} samples for {split_name} split")
        if len(dataset) > 0:
            print(f"\nExample sample from {split_name} split (first):")
            example_data = dataset[0]
            print(json.dumps(example_data, indent=2, ensure_ascii=False))
            
            if len(dataset) > 1:
                print(f"\nExample sample from {split_name} split (last):")
                example_data = dataset[-1]
                print(json.dumps(example_data, indent=2, ensure_ascii=False))
        
        # Save to parquet (add suffix if exclusion is used)
        suffix = '_filtered' if exclusion_set else ''
        output_file = local_dir / f'{split_name}{suffix}.parquet'
        dataset.to_parquet(str(output_file))
        print(f"Saved {len(dataset)} samples to {output_file}\n")
        
        # Track final size
        final_sizes[split_name] = len(dataset)
    
    # Save exclusion log if any exclusions were made
    if exclusion_log:
        exclusion_log_file = local_dir / 'exclusion_log.json'
        exclusion_summary = {
            'total_excluded': len(exclusion_log),
            'by_split': {},
            'excluded_questions': exclusion_log,
        }
        for entry in exclusion_log:
            split_name = entry['split']
            exclusion_summary['by_split'][split_name] = exclusion_summary['by_split'].get(split_name, 0) + 1
        
        with open(exclusion_log_file, 'w', encoding='utf-8') as f:
            json.dump(exclusion_summary, f, indent=2, ensure_ascii=False)
        print(f"\n=== Exclusion Summary ===")
        print(f"Total excluded: {len(exclusion_log)}")
        for split_name, count in exclusion_summary['by_split'].items():
            print(f"  {split_name}: {count}")
        print(f"Exclusion log saved to: {exclusion_log_file}")
    
    # Print final dataset summary
    print(f"\n{'='*60}")
    print("FINAL DATASET SUMMARY")
    print(f"{'='*60}")
    total_examples = sum(final_sizes.values())
    print(f"Total examples: {total_examples}")
    for split_name, count in final_sizes.items():
        print(f"  - {split_name}: {count}")
    print(f"Output directory: {local_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    fire.Fire(main)

"""
Example usage:
# Process all splits automatically
python examples/data_preprocess/mm_search/okvqa_qw3vl.py --qa_json_path=data/mm_deepsearch/OKVQA/qa_formatted.json --images_dir=data/mm_deepsearch/OKVQA/images --local_dir=data/mm_deepsearch/okvqa_qw3vl

# Process only train split
python examples/data_preprocess/mm_search/okvqa_qw3vl.py --qa_json_path=data/mm_deepsearch/OKVQA/qa_formatted.json --images_dir=data/mm_deepsearch/OKVQA/images --split=train --local_dir=data/mm_deepsearch/okvqa_qw3vl

# Process with content length filter
python examples/data_preprocess/mm_search/okvqa_qw3vl.py --qa_json_path=data/mm_deepsearch/OKVQA/qa_formatted.json --images_dir=data/mm_deepsearch/OKVQA/images --local_dir=data/mm_deepsearch/okvqa_qw3vl --filter_len=8192
"""

