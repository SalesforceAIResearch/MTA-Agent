#!/bin/bash

# Simple test script for multimodal deep research testing
# This script runs a few scenarios to test the multimodal research pipeline

echo "Starting Multimodal Deep Research Test..."

# Set environment variables
export OPENAI_API_KEY="dummy"

# Run the test with a limited number of scenarios
# --difficulty_filter="easy" \
# --prompt_tool "- You will use image_search as the first tool (<text_search_image>your image search query</text_search_image>)."
#python test_mm_deep_research_vllm.py run \
#    --server_url="http://127.0.0.1:8002/get_observation" \
#    --output_dir="simple_test_results_News_qwen25_after_1e6_ep3" \
#    --input_dir="/export/atlas-eval/mm-deep-research-rl/examples/train/mm_deep_research/News/" \
#    --runs_per_question=1 \
#    --max_scenarios=1000 \
#    --image_url_prefix="https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/News/images/" \
#    --prompt_tool "" \
#    --run_start_number=0 \
#    --gpu_id=1 \
#    --tensor_parallel_size=1 \
#    --model_path="/export/share/beckypeng/mm_dr/output/1e6_ep3"

python test_mm_deep_research_vllm.py run \
    --server_url="http://127.0.0.1:8002/get_observation" \
    --output_dir="simple_test_results_MM-BrowseComp_qwen3_before" \
    --input_dir="/export/atlas-eval/mm-deep-research-rl/examples/train/mm_deep_research/MM-BrowseComp/" \
    --runs_per_question=1 \
    --max_scenarios=1000 \
    --image_url_prefix="https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/MM-BrowseComp/images/" \
    --prompt_tool "" \
    --run_start_number=0 \
    --gpu_id=1 \
    --tensor_parallel_size=1 \
    --model_path="/export/share/beckypeng/models/Qwen3-VL-8B-Instruct"

echo "Test completed. Check the simple_test_results_mm directory for results."
