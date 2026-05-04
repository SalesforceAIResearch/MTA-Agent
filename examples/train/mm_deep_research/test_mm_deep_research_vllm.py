#!/usr/bin/env python3
"""
Multimodal Deep Research Test Cases - vLLM Implementation
Simulates actual deep research scenarios with multimodal data (text + images)
Uses Qwen2.5-VL model via vLLM for tool use generation and executes via agent tools
Uses the unified MultimodalDeepResearchTesterBase class with vLLM model type
"""
import fire
import logging
from typing import Optional, List

# Import the unified base class
from test_mm_deep_research import create_multimodal_tester

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main entry point for multimodal deep research testing using vLLM

    Usage:
        python test_mm_deep_research_vllm.py run --server_url=http://localhost:4000/get_observation
        python test_mm_deep_research_vllm.py run --output_dir=my_test_results_mm
        python test_mm_deep_research_vllm.py run --input_dir=data/mmsearch_plus_processed
        python test_mm_deep_research_vllm.py run --model_path=/path/to/qwen/model
        python test_mm_deep_research_vllm.py run --max_scenarios=5
        python test_mm_deep_research_vllm.py run --difficulty_filter=easy
        python test_mm_deep_research_vllm.py run --enabled_tools="web_text_to_text_search,web_text_to_img_search,web_url_reader,web_image_to_text,python_code"
        python test_mm_deep_research_vllm.py run --image_url_prefix="https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/MM-BrowseComp/images/"
        python test_mm_deep_research_vllm.py run --runs_per_question=3
        python test_mm_deep_research_vllm.py run --runs_per_question=3 --run_start_number=5
        python test_mm_deep_research_vllm.py run --max_model_len=10000 --tensor_parallel_size=1
        python test_mm_deep_research_vllm.py run --gpu_id=0  # Use GPU 0
        python test_mm_deep_research_vllm.py run --gpu_id=1  # Use GPU 1
        python test_mm_deep_research_vllm.py test_image --scenario_id=MMSearch-Plus_0 --gpu_id=0
        python test_mm_deep_research_vllm.py test_api --model_path=/path/to/qwen/model --gpu_id=0
        python test_mm_deep_research_vllm.py test_chat_format --model_path=/path/to/qwen/model --gpu_id=0
        python test_mm_deep_research_vllm.py test_chat_format --model_path=/path/to/qwen/model --gpu_id=0 --image_path=/path/to/image.jpg
    """
    def run(server_url="http://localhost:4000/get_observation", 
            output_dir="simple_test_results_mm", 
            input_dir="data/mmsearch_plus_processed",
            model_path="/export/home/becky/Qwen3-VL/qwen-vl-finetune/output/2e5_ep3/",
            max_scenarios=3,
            enabled_tools=None,
            difficulty_filter="",
            image_url_prefix=None,
            prompt_tool=None,
            runs_per_question=1,
            run_start_number=1,
            max_model_len=10000,
            tensor_parallel_size=1,
            gpu_id=None):
        """Run multimodal deep research test using vLLM"""
        
        # Parse enabled_tools if provided as comma-separated string
        if enabled_tools and isinstance(enabled_tools, str):
            enabled_tools = [t.strip() for t in enabled_tools.split(',')]
        
        # Create vLLM-based tester using the unified class
        tester = create_multimodal_tester(
            model_type="vllm",
            server_url=server_url,
            output_dir=output_dir,
            input_dir=input_dir,
            model_path=model_path,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_id=gpu_id,
            enabled_tools=enabled_tools,
            difficulty_filter=difficulty_filter,
            image_url_prefix=image_url_prefix,
            prompt_tool=prompt_tool,
            runs_per_question=runs_per_question,
            run_start_number=run_start_number
        )
        
        result = tester.run_multimodal_test(max_scenarios)
        
        # Don't return the result to prevent Fire from printing it
        print("Test completed successfully. Results saved to output directory.")
        return None
    
    def test_image(scenario_id="MMSearch-Plus_0", 
                   model_path="/export/home/becky/Qwen3-VL/qwen-vl-finetune/output/2e5_ep3/",
                   gpu_id=None):
        """Test image inclusion functionality"""
        
        tester = create_multimodal_tester(
            model_type="vllm",
            model_path=model_path,
            gpu_id=gpu_id
        )
        
        success = tester.test_image_inclusion(scenario_id)
        if success:
            print("✓ Image inclusion test passed")
        else:
            print("✗ Image inclusion test failed")
        return success
    
    def test_api(model_path="/export/home/becky/Qwen3-VL/qwen-vl-finetune/output/2e5_ep3/", 
                 gpu_id=None):
        """Test basic API functionality"""
        
        tester = create_multimodal_tester(
            model_type="vllm",
            model_path=model_path,
            gpu_id=gpu_id
        )
        
        # Test basic text generation
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        response = tester.generate_response_with_model(messages, max_tokens=100, temperature=0.7)
        
        print(f"API Test Response: {response}")
        return len(response) > 0
    
    def test_chat_format(model_path="/export/home/becky/Qwen3-VL/qwen-vl-finetune/output/2e5_ep3/",
                        image_path=None,
                        gpu_id=None):
        """Test chat format with optional image"""
        
        tester = create_multimodal_tester(
            model_type="vllm",
            model_path=model_path,
            gpu_id=gpu_id
        )
        
        if image_path:
            # Test with image
            response = tester.analyze_image_with_model(image_path, "Describe this image in detail.")
            print(f"Image Analysis Response: {response}")
        else:
            # Test text-only
            messages = [{"role": "user", "content": "Explain what machine learning is in simple terms."}]
            response = tester.generate_response_with_model(messages, max_tokens=200, temperature=0.7)
            print(f"Text Response: {response}")
        
        return len(response) > 0

    # Use Fire to create CLI
    fire.Fire({
        'run': run,
        'test_image': test_image,
        'test_api': test_api,
        'test_chat_format': test_chat_format
    })


if __name__ == "__main__":
    main()