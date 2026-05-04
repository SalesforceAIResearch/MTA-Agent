import openai
import os
import json
from pathlib import Path
from datetime import datetime
import time
import base64

# Initialize OpenAI client
openai.api_key = "dummy"
openai.default_headers = {"X-Api-Key": os.getenv("X_API_KEY")}
openai.base_url = "https://gateway.salesforceresearch.ai/openai/process/v1/"

client = openai.OpenAI(
    base_url="https://gateway.salesforceresearch.ai/openai/process/v1/",
    api_key="dummy",
    default_headers={"X-Api-Key": os.getenv("X_API_KEY")}
)

def load_qa_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)['scenarios']

def upload_image_file(image_path, base_dir, client):
    try:
        full_path = Path(base_dir) / image_path.lstrip('./')
        with open(full_path, 'rb') as f:
            return client.files.create(file=f, purpose="vision").id
    except:
        try:
            return client.files.create(file=image_path, purpose="vision").id
        except:
            raw_bytes = base64.b64decode(image_path)
            return client.files.create(file=("image.png", raw_bytes), purpose="vision").id

    return None

def save_response(response_data, output_dir, scenario_id):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"trajectory_{scenario_id}.json"
    file_path = output_path / filename
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, indent=2, ensure_ascii=False)
    
    print(f"Response saved to: {file_path}")
    return file_path

def process_scenarios(scenarios, start_idx=0, end_idx=None, image_base_dir=None, output_dir="./responses"):
    if end_idx is None:
        end_idx = len(scenarios)
    
    start_idx = max(0, start_idx)
    end_idx = min(len(scenarios), end_idx)
    selected_scenarios = scenarios[start_idx:end_idx]
    
    print(f"Processing {len(selected_scenarios)} questions ({start_idx} to {end_idx-1})...")
    
    for i, scenario in enumerate(selected_scenarios):
        scenario_id = scenario['id']
        question_data = scenario['question']
        ground_truth = scenario['ground_truth']
        question_text = question_data['text']
        
        print(f"\n--- Question {i+1}/{len(selected_scenarios)} (index: {start_idx + i}) ---")
        print(f"Scenario ID: {scenario_id}")
        
        # Upload image
        file_id = None
        if image_base_dir:
            if type(question_data['image']) == str:
                file_ids = []
                file_id = upload_image_file(question_data['image'], image_base_dir, client)
                if file_id:
                    print(f"Image uploaded: {file_id}")
            else:
                file_id = None
                file_ids = []
                for img_byte in question_data['image']:
                    file_ids.append(upload_image_file(img_byte, image_base_dir, client))
                if file_ids:
                    print("file_ids", file_ids)
        
        # Prepare API input
        content = [{"type": "input_text", "text": question_text}]
        if file_id:
            content.append({"type": "input_image", "image_url": None, "file_id": file_id})
        if file_ids:
            for file_id in file_ids:
                content.append({"type": "input_image", "image_url": None, "file_id": file_id})

        multimodal_input = [{"role": "user", "content": content}]
        print(multimodal_input)
        
        # Call API
        try:
            start_time = time.time()
            response = client.responses.create(
                model="gpt-5",
                tools=[{"type": "web_search"}],
                input=multimodal_input
            )
            execution_time = time.time() - start_time
            
            print(f"Response received in {execution_time:.2f}s")
            
            # Extract content
            reasoning_content = None
            web_search_results = []
            raw_content = []
            
            if hasattr(response, 'content') and response.content:
                for item in response.content:
                    item_type = getattr(item, 'type', 'unknown')
                    if item_type == 'reasoning' and hasattr(item, 'encrypted_content'):
                        reasoning_content = item.encrypted_content
                    elif item_type == 'web_search_call' and hasattr(item, 'results'):
                        web_search_results.append(item.results)
                    
                    raw_content.append({
                        "type": item_type,
                        "content": getattr(item, 'text', getattr(item, 'encrypted_content', 
                                         getattr(item, 'results', str(item))))
                    })
            
            # Save response
            response_data = {
                "scenario_id": scenario_id,
                "question": question_data,
                "ground_truth": ground_truth,
                "start_time": start_time,
                "end_time": time.time(),
                "duration": execution_time,
                "final_answer": response.output_text,
                "reasoning_encrypted_content": reasoning_content,
                "web_search_results": web_search_results,
                "raw_response_content": raw_content,
                "final_answer_tokens": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens+response.usage.output_tokens
                }
            }
            
            save_response(response_data, output_dir, scenario_id)
            
        except Exception as e:
            print(f"Error processing {scenario_id}: {str(e)}")
            error_data = {
                "scenario_id": scenario_id,
                "question": question_data,
                "ground_truth": ground_truth,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            save_response(error_data, output_dir, f"{scenario_id}_error")

# Main execution
if __name__ == "__main__":
    # Configuration
    json_path = "./FVQA/qa_formatted.json"
    image_base_dir = "./Paper/images"
    output_dir = "./gpt_5_responses_FVQA"
    
    # Set range (modify these as needed)
    start_idx = 0
    end_idx = 100  # None = process all remaining
    
    # Load and process
    scenarios = load_qa_data(json_path)
    print(f"Total questions available: {len(scenarios)}")
    
    process_scenarios(scenarios, start_idx, end_idx, image_base_dir, output_dir)
    print(f"\nCompleted processing!")