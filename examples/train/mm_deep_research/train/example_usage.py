#!/usr/bin/env python3
"""
Example usage of MultimodalDeepSearchRewardManager
Demonstrates how to use the reward system for multimodal research tasks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from verl_tool.workers.reward_manager.multimodal_deepsearch import MultimodalDeepSearchRewardManager
from transformers import AutoTokenizer
import torch


def create_example_responses():
    """Create example responses for testing"""
    
    # High-quality multimodal response
    excellent_response = """
    <think>
    I need to analyze the chart and search for additional information to provide a comprehensive answer.
    According to the research, the data shows clear trends. From the image analysis, I can see specific patterns.
    The evidence suggests a strong correlation between the variables.
    </think>
    
    <image_analysis>{"query": "Analyze the growth trends in this chart"}</image_analysis>
    <search>machine learning market growth 2024</search>
    <python>
    import pandas as pd
    import matplotlib.pyplot as plt
    data = pd.read_csv('market_data.csv')
    growth_rate = data['growth'].mean()
    print(f"Average growth rate: {growth_rate}%")
    </python>
    
    Based on my comprehensive analysis of the chart and extensive research, I can provide a detailed answer.
    According to the data analysis, the trends are clear. The research shows significant findings.
    From the visual analysis, I can see distinct patterns that support this conclusion.
    The evidence suggests a strong relationship between the factors.
    
    Therefore, the answer is \\boxed{25.3}.
    """
    
    # Good response with some tools
    good_response = """
    <think>I need to analyze this image and search for information</think>
    <image_analysis>{"query": "What does this chart show?"}</image_analysis>
    <search>chart analysis techniques</search>
    
    Based on the image analysis and research, the chart shows a clear upward trend.
    The answer is \\boxed{25.3}.
    """
    
    # Poor response with wrong format
    poor_response = """
    I think the answer is 25.3 based on looking at the chart.
    """
    
    # Wrong answer
    wrong_response = """
    <think>I need to solve this</think>
    <image_analysis>{"query": "Analyze the chart"}</image_analysis>
    <search>find the answer</search>
    
    Based on my analysis, the answer is \\boxed{30.5}.
    """
    
    return {
        'excellent': excellent_response,
        'good': good_response,
        'poor': poor_response,
        'wrong': wrong_response
    }


def create_mock_data_item(question_text, response_text, ground_truth):
    """Create a mock data item for testing"""
    return type('MockDataItem', (), {
        'non_tensor_batch': {
            'question': {'text': question_text},
            'reward_model': {'ground_truth': ground_truth},
            'data_source': 'example_source',
            'extra_info': {'id': 'example_1'}
        }
    })()


def demonstrate_reward_computation():
    """Demonstrate how the reward system works"""
    
    print("=" * 80)
    print("Multimodal Deep Research Reward System Demonstration")
    print("=" * 80)
    
    # Create a mock tokenizer
    tokenizer = type('MockTokenizer', (), {
        'decode': lambda x, **kwargs: x,
        'pad_token_id': 0
    })()
    
    # Initialize reward manager
    reward_manager = MultimodalDeepSearchRewardManager(
        tokenizer=tokenizer,
        num_examine=1,
    )
    
    # Get example responses
    responses = create_example_responses()
    question = "What is the growth rate shown in the chart?"
    ground_truth = "25.3"
    
    print(f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")
    print()
    
    # Test each response type
    for response_type, response in responses.items():
        print(f"--- {response_type.upper()} RESPONSE ---")
        print(f"Response: {response[:100]}...")
        
        # Create mock data item
        data_item = create_mock_data_item(question, response, ground_truth)
        
        # Compute base score
        base_score = reward_manager.compute_score(response, ground_truth)
        print(f"Base Score: {base_score:.3f}")
        
        # Add penalties and rewards
        scores = {'score': base_score, 'accuracy': 1 if base_score > 0 else 0}
        final_scores = reward_manager.add_additional_penalties(response, data_item, scores)
        
        print(f"Final Score: {final_scores['score']:.3f}")
        print(f"Accuracy: {final_scores['accuracy']}")
        
        # Show detailed breakdown
        print("Reward Breakdown:")
        for key, value in final_scores.items():
            if key.endswith('_reward') and value > 0:
                print(f"  {key}: +{value:.3f}")
            elif key.endswith('_penalty') and value > 0:
                print(f"  {key}: -{value:.3f}")
        
        print()


def demonstrate_tool_analysis():
    """Demonstrate tool usage analysis"""
    
    print("=" * 80)
    print("Tool Usage Analysis")
    print("=" * 80)
    
    from verl_tool.workers.reward_manager.multimodal_deepsearch import (
        has_image_analysis_actions,
        has_multimodal_synthesis,
        count_unique_tools_used,
        is_comprehensive_answer
    )
    
    responses = create_example_responses()
    
    for response_type, response in responses.items():
        print(f"--- {response_type.upper()} RESPONSE ANALYSIS ---")
        print(f"Has Image Analysis: {has_image_analysis_actions(response)}")
        print(f"Has Multimodal Synthesis: {has_multimodal_synthesis(response)}")
        print(f"Unique Tools Used: {count_unique_tools_used(response)}")
        print(f"Is Comprehensive: {is_comprehensive_answer(response, '25.3')}")
        print()


def main():
    """Main demonstration function"""
    try:
        demonstrate_reward_computation()
        demonstrate_tool_analysis()
        
        print("=" * 80)
        print("✅ Demonstration completed successfully!")
        print("=" * 80)
        
        print("\nKey Insights:")
        print("1. Excellent responses get high rewards for multimodal synthesis")
        print("2. Good responses get moderate rewards for tool usage")
        print("3. Poor responses get penalties for format violations")
        print("4. Wrong answers get low scores regardless of tool usage")
        print("5. The system encourages comprehensive, evidence-based research")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
