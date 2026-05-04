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

import json
import os
import regex as re
from collections import defaultdict
from typing import Union, List, Optional

import numpy as np
import torch

from verl import DataProto
from ..reward_score.torl_math import compute_score as torl_compute_score, is_equiv
from verl.workers.reward_manager import register
from ..torl import ToRLRewardManager, extract_answer
from ..utils import replace_consecutive_tokens

# LLM judge prompt for fallback when rule-based score is 0
LLM_JUDGE_PROMPT = """You are an expert answer evaluator. Compare the model's answer to the ground truth and determine if they are semantically equivalent.

Question: {question}

Ground Truth Answer: {ground_truth}

Model's Answer: {final_answer}

Instructions:
- The model's answer is correct if it conveys the same core information as the ground truth, even if worded differently.
- Minor differences in phrasing, formatting, or additional context are acceptable if the key answer is correct.
- Look for the actual answer, which might be inside \\boxed{{}}.
- Additionally, check if the ground truth answer appears anywhere in the model's reasoning/explanation outside of \\boxed{{}}. This indicates correct reasoning even if the final answer format is different.

Respond with a JSON object:
{{"is_correct": true/false, "is_correct_reasoning": true/false, "reason": "brief explanation"}}

Where:
- "is_correct": true if the answer (especially in \\boxed{{}}) matches the ground truth
- "is_correct_reasoning": true if the ground truth can be found in the reasoning/explanation outside of \\boxed{{}}, even if "is_correct" is false"""


def _call_llm_judge(
    question: str,
    ground_truth: Union[str, List[str]],
    model_answer: str,
    model: str = "gpt-5-mini",
    reasoning_score: float = 0.5,
) -> float:
    """
    Call LLM (e.g. GPT-5-mini) to judge answer correctness when rule-based score is 0.
    Returns 1.0 if is_correct, reasoning_score (default 0.5) if only is_correct_reasoning, else 0.0.
    """
    gt_str = ground_truth if isinstance(ground_truth, str) else " | ".join(ground_truth)
    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        ground_truth=gt_str,
        final_answer=model_answer,
    )
    api_key = os.getenv("X_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return 0.0
    try:
        from openai import OpenAI
        base_url = "https://gateway.salesforceresearch.ai/openai/process/v1/" if os.getenv("X_API_KEY") else None
        headers = {"X-Api-Key": api_key} if os.getenv("X_API_KEY") else None
        client = OpenAI(
            api_key=api_key if not headers else "dummy",
            base_url=base_url,
            default_headers=headers,
        ) if base_url else OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = (resp.choices[0].message.content or "").strip()
        # Parse JSON from response (may be wrapped in markdown code block)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        data = json.loads(content)
        is_correct = data.get("is_correct", False)
        is_correct_reasoning = data.get("is_correct_reasoning", False)
        # is_correct=True: score same as exact match (1.0)
        # is_correct=False and is_correct_reasoning=True: score=0.5 (reasoning_score)
        if is_correct:
            return 1.0
        if is_correct_reasoning:
            return reasoning_score
        return 0.0
    except Exception:
        return 0.0


def multimodal_deepsearch_compute_score(solution_str, ground_truth: Union[List[str], str]):
    """
    Scoring for multimodal deep research tasks with web search and code execution.
    Identical to deepsearch but explicitly named for multimodal context.
    Supports both text and image inputs.
    
    Handles multi-choice answers where the response might contain both the choice letter
    and the full text (e.g., "C. Churchill Downs" should match both "C" and "Churchill Downs").
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    score = 0.0
    
    # Extract boxed answer if present for multi-choice handling
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution_str)
    if boxed_match:
        boxed_answer = boxed_match.group(1).strip()
        
        # Handle multi-choice format: "C. Churchill Downs" or "C) Churchill Downs"
        # Try to extract both the choice letter and the text part
        choice_pattern = re.match(r"^([A-E])[\.\)]\s*(.+)$", boxed_answer)
        if choice_pattern:
            choice_letter = choice_pattern.group(1)
            choice_text = choice_pattern.group(2).strip()
            
            # Try matching the full answer, choice letter, and choice text separately
            candidates = [boxed_answer, choice_letter, choice_text]
            
            for candidate in candidates:
                for gt in ground_truth:
                    # Use is_equiv for direct matching (same logic as torl_compute_score)
                    if is_equiv(candidate, gt):
                        score = 1.0
                        break
                if score >= 1.0:  # Changed to >= to handle the case where score is already 1.0
                    break
            
            # Also try the standard torl_compute_score for the full answer (returns 1.0 or -1.0)
            if score < 1.0:
                for gt in ground_truth:
                    candidate_score = torl_compute_score(solution_str, gt)
                    score = max(score, candidate_score)
                    if score >= 1.0:
                        break
        else:
            # Not in multi-choice format, use standard matching
            for gt in ground_truth:
                score = max(score, torl_compute_score(solution_str, gt))
    else:
        # No boxed answer, use standard matching
        for gt in ground_truth:
            score = max(score, torl_compute_score(solution_str, gt))
    
    return score


@register("multimodal_deepsearch")
class MultimodalDeepSearchRewardManager(ToRLRewardManager):
    """
    A reward manager for multimodal deep research tasks.
    Follows deepsearch.py structure exactly, with added multimodal support.
    Supports both text and image inputs with web search and Python code execution.
    When rule-based score is 0, can optionally call an LLM (e.g. GPT-5-mini) to judge correctness.
    """
    name = "multimodal_deepsearch"

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key='data_source',
        enable_llm_judge: bool = True,
        llm_judge_model: str = "gpt-5-mini",
        llm_judge_reasoning_score: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer=tokenizer, num_examine=num_examine, compute_score=compute_score, reward_fn_key=reward_fn_key, **kwargs)
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = multimodal_deepsearch_compute_score
        self.reward_fn_key = reward_fn_key
        self.step = None
        self.add_tool_call_reward = True
        self.add_format_penalty = True
        self.enable_llm_judge = enable_llm_judge
        self.llm_judge_model = llm_judge_model
        self.llm_judge_reasoning_score = llm_judge_reasoning_score

    def __call__(self, data: DataProto, return_dict=False):
        """Same as ToRLRewardManager but when score is 0, optionally call LLM judge (e.g. GPT-5-mini)."""
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i in range(len(data)):
            score = {}
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extracted_answer = extract_answer(response_str, mode='math')

            torl_score = self.compute_score(
                solution_str=response_str,
                ground_truth=ground_truth,
            )
            # Log rule-based score before any LLM judge (for trajectory / reward_extra_info)
            score['score_before_llm_judge'] = float(torl_score)

            # When rule-based score is 0 or negative, optionally call LLM judge (e.g. GPT-5-mini)
            if torl_score <= 0 and self.enable_llm_judge:
                llm_judge_score = _call_llm_judge(
                    question=prompt_str,
                    ground_truth=ground_truth,
                    model_answer=response_str,
                    model=self.llm_judge_model,
                    reasoning_score=self.llm_judge_reasoning_score,
                )
                torl_score = llm_judge_score
                score['llm_judge_override'] = float(llm_judge_score)  # 0.0, 0.5, or 1.0 from judge
            else:
                # One value per sample so reward_extra_info length matches batch size
                score['llm_judge_override'] = None

            score['accuracy'] = 1.0 if torl_score > 0 else 0.0
            score['score'] = torl_score
            score['has_answer'] = 1.0 if extracted_answer else 0.0
            score = self.add_additional_penalties(response_str, data_item, score)

            if score['accuracy'] > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length)
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length)

            reward = score["score"]
            for key, value in score.items():
                reward_extra_info[key].append(value)
            if self.num_examine == 1:
                reward = score["accuracy"]
            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                for key, value in score.items():
                    print(f"[{key}]", value)

        correct_response_length_mean = np.mean(reward_extra_info['correct_response_length']) if reward_extra_info['correct_response_length'] else None
        wrong_response_length_mean = np.mean(reward_extra_info['wrong_response_length']) if reward_extra_info['wrong_response_length'] else None
        reward_extra_info['correct_response_length'] = [correct_response_length_mean] * len(reward_tensor)
        reward_extra_info['wrong_response_length'] = [wrong_response_length_mean] * len(reward_tensor)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(sorted(reward_extra_info.items()))}
        return reward_tensor

    def add_additional_penalties(self, response: str, data_i, scores_i: dict):
        """
        Add additional penalties and rewards.
        Exactly follows deepsearch.py logic.
        """
        # 1. Format penalty
        if self.add_format_penalty:
            # check if <think> exists in the response
            #  and if \\boxed{} exists in the response
            think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            answer_match = re.search(r"\\boxed\{.*?\}", response)
            if not think_match or not answer_match:
                scores_i['score'] = -1
                scores_i['format_penalty'] = 1
            else:
                scores_i['format_penalty'] = 0
        
        scores_i['score'] = scores_i['accuracy']
        
        # 2. Tool call reward
        if "tool_interact_info" in data_i.non_tensor_batch:
            if self.add_tool_call_reward:
                tool_interact_info = data_i.non_tensor_batch.get('tool_interact_info', None)
                num_turn = len(tool_interact_info) if tool_interact_info is not None else 0
                num_valid_action = sum([1 for t in tool_interact_info if t.get('valid_action', False)]) if tool_interact_info is not None else 0
                if num_valid_action > 0:
                    scores_i['score'] += 0.1
                    scores_i['tool_call_reward'] = 1
                else:
                    scores_i['tool_call_reward'] = 0
        
        return scores_i
    
    def _save_prompt_response_with_image_handling(self, prompt_ids, response_ids, valid_prompt_length, valid_response_length, valid_response_ids_with_loss_mask, data_item):
        """
        Helper method to save prompts and responses with proper image token handling.
        This is the multimodal-specific addition to deepsearch.py.
        """
        # Decode with image token replacement for readability
        to_save_prompt = self.tokenizer.decode(prompt_ids[-valid_prompt_length:], skip_special_tokens=False)
        to_save_response = self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=False)
        
        # Replace consecutive image pad tokens for cleaner logs
        to_save_prompt = replace_consecutive_tokens(to_save_prompt, token="<|image_pad|>")
        to_save_response = replace_consecutive_tokens(to_save_response, token="<|image_pad|>")
        
        # Handle loss mask if present
        if 'responses_with_loss_mask' in data_item.batch:
            to_save_response_with_loss_mask = self.tokenizer.decode(valid_response_ids_with_loss_mask, skip_special_tokens=False)
            to_save_response_with_loss_mask = replace_consecutive_tokens(to_save_response_with_loss_mask, token=self.tokenizer.pad_token)
        else:
            to_save_response_with_loss_mask = None
            
        return to_save_prompt, to_save_response, to_save_response_with_loss_mask
    
    def _process_tool_interact_info(self, tool_interact_info):
        """
        Process tool interaction info for storage, cropping images if present.
        This is the multimodal-specific addition to deepsearch.py.
        """
        if tool_interact_info is not None:
            # Crop the image for storage efficiency
            for tool_interact in tool_interact_info:
                if "image" in tool_interact:
                    if isinstance(tool_interact['image'], list):
                        tool_interact['image'] = [x[:50] for x in tool_interact['image']]
                    elif isinstance(tool_interact['image'], str):
                        tool_interact['image'] = tool_interact['image'][:50]
        return tool_interact_info
    