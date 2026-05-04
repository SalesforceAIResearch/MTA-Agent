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
Multimodal Deep Search Reward Manager

This package provides a reward system for training multimodal deep research models
that can analyze images and use web search and code execution tools.

Key Features:
- Base TORL math scoring (0 or 1)
- Tool call reward: +0.1 for using tools
- Format penalty: -1.0 for missing <think> or \boxed{}
- Multimodal support: handles images via <|image_pad|> tokens
- Tool interaction logging with image cropping

Usage:
    from verl_tool.workers.reward_manager.multimodal_deepsearch import MultimodalDeepSearchRewardManager
    
    reward_manager = MultimodalDeepSearchRewardManager(
        tokenizer=tokenizer,
        num_examine=1,
    )
"""

from .multimodal_deepsearch import (
    MultimodalDeepSearchRewardManager,
    multimodal_deepsearch_compute_score,
)

__all__ = [
    'MultimodalDeepSearchRewardManager',
    'multimodal_deepsearch_compute_score',
]