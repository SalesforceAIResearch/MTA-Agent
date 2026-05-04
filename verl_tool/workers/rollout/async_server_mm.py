"""
Multimodal Async Server with VerlToolChatCompletionSchedulerMM support.
This module extends async_server.py to use VerlToolChatCompletionSchedulerMM by default.
"""
import asyncio
import logging
from typing import Type
from verl_tool.agent_loop import AgentLoopManager
from .chat_scheduler_mm import VerlToolChatCompletionSchedulerMM
from verl.protocol import DataProto
logger = logging.getLogger(__file__)

class VerlToolAsyncLLMServerManagerMM(AgentLoopManager):
    """
    Multimodal version of AsyncLLMServerManager that uses VerlToolChatCompletionSchedulerMM by default.
    This provides ReAct reasoning pattern and GPT-4 summarization capabilities.
    """

    def _init_chat_scheduler(self):
        self.chat_scheduler_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.chat_scheduler_loop)
        print("VerlToolChatCompletionSchedulerMM")
        self.chat_scheduler = VerlToolChatCompletionSchedulerMM(
            config=self.full_config,
            server_addresses=self.server_addresses,
        )

        self.chat_scheduler_ready.set()
        self.chat_scheduler_loop.run_forever()
    
    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        self.wake_up()
        result = super().generate_sequences(prompts, **sampling_params)
        self.sleep()
        return result

# here are the hacky parts to replace the original AgentLoopManager with VerlToolAsyncLLMServerManagerMM
import verl_tool.agent_loop
import verl.experimental.agent_loop
verl_tool.agent_loop.AgentLoopManager = VerlToolAsyncLLMServerManagerMM # replace the original AgentLoopManager with VerlToolAsyncLLMServerManagerMM
verl.experimental.agent_loop.AgentLoopManager = VerlToolAsyncLLMServerManagerMM # replace the original AgentLoopManager with VerlToolAsyncLLMServerManagerMM

