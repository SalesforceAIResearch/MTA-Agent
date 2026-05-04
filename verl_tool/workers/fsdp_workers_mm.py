"""
Multimodal FSDP Workers that use agent loop registry directly.
This module extends fsdp_workers.py to use agent loops from the registry.
"""
import logging
import os
from verl.workers.fsdp_workers import ActorRolloutRefWorker, Worker, DictConfig
from verl.workers.fsdp_workers import *
from verl.utils.profiler import DistProfiler, DistProfilerExtension, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing
from verl.protocol import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from omegaconf import OmegaConf
from .utils import SiblingMetaClass, SiblingMarker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def dispatch_no_change(worker_group, *args, **kwargs):
    return args, kwargs

def collect_dp_compute(worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup
    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size
    return output

class AgentActorRolloutRefWorkerMM(Worker, DistProfilerExtension, ActorRolloutRefWorker, SiblingMarker, metaclass=SiblingMetaClass):
    """
    Multimodal version of AgentActorRolloutRefWorker that uses agent loop registry directly.
    The agent loop is loaded from the registry based on config.actor_rollout_ref.rollout.agent.default_agent_loop.
    """
    def __init__(self, config: DictConfig, role: str, **kwargs):
        # Call ActorRolloutRefWorker.__init__ directly (not Worker.__init__)
        # This properly handles profile_option and sets up self.config
        ActorRolloutRefWorker.__init__(self, config, role, **kwargs)
        
        # Extract agent config to check if agent mode is enabled
        agent_config = OmegaConf.select(self.config, 'actor_rollout_ref.rollout.agent', default={})
        self.enable_agent = OmegaConf.select(agent_config, 'enable_agent', default=False)
        self.default_agent_loop = OmegaConf.select(agent_config, 'default_agent_loop', default='verltool_agent')
        
        logger.info(f"[AgentActorRolloutRefWorkerMM] enable_agent={self.enable_agent}, default_agent_loop={self.default_agent_loop}")
        
        # The rollout system will handle agent loops through the registry
        # No need for manager classes

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            with simple_timer("generate_sequences", timing_generate):
                # The rollout system handles agent loops automatically through the registry
                # Just call the normal rollout.generate_sequences - it will use agent loops if enabled
                output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    # this is for issue https://github.com/volcengine/verl/issues/2613#issuecomment-3112156628
    # resume from checkpoint first val will have bad performance numbers without this modification
    # seems because of the fsdp weights not updated to vllm
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        # load the weight to vllm
        self.rollout_sharding_manager.__enter__()
        self.rollout_sharding_manager.__exit__(None, None, None)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

