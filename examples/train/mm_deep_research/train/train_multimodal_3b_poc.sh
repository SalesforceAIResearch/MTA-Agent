#!/bin/bash
set -x

# Multimodal Deep Research Training Configuration - 3B Model (TEST VERSION)
# Reduced rollout counts for quick testing of MM branch code
# Optimized for Qwen2.5-VL-3B with reduced resource requirements

dataset_name=mm_deepsearch/mmsearch_plus  # Using MMSearch-Plus data for multimodal_deepsearch training
train_data=[$(pwd)/data/${dataset_name}/train.parquet]
val_data=[$(pwd)/data/${dataset_name}/test.parquet]  # Using test split as validation data (or use FVQA test for validation)
model_name=Qwen/Qwen2.5-VL-3B-Instruct  # 3B parameter vision-language model for multimodal capabilities
rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
n_gpus_per_node=4  # Reduced from 8 for 3B model
n_nodes=1
# TEST MODE: Reduced rollout counts
# Note: batch_size must be large enough after normalization (divided by num_gpus)
# With n_gpus_per_node=4, batch_size will be divided by 4, so we need at least 16
n=2  # Reduced from 8 for testing (number of rollouts per prompt)
batch_size=16  # Increased from 8 to ensure enough samples after GPU normalization (number of prompts per batch)
ppo_mini_batch_size=8  # Increased from 4 to ensure divisibility (PPO mini-batch size)
# Total rollouts per batch = n * batch_size = 2 * 16 = 32 (vs original 8 * 64 = 512)
max_prompt_length=16384  # Should be big to avoid any truncation of image tokens which will cause error
max_response_length=16384  # Should be big for detailed responses with images
max_action_length=2048  # Sufficient for tool calls
max_obs_length=8192  # Should be big to avoid any truncation of image tokens which will cause error
temperature=1.0
top_p=1.0
enable_agent=True # enable agent for tool use
strategy="fsdp2"  # Changed to fsdp2 for better 3B model performance
action_stop_tokens='</python>,</text_search_text>,</text_search_image>,</web_read>,</image_search_text>,</bash>'  # Action stop tokens for all enabled tools
max_turns=4  # Multiple turns for complex multimodal research
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=2e-6  # Slightly higher learning rate for 3B model
reward_manager=multimodal_deepsearch  # Multimodal deep research reward manager
ppo_micro_batch_size_per_gpu=2  # Must be <= normalized ppo_mini_batch_size
log_prob_micro_batch_size_per_gpu=4  # Reduced from 16, but must be <= normalized batch_size per GPU
tensor_model_parallel_size=2  # Use 2 for better memory management with images
gpu_memory_utilization=0.8  # Higher gpu_memory_utilization for 3B model
do_offload=True  # Enable offload to prevent OOM with high gpu_memory_utilization
use_dynamic_bsz=False  # Set to False for stability with images
ulysses_sequence_parallel_size=1 # set to 1 for normal verl behavior, otherwise it will cause OOM
fsdp_size=-1
additional_eos_token_ids=[151645] # <|im_end|> token id
mask_observations=True # mask observations for kl loss and gradient descent
enable_mtrl=True # enable multi-turn training for complex research
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
max_num_batched_tokens=5000  # Limit batched tokens for memory efficiency
run_name_postfix="multimodal-research-3b-test"
if [ "$enable_agent" = "True" ]; then
    run_name="${reward_manager}-${strategy}-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
else
    run_name="${reward_manager}-${strategy}-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
fi
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
rollout_mode='async'

# temp file for action tokens as verl cannot pass special strs as params
action_stop_tokens_file="$(mktemp)"
mkdir -p $(dirname $action_stop_tokens_file)
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
# Start multimodal tool server with web search, code execution, and bash terminal
python -m verl_tool.servers.serve --host $host --port $port --tool_type "web_text_to_text_search,web_text_to_img_search,web_url_reader,web_image_to_text,ipython_code,bash_terminal" --workers_per_tool 2 --use_ray True &
server_pid=$!

echo "Multimodal tool server (pid=$server_pid) started at $tool_server_url"

unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo_mm \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.dataloader_num_workers=0 \
    data.train_batch_size=$batch_size \
    data.val_batch_size=16 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.max_concurrent_trajectories=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=multimodal_deepsearch \
    trainer.experiment_name=$run_name \
    trainer.resume_mode=disable \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=20

# Clean up
pkill -P -9 $server_pid
kill -9 $server_pid

