# Multimodal Deep Research Training Scripts

This directory contains training scripts and configurations for multimodal deep research models that can analyze images, search the web, and execute code to answer complex research questions.

## Files

- **`train_multimodal_7b.sh`**: Main training script for multimodal deep research with 4B parameter models

## Overview

The training system is designed to train vision-language models for comprehensive research tasks that require:
- **Visual Analysis**: Understanding and analyzing images/charts
- **Information Gathering**: Web search for additional context
- **Data Processing**: Code execution for analysis
- **Synthesis**: Combining multiple information sources

## Quick Start

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Keys**:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export SERPER_API_KEY="your-serper-key"
   ```

3. **Prepare Training Data**:
   Create your training data in the following format:
   ```json
   {
     "prompt": "Research question with image reference",
     "images": ["path/to/image.jpg"],
     "ground_truth": "Expected answer"
   }
   ```

### Running Training

1. **Make Script Executable**:
   ```bash
   chmod +x train_multimodal_4b.sh
   ```

2. **Start Training**:

   **Option A: 7B Model (Higher Performance)**:
   ```bash
   ./train_multimodal_7b.sh
   ```

   **Option B: 3B Model (Lower Resource Requirements)**:
   ```bash
   ./train_multimodal_3b.sh
   ```

## Training Configuration

### Model Settings

#### 7B Model Configuration
- **Model**: `Qwen/Qwen2.5-VL-7B` (7B parameter vision-language model)
- **Strategy**: `fsdp` (Fully Sharded Data Parallel)
- **GPUs**: 8 GPUs per node
- **Batch Size**: 128
- **Max Prompt Length**: 8192 tokens
- **Max Response Length**: 16384 tokens
- **Max Turns**: 6

#### 3B Model Configuration
- **Model**: `Qwen/Qwen2.5-VL-3B` (3B parameter vision-language model)
- **Strategy**: `fsdp2` (Fully Sharded Data Parallel v2)
- **GPUs**: 4 GPUs per node
- **Batch Size**: 64
- **Max Prompt Length**: 4096 tokens
- **Max Response Length**: 8192 tokens
- **Max Turns**: 4

### Hardware Requirements

#### 7B Model Requirements
- **GPUs**: 8 GPUs per node
- **Memory**: High GPU memory utilization (0.6) with offloading enabled
- **Strategy**: `fsdp` with parameter/optimizer offloading

#### 3B Model Requirements
- **GPUs**: 4 GPUs per node
- **Memory**: Higher GPU memory utilization (0.8) without offloading
- **Strategy**: `fsdp2` for better performance
- **Storage**: Sufficient space for model checkpoints and logs

### Training Parameters

```bash
# Core training settings
n=16                    # Number of samples per prompt
batch_size=128          # Training batch size
ppo_mini_batch_size=32  # PPO mini-batch size
lr=1e-6                 # Learning rate

# Multimodal settings
max_prompt_length=8192      # Increased for images
max_response_length=16384  # Increased for detailed research
max_obs_length=8192        # Increased for image observations
enable_mtrl=True           # Enable multi-turn training
max_turns=6                # Increased for complex research
```

### Tool Configuration

The training uses a multimodal tool server with:
- **Google Search**: Web search capabilities
- **Python Code**: Code execution environment
- **Image Analysis**: Visual analysis tools

```bash
# Tool server configuration
python -m verl_tool.servers.serve \
    --tool_type "google_search,python_code,image_analysis" \
    --workers_per_tool 4 \
    --use_ray True
```

## Expected Training Behavior

### High-Quality Responses

The system will learn to generate responses that:
- ✅ Use multiple tool types (image analysis + web search + code)
- ✅ Synthesize information from different sources
- ✅ Provide comprehensive, well-structured answers
- ✅ Follow proper format with `<think>` and `\boxed{answer}`
- ✅ Include evidence-based reasoning

### Response Format

The model learns to generate responses in this format:
```
<think>
Your reasoning process here. Consider the image, search results, and code outputs.
</think>

<image_analysis>{"query": "analyze the growth trends in this chart"}</image_analysis>
<search>machine learning trends 2024</search>
<python>import pandas as pd; data = pd.read_csv('data.csv')</python>

Based on my comprehensive analysis of the image and extensive research, I can provide a detailed answer.
According to the data analysis, the trends are clear. The research shows significant findings.
From the visual analysis, I can see distinct patterns that support this conclusion.

Therefore, the answer is \boxed{42}.
```

## Monitoring Training

### Logs and Outputs

The training process generates:
- **Console Output**: Real-time training progress
- **Wandb Logs**: Detailed metrics and visualizations
- **Checkpoints**: Model saves every 10 epochs
- **Step Records**: Detailed reward and response logs

### Key Metrics to Monitor

- **Reward Scores**: Overall reward distribution
- **Tool Usage**: Frequency of different tool types
- **Multimodal Synthesis**: Success rate of combining image + text
- **Format Compliance**: Adherence to response format
- **Research Depth**: Number of unique tools used per response

### Example Training Output

```
[prompt] Analyze this chart and research the latest trends in machine learning
[response] <think>I need to analyze the chart and search for current ML trends</think>
<image_analysis>{"query": "analyze the growth trends in this chart"}</image_analysis>
<search>machine learning trends 2024</search>
<python>import matplotlib.pyplot as plt; # analyze the data</python>
Based on my analysis of the chart and current research, the ML market is growing at 25% annually.
The answer is \boxed{25%}.
[ground_truth] 25%
[accuracy] 1
[score] 1.3
[image_analysis_reward] 1
[multimodal_synthesis_reward] 1
[research_depth_reward] 0.3
[python_reward] 0.1
[search_reward] 0.1
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM) Errors**:
   ```bash
   # Reduce GPU memory utilization
   gpu_memory_utilization=0.4
   
   # Enable offloading
   do_offload=True
   ```

2. **Tool Server Connection Issues**:
   - Check that the tool server is running
   - Verify API keys are set correctly
   - Check network connectivity

3. **Low Reward Scores**:
   - Verify multimodal synthesis is working
   - Check tool usage patterns
   - Ensure proper response format

4. **Training Instability**:
   - Reduce learning rate: `lr=5e-7`
   - Increase warmup steps: `lr_warmup_steps=20`
   - Adjust batch sizes

### Debug Mode

Enable detailed logging:
```bash
export VERL_DEBUG=1
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
```

### Performance Optimization

1. **Memory Optimization**:
   ```bash
   # Use dynamic batch sizing
   use_dynamic_bsz=True
   
   # Enable offloading for large models
   do_offload=True
   ```

2. **Speed Optimization**:
   ```bash
   # Use Ray for tool server
   --use_ray True
   
   # Optimize batch sizes
   ppo_micro_batch_size_per_gpu=1
   log_prob_micro_batch_size_per_gpu=8
   ```

## Customization

### Modifying Training Parameters

Edit the training script to adjust:
- Model architecture and size
- Learning rate and schedule
- Batch sizes and parallelism
- Tool server configuration
- Reward system parameters

### Adding New Tools

1. **Update Tool Server**:
   ```bash
   --tool_type "google_search,python_code,image_analysis,your_new_tool"
   ```

2. **Update Action Stop Tokens**:
   ```bash
   action_stop_tokens='</python>,</search>,</image_analysis>,</your_new_tool>'
   ```

3. **Update Reward System**:
   - Add tool detection in reward manager
   - Update reward weights
   - Add corresponding tests

### Custom Data Formats

Modify the data loading to support:
- Different image formats
- Custom prompt templates
- Alternative response formats
- Domain-specific evaluation metrics

## Advanced Usage

### Multi-Node Training

For larger models or datasets:
```bash
# Update node configuration
n_nodes=2
n_gpus_per_node=8

# Ensure proper networking
export NCCL_DEBUG=INFO
```

### Hyperparameter Tuning

Key parameters to tune:
- **Learning Rate**: `lr` (1e-6 to 1e-5)
- **Batch Size**: `batch_size` (64 to 256)
- **Temperature**: `temperature` (0.8 to 1.2)
- **Max Turns**: `max_turns` (3 to 8)
- **Reward Weights**: In reward manager configuration

### Model Evaluation

After training, evaluate the model:
```bash
# Run evaluation script
python ../test/example_usage.py

# Check reward system
cd ../../../verl_tool/workers/reward_manager/
python test_multimodal_deepsearch.py
```

## Related Documentation

- **Main README**: `../README.md`
- **Reward Manager**: `../../../verl_tool/workers/reward_manager/multimodal_deepsearch/`
- **Tool Server**: `../../../verl_tool/servers/README.md`
- **General Training**: `../../README.md`

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Verify all dependencies are installed
4. Ensure API keys are correctly set
5. Check hardware requirements are met

## License

This implementation follows the same license as the parent verl-tool project.
