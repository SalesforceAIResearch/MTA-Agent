# Multimodal Deep Research

A research system that can analyze images and text to answer questions by searching the web and running code.

## What it does

This system can:
- Look at images and understand what's in them
- Search the internet for information
- Run code to analyze data
- Write research reports based on what it finds

## Quick Start

### 1. Download Data
Download data from [Google Drive](https://drive.google.com/drive/folders/1tbux8pyUc3MUDXPmP-e-E-iZQcJZHzWm?usp=sharing) and save it as `examples/train/mm_deep_research/input_mm`.

### 2. Set up API Keys
```bash
# Required for GPT summarization (used in multimodal agent)
export OPENAI_API_KEY="your-api-key-here"
# OR for custom OpenAI-compatible API
export X_API_KEY="your-api-key-here"

# Required for web search tools (Tavily API)
export TAVILY_API_KEY="your-tavily-key-here"

# Required for reverse image search (SerpAPI Google Lens)
export SERPAPI_API_KEY="your-serpapi-key-here"

# Optional: Alternative search API (if not using Tavily)
export SERPER_API_KEY="your-serper-key-here"
```

**Note**: The training scripts use the following tools which require API keys:
- `web_text_to_text_search`: Requires `TAVILY_API_KEY`
- `web_text_to_img_search`: Requires `TAVILY_API_KEY`
- `web_url_reader`: Requires `TAVILY_API_KEY`
- `web_image_to_text`: Requires `SERPAPI_API_KEY`
- Multimodal agent summarization: Requires `OPENAI_API_KEY` or `X_API_KEY`

### 3. Start the System
```bash
./deploy_servers.sh
```
then:
```bash
cd cd examples/train/mm_deep_research
./test_mm_simple.sh
```

## How It Works

The system thinks step by step:

1. **Look at the question** - Understand what you're asking
2. **Decide what to do** - Choose a tool (search web, analyze image, etc.)
3. **Do it** - Execute the action
4. **Check results** - See if it found what it needs
5. **Repeat** - Keep going until it has enough information
6. **Write answer** - Create a final report

### Available Tools
- **Web Search** - Find information online
- **Image Analysis** - Look at and understand images
- **Code Execution** - Run Python code for analysis
- **File Operations** - Read and write files

The system stops when it has enough information or reaches the maximum number of steps (6).

## Results

The system creates several files when it runs:

- `trajectory_{scenario_id}.json` - Shows all the steps it took
- `evaluation_{scenario_id}.json` - How well it did
- `test_summary.json` - Overall results
- `test_execution.log` - Detailed log of what happened

### How Well It Performs
- **Success Rate**: Usually 85%+ of actions work correctly
- **Tool Accuracy**: 95%+ of tool uses are valid
- **Efficiency**: Stops early when it has enough information

## Configuration

### Input Format
Your research questions should be in JSON format:
```json
{
  "scenarios": [
    {
      "id": "scenario_1",
      "question": {
        "text": "Research question here",
        "image": "path/to/image.jpg"
      },
      "ground_truth": "Expected answer"
    }
  ]
}
```

### Settings
- `max_scenarios`: How many questions to test (default: 3)
- `gpt_model`: Which AI model to use (default: "gpt-4o")
- `max_iterations`: Maximum steps per question (default: 6)

## Example Usage

```python
from test_mm_deep_research import MultimodalDeepResearchTester

# Create tester
tester = MultimodalDeepResearchTester(
    server_url="http://localhost:4000/get_observation",
    gpt_model="gpt-4o"
)

# Run test
results = tester.run_multimodal_test(max_scenarios=5)
print(f"Quality Score: {results['overall_metrics']['avg_overall_quality_score']:.3f}")
```

## Use Cases

- **Academic Research** - Analyze research papers and images
- **Fact-Checking** - Verify information from multiple sources
- **Data Analysis** - Process images and text data together
- **Content Creation** - Research and write articles
- **Investigation** - Gather evidence from different sources

## Troubleshooting

### Common Problems

**"No valid tool found"**
- Make sure you ran `./deploy_servers.sh` first

**API Key Errors**
- Check that your OpenAI and SERPER API keys are set correctly

**Image Not Found**
- Make sure image paths in your scenario files are correct

**High Token Usage**
- Try using a smaller model like `gpt-3.5-turbo` for testing

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Related Documentation

- [Workflow Explanation](workflow_explanation.md) - How the system works
- [Tool Server Design](../../../assets/docs/tool_server.md) - Server setup
- [Training Guide](../../../assets/docs/training_guide.md) - Training models

## Contributing

Contributions welcome! See the main [Contributing Guide](../../../assets/docs/contributing.md) for details.

## License

This project is part of VerlTool. See the main repository for license information.
