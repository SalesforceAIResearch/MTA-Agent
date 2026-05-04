#!/bin/bash
# Deploy tool server for multimodal deep research
# This script starts the tool server with web search, image search, and code execution tools

# Load environment variables from .env only if not already set
if [ -f .env ]; then
    echo "Loading environment variables from .env file (if not already set)..."
    # Load .env file, only setting variables that aren't already set
    set -a
    source <(grep -v '^#' .env | grep -v '^$' | sed 's/^/export /')
    set +a
    echo "✓ Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Please create it with your API keys if they're not already set."
    echo "Example .env file content:"
    echo "TAVILY_API_KEY=your_tavily_key_here"
    echo "SERPAPI_API_KEY=your_serpapi_key_here"
    echo "OPENAI_API_KEY=your_openai_key_here"
    echo "X_API_KEY=your_x_api_key_here"
    echo "SERPER_API_KEY=your_serper_key_here  # Optional: alternative search API"
fi

# Export API keys to ensure they're available to the Python process
export TAVILY_API_KEY=${TAVILY_API_KEY:-""}
export SERPAPI_API_KEY=${SERPAPI_API_KEY:-""}
export OPENAI_API_KEY=${OPENAI_API_KEY:-""}
export X_API_KEY=${X_API_KEY:-""}

# Verify critical API keys are set
if [ -z "$TAVILY_API_KEY" ]; then
    echo "⚠️  ERROR: TAVILY_API_KEY is not set. web_text_to_text_search and web_text_to_img_search tools will not work."
    echo "   Please set TAVILY_API_KEY in your .env file or environment variables."
    echo "   The tool server will start but web search tools will fail."
fi
if [ -z "$SERPAPI_API_KEY" ]; then
    echo "⚠️  WARNING: SERPAPI_API_KEY is not set. web_image_to_text tool will not work."
    echo "   Please set SERPAPI_API_KEY in your .env file or environment variables."
fi

# Parse command-line argument for tool selection
# Usage: ./deploy_servers.sh [tool_name]
# Examples:
#   ./deploy_servers.sh                                    # Test all tools (default)
#   ./deploy_servers.sh web_text_to_text_search           # Test only web_text_to_text_search
#   ./deploy_servers.sh web_text_to_img_search            # Test only web_text_to_img_search
#   ./deploy_servers.sh web_url_reader                    # Test only web_url_reader
#   ./deploy_servers.sh web_image_to_text                 # Test only web_image_to_text
#   ./deploy_servers.sh ipython_code                      # Test only ipython_code
#   ./deploy_servers.sh bash_terminal                     # Test only bash_terminal
#   ./deploy_servers.sh web_text_to_text_search,web_text_to_img_search  # Test multiple tools

# All available tools
ALL_TOOLS="web_text_to_text_search,web_text_to_img_search,web_url_reader,web_image_to_text,ocr_tool,ipython_code,bash_terminal"

# Use provided tool(s) or default to all tools
if [ -n "$1" ]; then
    tool_type="$1"
    echo "🔧 Testing tool(s): $tool_type"
else
    tool_type="$ALL_TOOLS"
    echo "🔧 Testing all tools: $tool_type"
fi

echo "Starting tool server..."

# Activate conda environment
# eval "$(conda shell.bash hook)"
# conda activate verl-tool-qw3-1 || exit 1

# Start the tool server
host=127.0.0.2
port=8002

workers_per_tool=4

python -m verl_tool.servers.serve \
    --host $host \
    --port $port \
    --tool_type "$tool_type" \
    --workers_per_tool $workers_per_tool \
    --use_ray True \
    --max_concurrent_requests=1024 \
    --router_workers=1 