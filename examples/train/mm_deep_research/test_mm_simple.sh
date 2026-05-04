#!/bin/bash

# Check tool server
if ! curl -s http://127.0.0.2:8002/health > /dev/null 2>&1; then
    echo "⚠️  Tool server not running. Start with: ./deploy_servers.sh"
    exit 1
fi

# Load .env from project root
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/.env"
if [ -f "$ENV_FILE" ]; then
    set -a && source <(grep -v '^#' "$ENV_FILE" | grep -v '^$' | sed 's/^/export /') && set +a
fi

export OPENAI_API_KEY=${OPENAI_API_KEY:-"dummy"}
[ -z "$X_API_KEY" ] && echo "⚠️  WARNING: X_API_KEY not set (will cause 403 errors)"

# Run test
python test_mm_deep_research.py run \
    --server_url="http://127.0.0.2:8002/get_observation" \
    --gpt_model="gpt-4.1" \
    --output_dir="mmsearch_plus_processed_easy_gpt41_try" \
    --input_dir="/fsx/home/cqin/projects/code/mm-dr-rl/verl-tool-all/verl-tool-lastest/data/mmsearch_plus_processed/" \
    --runs_per_question=1 \
    --max_scenarios=10 \
    --image_url_prefix="https://cqin-public-data.s3.us-west-1.amazonaws.com/mm-deepsearch-sfr-2025/mmsearch_plus_processed/images/" \
    --prompt_tool "" \
    --difficulty_filter="easy" \
    --run_start_number 23
