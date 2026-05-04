#!/bin/bash

# Simple Deep Research Test Script
# Loads research scenarios (text-only or multimodal), tests them, and saves results
# 
# Usage:
#   ./simple_test.sh              # Run text-only scenarios (default)
#   ./simple_test.sh text         # Run text-only scenarios  
#   ./simple_test.sh mm           # Run multimodal scenarios (text + images)
#   ./simple_test.sh multimodal   # Run multimodal scenarios (text + images)

set -e  # Exit on any error

echo "=== Simple Deep Research Test ==="

# Configuration
HOST=localhost
PORT=5000
SCENARIO_FILE="research_scenarios.json"
MM_SCENARIO_FILE="research_scenarios_mm.json"
OUTPUT_DIR="examples/train/mm_deep_research/simple_test_results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if server is running
check_server() {
    local url="http://$HOST:$PORT/health"
    print_status "Checking if server is running at $url..."
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        print_success "Server is running and responding!"
        return 0
    else
        print_error "Server is not running or not responding at $url"
        print_status "Please start the server first with:"
        print_status "  conda run -n verl-tool-env python -m verl_tool.servers.serve --host $HOST --port $PORT --tool_type google_search,python_code --workers_per_tool 4"
        return 1
    fi
}

# Function to run the test
run_test() {
    local scenario_file="${1:-$SCENARIO_FILE}"
    print_status "Running deep research test with $scenario_file..."
    print_status "Results will be saved to: $OUTPUT_DIR"
    
    cd /fsx/home/cqin/projects/code/mm-dr-rl/verl-tool

    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Run the test
    if conda run -n verl-tool-env python examples/train/mm_deep_research/test_real_deep_research_archive.py run \
        --server_url=http://$HOST:$PORT/get_observation \
        --output_dir=$OUTPUT_DIR \
        --input_dir=examples/train/mm_deep_research/input \
        --scenario_files="[\"$scenario_file\"]"; then
        print_success "Test completed successfully!"
        
        # Show results summary
        if [ -d "$OUTPUT_DIR" ]; then
            print_status "Test results saved to: $OUTPUT_DIR"
            find "$OUTPUT_DIR" -name "test_summary.json" -exec echo "Summary: {}" \;
            find "$OUTPUT_DIR" -name "*.log" -exec echo "Logs: {}" \;
        fi
    else
        print_error "Test failed!"
        return 1
    fi
}

# Main execution
main() {
    print_status "Starting simple deep research test..."
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found. Please ensure conda is installed and available."
        exit 1
    fi
    
    # Check if server is running
    if ! check_server; then
        exit 1
    fi
    
    # Check if scenario file exists
    SCENARIO_PATH="examples/train/mm_deep_research/input/$SCENARIO_FILE"
    if [ ! -f "$SCENARIO_PATH" ]; then
        print_error "Scenario file not found: $SCENARIO_PATH"
        exit 1
    fi
    
    print_status "Found scenario file: $SCENARIO_PATH"
    
    # Run the test
    run_test "$SCENARIO_FILE"
    
    print_success "All done! Check the $OUTPUT_DIR directory for results."
}

# Parse command line arguments
case "${1:-}" in
    "mm"|"multimodal")
        print_status "Running multimodal research test..."
        SCENARIO_FILE="$MM_SCENARIO_FILE"
        SCENARIO_PATH="examples/train/mm_deep_research/input/$SCENARIO_FILE"
        if [ ! -f "$SCENARIO_PATH" ]; then
            print_error "Multimodal scenario file not found: $SCENARIO_PATH"
            exit 1
        fi
        print_status "Found multimodal scenario file: $SCENARIO_PATH"
        if check_server; then
            run_test "$SCENARIO_FILE"
            print_success "Multimodal test completed! Check the $OUTPUT_DIR directory for results."
        else
            exit 1
        fi
        ;;
    "text"|"regular")
        print_status "Running regular text-only research test..."
        main
        ;;
    *)
        main
        ;;
esac
