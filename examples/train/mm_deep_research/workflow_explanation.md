# Multimodal Deep Research - Technical Workflow

> 📖 **For quick start and usage, see the main [README.md](README.md)**

## 🎯 **System Overview**

The multimodal deep research system implements a **ReAct (Reasoning and Acting)** pattern for comprehensive research tasks combining text and image analysis.

## 📋 **1. Initialization Phase**

```python
MultimodalDeepResearchTester(
    server_url="http://localhost:4000/get_observation",
    output_dir="simple_test_results_mm", 
    input_dir="input_mm",
    gpt_model="gpt-4o"
)
```

**What happens:**
- Sets up OpenAI client for GPT-4o
- Creates timestamped output directories
- Initializes tool server connection
- Sets up logging and file structure

## 📂 **2. Scenario Loading**

```python
load_multimodal_scenarios(filename="qa_formatted.json", max_scenarios=3)
```

**What happens:**
- Loads research scenarios from JSON file
- Each scenario contains: question, image, ground_truth
- Limits to 3 scenarios for testing (configurable)

## 🔄 **3. ReAct Research Loop (Per Scenario)**

### **Main Loop Structure:**
```
FOR each scenario:
    WHILE iterations < max_iterations (6):
        1. REASON: Analyze current state, decide next action
        2. ACT: Execute the decided action  
        3. OBSERVE: Process results
        4. CHECK: Should we stop?
    Generate final comprehensive answer
```

### **Step 1: Reasoning (`_react_reasoning_step`)**
```python
def _react_reasoning_step(trajectory, scenario, iteration):
    # Collect previous observations
    # Analyze what information is still needed
    # Decide on next action with proper parameters
    # Determine if we should stop
    return {
        "reasoning": "Analysis of current state",
        "action": {"action_type": "web_search", "action_parameters": {...}},
        "should_stop": false,
        "confidence": 0.8
    }
```

**What happens:**
- Analyzes all previous research findings
- Tracks success/failure rates
- Generates next action with proper parameters
- Uses confidence scoring for stopping decisions

### **Step 2: Acting (`_react_action_step`)**
```python
def _react_action_step(action, scenario, iteration):
    # Validate action format
    # Convert to tool server format
    # Execute via tool server
    # Return results with metadata
```

**What happens:**
- Validates action has required fields
- Converts GPT action to tool server format:
  - `web_search` → `<search>query</search>`
  - `image_analysis` → `<python>print('GPT vision result')</python>`
  - `code_execution` → `<python>code</python>`
- Executes via tool server
- Tracks success/failure

### **Step 3: Observing**
- Tool server returns results
- Results are stored in trajectory
- Success/failure tracked for next iteration

### **Step 4: Stopping Criteria**
```python
# Stop if high confidence + sufficient info
if should_stop and confidence > 0.7:
    break

# Stop if too many consecutive failures  
if consecutive_failures >= 2:
    break
```

## 🛠️ **4. Tool Action Conversion**

The system converts GPT-generated actions to tool server format:

### **Web Search:**
```python
# GPT Action: {"action_type": "web_search", "action_parameters": {"query": "..."}}
# Tool Action: "<search>query</search>"
```

### **Image Analysis:**
```python
# GPT Action: {"action_type": "image_analysis", "action_parameters": {"query": "..."}}
# Process: Use GPT vision API to analyze image
# Tool Action: "<python>print('GPT vision analysis result')</python>"
```

### **Code Execution:**
```python
# GPT Action: {"action_type": "code_execution", "action_parameters": {"code": "..."}}
# Tool Action: "<python>code</python>"
```

## 📊 **5. Evaluation & Metrics**

```python
def evaluate_multimodal_research_quality(trajectory, scenario):
    # Calculate success rate, tool accuracy, relevance score
    # Measure action diversity, ReAct efficiency
    # Generate overall quality score
```

**Metrics calculated:**
- **Success Rate**: % of successful actions
- **Tool Accuracy**: % of valid tool executions
- **Relevance Score**: Keyword overlap with research question
- **Action Diversity**: Variety of action types used
- **ReAct Efficiency**: 1/iterations (higher = more efficient)
- **Overall Quality Score**: Weighted combination

## 💾 **6. Results Saving**

```python
def save_results(trajectories, evaluations):
    # Save individual trajectories as JSON
    # Save individual evaluations as JSON  
    # Save generated actions as JSON
    # Create summary report
    # Generate detailed execution log
```

**Files created:**
- `trajectory_{scenario_id}.json` - Complete research trajectory
- `evaluation_{scenario_id}.json` - Quality metrics
- `actions_{scenario_id}.json` - Generated actions
- `test_summary.json` - Overall results summary
- `test_execution.log` - Detailed execution log

## 🎯 **7. Final Answer Generation**

```python
def _generate_final_answer(trajectory, scenario):
    # Collect all successful observations
    # Combine research findings
    # Use GPT to synthesize comprehensive answer
    # Return structured research report
```

**What happens:**
- Gathers all successful research findings
- Uses GPT to synthesize comprehensive answer
- Structures as detailed research report
- Includes all relevant facts, dates, names

## 🔧 **Key Features**

### **Adaptive Planning:**
- Each iteration adapts based on previous findings
- Can change research direction dynamically
- Stops when sufficient information gathered

### **Robust Error Handling:**
- Validates action formats before execution
- Fallback mechanisms for failed reasoning
- Tracks consecutive failures

### **Multimodal Support:**
- Handles both text and image inputs
- Uses GPT vision API for image analysis
- Integrates visual and textual information

### **Comprehensive Evaluation:**
- Multiple quality metrics
- ReAct-specific efficiency measures
- Detailed logging and reporting

## 📈 **Performance Improvements**

### **Before (Static Approach):**
- Generated all 5 actions upfront
- No adaptation based on findings
- Always executed all actions
- Poor stopping criteria

### **After (ReAct Approach):**
- Dynamic action generation per iteration
- Adapts based on research findings
- Intelligent stopping (3-6 iterations)
- High success rate and efficiency

## 🚀 **Usage Example**

```python
# Run the test
tester = MultimodalDeepResearchTester()
results = tester.run_multimodal_test(max_scenarios=3)

# Results contain:
# - trajectories: Complete research paths
# - evaluations: Quality metrics  
# - overall_metrics: Summary statistics
# - test_dir: Path to saved results
```

This workflow creates a sophisticated, adaptive research system that mimics human research behavior while maintaining high efficiency and quality!
