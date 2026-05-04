#!/usr/bin/env python3
"""
Real Deep Research Test Cases
Simulates actual deep research scenarios with evaluation metrics
Based on deepsearch dataset structure and real research workflows
"""
import json
import requests
import fire
import logging
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepResearchTester:
    def __init__(self, server_url: str = "http://localhost:4000/get_observation", output_dir: str = "test_results", input_dir: str = "input"):
        self.server_url = server_url
        self.output_dir = Path(output_dir)
        self.input_dir = Path(input_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_dir = self.output_dir / f"deep_research_test_{timestamp}"
        self.test_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different outputs
        (self.test_dir / "trajectories").mkdir(exist_ok=True)
        (self.test_dir / "evaluations").mkdir(exist_ok=True)
        (self.test_dir / "logs").mkdir(exist_ok=True)
        
        logger.info(f"Test results will be saved to: {self.test_dir}")
        logger.info(f"Input scenarios will be loaded from: {self.input_dir}")
    
    def load_scenarios_from_file(self, filename: str) -> List[Dict[str, Any]]:
        """Load research scenarios from a JSON file"""
        file_path = self.input_dir / filename
        
        if not file_path.exists():
            logger.error(f"Scenario file not found: {file_path}")
            return []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            scenarios = data.get('scenarios', [])
            metadata = data.get('metadata', {})
            
            logger.info(f"Loaded {len(scenarios)} scenarios from {filename}")
            if metadata:
                logger.info(f"File metadata: {metadata}")
            
            return scenarios
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []
    
    def load_all_scenarios(self, scenario_files: List[str] = None) -> List[Dict[str, Any]]:
        """Load scenarios from multiple JSON files"""
        if scenario_files is None:
            # Default scenario files
            scenario_files = [
                "research_scenarios.json",
                "advanced_research_scenarios.json", 
                "quick_test_scenarios.json"
            ]
        
        all_scenarios = []
        
        for filename in scenario_files:
            scenarios = self.load_scenarios_from_file(filename)
            all_scenarios.extend(scenarios)
        
        logger.info(f"Total scenarios loaded: {len(all_scenarios)}")
        return all_scenarios
    
    def create_research_scenarios(self, scenario_files: List[str] = None) -> List[Dict[str, Any]]:
        """Load research scenarios from JSON files"""
        return self.load_all_scenarios(scenario_files)
    
    def simulate_research_trajectory(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a complete research trajectory for a given scenario"""
        
        logger.info(f"Starting research trajectory for: {scenario['id']}")
        logger.info(f"Question: {scenario['question']}")
        
        trajectory = {
            "scenario_id": scenario["id"],
            "question": scenario["question"],
            "ground_truth": scenario["ground_truth"],
            "steps": [],
            "start_time": time.time(),
            "total_actions": len(scenario["research_steps"])
        }
        
        # Execute each research step
        for i, step in enumerate(scenario["research_steps"]):
            logger.info(f"Step {i+1}/{len(scenario['research_steps'])}: {step['description']}")
            
            step_result = self._execute_research_step(step, scenario["id"], i)
            trajectory["steps"].append(step_result)
            
            # Add delay between steps to simulate thinking time
            time.sleep(0.5)
        
        trajectory["end_time"] = time.time()
        trajectory["duration"] = trajectory["end_time"] - trajectory["start_time"]
        
        return trajectory
    
    def _execute_research_step(self, step: Dict[str, Any], scenario_id: str, step_index: int) -> Dict[str, Any]:
        """Execute a single research step and return results"""
        
        trajectory_id = f"{scenario_id}_step_{step_index}"
        
        payload = {
            "trajectory_ids": [trajectory_id],
            "actions": [step["action"]],
            "extra_fields": [{}]
        }
        
        try:
            start_time = time.time()
            response = requests.post(self.server_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            processing_time = (time.time() - start_time) * 1000
            
            step_result = {
                "step_index": step_index,
                "action": step["action"],
                "expected_tool": step["expected_tool"],
                "description": step["description"],
                "processing_time_ms": processing_time,
                "response": result,
                "success": True,
                "error": None
            }
            
            # Extract observation
            if "observations" in result and len(result["observations"]) > 0:
                observation = result["observations"][0]
                step_result["observation"] = observation
                step_result["valid"] = result.get("valids", [False])[0]
                step_result["done"] = result.get("dones", [False])[0]
            else:
                step_result["observation"] = None
                step_result["valid"] = False
                step_result["done"] = False
            
            logger.info(f"✓ Step {step_index + 1} completed in {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"✗ Step {step_index + 1} failed: {str(e)}")
            step_result = {
                "step_index": step_index,
                "action": step["action"],
                "expected_tool": step["expected_tool"],
                "description": step["description"],
                "processing_time_ms": 0,
                "response": None,
                "success": False,
                "error": str(e),
                "observation": None,
                "valid": False,
                "done": False
            }
        
        return step_result
    
    def evaluate_research_quality(self, trajectory: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality of research based on multiple metrics"""
        
        evaluation = {
            "scenario_id": scenario["id"],
            "metrics": {},
            "analysis": {}
        }
        
        # 1. Success Rate
        successful_steps = sum(1 for step in trajectory["steps"] if step["success"])
        total_steps = len(trajectory["steps"])
        evaluation["metrics"]["success_rate"] = successful_steps / total_steps if total_steps > 0 else 0
        
        # 2. Tool Usage Accuracy
        correct_tool_usage = 0
        for step in trajectory["steps"]:
            if step["success"] and step["valid"]:
                # Check if the correct tool was used (this would need tool identification logic)
                correct_tool_usage += 1  # Simplified for now
        
        evaluation["metrics"]["tool_accuracy"] = correct_tool_usage / total_steps if total_steps > 0 else 0
        
        # 3. Response Time Analysis
        processing_times = [step["processing_time_ms"] for step in trajectory["steps"] if step["success"]]
        if processing_times:
            evaluation["metrics"]["avg_processing_time_ms"] = sum(processing_times) / len(processing_times)
            evaluation["metrics"]["max_processing_time_ms"] = max(processing_times)
            evaluation["metrics"]["min_processing_time_ms"] = min(processing_times)
        else:
            evaluation["metrics"]["avg_processing_time_ms"] = 0
            evaluation["metrics"]["max_processing_time_ms"] = 0
            evaluation["metrics"]["min_processing_time_ms"] = 0
        
        # 4. Content Quality Analysis (simplified)
        all_observations = []
        for step in trajectory["steps"]:
            if step["success"] and step["observation"]:
                obs_text = str(step["observation"])
                all_observations.append(obs_text)
        
        # Check for expected keywords in observations
        combined_observations = " ".join(all_observations).lower()
        found_keywords = []
        for keyword in scenario["expected_keywords"]:
            if keyword.lower() in combined_observations:
                found_keywords.append(keyword)
        
        evaluation["metrics"]["keyword_coverage"] = len(found_keywords) / len(scenario["expected_keywords"])
        evaluation["analysis"]["found_keywords"] = found_keywords
        evaluation["analysis"]["missing_keywords"] = [kw for kw in scenario["expected_keywords"] if kw not in found_keywords]
        
        # 5. Overall Quality Score
        quality_score = (
            evaluation["metrics"]["success_rate"] * 0.3 +
            evaluation["metrics"]["tool_accuracy"] * 0.2 +
            evaluation["metrics"]["keyword_coverage"] * 0.5
        )
        evaluation["metrics"]["overall_quality_score"] = quality_score
        
        return evaluation
    
    def save_results(self, trajectories: List[Dict[str, Any]], evaluations: List[Dict[str, Any]]):
        """Save all test results to organized files"""
        
        # Save individual trajectories
        for i, trajectory in enumerate(trajectories):
            trajectory_file = self.test_dir / "trajectories" / f"trajectory_{trajectory['scenario_id']}.json"
            with open(trajectory_file, 'w') as f:
                json.dump(trajectory, f, indent=2, default=str)
        
        # Save individual evaluations
        for i, evaluation in enumerate(evaluations):
            evaluation_file = self.test_dir / "evaluations" / f"evaluation_{evaluation['scenario_id']}.json"
            with open(evaluation_file, 'w') as f:
                json.dump(evaluation, f, indent=2, default=str)
        
        # Save summary report
        summary = {
            "test_timestamp": datetime.now().isoformat(),
            "total_scenarios": len(trajectories),
            "overall_metrics": self._calculate_overall_metrics(evaluations),
            "scenario_summaries": [
                {
                    "scenario_id": eval_data["scenario_id"],
                    "quality_score": eval_data["metrics"]["overall_quality_score"],
                    "success_rate": eval_data["metrics"]["success_rate"],
                    "keyword_coverage": eval_data["metrics"]["keyword_coverage"]
                }
                for eval_data in evaluations
            ]
        }
        
        summary_file = self.test_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed log
        log_file = self.test_dir / "logs" / "test_execution.log"
        with open(log_file, 'w') as f:
            f.write(f"Deep Research Test Execution Log\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Server URL: {self.server_url}\n")
            f.write(f"Total Scenarios: {len(trajectories)}\n\n")
            
            for i, (trajectory, evaluation) in enumerate(zip(trajectories, evaluations)):
                f.write(f"=== Scenario {i+1}: {trajectory['scenario_id']} ===\n")
                f.write(f"Question: {trajectory['question']}\n")
                f.write(f"Duration: {trajectory['duration']:.2f}s\n")
                f.write(f"Quality Score: {evaluation['metrics']['overall_quality_score']:.3f}\n")
                f.write(f"Success Rate: {evaluation['metrics']['success_rate']:.3f}\n")
                f.write(f"Keyword Coverage: {evaluation['metrics']['keyword_coverage']:.3f}\n")
                f.write(f"Found Keywords: {evaluation['analysis']['found_keywords']}\n")
                f.write(f"Missing Keywords: {evaluation['analysis']['missing_keywords']}\n\n")
        
        logger.info(f"Results saved to: {self.test_dir}")
        logger.info(f"Summary: {summary_file}")
        logger.info(f"Log: {log_file}")
    
    def _calculate_overall_metrics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall metrics across all scenarios"""
        
        if not evaluations:
            return {}
        
        metrics = ["success_rate", "tool_accuracy", "keyword_coverage", "overall_quality_score"]
        overall = {}
        
        for metric in metrics:
            values = [eval_data["metrics"][metric] for eval_data in evaluations]
            overall[f"avg_{metric}"] = sum(values) / len(values)
            overall[f"max_{metric}"] = max(values)
            overall[f"min_{metric}"] = min(values)
        
        return overall
    
    def run_comprehensive_test(self, scenario_files: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive deep research test with specified scenario files"""
        
        logger.info("=== Starting Comprehensive Deep Research Test ===")
        
        # Load research scenarios from JSON files
        scenarios = self.create_research_scenarios(scenario_files)
        logger.info(f"Loaded {len(scenarios)} research scenarios")
        
        if not scenarios:
            logger.error("No scenarios loaded. Check input files.")
            return {"error": "No scenarios loaded"}
        
        # Execute all scenarios
        trajectories = []
        evaluations = []
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"\n--- Executing Scenario {i+1}/{len(scenarios)}: {scenario['id']} ---")
            
            # Simulate research trajectory
            trajectory = self.simulate_research_trajectory(scenario)
            trajectories.append(trajectory)
            
            # Evaluate research quality
            evaluation = self.evaluate_research_quality(trajectory, scenario)
            evaluations.append(evaluation)
            
            logger.info(f"Quality Score: {evaluation['metrics']['overall_quality_score']:.3f}")
            logger.info(f"Success Rate: {evaluation['metrics']['success_rate']:.3f}")
            logger.info(f"Keyword Coverage: {evaluation['metrics']['keyword_coverage']:.3f}")
        
        # Save all results
        self.save_results(trajectories, evaluations)
        
        # Calculate and return overall results
        overall_metrics = self._calculate_overall_metrics(evaluations)
        
        logger.info("\n=== Test Complete ===")
        logger.info(f"Overall Quality Score: {overall_metrics.get('avg_overall_quality_score', 0):.3f}")
        logger.info(f"Average Success Rate: {overall_metrics.get('avg_success_rate', 0):.3f}")
        logger.info(f"Average Keyword Coverage: {overall_metrics.get('avg_keyword_coverage', 0):.3f}")
        
        return {
            "trajectories": trajectories,
            "evaluations": evaluations,
            "overall_metrics": overall_metrics,
            "test_dir": str(self.test_dir)
        }

def main():
    """Main entry point for deep research testing
    
    Usage:
        python test_real_deep_research.py run --server_url=http://localhost:4000/get_observation
        python test_real_deep_research.py run --output_dir=my_test_results
        python test_real_deep_research.py run --scenario_files='["quick_test_scenarios.json"]'
        python test_real_deep_research.py run --scenario_files='["research_scenarios.json", "advanced_research_scenarios.json"]'
    """
    fire.Fire({
        "run": lambda server_url="http://localhost:4000/get_observation", 
                output_dir="test_results", 
                input_dir="input",
                scenario_files=None: 
            DeepResearchTester(server_url, output_dir, input_dir).run_comprehensive_test(scenario_files)
    })

if __name__ == "__main__":
    main()
