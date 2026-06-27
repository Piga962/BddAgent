#!/usr/bin/env python3
"""
BddAgent Experiment Runner

A rigorous experimental framework for evaluating BDD-driven LLM code generation.
Supports multiple models, comprehensive metrics collection, and reproducible experiments.

Usage:
    python run_experiment.py --dataset data/LM_prompt_elements.jsonl --mode local_file_completion
    python run_experiment.py --config experiment_config.yaml
    python run_experiment.py --dataset data/test.jsonl --models azure/gpt-4.1-mini openai/gpt-4o

For more options:
    python run_experiment.py --help
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import ExperimentConfig, ModelConfig, DatasetConfig, create_default_config
from metrics import MetricsCollector, CodeValidator, ValidationLevel
from utils import setup_logging, get_logger, set_seeds, ExperimentManifest, create_manifest
from analysis import ExperimentAnalyzer

# Import existing BddAgent components
from game.actionContext import create_action_context_with_registry
from game.actions import DecoratorActionRegistry
from game.agent import Agent, AgentRegistry
from game.environment import ActionContextEnvironment
from game.llms import create_simple_llm_function
from game.memory import Goal, Memory
from game.agentLanguage import AgentFunctionCallingActionLanguage

import tools.agentTools, tools.fileTools, tools.promptTools, tools.otherTools, tools.devEvalTools


class ExperimentRunner:
    """
    Rigorous experiment runner for BddAgent evaluation.

    Features:
    - Multi-model comparison
    - Comprehensive metrics collection
    - Reproducibility controls
    - Structured logging
    - Statistical analysis
    """

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Optional[str] = None
    ):
        self.config = config
        self.output_dir = Path(output_dir or config.output.base_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up reproducibility
        set_seeds(config.random_seed)

        # Set up logging
        self.logger = setup_logging(
            experiment_id=config.experiment_id,
            log_dir=str(self.output_dir / "logs")
        )

        # Create manifest
        self.manifest = create_manifest(
            experiment_id=config.experiment_id,
            experiment_name=config.experiment_name,
            description=config.description,
            config=config.__dict__,
            output_dir=str(self.output_dir)
        )

        # Validator
        self.validator = CodeValidator(ValidationLevel.STANDARD)

        # Load tests
        self.tests = self._load_tests()

        self.logger.info(f"Initialized experiment: {config.experiment_id}")
        self.logger.info(f"Loaded {len(self.tests)} test cases")

    def _load_tests(self) -> List[Dict]:
        """Load test cases from dataset."""
        tests = []
        dataset_path = self.config.dataset.path

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    test_case = json.loads(line)
                    processed = {
                        "namespace": test_case['namespace'],
                        "input_code": test_case['input_code'],
                    }

                    mode = self.config.dataset.mode
                    if mode == 'local_file_completion':
                        processed['context_above'] = test_case.get('contexts_above', '')
                    elif mode == 'local_file_infiling':
                        processed['context_above'] = test_case.get('contexts_above', '')
                        processed['context_below'] = test_case.get('contexts_below', '')

                    tests.append(processed)

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse test case: {e}")

        # Apply sample size limit if specified
        if self.config.dataset.sample_size:
            import random
            random.seed(self.config.dataset.sample_seed)
            if len(tests) > self.config.dataset.sample_size:
                tests = random.sample(tests, self.config.dataset.sample_size)
                self.logger.info(f"Sampled {len(tests)} test cases")

        return tests

    def _create_task(self, test: Dict, mode: str) -> str:
        """Create the task prompt for an agent."""
        namespace = test['namespace']
        requirements = test['input_code']

        context_str = ""
        context_instruction = ""

        if mode == 'without_context':
            context_instruction = "Generate code based only on requirements and common Python patterns."
        elif mode == 'local_file_completion':
            context_str = f"Context above: {test.get('context_above', '')}"
            context_instruction = "Use patterns, imports, and helper functions from the context above."
        elif mode == 'local_file_infiling':
            context_str = f"Context above: {test.get('context_above', '')}\nContext below: {test.get('context_below', '')}"
            context_instruction = "Use patterns from both context above and below. Ensure code fits between them."

        task = f"""
DEVEVAL COORDINATION: {namespace}

Requirements: {requirements}
{context_str}

STRATEGY: {context_instruction}

EXECUTE WORKFLOW:
1. Use analyze_deveval_requirements to understand the task
2. Generate the bdd_tests needed for validation
3. Use call_agent_with_reflection to call 'DevEvalCoder' with coding task and the bdd_tests as validation criteria
4. Use call_agent_with_selected_context to call 'DevEvalReviewer' for validation based on the bdd_tests
5. Extract final clean function body
6. validate_function_body(function_body=<step 3 result>, requirements="{requirements}")
   - If validation fails, regenerate from step 2 with fixes
7. terminate(message=<validated function body from step 3>)

Coordinate the team to produce working Python code.
CRITICAL: Final output must be ONLY function body, 4-space indented, no 'def' line.
"""
        return task

    def _create_agents(self, llm_function) -> tuple:
        """Create the agent pipeline."""
        # Main coordinator
        main_agent = Agent(
            goals=[
                Goal(1, "DevEval Analysis", "Use tools to analyze DevEval requirements thoroughly"),
                Goal(2, "Agent Coordination", "Delegate to coding agent and coordinate review process"),
                Goal(3, "Quality Assurance", "Ensure code meets DevEval standards through review tools")
            ],
            agent_language=AgentFunctionCallingActionLanguage(),
            action_registry=DecoratorActionRegistry(tags=["selective", "deveval", "analysis"]),
            generate_response=llm_function,
            environment=ActionContextEnvironment(),
            agent_name="DevEvalCoordinator",
            max_iterations=12
        )

        # Coding agent
        coding_agent = Agent(
            goals=[
                Goal(1, "Complete Function Generation", "Generate complete Python functions with proper signatures"),
                Goal(2, "Working Implementation", "Write actual working Python code that solves the given requirements"),
            ],
            agent_language=AgentFunctionCallingActionLanguage(),
            action_registry=DecoratorActionRegistry(tags=["deveval"]),
            generate_response=llm_function,
            environment=ActionContextEnvironment(),
            agent_name="DevEvalCoder",
            max_iterations=8
        )

        # Code reviewer
        reviewer_agent = Agent(
            goals=[
                Goal(1, "Code Quality Review", "Review code for quality, best practices, and potential issues"),
                Goal(2, "Security Analysis", "Identify security vulnerabilities and suggest improvements"),
                Goal(3, "Performance Review", "Analyze code for performance optimization opportunities")
            ],
            agent_language=AgentFunctionCallingActionLanguage(),
            action_registry=DecoratorActionRegistry(tags=["deveval"]),
            generate_response=llm_function,
            environment=ActionContextEnvironment(),
            agent_name="DevEvalReviewer",
            max_iterations=10
        )

        # Register terminate tools
        main_agent.action_registry.register_terminate_tool()
        coding_agent.action_registry.register_terminate_tool()
        reviewer_agent.action_registry.register_terminate_tool()

        return main_agent, coding_agent, reviewer_agent

    def _extract_code_from_memory(self, memory, namespace: str) -> str:
        """Extract generated code from agent memory."""
        import re

        def extract_function_body(text: str) -> Optional[str]:
            """Extract function body from text."""
            # Try to parse JSON first
            for _ in range(5):
                try:
                    data = json.loads(text)
                    if isinstance(data, dict) and 'result' in data:
                        text = str(data['result'])
                    elif isinstance(data, str):
                        text = data
                    else:
                        break
                except json.JSONDecodeError:
                    break

            # Clean markdown
            text = re.sub(r'```python\s*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
            text = re.sub(r'```\s*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)

            # Find function definition
            lines = text.split('\n')
            function_start = -1
            for i, line in enumerate(lines):
                if 'def ' in line and '(' in line and ':' in line:
                    function_start = i
                    break

            if function_start == -1:
                return None

            # Extract body
            function_lines = lines[function_start:]
            body_lines = []
            found_def = False
            main_indent = 0
            in_docstring = False
            docstring_char = None

            for line in function_lines:
                stripped = line.strip()
                current_indent = len(line) - len(line.lstrip())

                if not found_def and stripped.startswith('def '):
                    found_def = True
                    main_indent = current_indent
                    continue

                if found_def:
                    # Skip docstrings
                    if not in_docstring:
                        if stripped.startswith('"""') or stripped.startswith("'''"):
                            docstring_char = '"""' if stripped.startswith('"""') else "'''"
                            if stripped.count(docstring_char) >= 2:
                                continue
                            else:
                                in_docstring = True
                                continue
                    else:
                        if docstring_char and docstring_char in line:
                            in_docstring = False
                            docstring_char = None
                        continue

                    if stripped:
                        if current_indent <= main_indent and stripped.startswith('def '):
                            break

                        # Normalize indentation
                        if current_indent <= main_indent:
                            body_lines.append('    ' + stripped)
                        else:
                            relative = current_indent - main_indent
                            new_indent = 4 + (relative // 4) * 4
                            body_lines.append(' ' * new_indent + stripped)
                    else:
                        body_lines.append('')

            # Clean result
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()

            result = '\n'.join(body_lines) if body_lines else None
            if result:
                result = re.sub(r'\s*Agent session completed.*$', '', result, flags=re.DOTALL)
                result = result.rstrip()

            return result

        # Search memory for code
        for item in reversed(memory.items):
            content = str(item.get("content", ""))

            # Try JSON result
            try:
                data = json.loads(content)
                if "result" in data:
                    result_text = str(data["result"])
                    if "def " in result_text:
                        body = extract_function_body(result_text)
                        if body and len(body.strip()) > 10:
                            return body
            except json.JSONDecodeError:
                pass

            # Try plain text
            if "def " in content and len(content) > 50:
                body = extract_function_body(content)
                if body and len(body.strip()) > 10:
                    return body

        return "    pass  # No implementation found"

    def run_single_model(self, model_config: ModelConfig) -> Dict[str, Any]:
        """Run experiment with a single model."""
        model_id = model_config.get_litellm_id()
        self.logger.info(f"Starting experiment with model: {model_id}")

        # Create metrics collector
        collector = MetricsCollector(
            experiment_id=self.config.experiment_id,
            experiment_name=self.config.experiment_name,
            model_name=model_config.name,
            mode=self.config.dataset.mode,
            output_dir=str(self.output_dir)
        )

        # Create LLM function
        llm_function = create_simple_llm_function(model_id)

        # Results for DevEval format
        deveval_results = []

        for idx, test in enumerate(self.tests):
            namespace = test['namespace']

            with collector.track_test(namespace, idx) as metrics:
                self.logger.test_start(namespace, idx)

                try:
                    # Create fresh agents for each test
                    main_agent, coding_agent, reviewer_agent = self._create_agents(llm_function)

                    # Set up registry
                    registry = AgentRegistry()
                    registry.register_agent("DevEvalCoder", coding_agent.run)
                    registry.register_agent("DevEvalReviewer", reviewer_agent.run)

                    # Create shared memory
                    shared_memory = Memory()

                    # Create task
                    task = self._create_task(test, self.config.dataset.mode)

                    # Run main agent
                    action_context = {
                        "agent_registry": registry,
                        "target_language": "python",
                        "project_type": "deveval_function",
                        "namespace": namespace,
                        "shared_memory": shared_memory
                    }

                    result_memory = main_agent.run(
                        user_input=task,
                        memory=shared_memory,
                        action_context_props=action_context
                    )

                    # Extract code
                    final_code = self._extract_code_from_memory(result_memory, namespace)

                    # Validate code
                    validation = self.validator.validate_function_body(
                        final_code,
                        requirements=test['input_code']
                    )
                    collector.record_validation(validation)
                    collector.record_generated_code(final_code)

                    # Record iterations
                    collector.record_iterations(len(result_memory.items))

                    # Add to DevEval results
                    deveval_results.append({
                        "namespace": namespace,
                        "completion": final_code
                    })

                    self.logger.test_end(True, metrics.timing.duration_seconds)

                except Exception as e:
                    self.logger.exception(f"Error processing {namespace}")
                    collector.record_error(type(e).__name__, str(e))
                    self.logger.test_end(False, metrics.timing.duration_seconds)

                # Progress update
                progress = collector.get_progress()
                if (idx + 1) % 10 == 0:
                    self.logger.info(
                        f"Progress: {progress['completed']}/{len(self.tests)} "
                        f"({progress['success_rate']:.1%} success)"
                    )

        # Finalize and save
        experiment_metrics = collector.finalize()
        result_files = collector.save_results(prefix=self.config.experiment_id)
        collector.print_summary()

        # Save DevEval format results
        deveval_path = self.output_dir / f"{model_config.name}_{self.config.dataset.mode}_results.jsonl"
        with open(deveval_path, 'w', encoding='utf-8') as f:
            for result in deveval_results:
                f.write(json.dumps(result) + '\n')

        return {
            "model": model_config.name,
            "metrics": experiment_metrics.to_dict(),
            "result_files": result_files,
            "deveval_results_path": str(deveval_path)
        }

    def run(self) -> Dict[str, Any]:
        """Run the complete experiment."""
        self.manifest.start()
        self.logger.experiment_start(self.config.__dict__)

        all_results = {}

        for model_config in self.config.models:
            try:
                results = self.run_single_model(model_config)
                all_results[model_config.name] = results
                self.manifest.add_result_file(results["deveval_results_path"])
            except Exception as e:
                self.logger.exception(f"Failed to run model {model_config.name}")
                all_results[model_config.name] = {"error": str(e)}

        # Finalize manifest
        summary = {
            "models_tested": len(self.config.models),
            "total_tests": len(self.tests),
            "models": {
                name: result.get("metrics", {}).get("success_rate", 0)
                for name, result in all_results.items()
                if "metrics" in result
            }
        }
        self.manifest.complete(summary)
        manifest_path = self.manifest.save()

        self.logger.experiment_end(summary)
        self.logger.info(f"Manifest saved to: {manifest_path}")

        # Run analysis if multiple models
        if len(self.config.models) >= 2:
            analyzer = ExperimentAnalyzer(str(self.output_dir))
            analyzer.load_results()
            report = analyzer.generate_report(
                str(self.output_dir / f"{self.config.experiment_id}_analysis.json")
            )
            analyzer.print_summary()

        return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BddAgent Experiment Runner - Rigorous LLM Code Generation Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a single model
  python run_experiment.py --dataset data/LM_prompt_elements.jsonl --mode local_file_completion

  # Run with multiple models
  python run_experiment.py --dataset data/test.jsonl --models azure/gpt-4.1-mini openai/gpt-4o

  # Run from config file
  python run_experiment.py --config experiment_config.yaml

  # Generate config file
  python run_experiment.py --generate-config --dataset data/test.jsonl --output my_config.yaml
        """
    )

    # Config file option
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to experiment configuration YAML file"
    )

    # Dataset options
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Path to dataset JSONL file"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="local_file_completion",
        choices=["without_context", "local_file_completion", "local_file_infiling"],
        help="Experiment mode (default: local_file_completion)"
    )
    parser.add_argument(
        "--sample-size", "-n",
        type=int,
        default=None,
        help="Number of test cases to sample (default: all)"
    )

    # Model options
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["azure/gpt-4.1-mini"],
        help="Model IDs to evaluate (e.g., azure/gpt-4.1-mini openai/gpt-4o)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Model temperature (default: 0.2)"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="BddAgent Evaluation",
        help="Experiment name for tracking"
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    # Config generation
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate a config file instead of running experiment"
    )

    # Analysis only
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis on existing results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory containing results to analyze"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Analysis only mode
    if args.analyze_only:
        results_dir = args.results_dir or args.output
        print(f"Analyzing results in: {results_dir}")
        analyzer = ExperimentAnalyzer(results_dir)
        analyzer.load_results()
        report = analyzer.generate_report(
            str(Path(results_dir) / "analysis_report.json")
        )
        analyzer.print_summary()
        return

    # Config generation mode
    if args.generate_config:
        if not args.dataset:
            print("Error: --dataset is required for config generation")
            sys.exit(1)

        config = create_default_config(
            dataset_path=args.dataset,
            mode=args.mode,
            experiment_name=args.experiment_name
        )

        output_path = args.output if args.output.endswith('.yaml') else "experiment_config.yaml"
        config.to_yaml(output_path)
        print(f"Configuration saved to: {output_path}")
        return

    # Load or create config
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        if not args.dataset:
            print("Error: --dataset or --config is required")
            sys.exit(1)

        # Create config from CLI args
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        models = []
        for model_id in args.models:
            parts = model_id.split('/')
            provider = parts[0] if len(parts) > 1 else "openai"
            model_name = parts[-1]

            models.append(ModelConfig(
                name=model_id.replace('/', '_'),
                provider=provider,
                model_id=model_name,
                temperature=args.temperature,
                seed=args.seed
            ))

        dataset = DatasetConfig(
            path=args.dataset,
            mode=args.mode,
            sample_size=args.sample_size,
            sample_seed=args.seed
        )

        config = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name=args.experiment_name,
            description=f"BddAgent evaluation with {len(models)} model(s)",
            models=models,
            dataset=dataset,
            random_seed=args.seed
        )

    # Validate config
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    # Run experiment
    runner = ExperimentRunner(config, output_dir=args.output)

    try:
        results = runner.run()
        print("\nExperiment completed successfully!")
        print(f"Results saved to: {args.output}")
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
