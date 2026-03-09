"""
Main inference script for running model inference in the agent environment.
"""

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import yaml

from agents import BaseAgent, CocoaAgent, OpenAIDeepResearchAgent, GeminiDeepResearchAgent
from executor.utils import setup_logging, load_config, get_logger
from decrypt import decrypt_file_to_memory, read_canary


def parse_arguments() -> dict:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run model inference on tasks")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Path to configuration file")
    parser.add_argument("--tasks-dir", type=str, default="tasks/",
                       help="Path to tasks directory (containing task subdirectories)")
    parser.add_argument("--output-dir", type=str, default="results/",
                       help="Output directory for results (one JSON file per task)")
    parser.add_argument("--model", type=str,
                       help="Override model name from config")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers (each uses a separate Docker port starting from docker_port)")

    return parser.parse_args()


def load_tasks(tasks_dir: str, use_encrypted: bool = False) -> List[Dict[str, Any]]:
    """Load tasks from directory structure (tasks/task-name/task.yaml or task.yaml.enc).

    Args:
        tasks_dir: Path to tasks directory
        use_encrypted: If True, load from encrypted .enc files (decrypt to memory only)
                      If False, load from plaintext .yaml files

    Returns:
        List of task dictionaries
    """
    logger = get_logger("inference")
    tasks = []
    tasks_path = Path(tasks_dir)

    mode = "encrypted" if use_encrypted else "plaintext"
    logger.info(f"Loading tasks from {tasks_dir} (mode: {mode})")

    if not tasks_path.is_dir():
        raise ValueError(f"Tasks directory not found: {tasks_dir}")

    # Iterate through task subdirectories
    for task_dir in sorted(tasks_path.iterdir()):
        if not task_dir.is_dir():
            continue

        if use_encrypted:
            # Load from encrypted file (decrypt to memory only)
            task_file_enc = task_dir / "task.yaml.enc"
            if not task_file_enc.exists():
                logger.warning(f"No task.yaml.enc found in {task_dir}, skipping")
                continue

            # Read canary for decryption
            canary = read_canary(task_dir)
            if canary is None:
                logger.warning(f"No canary.txt found in {task_dir}, skipping")
                continue

            try:
                # Decrypt to memory
                task_yaml_content = decrypt_file_to_memory(task_file_enc, canary)
                task_data = yaml.safe_load(task_yaml_content)
            except Exception as e:
                logger.error(f"Failed to decrypt task.yaml.enc in {task_dir}: {e}")
                continue
        else:
            # Load from plaintext file
            task_file = task_dir / "task.yaml"
            if not task_file.exists():
                logger.warning(f"No task.yaml found in {task_dir}, skipping")
                continue

            with open(task_file, 'r') as f:
                task_data = yaml.safe_load(f)

        if task_data is None:
            logger.warning(f"Empty task data in {task_dir}, skipping")
            continue

        # Add task directory path and test file path
        task_data["task_dir"] = str(task_dir)
        task_data["task_name"] = task_dir.name

        # Check for test file (encrypted or plaintext based on mode)
        if use_encrypted:
            test_file_enc = task_dir / "test.py.enc"
            task_data["test_file_path"] = str(test_file_enc) if test_file_enc.exists() else None
            task_data["use_encrypted"] = True
        else:
            test_file = task_dir / "test.py"
            task_data["test_file_path"] = str(test_file) if test_file.exists() else None
            task_data["use_encrypted"] = False

        tasks.append(task_data)

    logger.info(f"Loaded {len(tasks)} tasks from {tasks_dir}")
    return tasks


def run_single_task(task: Dict[str, Any], config: Dict[str, Any], output_dir: str) -> str:
    """Run a single task in its own process with a dedicated Docker port.

    This function is designed to be called in a subprocess via ProcessPoolExecutor.
    Each invocation creates its own agent, runs the task, writes the result JSON,
    and returns the task name.

    Args:
        task: Task dictionary (already loaded, includes task_name and task_dir)
        config: Full config dictionary (already has docker_port set for this worker)
        output_dir: Directory to write the result JSON into

    Returns:
        task_name string (used by the caller for progress logging)
    """
    # Re-initialize logging inside the subprocess (each process has its own log state)
    log_level = config.get("log_level", "INFO")
    setup_logging(log_level)
    logger = get_logger("inference.worker")

    task_name = task.get("task_name", "unknown")
    port = config["sandbox"]["docker_port"]
    logger.info(f"[port={port}] Starting task: {task_name}")

    agent_type = config.get("agent_type", "cocoa")
    if agent_type == "openai_deep_research":
        agent = OpenAIDeepResearchAgent(config)
    elif agent_type == "gemini_deep_research":
        agent = GeminiDeepResearchAgent(config)
    else:
        agent = CocoaAgent(config)

    output_file = Path(output_dir) / f"{task_name}.json"

    agent.setup_environment(task)
    try:
        result = agent.run_task(task)
        test_result = agent.run_eval(task, result)
        if test_result is not None:
            result["eval"] = test_result
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"[port={port}] Task {task_name} done. Result saved to {output_file}")
    except Exception as e:
        logger.error(f"[port={port}] Task {task_name} failed: {e}")
        with open(output_file, "w") as f:
            json.dump({"status": "error", "error": str(e), "task_name": task_name}, f, indent=2)
    finally:
        agent.cleanup_environment()

    return task_name


def compute_statistics(output_dir: str) -> str:
    """Compute and write statistics.txt from result JSON files."""
    logger = get_logger("inference")
    total_tasks = passed_tasks = error_tasks = 0
    passed_list: List[str] = []
    error_list: List[str] = []

    output_path = Path(output_dir)
    for json_file in output_path.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            total_tasks += 1
            if data.get("status") == "error":
                error_tasks += 1
                error_list.append(json_file.stem)
            elif data.get("eval", {}).get("passed", False) is True:
                passed_tasks += 1
                passed_list.append(json_file.stem)
        except Exception as e:
            logger.error(f"Error reading {json_file}: {e}")

    success_rate = (passed_tasks / total_tasks * 100) if total_tasks > 0 else 0.0
    lines = [
        f"Total Tasks: {total_tasks}",
        f"Passed: {passed_tasks}",
        f"Failed: {total_tasks - passed_tasks - error_tasks}",
        f"Errors: {error_tasks}",
        f"Success Rate: {success_rate:.2f}%",
    ]
    if passed_list:
        lines += ["\nPassed Tasks:"] + [f"  - {t}" for t in sorted(passed_list)]
    if error_list:
        lines += ["\nError Tasks:"] + [f"  - {t}" for t in sorted(error_list)]

    stats_content = "\n".join(lines) + "\n"
    stats_file = output_path / "statistics.txt"
    with open(stats_file, "w") as f:
        f.write(stats_content)
    logger.info(f"Statistics saved to {stats_file}")
    return stats_content


def main():
    """Main function."""
    args = parse_arguments()

    config = load_config(args.config)

    # Setup logging with specified level FIRST before getting any loggers
    log_level = config.get("log_level", "INFO")
    setup_logging(log_level)

    logger = get_logger("inference")
    logger.info("Starting inference")

    if args.model:
        config["controller"]["args"]["model"] = args.model
        logger.info(f"Model overridden to: {args.model}")

    os.makedirs(args.output_dir, exist_ok=True)

    use_encrypted = config.get("use_encrypted_tasks", False)
    logger.info(f"Use encrypted tasks: {use_encrypted}")

    tasks = load_tasks(args.tasks_dir, use_encrypted=use_encrypted)
    n_workers = max(1, args.workers)
    base_port = config.get("sandbox", {}).get("docker_port", 8080)

    logger.info(f"Running {len(tasks)} tasks with {n_workers} worker(s), base port {base_port}")

    if n_workers == 1:
        # Serial path — no subprocess overhead
        agent_type = config.get("agent_type", "cocoa")
        if agent_type == "openai_deep_research":
            agent = OpenAIDeepResearchAgent(config)
        elif agent_type == "gemini_deep_research":
            agent = GeminiDeepResearchAgent(config)
        else:
            agent = CocoaAgent(config)

        for i, task in enumerate(tasks, 1):
            task_name = task.get("task_name", f"task_{i}")
            logger.info(f"Processing task {i}/{len(tasks)}: {task_name}")
            agent.setup_environment(task)
            try:
                result = agent.run_task(task)
                test_result = agent.run_eval(task, result)
                if test_result is not None:
                    result["eval"] = test_result
                output_file = Path(args.output_dir) / f"{task_name}.json"
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                logger.debug(f"Task {task_name} result saved to {output_file}")
            except Exception as e:
                logger.error(f"Task {task_name} failed with error: {e}")
                output_file = Path(args.output_dir) / f"{task_name}.json"
                with open(output_file, "w") as f:
                    json.dump({"status": "error", "error": str(e), "task_name": task_name}, f, indent=2)
            finally:
                agent.cleanup_environment()
    else:
        # Parallel path — each worker gets its own port and subprocess
        # Build per-worker configs: worker i uses port base_port + i
        worker_configs = []
        for i in range(n_workers):
            cfg = copy.deepcopy(config)
            cfg.setdefault("sandbox", {})["docker_port"] = base_port + i
            worker_configs.append(cfg)

        # Submit all tasks; round-robin assign ports so concurrent tasks use different ports
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for idx, task in enumerate(tasks):
                worker_cfg = worker_configs[idx % n_workers]
                future = pool.submit(run_single_task, task, worker_cfg, args.output_dir)
                futures[future] = task.get("task_name", f"task_{idx+1}")

            completed = 0
            for future in as_completed(futures):
                task_name = futures[future]
                completed += 1
                try:
                    future.result()
                    logger.info(f"[{completed}/{len(tasks)}] Completed: {task_name}")
                except Exception as e:
                    logger.error(f"[{completed}/{len(tasks)}] Worker error for {task_name}: {e}")

    logger.info(f"All {len(tasks)} tasks processed. Results saved to {args.output_dir}")

    stats = compute_statistics(args.output_dir)
    print(stats)


if __name__ == "__main__":
    main()
