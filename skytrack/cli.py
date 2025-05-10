import argparse
import yaml
import itertools
import time
from typing import Any, Dict, List, Optional

import sky

def load_config(path: str) -> Dict[str, Any]:
    """Load *YAML* config file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_grid(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from a dict of lists."""
    keys = list(params.keys())
    values = list(params.values())
    combos = []
    for prod in itertools.product(*values):
        combo = dict(zip(keys, prod))
        combos.append(combo)
    return combos

def apply_resource_rules(params: Dict[str, Any], rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a parameter dict and a list of rules, return resource overrides.
    Each rule has 'if' (dict of param:value) and 'resources' (dict of sky.Resources attrs).
    The first matching rule is applied. Example rule:
      {"if": {"model": "large"}, "resources": {"accelerators": "A100:4", "cpus": 16}}
    """
    for rule in rules:
        cond = rule.get("if", {})
        if all(params.get(k) == v for k, v in cond.items()):
            return rule.get("resources", {})
    return {}

class JobState:
    """Class to track job state information."""
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.status = "PENDING"   # one of PENDING, RUNNING, DONE, FAILED
        self.attempts = 0
        self.request_id: Optional[str] = None
        self.cluster_name: Optional[str] = None
        self.error: Optional[str] = None

    def to_dict(self):
        return {
            "params": self.params,
            "status": self.status,
            "attempts": self.attempts,
            "request_id": self.request_id,
            "cluster_name": self.cluster_name,
            "error": self.error,
        }

def load_job_states(path: str) -> List[JobState]:
    """Load job states from a JSON file; return empty list if file not found."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    states = []
    for item in data.get("jobs", []):
        job = JobState(item["params"])
        job.status = item.get("status", job.status)
        job.attempts = item.get("attempts", job.attempts)
        job.request_id = item.get("request_id", job.request_id)
        job.cluster_name = item.get("cluster_name", job.cluster_name)
        job.error = item.get("error", job.error)
        states.append(job)
    return states

def save_job_states(path: str, states: List[JobState]):
    """Save job states to a JSON file."""
    data = {"jobs": [job.to_dict() for job in states]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def run_sweep(config: Dict[str, Any]):
    """Main sweep logic: launch tasks, track status, retry failures, enforce concurrency."""
    sweep_type = config.get("type", "grid")
    command = config.get("command", "")
    params = config.get("params", {}) if sweep_type == "grid" else {}
    rules = config.get("resources_rules", [])
    retry_limit = config.get("retry_limit", 0)
    max_concurrent = config.get("max_concurrent", 1)
    reuse_cluster = config.get("reuse_cluster", False)
    state_file = config.get("state_file", "skytrack_jobs.json")

    # Build list of parameter combinations or commands
    if sweep_type == "grid":
        combos = generate_grid(params)
    else:
        raw_items = config.get("benchmarks", [])
        if all(isinstance(item, str) for item in raw_items):
            # Treat list of strings as raw commands
            combos = [{"__cmd": item} for item in raw_items]
        else:
            # Treat list of dicts as parameter sets
            combos = raw_items

    # Load existing job states if any (allows resume), else initialize
    states = load_job_states(state_file)
    if not states:
        for combo in combos:
            states.append(JobState(combo))
        save_job_states(state_file, states)

    # -- Reuse a single cluster for all jobs (sequential execution) --
    if reuse_cluster:
        cluster_name = config.get("cluster_name", "skytrack-tune")
        print(f"Reusing single cluster '{cluster_name}' for all jobs.")
        for job in states:
            if job.status == "DONE":
                # skip already completed jobs (if resuming)
                continue
            # Attempt job up to (retry_limit+1) times
            while job.attempts < retry_limit + 1:
                job.attempts += 1
                action = "Retrying" if job.attempts > 1 else "Launching"
                print(f"{action} job for params {job.params} (attempt {job.attempts})")
                job.status = "RUNNING"
                # Determine run command
                if "__cmd" in job.params:
                    run_command = job.params["__cmd"]
                else:
                    run_command = command.format(**job.params)
                # Apply resource rules
                resource_kwargs = apply_resource_rules(job.params, rules)
                resources = sky.Resources()
                for attr, val in resource_kwargs.items():
                    if hasattr(resources, attr):
                        setattr(resources, attr, val)
                task = sky.Task(run=run_command)
                task.set_resources(resources)
                # Launch or exec on the same cluster
                if job.attempts == 1:
                    request_id = sky.launch(task, cluster_name=cluster_name, down=False)
                else:
                    request_id = sky.exec(task, cluster_name=cluster_name, down=False)
                job.request_id = request_id
                print(f" -> Request {request_id} on cluster '{cluster_name}'")
                # Poll job status until DONE or FAILED
                while True:
                    try:
                        result = sky.api_status(request_ids=[request_id])
                        if result:
                            status_str = result[0].get("status")
                            if status_str == "SUCCEEDED":
                                job.status = "DONE"
                                print(f"Job {request_id} DONE.")
                                break
                            elif status_str == "FAILED":
                                job.status = "FAILED"
                                job.error = result[0].get("error", "")
                                print(f"Job {request_id} FAILED: {job.error}")
                                break
                        time.sleep(5)
                    except Exception as e:
                        print(f"Error checking status for job {request_id}: {e}")
                        time.sleep(5)
                # If succeeded or reached retry limit, stop retrying this job
                if job.status == "DONE" or job.attempts >= retry_limit + 1:
                    break
            save_job_states(state_file, states)
        print(f"All jobs complete on reuse cluster.\nCluster '{cluster_name}' is still running (reuse mode).")
        return

    # -- Non-reuse mode: each job gets its own cluster (possibly concurrently) --
    while True:
        # Launch pending jobs up to the concurrency limit
        running_count = sum(1 for job in states if job.status == "RUNNING")
        slots = max_concurrent - running_count
        for idx, job in enumerate(states):
            if slots <= 0:
                break
            if job.status in ("PENDING", "FAILED") and job.attempts < retry_limit + 1:
                job.attempts += 1
                action = "Retrying" if job.attempts > 1 else "Launching"
                print(f"{action} job for params {job.params} (attempt {job.attempts})")
                job.status = "RUNNING"
                # Determine run command
                if "__cmd" in job.params:
                    run_command = job.params["__cmd"]
                else:
                    run_command = command.format(**job.params)
                # Apply resource rules
                resource_kwargs = apply_resource_rules(job.params, rules)
                resources = sky.Resources()
                for attr, val in resource_kwargs.items():
                    if hasattr(resources, attr):
                        setattr(resources, attr, val)
                task = sky.Task(run=run_command)
                task.set_resources(resources)
                cluster_name = f"tune-job{idx}-att{job.attempts}"
                job.cluster_name = cluster_name
                # Launch with down=True to auto-shutdown after job
                request_id = sky.launch(task, cluster_name=cluster_name, down=True)
                job.request_id = request_id
                print(f" -> Launched request {request_id} on cluster '{cluster_name}'")
                save_job_states(state_file, states)
                slots -= 1

        # Poll running jobs for completion or failure
        all_done = True
        for job in states:
            if job.status == "RUNNING":
                try:
                    result = sky.api_status(request_ids=[job.request_id])
                    if result:
                        status_str = result[0].get("status")
                        if status_str == "SUCCEEDED":
                            job.status = "DONE"
                            print(f"Job {job.request_id} DONE.")
                        elif status_str == "FAILED":
                            job.status = "FAILED"
                            job.error = result[0].get("error", "")
                            print(f"Job {job.request_id} FAILED: {job.error}")
                        else:
                            all_done = False
                    else:
                        all_done = False
                except Exception as e:
                    print(f"Error checking status for job {job.request_id}: {e}")
                    all_done = False
            elif job.status in ("PENDING", "FAILED") and job.attempts < retry_limit + 1:
                # Still pending or retryable failed jobs remain
                all_done = False
        save_job_states(state_file, states)

        if all_done:
            break
        time.sleep(5)

    print("All jobs are complete.")
    print("Clusters were torn down after each job (down=True).")

def main():
    parser = argparse.ArgumentParser(description="SkyTrack CLI")
    subparsers = parser.add_subparsers(dest='command')
    tune_parser = subparsers.add_parser('tune', help='Run hyperparameter tuning or benchmarks via SkyPilot')
    tune_parser.add_argument('--config', type=str, required=True,
                             help='Path to tuning config YAML file')
    args = parser.parse_args()
    if args.command == 'tune':
        config = load_config(args.config)
        run_sweep(config)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
