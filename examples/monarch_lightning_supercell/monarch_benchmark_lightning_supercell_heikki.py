"""
Benchmarking latency and throughput of a mock RL environment workers in a SEED RL like architecture.
To be run in the "Large-Scale Interactive Training with Monarch" (https://lightning.ai/meta-ai/templates/large-scale-interactive-training-with-monarch)
Studio (depends on the utils modules therein).
Problems will arise if max(node_counts) exceeds 16. Possibly an issue with port ranges.
"""

"""
Benchmarking latency and throughput of a mock RL environment workers in a SEED RL like architecture.
To be run in the "Large-Scale Interactive Training with Monarch" (https://lightning.ai/meta-ai/templates/large-scale-interactive-training-with-monarch)
Studio (depends on the utils modules therein).
Problems will arise if max(node_counts) exceeds 16. Possibly an issue with port ranges.
"""


import os
import traceback

# === IMPORTANT: Set hyperactor / monarch env vars BEFORE importing monarch ===
# V0 forced + file logging
os.environ["MONARCH_V0_WORKAROUND_DO_NOT_USE"] = "1"  # V1 seems to work fine?
os.environ["MONARCH_FILE_LOG"] = "debug"
# Intentionally NOT setting MONARCH_STDERR_LOG=debug -- it streams debug from
# every node to stdout and drowns useful output. Logs land in /tmp/*.log on
# each node and can be fetched via EnvActor.dump_log_file().
# Bigger timeouts (defaults 30s) so 16/32-node ack storms don't time out.

os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT"] = "300s"
os.environ["HYPERACTOR_HOST_SPAWN_READY_TIMEOUT"] = "180s"
os.environ["HYPERACTOR_MESH_PROC_SPAWN_MAX_IDLE"] = "180s"
os.environ["HYPERACTOR_MESH_ACTOR_SPAWN_MAX_IDLE"] = "180s"

# Critical: remote nodes use tcp, not unix sockets
os.environ["HYPERACTOR_MESH_DEFAULT_TRANSPORT"] = "tcp"

# Give the controller's Tokio runtime more workers (>= max NODE_COUNTS)
os.environ["TOKIO_WORKER_THREADS"] = "64"  # default=num_cpus

import torch
from monarch.actor import Actor, endpoint, current_rank
import asyncio
import time
import numpy as np
import sys

sys.path.append('../')

from utils.mmt_utils import launch_mmt_job
from utils.mesh_utils import setup_proc_mesh_from_job


# Configuration (same as local benchmark)
OBSERVATION_SIZE = 128
NUM_WARMUP_STEPS = 10
NUM_BENCHMARK_STEPS = 300
NUM_ENVS_PER_NODE = 128
NUM_CPUS = 8  # CPUs per machine on a CPU node
STEP_TIMEOUT_SEC = 60.0  # Timeout per step to detect stuck nodes
NODE_COUNTS = list(range(2, 5))


class EnvActor(Actor):
    """Actor that runs RogueEnv on a remote node."""
    
    def __init__(self, num_envs: int):
        self.rank = current_rank().rank
        self.num_envs = num_envs
        self.env = None
        self.cached_action_ids = None
        self.step_count = 0
        self.last_error = None
    
    @endpoint
    async def step(self, action: torch.Tensor):
        try:
            
            # Assert shape is [NUM_ENVS_PER_NODE, OBSERVATION_SIZE] for sanity (matches local benchmark)
            print(action.shape)

            # Do some fake stuff with action:
            action_sum = action.sum()

            # Generate random tensor for simulating the state:
            obs = torch.randn(self.num_envs, OBSERVATION_SIZE)

            self.step_count += 1
            return obs
        
        except Exception as e:
            tb = traceback.format_exc()
            self.last_error = {"error": str(e), "traceback": tb, "step": self.step_count}
            print(f"[Node {self.rank}] STEP ERROR at step {self.step_count}: {e}\n{tb}")
            return {
                "rank": self.rank,
                "step": self.step_count,
                "status": "error",
                "error": str(e),
                "traceback": tb,
            }


def get_job_name():
    with open("/teamspace/studios/this_studio/job_name") as f:
        name = f.read().strip()
    return name


def _stats_ms(arr):
    arr = np.asarray(arr) * 1000.0  # to ms
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "p50":  float(np.percentile(arr, 50)),
        "p95":  float(np.percentile(arr, 95)),
        "p99":  float(np.percentile(arr, 99)),
        "max":  float(arr.max()),
    }


async def setup_benchmark_cluster(max_nodes: int):
    """Launch a single MMT job sized for max_nodes, build a proc_mesh, spawn
    one EnvActor per node, and initialize each env. Returns
    (job, studio, proc_mesh, env_actors)."""
    print(f"\n{'='*60}\nSetting up cluster for up to {max_nodes} nodes...\n{'='*60}")

    job, studio = launch_mmt_job(
        num_nodes=max_nodes,
        num_gpus=NUM_CPUS,
        machine="CPU",
    )

    print('Waiting for job to start...')
    max_retries = 100
    retry_interval = 10
    for i in range(max_retries):
        if str(job.status) == 'Running':
            print(f'Job is running after {i * retry_interval}s')
            break
        print(f'\r  Job status: {job.status}, retrying in {retry_interval}s... ({i+1}/{max_retries})', end='')
        await asyncio.sleep(retry_interval)
    else:
        raise RuntimeError(f'Job did not start after {max_retries * retry_interval}s. Status: {job.status}')

    print('\nWaiting additional 60s for processes to stabilize...')
    await asyncio.sleep(60)

    print('Setting up process mesh and actors...')
    proc_mesh = setup_proc_mesh_from_job(job, max_nodes, 1)  # 1 actor per node
    env_actors = proc_mesh.spawn("env_actor", EnvActor, NUM_ENVS_PER_NODE)
    
    print("Cluster ready.")
    return job, studio, proc_mesh, env_actors


async def run_distributed_benchmark(env_actors, num_nodes: int):
    """Run the round-trip benchmark using the first `num_nodes` actors of an
    already-initialized env_actors mesh. Uses batched broadcast calls for
    better scaling performance."""
    total_envs = num_nodes * NUM_ENVS_PER_NODE
    print(f"\n{'-'*60}")
    print(f"Benchmark: {total_envs} envs across {num_nodes} nodes"
          f" ({NUM_ENVS_PER_NODE}/node)")
    print(f"{'-'*60}")

    # Use a sliced actor mesh for batched broadcast calls (more efficient than per-node calls)
    active_actors = env_actors.slice(hosts=slice(0, num_nodes))

    def make_random_actions():
        actions = torch.randint(15, size=(NUM_ENVS_PER_NODE, OBSERVATION_SIZE), dtype=torch.int32)
        return actions

    # Warmup calls (more efficient than per-node dispatch)
    print(f"Warming up ({NUM_WARMUP_STEPS} steps)...")
    for step in range(NUM_WARMUP_STEPS):
        t0 = time.perf_counter()
        actions = make_random_actions()
        
        results = await active_actors.step.call(actions)
        dt = time.perf_counter() - t0

        if step == 0 or (step + 1) % 5 == 0:
            print(f"  Warmup step {step+1}/{NUM_WARMUP_STEPS} completed in {dt:.2f}s")

    # Benchmark using batched broadcast calls
    print(f"Benchmarking ({NUM_BENCHMARK_STEPS} steps) using batched broadcast...")
    gen_times        = np.zeros(NUM_BENCHMARK_STEPS)
    dispatch_times   = np.zeros(NUM_BENCHMARK_STEPS)
    wait_times       = np.zeros(NUM_BENCHMARK_STEPS)  # Single wait time for broadcast

    for step_idx in range(NUM_BENCHMARK_STEPS):
        t0 = time.perf_counter()
        actions = make_random_actions()
        t1 = time.perf_counter()
        
        # Single broadcast dispatch to all nodes - Monarch optimizes internally
        result_future = active_actors.step.call(actions)
        t2 = time.perf_counter()
        
        try:
            results = await asyncio.wait_for(result_future, timeout=STEP_TIMEOUT_SEC)
            t3 = time.perf_counter()
            
        except asyncio.TimeoutError:
            print(f"  TIMEOUT at benchmark step {step_idx+1}!")
            raise

        gen_times[step_idx]      = t1 - t0
        dispatch_times[step_idx] = t2 - t1
        wait_times[step_idx]     = t3 - t2
        
        # Progress indicator
        if (step_idx + 1) % 50 == 0:
            print(f"  Step {step_idx+1}/{NUM_BENCHMARK_STEPS}, "
                  f"last total={1000*(t3-t1):.1f}ms")

    total_times = dispatch_times + wait_times

    stats = {
        "gen":        _stats_ms(gen_times),
        "dispatch":   _stats_ms(dispatch_times),    # outbound (single broadcast)
        "wait":       _stats_ms(wait_times),        # inbound (all nodes)
        "total":      _stats_ms(total_times),
    }

    # Compact per-phase report
    print(f"\n  Results for {num_nodes} nodes ({total_envs} envs):")
    print(f"  {'phase':<11} {'mean':>8} {'std':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}   (ms)")
    for phase in ("gen", "dispatch", "wait", "total"):
        s = stats[phase]
        print(f"  {phase:<11} {s['mean']:>8.3f} {s['std']:>8.3f} {s['p50']:>8.3f} "
              f"{s['p95']:>8.3f} {s['p99']:>8.3f} {s['max']:>8.3f}")

    throughput = total_envs / total_times.mean()
    print(f"  Throughput: {throughput:.1f} envs/sec")

    return {
        "total_envs":        total_envs,
        "num_nodes":         num_nodes,
        "envs_per_node":     NUM_ENVS_PER_NODE,
        # Backwards-compatible scalar keys
        "mean_ms":           stats["total"]["mean"],
        "std_ms":            stats["total"]["std"],
        "throughput":        throughput,
        "times":             total_times,
        # Per-phase arrays (simplified for broadcast mode)
        "gen_times":         gen_times,
        "dispatch_times":    dispatch_times,
        "wait_times":        wait_times,
        "stats":             stats,
    }



# Run benchmark across NODE_COUNTS using a SINGLE shared MMT job sized for MAX_NODES.
# Start with smaller node counts and scale up to identify the breaking point.

MAX_NODES = max(NODE_COUNTS)

distributed_results = {}

job = studio = proc_mesh = env_actors = None


# Set to False to keep nodes alive on failure so you can SSH in and inspect
# /tmp/*.log (MONARCH_FILE_LOG=debug). Run the manual `job.stop()` cell below
# once you're done debugging.
STOP_JOB_ON_FAILURE = True

benchmark_failed = False
job, studio, proc_mesh, env_actors = await setup_benchmark_cluster(MAX_NODES)


# Do actual benchmark loop:
for n in NODE_COUNTS:
    print(f"\n{'='*60}")
    print(f"Testing with {n} nodes...")
    print(f"{'='*60}")

    try:
        distributed_results[n] = await run_distributed_benchmark(env_actors, n)
    except Exception as e:
        print(f"Benchmark failed for {n} nodes: {e}")
        benchmark_failed = True
        print("Stopping further tests to avoid wasting time.")
        break



# Plot results:
import matplotlib.pyplot as plt

# 2x2 plot: per-phase p50, per-phase p99, total p50/p95/p99, throughput.
node_counts = sorted(distributed_results.keys())

phase_colors = {
    "gen":      ("gray",       "gen (controller)"),
    "dispatch": ("tab:blue",   "dispatch (outbound)"),
    "wait":     ("tab:red",    "wait (inbound)"),
    "total":    ("tab:purple", "total (dispatch+wait)"),
}

def _series(metric, phase):
    return [distributed_results[n]["stats"][phase][metric] for n in node_counts]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: per-phase p50
ax = axes[0, 0]
for phase, (color, label) in phase_colors.items():
    ax.plot(node_counts, _series("p50", phase), marker='o', linewidth=2,
            color=color, label=label)
ax.set_xlim(0, max(node_counts) + 1)
ax.set_xticks(node_counts)
ax.set_xlabel('num_nodes'); ax.set_ylabel('time (ms)')
ax.set_title('Per-phase p50 latency')
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

# Top-right: per-phase p99
ax = axes[0, 1]
for phase, (color, label) in phase_colors.items():
    ax.plot(node_counts, _series("p99", phase), marker='o', linewidth=2,
            color=color, label=label)
ax.set_xlim(0, max(node_counts) + 1)
ax.set_xticks(node_counts)
ax.set_xlabel('num_nodes'); ax.set_ylabel('time (ms)')
ax.set_title('Per-phase p99 latency')
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

# Bottom-left: total p50/p95/p99
ax = axes[1, 0]
for metric, color in [("p50", "tab:blue"), ("p95", "tab:orange"), ("p99", "tab:red")]:
    ax.plot(node_counts, _series(metric, "total"), marker='o', linewidth=2,
            color=color, label=f"total {metric}")
ax.set_xlim(0, max(node_counts) + 1)
ax.set_xticks(node_counts)
ax.set_xlabel('num_nodes'); ax.set_ylabel('time (ms)')
ax.set_title('Total round-trip p50 / p95 / p99')
ax.grid(True, alpha=0.3); ax.legend()

# Bottom-right: throughput
ax = axes[1, 1]
throughputs = [distributed_results[n]["throughput"] for n in node_counts]
ax.plot(node_counts, throughputs, marker='s', linewidth=2, markersize=8, color='tab:green', label='actual')

# Linear scaling reference (based on smallest node count result)
base_nodes = min(node_counts)
base_throughput = distributed_results[base_nodes]["throughput"]
linear_throughputs = [base_throughput * (n / base_nodes) for n in node_counts]
ax.plot(node_counts, linear_throughputs, linestyle='--', linewidth=2, color='gray', alpha=0.7, label=f'linear (from {base_nodes} nodes)')

ax.set_xlim(0, max(node_counts) + 1)
ax.set_xticks(node_counts)
ax.set_xlabel('num_nodes'); ax.set_ylabel('envs/sec')
ax.set_title('Throughput (total_envs / mean total time)')
ax.grid(True, alpha=0.3); ax.legend()

plt.tight_layout()
plt.show()