"""
Monarch v0.3.0 Benchmark: SEED RL-style environment workers on Lightning AI.

Benchmarks round-trip latency and throughput of mock RL environment actors
using the current Monarch attach_to_workers + bootstrap API.

Usage:
  Run cells in a Jupyter notebook on Lightning AI, or execute with:
      python -m asyncio monarch_benchmark_v3.py

Files required in the same directory (or importable from worker path):
  - utils.py       (bootstrap + IP helpers)
  - mmt_utils.py   (Lightning MMT launcher)

Changes from monarch_benchmark_2.py (Supercell original):
  - Removed MONARCH_V0_WORKAROUND_DO_NOT_USE
  - Uses enable_transport() on the controller
  - Uses attach_to_workers + host_mesh.spawn_procs() instead of
    the old setup_proc_mesh_from_job helper
  - Uses current mmt_utils / utils from the Monarch v0.3.0 Lightning examples
  - Proper env vars for log forwarding
"""

# ============================================================================
# Cell 1: Environment variables (MUST be set before importing monarch)
# ============================================================================
import os
import traceback

os.environ["XDG_RUNTIME_DIR"] = "/tmp"
os.environ["MONARCH_FILE_LOG"] = "debug"
os.environ["HYPERACTOR_MESH_ENABLE_LOG_FORWARDING"] = "true"
os.environ["HYPERACTOR_MESH_ENABLE_FILE_CAPTURE"] = "true"
os.environ["HYPERACTOR_MESH_TAIL_LOG_LINES"] = "100"

# TCP transport for cross-machine communication
os.environ["HYPERACTOR_MESH_DEFAULT_TRANSPORT"] = "tcp"

import asyncio
import sys
import time

import numpy as np
import torch
from mmt_utils import launch_mmt_job
from monarch._src.actor.bootstrap import attach_to_workers
from monarch.actor import Actor, current_rank, enable_transport, endpoint
from utils import get_host_ip_addr


# ============================================================================
# Cell 2: Configuration
# ============================================================================
OBSERVATION_SIZE = 128
NUM_WARMUP_STEPS = 10
NUM_BENCHMARK_STEPS = 300
NUM_ENVS_PER_NODE = 128
STEP_TIMEOUT_SEC = 60.0

# Node counts to benchmark. Each count uses a slice of the same cluster.
# range(2, 17) tests 2..16 nodes; range(2, 33) tests up to 32 nodes.
# Supercell hit a 16-node wall with the old API — this should go further.
NODE_COUNTS = list(range(2, 33))

# Port for Monarch transport (controller + all workers use the same port,
# each on its own machine).
PORT = 26600


# ============================================================================
# Cell 3: Actor definition (unchanged from Supercell's original)
# ============================================================================
class EnvActor(Actor):
    """Mock RL environment actor running on a remote CPU node."""

    def __init__(self, num_envs: int):
        self.rank = current_rank().rank
        self.num_envs = num_envs
        self.step_count = 0
        self.last_error = None

    @endpoint
    async def step(self, action: torch.Tensor):
        try:
            action_sum = action.sum()
            obs = torch.randn(self.num_envs, OBSERVATION_SIZE)
            self.step_count += 1
            return obs
        except Exception as e:
            tb = traceback.format_exc()
            self.last_error = {
                "error": str(e),
                "traceback": tb,
                "step": self.step_count,
            }
            print(f"[Node {self.rank}] STEP ERROR at step {self.step_count}: {e}\n{tb}")
            return {
                "rank": self.rank,
                "step": self.step_count,
                "status": "error",
                "error": str(e),
                "traceback": tb,
            }

    @endpoint
    async def ping(self) -> dict:
        """Health check endpoint."""
        return {"rank": self.rank, "step_count": self.step_count, "status": "ok"}


# ============================================================================
# Cell 4: Helper functions
# ============================================================================
def _stats_ms(arr):
    arr = np.asarray(arr) * 1000.0
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


# ============================================================================
# Cell 5: Cluster setup using current Monarch API
# ============================================================================
async def setup_benchmark_cluster(max_nodes: int, port: int = PORT):
    """Launch MMT workers, attach, create proc_mesh, spawn EnvActors.

    Key differences from the old setup_proc_mesh_from_job approach:
      1. enable_transport() on the controller
      2. Get worker IPs from job.machines
      3. attach_to_workers() to connect to the bootstrap loops
      4. host_mesh.spawn_procs() to create the process mesh
    """
    print(f"\n{'='*60}")
    print(f"Setting up cluster for {max_nodes} nodes...")
    print(f"{'='*60}")

    # --- Step 1: Enable transport on the controller ---
    host_ip = get_host_ip_addr(addr_type="public")
    enable_transport(f"tcp://{host_ip}:{port}@tcp://0.0.0.0:{port}")
    print(f"Controller transport enabled at {host_ip}:{port}")

    # --- Step 2: Launch MMT job ---
    mmt_job_name = f"supercell-benchmark-{max_nodes}n"
    job, studio = launch_mmt_job(
        num_nodes=max_nodes,
        mmt_job_name=mmt_job_name,
        port=port,
        use_cpu=True,
    )

    # --- Step 3: Wait for job to be running ---
    print("Waiting for job to start...")
    from lightning_sdk import Status

    max_retries = 100
    retry_interval = 10
    for i in range(max_retries):
        if job.status == Status("Running"):
            print(f"Job is running after {i * retry_interval}s")
            break
        print(f"\r  Job status: {job.status} ({i+1}/{max_retries})", end="")
        await asyncio.sleep(retry_interval)
    else:
        raise RuntimeError(
            f"Job did not start after {max_retries * retry_interval}s. "
            f"Status: {job.status}"
        )

    print("Waiting 30s for worker bootstraps to stabilize...")
    await asyncio.sleep(30)

    # --- Step 4: Get worker IPs and attach ---
    ip_list = [machine.public_ip for machine in job.machines]
    print(f"Worker IPs: {ip_list}")

    worker_addrs = [f"tcp://{ip}:{port}@tcp://0.0.0.0:{port}" for ip in ip_list]

    host_mesh = attach_to_workers(
        name="host_mesh", ca="trust_all_connections", workers=worker_addrs
    )

    # --- Step 5: Create process mesh (CPU-only, 1 proc per host) ---
    proc_mesh = host_mesh.spawn_procs()
    await proc_mesh.logging_option(stream_to_client=True, aggregate_window_sec=3)

    # --- Step 6: Spawn EnvActors ---
    env_actors = proc_mesh.spawn("env_actor", EnvActor, NUM_ENVS_PER_NODE)

    # --- Step 7: Health check ---
    print("Running health check (ping all actors)...")
    health = await env_actors.ping.call()
    print(f"All {max_nodes} actors healthy: {health}")

    print("Cluster ready.\n")
    return job, studio, host_mesh, proc_mesh, env_actors


# ============================================================================
# Cell 6: Benchmark runner (logic preserved from Supercell's original)
# ============================================================================
async def run_distributed_benchmark(env_actors, num_nodes: int):
    """Run the round-trip latency benchmark using the first num_nodes actors."""
    total_envs = num_nodes * NUM_ENVS_PER_NODE
    print(f"\n{'-'*60}")
    print(
        f"Benchmark: {total_envs} envs across {num_nodes} nodes "
        f"({NUM_ENVS_PER_NODE}/node)"
    )
    print(f"{'-'*60}")

    active_actors = env_actors.slice(hosts=slice(0, num_nodes))

    def make_random_actions():
        return torch.randint(
            15, size=(NUM_ENVS_PER_NODE, OBSERVATION_SIZE), dtype=torch.int32
        )

    # --- Warmup ---
    print(f"Warming up ({NUM_WARMUP_STEPS} steps)...")
    for step in range(NUM_WARMUP_STEPS):
        t0 = time.perf_counter()
        actions = make_random_actions()
        results = await active_actors.step.call(actions)
        dt = time.perf_counter() - t0
        if step == 0 or (step + 1) % 5 == 0:
            print(f"  Warmup step {step+1}/{NUM_WARMUP_STEPS} in {dt:.2f}s")

    # --- Timed benchmark ---
    print(f"Benchmarking ({NUM_BENCHMARK_STEPS} steps)...")
    gen_times = np.zeros(NUM_BENCHMARK_STEPS)
    dispatch_times = np.zeros(NUM_BENCHMARK_STEPS)
    wait_times = np.zeros(NUM_BENCHMARK_STEPS)

    for step_idx in range(NUM_BENCHMARK_STEPS):
        t0 = time.perf_counter()
        actions = make_random_actions()
        t1 = time.perf_counter()

        result_future = active_actors.step.call(actions)
        t2 = time.perf_counter()

        try:
            results = await asyncio.wait_for(result_future, timeout=STEP_TIMEOUT_SEC)
            t3 = time.perf_counter()
        except asyncio.TimeoutError:
            print(f"  TIMEOUT at benchmark step {step_idx+1}!")
            raise

        gen_times[step_idx] = t1 - t0
        dispatch_times[step_idx] = t2 - t1
        wait_times[step_idx] = t3 - t2

        if (step_idx + 1) % 50 == 0:
            print(
                f"  Step {step_idx+1}/{NUM_BENCHMARK_STEPS}, "
                f"last total={1000*(t3-t1):.1f}ms"
            )

    total_times = dispatch_times + wait_times

    stats = {
        "gen": _stats_ms(gen_times),
        "dispatch": _stats_ms(dispatch_times),
        "wait": _stats_ms(wait_times),
        "total": _stats_ms(total_times),
    }

    print(f"\n  Results for {num_nodes} nodes ({total_envs} envs):")
    print(
        f"  {'phase':<11} {'mean':>8} {'std':>8} {'p50':>8} "
        f"{'p95':>8} {'p99':>8} {'max':>8}   (ms)"
    )
    for phase in ("gen", "dispatch", "wait", "total"):
        s = stats[phase]
        print(
            f"  {phase:<11} {s['mean']:>8.3f} {s['std']:>8.3f} "
            f"{s['p50']:>8.3f} {s['p95']:>8.3f} {s['p99']:>8.3f} "
            f"{s['max']:>8.3f}"
        )

    throughput = total_envs / total_times.mean()
    print(f"  Throughput: {throughput:.1f} envs/sec")

    return {
        "total_envs": total_envs,
        "num_nodes": num_nodes,
        "envs_per_node": NUM_ENVS_PER_NODE,
        "mean_ms": stats["total"]["mean"],
        "std_ms": stats["total"]["std"],
        "throughput": throughput,
        "times": total_times,
        "gen_times": gen_times,
        "dispatch_times": dispatch_times,
        "wait_times": wait_times,
        "stats": stats,
    }


# ============================================================================
# Cell 7: Run benchmarks
# ============================================================================
async def run_all_benchmarks():
    MAX_NODES = max(NODE_COUNTS)
    distributed_results = {}
    benchmark_failed = False

    job, studio, host_mesh, proc_mesh, env_actors = await setup_benchmark_cluster(
        MAX_NODES
    )

    for n in NODE_COUNTS:
        print(f"\n{'='*60}")
        print(f"Testing with {n} nodes...")
        print(f"{'='*60}")
        try:
            distributed_results[n] = await run_distributed_benchmark(env_actors, n)
        except Exception as e:
            print(f"Benchmark failed for {n} nodes: {e}")
            traceback.print_exc()
            benchmark_failed = True
            print("Stopping further tests.")
            break

    return distributed_results, benchmark_failed, job, studio, host_mesh


distributed_results, benchmark_failed, job, studio, host_mesh = asyncio.run(
    run_all_benchmarks()
)


# ============================================================================
# Cell 8: Plot results
# ============================================================================
import matplotlib.pyplot as plt

node_counts = sorted(distributed_results.keys())

phase_colors = {
    "gen": ("gray", "gen (controller)"),
    "dispatch": ("tab:blue", "dispatch (outbound)"),
    "wait": ("tab:red", "wait (inbound)"),
    "total": ("tab:purple", "total (dispatch+wait)"),
}


def _series(metric, phase):
    return [distributed_results[n]["stats"][phase][metric] for n in node_counts]


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: per-phase p50
ax = axes[0, 0]
for phase, (color, label) in phase_colors.items():
    ax.plot(
        node_counts,
        _series("p50", phase),
        marker="o",
        linewidth=2,
        color=color,
        label=label,
    )
ax.set_xlim(0, max(node_counts) + 1)
ax.set_xticks(node_counts)
ax.set_xlabel("num_nodes")
ax.set_ylabel("time (ms)")
ax.set_title("Per-phase p50 latency")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Top-right: per-phase p99
ax = axes[0, 1]
for phase, (color, label) in phase_colors.items():
    ax.plot(
        node_counts,
        _series("p99", phase),
        marker="o",
        linewidth=2,
        color=color,
        label=label,
    )
ax.set_xlim(0, max(node_counts) + 1)
ax.set_xticks(node_counts)
ax.set_xlabel("num_nodes")
ax.set_ylabel("time (ms)")
ax.set_title("Per-phase p99 latency")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Bottom-left: total p50/p95/p99
ax = axes[1, 0]
for metric, color in [("p50", "tab:blue"), ("p95", "tab:orange"), ("p99", "tab:red")]:
    ax.plot(
        node_counts,
        _series(metric, "total"),
        marker="o",
        linewidth=2,
        color=color,
        label=f"total {metric}",
    )
ax.set_xlim(0, max(node_counts) + 1)
ax.set_xticks(node_counts)
ax.set_xlabel("num_nodes")
ax.set_ylabel("time (ms)")
ax.set_title("Total round-trip p50 / p95 / p99")
ax.grid(True, alpha=0.3)
ax.legend()

# Bottom-right: throughput
ax = axes[1, 1]
throughputs = [distributed_results[n]["throughput"] for n in node_counts]
ax.plot(
    node_counts,
    throughputs,
    marker="s",
    linewidth=2,
    markersize=8,
    color="tab:green",
    label="actual",
)

base_n = min(node_counts)
base_tp = distributed_results[base_n]["throughput"]
linear_tp = [base_tp * (n / base_n) for n in node_counts]
ax.plot(
    node_counts,
    linear_tp,
    linestyle="--",
    linewidth=2,
    color="gray",
    alpha=0.7,
    label=f"linear (from {base_n} nodes)",
)

ax.set_xlim(0, max(node_counts) + 1)
ax.set_xticks(node_counts)
ax.set_xlabel("num_nodes")
ax.set_ylabel("envs/sec")
ax.set_title("Throughput (total_envs / mean total time)")
ax.grid(True, alpha=0.3)
ax.legend()

plt.suptitle(
    f"Monarch v0.3.0 RL Benchmark  |  {NUM_ENVS_PER_NODE} envs/node  |  "
    f"obs_size={OBSERVATION_SIZE}  |  {NUM_BENCHMARK_STEPS} steps",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("benchmark_results.png", dpi=150)
plt.show()


# ============================================================================
# Cell 9: Cleanup
# ============================================================================
# host_mesh.shutdown().get()
# job.stop()
# print("Cleanup complete.")
