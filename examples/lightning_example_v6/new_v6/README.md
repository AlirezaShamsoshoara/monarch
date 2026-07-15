# Monarch × Lightning AI Studios

Interactive, multi-node distributed training and debugging with
[**Monarch**](https://github.com/meta-pytorch/monarch) (Meta's distributed
actor framework) + [**TorchTitan**](https://github.com/pytorch/torchtitan) on
[**Lightning AI**](https://lightning.ai).

> **Monarch version:** these studios were built and verified against
> **`torchmonarch 0.6.0.dev20260606`**. Install with `pip install torchmonarch`
> (or pin that exact version for a byte-for-byte match).

## Notebooks

| Notebook | What it covers | Machines |
|---|---|---|
| [`studio_0_monarch_basics.ipynb`](./studio_0_monarch_basics.ipynb) | Actors, endpoints, process meshes, ping-pong | Local (no cluster) |
| [`studio_1_getting_started.ipynb`](./studio_1_getting_started.ipynb) | Multi-node TorchTitan training (Qwen3-0.6B) | GPU (L40S) |
| [`studio_2_workspace_sync.ipynb`](./studio_2_workspace_sync.ipynb) | Hot-reload files to workers via a Monarch actor (edit → sync → no restart) | Local + CPU |
| [`studio_3_interactive_debugging.ipynb`](./studio_3_interactive_debugging.ipynb) | Env-var management + `monarch debug` breakpoints | CPU |
| [`monarch_lightning.ipynb`](./monarch_lightning.ipynb) | Hero notebook: training + all advanced features (Llama-3.1-8B) | GPU (L40S) |

## Scripts

- **[`run_native_sync.py`](./run_native_sync.py)** — Monarch's **native**
  `sync_workspace` (rsync-based, delta transfers) as a runnable script:
  `python run_native_sync.py`. Native `sync_workspace` works from a Python
  script but **not** from inside a Jupyter kernel on this build, so use the
  script for the native path and `studio_2_workspace_sync.ipynb` (actor-based)
  for the in-notebook path. Details in
  [`SYNC_WORKSPACE_ISSUE.md`](./SYNC_WORKSPACE_ISSUE.md).

## Supporting files

- **`utils.py`** — worker `bootstrap()` (runs the Monarch worker loop) + client IP helpers. Imported on **both** the workers and the client.
- **`mmt_utils.py`** — `launch_mmt_job()` for Lightning MMT (GPU or CPU).
- **`SYNC_WORKSPACE_ISSUE.md`** — engineering notes on Monarch's native `sync_workspace` on this build (what works, what doesn't, and why). Not needed to run the studios.
- **`assets/`** — architecture diagram + screenshots used by the notebooks.

## Recommended path

Run **Studio 0** locally first (no cluster). Then **Studio 1** to launch a GPU
job and train. **Studios 2 & 3** can re-attach to a running job (or spin up a
cheap CPU job) — keep the same `MMT_JOB_NAME` to reconnect after a kernel
restart instead of launching a new job.

## Prerequisites

- A Lightning AI account with access to the machine types you request
  (L40S GPUs for training, `CPU_X_4` for the CPU studios).
- A recent `lightning_sdk`: `pip install -U lightning_sdk`
- Monarch installed on the studio (and snapshotted to workers): `pip install torchmonarch`
- For training studios: TorchTitan + a Hugging Face token for the tokenizer.

## The connection flow in one glance

```python
# CLIENT (this notebook)
enable_transport(f"tcp://{my_public_ip}:{PORT}@tcp://0.0.0.0:{PORT}")
job, studio = launch_mmt_job(num_nodes=N, mmt_job_name=..., port=PORT, num_gpus=8)  # workers run bootstrap()
worker_addrs = [f"tcp://{m.public_ip}:{PORT}@tcp://0.0.0.0:{PORT}" for m in job.machines]
host_mesh = attach_to_workers(name="host_mesh", ca="trust_all_connections", workers=worker_addrs)
proc_mesh = host_mesh.spawn_procs(per_host={"gpus": 8})
await proc_mesh.logging_option(stream_to_client=True, aggregate_window_sec=3)
actors = proc_mesh.spawn("my_actor", MyActor, ...)
```

## Workspace sync — two options

- **In a notebook →** `studio_2_workspace_sync.ipynb` (a small Monarch actor
  pushes files over the mesh — no rsync, reliable in Jupyter and scripts).
- **Native `sync_workspace` (rsync) →** run `python run_native_sync.py`. It
  installs a one-line `rsync` shim (rewrites the daemon's `hosts allow` to
  numeric loopback, which this container needs) and runs the sync as an asyncio
  task. Keep IPv6 loopback enabled (don't `disable_ipv6`). Native
  `sync_workspace` does **not** work inside a Jupyter kernel on this build — see
  [`SYNC_WORKSPACE_ISSUE.md`](./SYNC_WORKSPACE_ISSUE.md).

## Cleanup

Every studio ends with a cleanup cell:

```python
host_mesh.shutdown().get()   # stop all remote procs
job.stop()                   # stop the Lightning MMT job
```
