# Monarch × Lightning AI Studios — v6

Interactive, multi-node distributed training and debugging with
[**Monarch**](https://github.com/meta-pytorch/monarch) (Meta's distributed
actor framework) + [**TorchTitan**](https://github.com/pytorch/torchtitan) on
[**Lightning AI**](https://lightning.ai) — updated for the current Monarch API
("v6", v0.4+).

These are the same studios as `../old_v0/`, ported from the deprecated Monarch
**v0** API to the current one. They mirror the flow proven out in the working
`../../monarch_lightning_supercell/monarch_benchmark_lightning_supercell_ali.py`
benchmark.

## Notebooks

| Notebook | What it covers | Machines |
|---|---|---|
| [`studio_0_monarch_basics.ipynb`](./studio_0_monarch_basics.ipynb) | Actors, endpoints, process meshes, ping-pong | Local (no cluster) |
| [`studio_1_getting_started.ipynb`](./studio_1_getting_started.ipynb) | Multi-node TorchTitan training (Qwen3-0.6B) | GPU (L40S) |
| [`studio_2_workspace_sync.ipynb`](./studio_2_workspace_sync.ipynb) | Hot-reload files with `sync_workspace` | CPU |
| [`studio_3_interactive_debugging.ipynb`](./studio_3_interactive_debugging.ipynb) | Env-var management + `monarch debug` breakpoints | CPU |
| [`monarch_lightning.ipynb`](./monarch_lightning.ipynb) | Hero notebook: training + all advanced features (Llama-3.1-8B) | GPU (L40S) |

## Supporting files

- **`utils.py`** — worker `bootstrap()` (runs the Monarch worker loop) + client IP helpers. Imported on **both** the workers and the client.
- **`mmt_utils.py`** — `launch_mmt_job()` for Lightning MMT (GPU or CPU).
- **`SYNC_WORKSPACE_ISSUE.md`** — the known `sync_workspace` limitation on Lightning (read before Studio 2).
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

---

## What changed from `old_v0` (Monarch v0 → v6)

| Concern | old_v0 (Monarch v0) | new_v6 (current) |
|---|---|---|
| Force old runtime | `os.environ["MONARCH_V0_WORKAROUND_DO_NOT_USE"] = "1"` | removed |
| Worker command | `process_allocator` | `python -c 'from utils import bootstrap; bootstrap(PORT)'` |
| Client setup | `RemoteAllocator` + `StaticRemoteAllocInitializer` + `MasterNodeServer` HTTP registration | `enable_transport(...)` + `attach_to_workers(...)` |
| Address format | `tcp!IP:PORT` | `tcp://IP:PORT@tcp://0.0.0.0:PORT` (dial\@bind alias) |
| Build the mesh | `setup_proc_mesh_from_job(job, ...)` → `ProcMesh.from_alloc(alloc)` | `host_mesh = attach_to_workers(...)` → `host_mesh.spawn_procs(per_host={"gpus": N})` |
| Local mesh (Studio 0) | `proc_mesh(gpus=N)` | `this_host().spawn_procs(per_host={"gpus": N})` |
| Distributed env vars | `from monarch.utils import setup_env_for_distributed` | `from monarch.spmd import setup_torch_elastic_env_async` (notebooks import the new name and fall back to the old one) |
| Workspace sync | `proc_mesh.sync_workspace(...)` | `host_mesh.sync_workspace(...)` (`proc_mesh.sync_workspace` now raises) |
| Utils layout | `utils/` package (`ip_utils`, `mesh_utils`, `master_node`, `mmt_utils`) | flat `utils.py` + `mmt_utils.py` |

### The v6 connection flow in one glance

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

## Known limitations

- **`sync_workspace` hangs on Lightning remote workers.** The client's rsync
  daemon binds to `127.0.0.1`, which the remote workers can't reach. Studio 2
  shows the intended API but flags this prominently; see
  [`SYNC_WORKSPACE_ISSUE.md`](./SYNC_WORKSPACE_ISSUE.md) for the root cause and
  workarounds (manual `rsync`/`scp`, shared cloud storage, or baking files into
  the studio snapshot).
- When spawning **CPU** procs that you later want to `sync_workspace`, name the
  per-host dimension `"gpus"` (e.g. `per_host={"gpus": 4}`) — `sync_workspace`
  asserts the mesh labels are a subset of `{"gpus", "hosts"}`.

## Cleanup

Every studio ends with a cleanup cell:

```python
host_mesh.shutdown().get()   # stop all remote procs
job.stop()                   # stop the Lightning MMT job
```
