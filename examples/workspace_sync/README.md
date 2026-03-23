# Workspace Sync

## Overview

Workspace sync uses **rsync** to synchronize local files/directories from your machine (the "client") to remote worker processes. It syncs **all files** in the specified directories — Python scripts, configs, model weights, binaries, anything.

## Architecture

```
┌─────────────────────┐         TCP (tunneled via actor mesh)        ┌──────────────────────┐
│     Client Side     │ ──────────────────────────────────────────── │    Worker Side(s)    │
│                     │                                              │                      │
│  RsyncDaemon        │   1. Daemon starts, serves local workspace   │  RsyncActor /        │
│  (serves files)     │   2. Workers connect back and pull files     │  CodeSyncManager     │
│                     │   3. Files land in $WORKSPACE_DIR/<name>     │  (receives files)    │
└─────────────────────┘                                              └──────────────────────┘
```

### Key Components

| Component | Location | Role |
|---|---|---|
| `Workspace` | `python/monarch/tools/config/workspace.py` | Python config class — defines local dirs to sync |
| `CodeSyncMeshClient` | `monarch_extension/src/code_sync.rs` | Python-facing client that triggers sync |
| `RsyncDaemon` | `monarch_hyperactor/src/code_sync/rsync.rs` | Starts rsync server on the client side |
| `RsyncActor` | `monarch_hyperactor/src/code_sync/rsync.rs` | Runs rsync client on each worker |
| `CodeSyncManager` | `monarch_hyperactor/src/code_sync/manager.rs` | Orchestrates rsync + optional hot-reload across a mesh |

### How `Workspace` Works

`Workspace` is **not** a config file format — it's a Python class that defines which local directories to sync. It syncs all file types (`.py`, `.yaml`, `.bin`, etc.) using `rsync --archive --delete`.

```python
from pathlib import Path
from monarch.tools.config.workspace import Workspace

# Single directory — syncs everything under it
workspace = Workspace(dirs=[Path.home() / "github" / "torchtitan"])

# Multiple directories
workspace = Workspace(dirs=[
    Path.home() / "torch",             # -> $WORKSPACE_DIR/torch
    Path.home() / "github" / "torchtitan",  # -> $WORKSPACE_DIR/torchtitan
])

# Explicit local -> remote name mapping
workspace = Workspace(dirs={
    Path.home() / "torch": "github/pytorch",           # -> $WORKSPACE_DIR/github/pytorch
    Path.home() / "github" / "torchtitan": "torchtitan", # -> $WORKSPACE_DIR/torchtitan
})
```

## Prerequisites

- `rsync` installed (macOS has it by default at `/usr/bin/rsync`)
- Monarch Python package installed

## Examples

### Single Laptop (Local Processes)

Uses `ProcessJob` to spawn local subprocesses that simulate remote hosts. Everything runs on `localhost`.

```bash
python examples/workspace_sync/single_laptop_sync.py
```

See [single_laptop_sync.py](single_laptop_sync.py).

### Two Hosts (Simulated Locally)

Uses `ProcessJob` with multiple hosts. Still runs locally but simulates a multi-host scenario.

```bash
python examples/workspace_sync/multi_host_sync.py
```

See [multi_host_sync.py](multi_host_sync.py).

### Two Machines (Real Remote)

For actual multi-machine sync, you need a job scheduler (SkyPilot, Slurm) or manual allocator setup. See [Existing Examples](#existing-examples) below.

## Testing

### Existing Tests

| Test | Location | Description |
|---|---|---|
| Python unit test | `python/tests/_monarch/test_sync_workspace.py` | End-to-end test using `ProcessJob` |
| Rust rsync test | `monarch_hyperactor/src/code_sync/rsync.rs` (line 466) | Tests `RsyncDaemon` + `do_rsync` directly |
| Rust mesh test | `monarch_hyperactor/src/code_sync/rsync.rs` (line 485) | Tests `RsyncActor` in an actor mesh |
| Rust CodeSyncManager test | `monarch_hyperactor/src/code_sync/manager.rs` (line 597) | Full `CodeSyncManager` mesh test |

Run the Python test:

```bash
python -m pytest python/tests/_monarch/test_sync_workspace.py -v
```

### Existing Examples

- **SkyPilot quickstart**: `examples/skypilot/monarch_quickstart.sky.yaml` — provisions remote hosts and runs Monarch

## Test Matrix

| Scenario | How | What You Need |
|---|---|---|
| Single laptop | `ProcessJob({"hosts": 1})` | `rsync`, Monarch |
| Multi-host (local) | `ProcessJob({"hosts": N})` | `rsync`, Monarch |
| Two real machines | SkyPilot / Slurm / manual allocator | `rsync` on all machines, Monarch on all, network connectivity |
