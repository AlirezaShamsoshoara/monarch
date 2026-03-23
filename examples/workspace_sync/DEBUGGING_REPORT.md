# Workspace Sync Debugging Report

## Overview

This document summarizes a debugging session investigating issues with Monarch's `sync_workspace` feature. Two distinct bugs were identified and one environment limitation was documented.

---

## Issue 1: rsync daemon rejects IPv6 loopback connections on some Linux hosts

### Symptom

`await host.sync_workspace(workspace)` hangs indefinitely on certain Linux nodes but works on others. No error message is produced — the call simply never returns.

### Investigation

1. Verified `rsync` is installed and functional on the affected node
2. Verified IPv4 and IPv6 loopback bind/connect works
3. Tested rsync daemon mode manually — discovered rsync daemon rejects connections with:
   ```
   @ERROR: access denied to workspace from ip6-localhost (::1)
   ```
4. Inspected `/etc/hosts` on the affected node:
   ```
   127.0.0.1   localhost
   ::1         ip6-localhost ip6-loopback
   ```

### Root Cause

In `monarch_hyperactor/src/code_sync/rsync.rs` (line 261), the rsync daemon config is written with:

```
hosts allow = localhost
```

The daemon binds to `[::1]:0` (IPv6 loopback). When the rsync client connects from `::1`, rsync does a reverse DNS lookup using `/etc/hosts`. On many Linux distributions (Ubuntu, cloud VMs), `::1` maps to `ip6-localhost`, **not** `localhost`. rsync checks the resolved hostname against `hosts allow`, finds no match, and rejects the connection.

### Why it hangs instead of erroring

In `code_sync_mesh` (`manager.rs:485`), a `try_join!` waits for two concurrent futures:

1. **Client side** (`method_fut`): waiting for `Connect` messages from the actor, then `copy_bidirectional` to relay data to the rsync daemon
2. **Actor side**: waiting for `result_rx` with the sync result

When the rsync client on the actor side gets "access denied", `do_rsync` fails and the error is sent back via the `result` port. However, the client's `method_fut` is still waiting for the `Connect` stream's `copy_bidirectional` to complete, which never finishes because the rsync session was rejected. Neither side cleanly tears down, so both futures hang indefinitely.

### Fix

**File:** `monarch_hyperactor/src/code_sync/rsync.rs`, line 261

```rust
// Before:
hosts allow = localhost

// After:
hosts allow = localhost ip6-localhost
```

This covers both `/etc/hosts` naming conventions across Linux distributions.

### Recommendations for Monarch developers

1. **Use IP addresses instead of hostnames:** `hosts allow = 127.0.0.1 ::1` would be even more robust as it avoids DNS lookup entirely
2. **Add a timeout to `try_join!`:** The `code_sync_mesh` function should have a timeout so that errors like this surface as failures instead of infinite hangs. Currently, if any part of the rsync flow fails without cleanly closing the `Connect` stream, the entire operation hangs forever
3. **Log rsync stderr:** When `do_rsync` fails, the error message ("access denied") should be logged or propagated clearly rather than silently causing a hang

---

## Issue 2: `sync_workspace` hangs in Jupyter but works from Python scripts

### Symptom

`await host.sync_workspace(workspace)` hangs indefinitely when called from a Jupyter notebook cell on certain nodes. The same code wrapped in `asyncio.run()` in a standalone Python script completes successfully on the same node.

### Investigation

1. Confirmed the rsync fix (Issue 1) was applied and present in the compiled binary using `strings` on the `.so` file
2. Added step-by-step diagnostics to isolate where the hang occurs:
   - **Step 1:** `_code_sync_proc_mesh.get()` — works
   - **Step 2:** Await proc mesh future — works
   - **Step 3:** `CodeSyncMeshClient.spawn_blocking()` — works
   - **Step 4:** `sync_workspaces()` — **hangs here**
3. Checked Monarch logs: the `code_sync_manager` actor spawns successfully and goes Idle, but never receives a `Sync` message. No rsync daemon temp files are created, confirming the hang occurs before `TcpListener::bind` inside the Rust `code_sync_mesh` function
4. Confirmed the same test works perfectly via `python test_sync.py`

### Root Cause

The `sync_workspaces` method in `monarch_extension/src/code_sync.rs` uses `monarch_hyperactor::runtime::future_into_py` to bridge a Rust tokio future into a Python awaitable. This works correctly with `asyncio.run()` (which creates a clean event loop).

Jupyter runs its own persistent `asyncio` event loop (via `IPKernel`). On some Jupyter configurations, the `future_into_py` bridge doesn't properly re-poll the tokio future after the first yield. The async call hangs at the first `.await` point inside the Rust code (`TcpListener::bind(...).await` inside `code_sync_mesh`).

Evidence: `"Syncing workspace:"` is printed (happens synchronously before the first `.await`), but no rsync daemon temp files are ever created (would happen after the `.await` completes).

### Workarounds Tried

| Approach | Result |
|---|---|
| `nest_asyncio.apply()` at top of notebook | Did not work |
| `concurrent.futures.ThreadPoolExecutor` wrapping `asyncio.run()` | Did not work |
| `threading.Thread` with `asyncio.new_event_loop()` | Did not work |
| Subprocess calling a separate Python script | Did not work (can't share `ProcessJob` across processes) |
| **Standalone Python script with `asyncio.run()`** | **Works** |

### Working Solution

Run workspace sync from a Python script instead of Jupyter:

```python
# test_sync.py
import asyncio
import tempfile, shutil
from pathlib import Path
from monarch._src.job.process import ProcessJob
from monarch.tools.config.workspace import Workspace

async def main():
    tmpdir = Path(tempfile.mkdtemp(prefix="sync_test_"))
    local_ws = tmpdir / "local" / "my_project"
    local_ws.mkdir(parents=True)
    (local_ws / "test.txt").write_text("hello\n")
    remote_root = tmpdir / "remote" / "workspace"

    workspace = Workspace(dirs=[local_ws])
    job = ProcessJob({"hosts": 1}, env={"WORKSPACE_DIR": str(remote_root)})
    host = job.state(cached_path=None).hosts

    await host.sync_workspace(workspace)
    print("Sync complete!")

    host.shutdown().get()
    shutil.rmtree(tmpdir)

asyncio.run(main())
```

```sh
python test_sync.py
```

### Recommendations for Monarch developers

1. **Test `future_into_py` in Jupyter environments:** The bridge between tokio and Python's asyncio does not work reliably in Jupyter on all nodes
2. **Provide a synchronous API:** A `sync_workspace_blocking()` method that doesn't require async would work in all environments
3. **Document Jupyter limitations:** Note that async Monarch APIs may not work in Jupyter notebooks and recommend using Python scripts instead

---

## Environment Details

- **Working node:** Linux, Python 3.10, `torchmonarch` installed via pip
- **Faulty node:** Linux (Lightning AI cloud), Python 3.12, Monarch built from source
- **rsync version on faulty node:** 3.2.7, protocol version 31
- **Monarch version:** Built from source (main branch, ~March 2026)

## Files Changed

| File | Change |
|---|---|
| `monarch_hyperactor/src/code_sync/rsync.rs:261` | `hosts allow = localhost` → `hosts allow = localhost ip6-localhost` |
