# Issue: `sync_workspace` Hangs When Using `attach_to_workers` with Remote Machines

## Summary

When using `host_mesh.sync_workspace()` with a `HostMesh` created via `attach_to_workers()` connected to remote machines (e.g., Lightning AI MMT nodes), the sync operation hangs indefinitely. The rsync daemon binds to localhost addresses, making it unreachable from remote workers.

---

## Environment

- **Monarch Version**: 0.3.0 (installed from PyPI as `torchmonarch`)
- **Platform**: Lightning AI Studios
- **Setup**: Client notebook connecting to remote MMT worker nodes via `attach_to_workers()`

---

## Steps to Reproduce

1. Launch remote worker nodes using Lightning AI MMT
2. Connect to workers using `attach_to_workers()`:
   ```python
   from monarch._src.actor.bootstrap import attach_to_workers

   worker_addrs = [f"tcp://{ip}:{port}@tcp://0.0.0.0:{port}" for ip in worker_ips]
   host_mesh = attach_to_workers(
       name="host_mesh", ca="trust_all_connections", workers=worker_addrs
   )
   proc_mesh = host_mesh.spawn_procs(per_host={"cpus": 4})
   ```

3. Attempt to sync workspace:
   ```python
   from monarch.tools.config.workspace import Workspace
   from pathlib import Path

   workspace = Workspace(dirs=[Path("/path/to/local/directory")])
   await host_mesh.sync_workspace(workspace=workspace)  # <-- Hangs here
   ```

---

## Root Cause Analysis

### 1. Rsync Daemon Binds to Localhost

**File**: `monarch_hyperactor/src/code_sync/manager.rs` (lines 451-456)

```rust
let ipv6_lo: SocketAddr = "[::1]:0".parse()?;
let ipv4_lo: SocketAddr = "127.0.0.1:0".parse()?;
let addrs: [SocketAddr; 2] = [ipv6_lo, ipv4_lo];
let daemon =
    RsyncDaemon::spawn(TcpListener::bind(&addrs[..]).await?, &local_workspace).await?;
```

The rsync daemon spawned on the client machine binds to **localhost addresses only** (`127.0.0.1` and `::1`). This means the daemon is only accessible from the local machine.

### 2. Remote Workers Cannot Connect

When `sync_workspace()` is called:

1. The client spawns an rsync daemon on localhost (e.g., `127.0.0.1:random_port`)
2. The client sends the daemon address to remote workers
3. Remote workers attempt to connect to `127.0.0.1:port` on **their own machines**
4. The connection fails silently or hangs because there's no rsync daemon on the remote machines at that address
5. The sync operation waits indefinitely for rsync results that never arrive

### 3. Architecture Mismatch

The `sync_workspace` feature appears to be designed for scenarios where:
- Workers are allocated via allocators (Kubernetes, local process allocator, etc.)
- Network topology allows workers to reach the client's rsync daemon
- Or the rsync daemon binds to a publicly accessible address

With `attach_to_workers()` connecting to remote machines over the internet (Lightning AI), there's no network path from the remote workers back to the client's localhost.

---

## Code Flow Reference

### Entry Point
**File**: `python/monarch/_src/actor/host_mesh.py` (lines 308-329)

```python
async def sync_workspace(
    self,
    workspace: Workspace,
    conda: bool = False,
    auto_reload: bool = False,
) -> None:
    if self._code_sync_proc_mesh:
        await self._code_sync_proc_mesh.get()._sync_workspace(
            workspace, conda, auto_reload
        )
```

### Sync Implementation
**File**: `python/monarch/_src/actor/proc_mesh.py` (lines 674-805)

```python
async def _sync_workspace(self, workspace, conda, auto_reload):
    # ...
    await self._code_sync_client.sync_workspaces(
        instance=context().actor_instance._as_rust(),
        workspaces=list(workspaces.values()),
        auto_reload=auto_reload,
    )
```

### Rsync Daemon Spawn (The Problem)
**File**: `monarch_hyperactor/src/code_sync/manager.rs` (lines 449-466)

```rust
CodeSyncMethod::Rsync => {
    // Spawn a rsync daemon to accept incoming connections from actors.
    let ipv6_lo: SocketAddr = "[::1]:0".parse()?;
    let ipv4_lo: SocketAddr = "127.0.0.1:0".parse()?;
    let addrs: [SocketAddr; 2] = [ipv6_lo, ipv4_lo];
    let daemon =
        RsyncDaemon::spawn(TcpListener::bind(&addrs[..]).await?, &local_workspace).await?;
    // ...
}
```

---

## Potential Fixes

### Option 1: Bind to Public Interface

Modify the rsync daemon to bind to `0.0.0.0` (all interfaces) instead of localhost when remote workers are detected:

```rust
// In manager.rs
let bind_addr = if is_remote_connection {
    "0.0.0.0:0".parse()?
} else {
    "127.0.0.1:0".parse()?
};
let daemon = RsyncDaemon::spawn(TcpListener::bind(bind_addr).await?, &local_workspace).await?;
```

**Considerations**:
- Security implications of exposing rsync daemon to network
- Firewall/NAT traversal issues
- Need to communicate the client's public IP to workers

### Option 2: Reverse Connection Model

Instead of workers connecting to the client's rsync daemon, have the client push files to each worker:

1. Each worker spawns an rsync daemon on their machine
2. Client connects to each worker's daemon and pushes files
3. Workers report success/failure back to client

### Option 3: Use SSH/SCP Tunneling

Leverage existing SSH connections (if available) to tunnel rsync traffic:

```rust
// Create SSH tunnel to each worker, then rsync through tunnel
```

### Option 4: Alternative Sync Method for Remote Workers

Add a new `CodeSyncMethod` specifically for remote worker scenarios that uses a different transport (e.g., HTTP-based file transfer, cloud storage as intermediary).

### Option 5: Document as Limitation

If the above fixes are complex, document this as a known limitation:
- `sync_workspace` works with local allocators and Kubernetes deployments
- `sync_workspace` does NOT work with `attach_to_workers` to remote machines
- Recommend manual file sync (scp, rsync CLI) for remote worker scenarios

---

## Workarounds (For Users)

Until this is fixed, users can:

1. **Manual rsync**: Use command-line rsync/scp to sync files to remote workers before running training

2. **Shared storage**: Use cloud storage (S3, GCS) as an intermediary - upload from client, download on workers

3. **Include files in environment**: Package necessary files in the worker environment/image before launching

---

## Additional Notes

### Dimension Label Issue

There's also a secondary issue in `proc_mesh.py` (line 698):

```python
assert set(self._region.labels).issubset({"gpus", "hosts"})
```

And line 710:
```python
shape=WorkspaceShape.shared("gpus"),
```

This assumes a "gpus" dimension exists, which may not be true for CPU-only deployments. However, this is separate from the hanging issue.

---

## Contact

If you need additional logs or testing, please let me know. The client logs showing the hang are available in `monarch_client_log.log`.
