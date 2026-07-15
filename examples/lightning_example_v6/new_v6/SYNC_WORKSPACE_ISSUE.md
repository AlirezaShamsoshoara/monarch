# Issue: `sync_workspace` Hangs (rsync daemon `hosts allow` name resolution)

> ## 2026-07 UPDATE — actual root cause (supersedes the "network topology" analysis below)
>
> After live debugging on a Lightning studio (`torchmonarch 0.6.0.dev20260606`), the
> real cause is **not** the client/remote network topology. The rsync data path is
> bridged over hyperactor channels — the worker's rsync never dials the client's
> localhost directly — so localhost binding is not the blocker.
>
> The daemon Monarch spawns is hardcoded with:
> ```
> hosts allow = localhost ip6-localhost
> ```
> On the affected container, rsync fails to resolve those loopback names when
> evaluating the allow-list, logging (in `/tmp/rsyncd.*/log`):
> ```
> malformed address localhost: Name or service not known
> connect from UNKNOWN (localhost)
> ```
> so it **denies the bridged connection**, and the client blocks forever waiting
> for a transfer that never starts.
>
> **What we established:**
> - `rsync` is installed (`/usr/bin/rsync` 3.2.7); `attach_to_workers` and mesh
>   comms work — the hang is purely in the rsync `hosts allow` check.
> - A **direct** manual test of an rsync daemon with `hosts allow = localhost
>   ip6-localhost` bound to `127.0.0.1` **allows** the connection
>   (`rsync allowed access ... from localhost (127.0.0.1)`). So hostname-based
>   `hosts allow` is *not* fundamentally broken.
> - Monarch's daemon fails because it binds **`::1` (IPv6) first**, and when
>   `localhost` doesn't resolve in the daemon's IPv6 family (pristine `/etc/hosts`
>   had `localhost` only on the IPv4 line), rsync logs `malformed address
>   localhost` on the first allow token and bails → denies → hang.
> - A manual daemon with **numeric** `hosts allow = 127.0.0.1 ::1` always allows
>   the connection with no name lookup at all.
>
> **Blocker #1 fix (works): numeric `hosts allow` via an rsync shim.** A tiny
> `rsync` shim earlier on `PATH` rewrites the daemon config's `hosts allow` to
> numeric `127.0.0.1 ::1`. Verified on the live studio: the shim is invoked
> (`[shim] ... --daemon ... --config=...`) and the daemon config ends up as
> `hosts allow = 127.0.0.1 ::1` — the exact config that a *manual* `rsync` test
> accepts.
>
 ## Blocker #2 (SCRIPT: fixed; JUPYTER: broken)
>
> With the shim fixing `hosts allow`, the second requirement is **how the call is
> awaited**:
>
> - **Plain `python` script — WORKS RELIABLY.** Run `sync_workspace` as an
>   asyncio **Task** (`asyncio.wait_for(...)`), NOT a bare `await`. Driving it as
>   a task lets the event loop service Monarch's rsync-bridge / subprocess
>   background work. A bare `await` fails with `unexpected early exit: ... 127`.
>   A tiny retry loop covers the occasional transient. Verified repeatedly:
>   `SYNC OK`, files transferred (`run_native_sync.py`).
>   (An `ENOENT`/`NotFound` seen during debugging was an artifact of a diagnostic
>   shim redirecting the daemon `--log-file` out of Monarch's temp dir — not a
>   real bug. Source dir `/tmp` vs `/teamspace` and `RUST_LOG` were both ruled
>   out.)
> - **Jupyter kernel — BROKEN.** Reproduced 5× via nbconvert in a clean
>   environment: ipykernel's persistent asyncio event loop reaps Monarch's rsync
>   daemon subprocess → `unexpected early exit 127` (or, with a wrapper shim, the
>   transfer completes but the `await` never returns). `asyncio.wait_for`, a
>   worker-thread `asyncio.run`, `SIGCHLD=SIG_DFL`, and killing all stray
>   processes were tried — none fixed it. This is a Monarch/ipykernel subprocess
>   conflict, not user-fixable.
>
> **Bottom line:**
> - **Native `sync_workspace` in a script:** ✅ works — `run_native_sync.py`
>   (shim + `wait_for` + retry). Run with `python run_native_sync.py`.
> - **Native `sync_workspace` in a notebook:** ❌ not viable on this build.
> - **In-notebook workflow:** use the **actor-based push**
>   (`studio_2_workspace_sync.ipynb`) — no rsync, reliable in notebook and script.
>
> **Upstream fix suggestion:** Monarch's rsync daemon should use numeric loopback
> (or a secrets file) for `hosts allow` instead of hostnames, so it doesn't
> depend on the container resolving `localhost` to a native `::1`.
>
> **Upstream fix suggestion:** Monarch's rsync daemon should use numeric loopback
> (`127.0.0.1`/`::1`) or a secrets file for `hosts allow`, not hostnames, so it
> doesn't depend on container name resolution.
>
> The Monarch transport port (e.g. `26600`) is unrelated — the rsync daemon picks
> its own ephemeral port.

---

## Original report (historical — root cause since corrected above)

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
