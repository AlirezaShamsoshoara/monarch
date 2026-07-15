#!/usr/bin/env python3
"""Native Monarch `sync_workspace` over rsync — WORKING, run as a plain script.

    python run_native_sync.py

Verified on the live Lightning studio (`torchmonarch 0.6.0.dev20260606`):
`SYNC OK`, files transferred, reproduced reliably.

Two things make native `sync_workspace` work here:

  1. A numeric-`hosts allow` rsync shim. Monarch's rsync daemon is hardcoded with
     `hosts allow = localhost ip6-localhost` and binds ::1; on this container
     `localhost` doesn't resolve to a native ::1, so the daemon denies the
     (channel-bridged) connection and the call hangs. The shim rewrites the
     daemon config's allow-list to numeric `127.0.0.1 ::1` (no name lookup).
     Monarch calls `rsync` via PATH, so it picks up the shim.

  2. Running the call as an asyncio Task (`asyncio.wait_for(...)`), NOT a bare
     `await`. Driving it as a task lets the event loop service Monarch's rsync
     bridge / subprocess background work; a bare `await` fails with
     "unexpected early exit". A small retry loop covers the occasional race.

  Also: keep IPv6 loopback enabled (do NOT `sysctl ... disable_ipv6=1`) — the
  worker's rsync connects to the bridge at tcp://[::1]:PORT.

NOTE: This does NOT work inside a Jupyter kernel — ipykernel's event loop reaps
Monarch's rsync daemon subprocess ("unexpected early exit 127"). For an
in-notebook workflow use the actor-based sync in `studio_2_workspace_sync.ipynb`.
See SYNC_WORKSPACE_ISSUE.md.
"""
import os, socket, subprocess, sys, time, asyncio

os.environ["XDG_RUNTIME_DIR"] = "/tmp"
os.environ["WORKSPACE_DIR"] = "/tmp/monarch_workspace"
os.environ["MONARCH_FILE_LOG"] = "debug"
os.environ["HYPERACTOR_MESH_DEFAULT_TRANSPORT"] = "tcp"
os.makedirs(os.environ["WORKSPACE_DIR"], exist_ok=True)

# --- numeric hosts-allow rsync shim ---
shim_dir = "/tmp/rsync_shim"
os.makedirs(shim_dir, exist_ok=True)
with open(os.path.join(shim_dir, "rsync"), "w") as f:
    f.write(
        "#!/usr/bin/env bash\n"
        'for a in "$@"; do case "$a" in --config=*) cfg="${a#--config=}"; '
        "[ -f \"$cfg\" ] && sed -i 's/^[[:space:]]*hosts allow.*/    hosts allow = 127.0.0.1 ::1/' \"$cfg\";; esac; done\n"
        'exec /usr/bin/rsync "$@"\n'
    )
os.chmod(os.path.join(shim_dir, "rsync"), 0o755)
os.environ["PATH"] = shim_dir + os.pathsep + os.environ["PATH"]

from monarch.actor import Actor, current_rank, endpoint, enable_transport
from monarch._src.actor.bootstrap import attach_to_workers
from monarch.tools.config.workspace import Workspace
from pathlib import Path

WP, CP = 26700, 26600
enable_transport(f"tcp://127.0.0.1:{CP}@tcp://0.0.0.0:{CP}")

wc = ("from monarch.actor import run_worker_loop_forever; "
      f"run_worker_loop_forever(address='tcp://127.0.0.1:{WP}@tcp://0.0.0.0:{WP}', ca='trust_all_connections')")
wp = subprocess.Popen([sys.executable, "-c", wc], env=os.environ.copy())
time.sleep(12)

host = attach_to_workers(name="h", ca="trust_all_connections",
                        workers=[f"tcp://127.0.0.1:{WP}@tcp://0.0.0.0:{WP}"])
pm = host.spawn_procs(per_host={"gpus": 2})

src = "/teamspace/studios/this_studio/monarch_sync_example"
os.makedirs(src, exist_ok=True)
with open(os.path.join(src, "training_config.toml"), "w") as f:
    f.write("[training]\nlearning_rate = 0.001\n")
ws = Workspace(dirs=[Path(src)])


async def sync_with_retry(attempts=6):
    for i in range(1, attempts + 1):
        try:
            # Run as a task (wait_for), NOT a bare await.
            await asyncio.wait_for(
                host.sync_workspace(workspace=ws, conda=False, auto_reload=False),
                timeout=30,
            )
            print(f"SYNC OK on attempt {i}", flush=True)
            return True
        except Exception as e:
            print(f"attempt {i} failed: {repr(e)[:80]}", flush=True)
            await asyncio.sleep(1.5)
    print("ALL ATTEMPTS FAILED", flush=True)
    return False


ok = asyncio.run(sync_with_retry())
dest = os.path.join(os.environ["WORKSPACE_DIR"], "monarch_sync_example", "training_config.toml")
print("file present on worker:", os.path.exists(dest), flush=True)
wp.terminate()
