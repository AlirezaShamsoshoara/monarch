import os, socket, subprocess, sys, time, asyncio

os.environ["XDG_RUNTIME_DIR"] = "/tmp"
os.environ["WORKSPACE_DIR"] = "/tmp/monarch_workspace"
os.environ["MONARCH_FILE_LOG"] = "debug"
os.environ["HYPERACTOR_MESH_DEFAULT_TRANSPORT"] = "tcp"
# Verbose rust logging for the code-sync path (adjust module filters as needed).
os.environ["RUST_LOG"] = os.environ.get("RUST_LOG", "monarch_hyperactor=debug,hyperactor_mesh=debug")
os.makedirs(os.environ["WORKSPACE_DIR"], exist_ok=True)

# --- rsync shim: numeric hosts allow + log EVERY invocation (daemon AND client) ---
shim_dir = "/tmp/rsync_shim"
os.makedirs(shim_dir, exist_ok=True)
with open(os.path.join(shim_dir, "rsync"), "w") as f:
    f.write(
        "#!/usr/bin/env bash\n"
        'echo "[shim] $(date +%T) pid=$$ ppid=$PPID $*" >> /tmp/rsync_shim.log\n'
        'for a in "$@"; do case "$a" in --config=*) cfg="${a#--config=}"; '
        "[ -f \"$cfg\" ] && sed -i 's/^[[:space:]]*hosts allow.*/    hosts allow = 127.0.0.1 ::1/' \"$cfg\";; esac; done\n"
        'exec /usr/bin/rsync "$@"\n'
    )
os.chmod(os.path.join(shim_dir, "rsync"), 0o755)
try:
    os.remove("/tmp/rsync_shim.log")
except OSError:
    pass
os.environ["PATH"] = shim_dir + os.pathsep + os.environ["PATH"]

from monarch.actor import Actor, current_rank, endpoint, enable_transport
from monarch._src.actor.bootstrap import attach_to_workers
from monarch.tools.config.workspace import Workspace
from pathlib import Path

WP, CP = 26700, 26600
enable_transport(f"tcp://127.0.0.1:{CP}@tcp://0.0.0.0:{CP}")

wc = (
    "from monarch.actor import run_worker_loop_forever; "
    f"run_worker_loop_forever(address='tcp://127.0.0.1:{WP}@tcp://0.0.0.0:{WP}', ca='trust_all_connections')"
)
wp = subprocess.Popen([sys.executable, "-c", wc], env=os.environ.copy())
print("worker loop pid", wp.pid, flush=True)
time.sleep(12)

host = attach_to_workers(
    name="h", ca="trust_all_connections",
    workers=[f"tcp://127.0.0.1:{WP}@tcp://0.0.0.0:{WP}"],
)
pm = host.spawn_procs(per_host={"gpus": 2})
print("mesh built", flush=True)

src = "/tmp/dbg_ws"
os.makedirs(src, exist_ok=True)
with open(os.path.join(src, "a.txt"), "w") as f:
    f.write("hello\n")
ws = Workspace(dirs=[Path(src)])


async def main():
    print("CALLING sync_workspace...", flush=True)
    try:
        await asyncio.wait_for(
            host.sync_workspace(workspace=ws, conda=False, auto_reload=False),
            timeout=60,
        )
        print("RESULT: SYNC OK", flush=True)
    except asyncio.TimeoutError:
        print("RESULT: SYNC TIMEOUT", flush=True)
    except Exception as e:
        print("RESULT: EXC", repr(e), flush=True)


asyncio.run(main())

print("=== shim log ===", flush=True)
try:
    print(open("/tmp/rsync_shim.log").read(), flush=True)
except Exception:
    print("(no shim log)", flush=True)

wp.terminate()
