"""Worker-side + client-side helpers for the Monarch Lightning studios.

This replaces the old ``utils/`` package (``ip_utils.py``, ``mesh_utils.py``,
``master_node.py``) that the ``old_v0`` notebooks used with the Monarch **v0**
API (``RemoteAllocator`` + ``process_allocator`` + a custom master-node HTTP
registration server).

In current Monarch the flow is much simpler:

  * every remote worker runs ``bootstrap(port)`` which starts a long-lived
    Monarch worker loop (``run_worker_loop_forever``);
  * the client (notebook) calls ``enable_transport(...)`` and then
    ``attach_to_workers(...)`` to connect to those loops.

This module is imported on BOTH sides:

  * On the workers it is invoked as
    ``python -c 'from utils import bootstrap; bootstrap(26600)'``
    (see ``mmt_utils.launch_mmt_job``), so it must be importable from the
    worker working directory. On Lightning the studio is snapshotted to the
    workers, so keeping ``utils.py`` next to the notebooks is enough.
  * On the client the notebooks import ``get_host_ip_addr`` /
    ``bootstrap_addr``.

This is the same, tested helper used by the working
``monarch_lightning_supercell`` benchmark.
"""

import os
import subprocess

os.environ["XDG_RUNTIME_DIR"] = "/tmp"

from monarch.actor import run_worker_loop_forever


def get_host_ip_addr(addr_type: str = "public") -> str | None:
    """Return this machine's public IPv4 address using ``curl ifconfig.me``.

    Works on Lightning AI worker nodes (and the studio) where the instance
    metadata service is not always reachable. ``addr_type`` is accepted for
    API compatibility; only the public address is returned.
    """
    try:
        result = subprocess.run(
            ["curl", "-4", "ifconfig.me"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            public_ip = result.stdout.strip()
            if "." in public_ip and len(public_ip.split(".")) == 4:
                return public_ip
        return None
    except Exception as e:
        print(f"Error getting public IP with curl: {e}")
        return None


def bootstrap_addr(ip: str | None, port: int) -> str:
    """Build the alias listen/dial address ``tcp://IP:PORT@tcp://0.0.0.0:PORT``.

    The alias form encodes two things:
      * ``dial_to`` = the public IP (how the client reaches this worker), and
      * ``bind_to`` = ``0.0.0.0`` (AWS/Lightning does not allow binding
        directly to the public IP).
    """
    if ip is not None:
        return f"tcp://{ip}:{port}@tcp://0.0.0.0:{port}"
    return ""


def bootstrap(port: int, addr_type: str = "public") -> None:
    """Entry point run on each MMT worker node to start the Monarch worker loop.

    Blocks forever serving the worker loop; the client attaches to it via
    ``attach_to_workers``.
    """
    ip = get_host_ip_addr(addr_type)
    address = bootstrap_addr(ip, port)
    print(f"Worker bootstrap starting at {address}")
    run_worker_loop_forever(address=address, ca="trust_all_connections")
