"""Lightning AI Multi-Machine-Training (MMT) launcher for Monarch.

Replaces the old ``utils/mmt_utils.py`` that launched workers with
``command="process_allocator"`` (the Monarch v0 remote-allocator daemon).

Each worker instead runs the Monarch worker loop directly via
``python -c 'from utils import bootstrap; bootstrap(<port>)'`` (see
``utils.py``). The client then connects with ``attach_to_workers`` — there is
no more ``RemoteAllocator`` / ``StaticRemoteAllocInitializer`` / master-node
registration server.

IMPORTANT: ``utils.py`` must be importable from the worker's working
directory. On Lightning the studio environment is snapshotted to the workers
at the same absolute path, so the worker command below ``cd``s into the
directory that holds this module (``utils.py`` lives next to it) and also puts
it on ``PYTHONPATH`` before running the bootstrap.
"""

import os

from lightning_sdk import Machine, MMT, Status, Studio


# Directory holding this module (and utils.py). The Lightning studio is
# snapshotted to the workers at the same absolute path, so the worker can cd
# here / import utils from here.
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))

# Where synced workspaces land on the workers (used by Studio 2 / sync_workspace).
WORKSPACE_DIR = "/tmp"


def _base_env(use_cpu: bool) -> dict:
    """Env vars forwarded to every worker.

    Log-forwarding + file-capture let the client stream aggregated worker logs
    back into the notebook (``proc_mesh.logging_option(...)``). ``PYTHONPATH``
    makes ``utils.py`` importable regardless of the worker's working directory.
    """
    env = {
        "XDG_RUNTIME_DIR": "/tmp",
        "WORKSPACE_DIR": WORKSPACE_DIR,
        "PYTHONPATH": _UTILS_DIR,
        "MONARCH_FILE_LOG": "debug",
        "HYPERACTOR_MESH_ENABLE_LOG_FORWARDING": "true",
        "HYPERACTOR_MESH_ENABLE_FILE_CAPTURE": "true",
        "HYPERACTOR_MESH_TAIL_LOG_LINES": "100",
    }
    return env


def _bootstrap_command(port: int) -> str:
    """Worker command: cd into the utils dir (so ``from utils import`` works
    regardless of the worker's CWD), then run the Monarch worker loop."""
    return f"cd {_UTILS_DIR} && python -c 'from utils import bootstrap; bootstrap({port})'"


def launch_mmt_job(
    num_nodes: int = 2,
    mmt_job_name: str = "",
    port: int = 26600,
    num_gpus: int = 8,
    machine: str = "L40S",
    use_cpu: bool = False,
):
    """Launch (or re-attach to) a Lightning MMT job running the Monarch worker loop.

    Args:
        num_nodes: number of worker machines.
        mmt_job_name: name of the MMT job. Pass the SAME name again (e.g. after
            a kernel restart) to re-attach to an already-running job instead of
            launching a new one.
        port: TCP port the Monarch worker loop listens on (client + workers use
            the same port, each on its own machine).
        num_gpus: GPUs per node (ignored when ``use_cpu=True``).
        machine: GPU family, e.g. ``"L40S"`` -> ``Machine.L40S_X_<num_gpus>``.
        use_cpu: if True, use ``Machine.CPU_X_4`` machines (handy for the
            Studio 2 / Studio 3 demos that don't need GPUs).

    Returns:
        (job, studio) tuple.
    """
    studio = Studio()

    # Re-attach to an existing job with the same name if it's already up.
    try:
        job = MMT(name=mmt_job_name, _fetch_job=True)
        if job.status in (Status("Running"), Status("Pending")):
            print(f"MMT job '{mmt_job_name}' already running/pending -- returning it")
            return job, studio
    except Exception:
        print("No existing job found, creating a new one")

    studio.install_plugin("multi-machine-training")
    print(f"Launching MMT job '{mmt_job_name}' with {num_nodes} nodes (use_cpu={use_cpu})...")

    # workers run the Monarch worker loop, NOT process_allocator.
    python_command = _bootstrap_command(port)

    if use_cpu:
        machine_type = Machine.CPU_X_4
        print("Using CPU machines (CPU_X_4)")
    else:
        machine_type = getattr(Machine, f"{machine}_X_{num_gpus}")
        print(f"Using GPU machines ({machine}_X_{num_gpus})")

    job = MMT.run(
        command=python_command,
        name=mmt_job_name,
        machine=machine_type,
        studio=studio,
        num_machines=num_nodes,
        env=_base_env(use_cpu),
    )

    print(f"Job started: {job.name}  status={job.status}")
    return job, studio


def launch_mmt_job_gcp(
    num_nodes: int = 2,
    mmt_job_name: str = "",
    port: int = 26600,
    num_gpus: int = 8,
    machine: str = "L40S",
    use_cpu: bool = False,
):
    """Same as :func:`launch_mmt_job` but adds the NCCL env vars needed when the
    underlying Lightning infra is GCP (disable IB / P2P, restrict the socket
    interface). Use this variant only if you see NCCL init hangs on GCP nodes.
    """
    studio = Studio()

    try:
        job = MMT(name=mmt_job_name, _fetch_job=True)
        if job.status in (Status("Running"), Status("Pending")):
            print(f"MMT job '{mmt_job_name}' already running/pending -- returning it")
            return job, studio
    except Exception:
        print("No existing job found, creating a new one")

    studio.install_plugin("multi-machine-training")
    print(f"Launching MMT job (GCP) '{mmt_job_name}' with {num_nodes} nodes...")

    python_command = _bootstrap_command(port)

    if use_cpu:
        machine_type = Machine.CPU_X_4
        print("Using CPU machines (CPU_X_4)")
    else:
        machine_type = getattr(Machine, f"{machine}_X_{num_gpus}")
        print(f"Using GPU machines ({machine}_X_{num_gpus})")

    env = _base_env(use_cpu)
    env.update(
        {
            "NCCL_NET_PLUGIN": "none",
            "NCCL_SOCKET_IFNAME": "^lo,docker",
            "NCCL_IB_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_DEBUG": "INFO",
        }
    )

    job = MMT.run(
        command=python_command,
        name=mmt_job_name,
        machine=machine_type,
        studio=studio,
        num_machines=num_nodes,
        env=env,
    )

    print(f"Job started: {job.name}  status={job.status}")
    return job, studio
