from lightning_sdk import Machine, MMT, Status, Studio


def launch_mmt_job_gcp(
    num_nodes: int = 2,
    mmt_job_name: str = "",
    port: int = 26600,
    num_gpus: int = 8,
    use_cpu: bool = False,
):
    """Launch an MMT job on Lightning AI when the underlying infra is GCP.

    Same as launch_mmt_job but includes NCCL env vars needed for GCP
    networking (disable IB, P2P, restrict socket interface).

    The MMT command runs the Monarch bootstrap on each worker node via
    ``from utils import bootstrap; bootstrap(port)``.

    IMPORTANT: For this command to work, ``utils.py`` must be importable
    from the worker's working directory. On Lightning, the studio
    environment is snapshotted to workers, so place ``utils.py`` in the
    studio root or adjust the command below to include the correct path.
    """
    studio = Studio()

    try:
        job = MMT(name=mmt_job_name, _fetch_job=True)
        if job.status == Status("Running") or job.status == Status("Pending"):
            print(f"MMT job '{mmt_job_name}' already running/pending -- returning it")
            return job, studio
    except Exception:
        print("No existing job found, creating a new one")

    studio.install_plugin("multi-machine-training")
    print(f"Launching MMT job (GCP) with {num_nodes} nodes (use_cpu={use_cpu})...")

    python_command = f"python -c 'from utils import bootstrap; bootstrap({port})'"

    if use_cpu:
        machine_type = Machine.CPU_X_4
        print("Using CPU machines (CPU_X_4)")
    else:
        # Machine with T4 GPUs
        # machine_type = getattr(Machine, f"T4_X_{num_gpus}")

        # Machine with L4 GPUs
        # machine_type = getattr(Machine, f"L4_X_{num_gpus}")

        # Machine with L40S GPUs
        machine_type = getattr(Machine, f"L40S_X_{num_gpus}")
        print(f"Using GPU machines (L40S_X_{num_gpus})")

    job = MMT.run(
        command=python_command,
        name=mmt_job_name,
        machine=machine_type,
        studio=studio,
        num_machines=num_nodes,
        env={
            "XDG_RUNTIME_DIR": "/tmp",
            "MONARCH_FILE_LOG": "debug",
            "HYPERACTOR_MESH_ENABLE_LOG_FORWARDING": "true",
            "HYPERACTOR_MESH_ENABLE_FILE_CAPTURE": "true",
            "HYPERACTOR_MESH_TAIL_LOG_LINES": "100",
            "NCCL_NET_PLUGIN": "none",
            "NCCL_SOCKET_IFNAME": "^lo,docker",
            "NCCL_IB_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_DEBUG": "INFO",
        },
    )

    print(f"Job started: {job.name}  status={job.status}")
    return job, studio


def launch_mmt_job(
    num_nodes: int = 2,
    mmt_job_name: str = "",
    port: int = 26600,
    num_gpus: int = 8,
    use_cpu: bool = False,
):
    """Launch an MMT job on Lightning AI (AWS infra).

    The MMT command runs the Monarch bootstrap on each worker node via
    ``from utils import bootstrap; bootstrap(port)``.

    IMPORTANT: For this command to work, ``utils.py`` must be importable
    from the worker's working directory. On Lightning, the studio
    environment is snapshotted to workers, so place ``utils.py`` in the
    studio root or adjust the command below to include the correct path.
    """
    studio = Studio()

    try:
        job = MMT(name=mmt_job_name, _fetch_job=True)
        if job.status == Status("Running") or job.status == Status("Pending"):
            print(f"MMT job '{mmt_job_name}' already running/pending -- returning it")
            return job, studio
    except Exception:
        print("No existing job found, creating a new one")

    studio.install_plugin("multi-machine-training")
    print(f"Launching MMT job with {num_nodes} nodes (use_cpu={use_cpu})...")

    python_command = f"python -c 'from utils import bootstrap; bootstrap({port})'"

    if use_cpu:
        # Machine with 4 X CPU
        machine_type = Machine.CPU_X_4
        print("Using CPU machines (CPU_X_4)")
    else:
        # Machine with T4 GPUs
        # machine_type = getattr(Machine, f"T4_X_{num_gpus}")

        # Machine with L4 GPUs
        # machine_type = getattr(Machine, f"L4_X_{num_gpus}")

        # Machine with L40S GPUs
        machine_type = getattr(Machine, f"L40S_X_{num_gpus}")
        print(f"Using GPU machines (L40S_X_{num_gpus})")

    job = MMT.run(
        command=python_command,
        name=mmt_job_name,
        machine=machine_type,
        studio=studio,
        num_machines=num_nodes,
        env={
            "XDG_RUNTIME_DIR": "/tmp",
            "MONARCH_FILE_LOG": "debug",
            "HYPERACTOR_MESH_ENABLE_LOG_FORWARDING": "true",
            "HYPERACTOR_MESH_ENABLE_FILE_CAPTURE": "true",
            "HYPERACTOR_MESH_TAIL_LOG_LINES": "100",
        },
    )

    print(f"Job started: {job.name}  status={job.status}")
    return job, studio
