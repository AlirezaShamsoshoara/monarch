from lightning_sdk import Machine, MMT, Status, Studio


def launch_mmt_job_gcp(num_nodes=2, mmt_job_name="", port=26600, num_gpus: int = 8):
    """
    Launch a multi-machine training job using Lightning SDK's MMT API.
    """

    studio = Studio()

    try:
        job = MMT(name=mmt_job_name, _fetch_job=True)

        if job.status == Status("Running") or job.status == Status("Pending"):
            print(
                f"MMT job with {num_nodes} nodes is already created! Returning the the job"
            )
            return job, studio

    except:
        print("Job has not been created by the user")

    # Install the MMT plugin befor running the actual job
    studio.install_plugin("multi-machine-training")

    print(f"Launching MMT job with {num_nodes} nodes...")

    python_command: str = f"python -c 'from utils import bootstrap; bootstrap({port})'"

    # Machine with T4 GPUs
    # machine_type = getattr(Machine, f"T4_X_{num_gpus}")

    # Machine with L4 GPUs
    # machine_type = getattr(Machine, f"L4_X_{num_gpus}")

    # Machine with L40S GPUs
    machine_type = getattr(Machine, f"L40S_X_{num_gpus}")

    # Machine with 4 X CPU
    # machine_type = getattr(Machine, "CPU_X_4")

    job = MMT.run(
        # command="process_allocator",
        command=python_command,
        name=mmt_job_name,
        # machine=Machine.CPU_X_4,
        machine=machine_type,
        studio=studio,
        num_machines=num_nodes,
        env={
            # "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",  # Make all GPUs visible # TODO: Should make this one dynamic
            "MONARCH_FILE_LOG": "debug",
            "HYPERACTOR_MESH_ENABLE_LOG_FORWARDING": "true",
            "HYPERACTOR_MESH_ENABLE_FILE_CAPTURE": "true",
            "HYPERACTOR_MESH_TAIL_LOG_LINES": "100",
            "NCCL_NET_PLUGIN": "none",
            "NCCL_SOCKET_IFNAME": "^lo,docker",
            "NCCL_IB_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_DEBUG": "INFO",
            # "HYPERACTOR_REMOTE_ALLOC_ALLOWED_PORT_RANGE": REMOTE_ALLOWED_PORT_RANGE,
            # "HYPERACTOR_REMOTE_ALLOC_BIND_TO_INADDR_ANY": "true",
            # "WORKSPACE_DIR": "/tmp",
            # "HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT": "300sec"
        },
    )

    print(f"Job started with ID: {job.name}")
    print(f"Job status: {job.status}")

    # Monitor job status
    return job, studio


def launch_mmt_job(num_nodes=2, mmt_job_name="", port=26600, num_gpus: int = 8):
    """
    Launch a multi-machine training job using Lightning SDK's MMT API.
    """

    studio = Studio()

    try:
        job = MMT(name=mmt_job_name, _fetch_job=True)

        if job.status == Status("Running") or job.status == Status("Pending"):
            print(
                f"MMT job with {num_nodes} nodes is already created! Returning the the job"
            )
            return job, studio

    except:
        print("Job has not been created by the user")

    # Install the MMT plugin befor running the actual job
    studio.install_plugin("multi-machine-training")

    print(f"Launching MMT job with {num_nodes} nodes...")

    python_command: str = f"python -c 'from utils import bootstrap; bootstrap({port})'"

    # Machine with T4 GPUs
    # machine_type = getattr(Machine, f"T4_X_{num_gpus}")

    # Machine with L4 GPUs
    # machine_type = getattr(Machine, f"L4_X_{num_gpus}")

    # Machine with L40S GPUs
    machine_type = getattr(Machine, f"L40S_X_{num_gpus}")

    # Machine with 4 X CPU
    # machine_type = getattr(Machine, "CPU_X_4")

    job = MMT.run(
        # command="process_allocator",
        command=python_command,
        name=mmt_job_name,
        # machine=Machine.CPU_X_4,
        machine=machine_type,
        studio=studio,
        num_machines=num_nodes,
        env={
            # "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",  # Make all GPUs visible # TODO: Should make this one dynamic
            "MONARCH_FILE_LOG": "debug",
            "HYPERACTOR_MESH_ENABLE_LOG_FORWARDING": "true",
            "HYPERACTOR_MESH_ENABLE_FILE_CAPTURE": "true",
            "HYPERACTOR_MESH_TAIL_LOG_LINES": "100",
            # "NCCL_NET_PLUGIN": "none",
            # "NCCL_SOCKET_IFNAME": "^lo,docker",
            # "NCCL_IB_DISABLE": "1",
            # "NCCL_P2P_DISABLE": "1",
            # "NCCL_DEBUG": "INFO",
            # "HYPERACTOR_REMOTE_ALLOC_ALLOWED_PORT_RANGE": REMOTE_ALLOWED_PORT_RANGE,
            # "HYPERACTOR_REMOTE_ALLOC_BIND_TO_INADDR_ANY": "true",
            # "WORKSPACE_DIR": "/tmp",
            # "HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT": "300sec"
        },
    )

    print(f"Job started with ID: {job.name}")
    print(f"Job status: {job.status}")

    # Monitor job status
    return job, studio
