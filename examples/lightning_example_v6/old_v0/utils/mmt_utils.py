from lightning_sdk import Machine, MMT, Studio, Status
from utils.mesh_utils import REMOTE_ALLOWED_PORT_RANGE

def launch_mmt_job(num_nodes=2, num_gpus=1, machine="L40S", mmt_job_name=""):
    """
    Launch a multi-machine training job using Lightning SDK's MMT API.
    """

    studio = Studio()

    try:
        job = MMT(
            name=mmt_job_name,
            _fetch_job = True
            )
        
        if job.status == Status('Running') or job.status == Status('Pending'):
            print(f"MMT job with {num_nodes} nodes is already created! Returning the the job")
            return job, studio

    except:
        print("Job has not been created by the user")


    # Install the MMT plugin befor running the actual job
    studio.install_plugin("multi-machine-training")

    print(f"Launching MMT job with {num_nodes} nodes...")

    machine_type = getattr(Machine, f"{machine}_X_{num_gpus}" if num_gpus > 1 else machine)

    job = MMT.run(
        command="process_allocator",
        name=f"Monarch-Titan-{num_nodes}-nodes",
        machine=machine_type,
        studio=studio,
        num_machines=num_nodes,
        env={
            "CUDA_VISIBLE_DEVICES": ",".join([str(el) for el in range(num_gpus)]),
            "MONARCH_FILE_LOG": "debug",
            "HYPERACTOR_REMOTE_ALLOC_ALLOWED_PORT_RANGE": REMOTE_ALLOWED_PORT_RANGE,
            "HYPERACTOR_REMOTE_ALLOC_BIND_TO_INADDR_ANY": "true",
            "WORKSPACE_DIR": "/tmp",
        },
    )

    print(f"Job started with ID: {job.name}")
    print(f"Job status: {job.status}")

    with open("/teamspace/studios/this_studio/job_name", "w") as f:
        f.write(job.name)

    # Monitor job status
    return job, studio