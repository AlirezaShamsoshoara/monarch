# Everything is like a Job API, instead it is MMT

# Install the Lightning SDK
# pip install lightning-sdk

# Export authentication variables
# export LIGHTNING_USER_ID=your-user-id
# export LIGHTNING_API_KEY=your-api-key

from lightning_sdk import Machine, MMT, Studio

# Step 1: Initialize a Studio
studio = Studio(name="batch-processing", teamspace="my-teamspace", user="my-user")
studio.start()

# Step 3: Run a Multi-machine job
job = MMT.run(
    command="python process_data.py",
    name="data-job",
    machine=Machine.T4,
    studio=studio,
    num_machines=2,
)

# Step 4: Monitor Job Status
print(job.status)  # Running, Completed, Failed

# Step 5: Stop or Delete a Job
# job.stop()  # Gracefully stop a running job
# job.delete()  # Cancel and remove the job

# Step 6: Shut Down the Studio
studio.stop()

# List running multi-machine jobs
teamspace = Teamspace("my-teamspace", user="my-user")

for job in teamspace.multi_machine_jobs:
    print(job.name, job.status)


# Check multi-machine status
job = MMT("data-job", teamspace="my-teamspace", user="my-user")
print(job.status)  # Running, Completed, Failed
