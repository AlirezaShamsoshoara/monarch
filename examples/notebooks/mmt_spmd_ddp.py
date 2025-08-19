"""
Multi-Machine Training (MMT) SPMD DDP Example

This file integrates the Lightning SDK's MMT API with Monarch's SPMD DDP training
to enable distributed training across multiple nodes.
"""

import asyncio
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from lightning_sdk import Machine, MMT, Studio
from monarch.actor import Actor, current_rank, endpoint, proc_mesh
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=512, output_size=64):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.net2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.net3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu1(self.net1(x))
        x = self.dropout(x)
        x = self.relu2(self.net2(x))
        x = self.dropout(x)
        return self.net3(x)


class MultiNodeDDPActor(Actor):
    """
    Multi-node DDP Actor that can run across multiple machines using MMT.
    Adapted from the single-machine DDPActor to work in a multi-node environment.
    """

    def __init__(self, gpus_per_node=8):
        self.rank = current_rank().rank
        self.gpus_per_node = gpus_per_node

        # Get distributed environment variables set by MMT
        self.world_size = int(os.environ.get("WORLD_SIZE", gpus_per_node))
        self.node_rank = int(os.environ.get("NODE_RANK", 0))
        self.master_addr = os.environ.get("MASTER_ADDR", "localhost")
        self.master_port = os.environ.get("MASTER_PORT", "12355")

        # Calculate global rank
        self.global_rank = self.node_rank * self.gpus_per_node + self.rank

    def _rprint(self, msg):
        print(
            f"Node {self.node_rank}, Local Rank {self.rank}, Global Rank {self.global_rank}: {msg}"
        )

    @endpoint
    async def setup(self):
        self._rprint("Initializing torch distributed for multi-node training")
        self._rprint(
            f"World size: {self.world_size}, Master: {self.master_addr}:{self.master_port}"
        )

        # Initialize the process group for multi-node training
        dist.init_process_group(
            backend="nccl",  # Use NCCL for multi-GPU/multi-node
            rank=self.global_rank,
            world_size=self.world_size,
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
        )

        # Set the device for this process
        torch.cuda.set_device(self.rank)

        self._rprint("Finished initializing torch distributed")

    @endpoint
    async def cleanup(self):
        self._rprint("Cleaning up torch distributed")
        dist.destroy_process_group()

    @endpoint
    async def demo_basic(
        self,
        num_epochs=10,
        batch_size=64,
        input_size=128,
        hidden_size=512,
        output_size=64,
    ):
        self._rprint("Running multi-node DDP example")

        # Create model and move it to the appropriate GPU
        device = torch.device(f"cuda:{self.rank}")
        model = ToyModel(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size
        ).to(device)

        # Wrap model with DDP
        ddp_model = DDP(model, device_ids=[self.rank])

        # Print model size information (only from rank 0)
        if self.global_rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            total_size_gb = total_params * 4 / (1024**3)
            self._rprint(
                f"Model has {total_params:,} parameters ({total_size_gb:.2f} GB)"
            )

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Generate random input data
            inputs = torch.randn(batch_size, input_size).to(device)
            labels = torch.randn(batch_size, output_size).to(device)

            # Forward pass
            outputs = ddp_model(inputs)
            loss = loss_fn(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Print progress from all ranks every 5 epochs
            if epoch % 5 == 0:
                self._rprint(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self._rprint("Finished multi-node DDP training")


async def run_multi_node_training(gpus_per_node=8):
    """
    Run multi-node training using the local process mesh.
    This function is called on each node by the MMT job.
    """
    # Get the number of GPUs available on this node
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else gpus_per_node

    print(f"Starting training on node with {num_gpus} GPUs")

    # Create process mesh for this node
    local_proc_mesh = await proc_mesh(
        gpus=num_gpus,
        env={
            "MASTER_ADDR": os.environ.get("MASTER_ADDR", "localhost"),
            "MASTER_PORT": os.environ.get("MASTER_PORT", "12355"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE", str(num_gpus)),
            "NODE_RANK": os.environ.get("NODE_RANK", "0"),
        },
    )

    # Spawn our actor mesh on top of the process mesh
    ddp_actor = await local_proc_mesh.spawn("ddp_actor", MultiNodeDDPActor, num_gpus)

    # Setup torch Distributed
    await ddp_actor.setup.call()

    # Run the training with larger model for multi-node setup
    await ddp_actor.demo_basic.call(
        num_epochs=100,
        batch_size=256,
        input_size=512,
        hidden_size=1024 * 16,  # Large model that benefits from multi-node
        output_size=256,
    )

    # Cleanup
    await ddp_actor.cleanup.call()


def launch_mmt_job(num_nodes=3, teamspace="my-teamspace", org="my-org"):
    """
    Launch a multi-machine training job using Lightning SDK's MMT API.
    """
    # Initialize a Studio
    # studio = Studio(name="multi-node-ddp-training", teamspace=teamspace, org=org)
    studio = Studio()
    # studio.start()

    print(f"Launching MMT job with {num_nodes} nodes...")

    # Run a Multi-machine job
    job = MMT.run(
        command="python multi_node_train.py",
        name="multi-node-ddp-training",
        machine=Machine.T4_X_4,  # Use GPU machines for training
        studio=studio,
        num_machines=num_nodes,
        env={
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",  # Make all GPUs visible
        },
    )

    print(f"Job started with ID: {job.name}")
    print(f"Job status: {job.status}")

    # Monitor job status
    return job, studio


def monitor_job(job, studio):
    """
    Monitor the job status and provide updates.
    """
    import time

    print("Monitoring job status...")
    while job.status in ["Running", "Pending"]:
        print(f"Job status: {job.status}")
        time.sleep(30)  # Check every 30 seconds

    print(f"Final job status: {job.status}")

    # Clean up
    if job.status == "Completed":
        print("Training completed successfully!")
    else:
        print(f"Training finished with status: {job.status}")

    # Shut down the studio
    studio.stop()


if __name__ == "__main__":
    # Check if this is being run as part of an MMT job or as the launcher
    if "NODE_RANK" in os.environ:
        # This is being executed on a compute node by MMT
        print("Running as part of MMT job...")
        asyncio.run(run_multi_node_training())
    else:
        # This is the launcher script
        print("Launching multi-node training job...")

        # Configuration
        NUM_NODES = 3
        TEAMSPACE = "general"  # Replace with your teamspace
        ORG = "meta-ai"  # Replace with your username

        # Launch the job
        job, studio = launch_mmt_job(num_nodes=NUM_NODES, teamspace=TEAMSPACE, org=ORG)

        # Monitor the job (optional - you can also monitor separately)
        # monitor_job(job, studio)

        print(f"Job launched. You can monitor it using: job.status")
        print(f"To stop the job: job.stop()")
        print(f"To clean up: studio.stop()")
