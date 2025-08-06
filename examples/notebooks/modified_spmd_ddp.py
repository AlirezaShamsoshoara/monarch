import asyncio
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from monarch.actor import Actor, current_rank, endpoint, proc_mesh

from torch.nn.parallel import DistributedDataParallel as DDP

WORLD_SIZE = 8


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


class DDPActor(Actor):
    """This Actor wraps the basic functionality from Torch's DDP example. Conveniently, all of the
    methods we need are already laid out for us, so we can just wrap them in the usual Actor endpoint semantic with some light modifications

    # copy pasta from https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#basic-use-case
    """

    def __init__(self):
        self.rank = current_rank().rank

    def _rprint(self, msg):
        print(f"{self.rank=} {msg}")

    @endpoint
    async def setup(self):
        self._rprint("Initializing torch distributed")

        # initialize the process group
        dist.init_process_group("gloo", rank=self.rank, world_size=WORLD_SIZE)
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
        self._rprint("Running basic DDP example with huge model")

        # create model and move it to GPU with id rank
        model = ToyModel(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size
        ).to(self.rank)
        ddp_model = DDP(model, device_ids=[self.rank])

        # Print model size information
        total_params = sum(p.numel() for p in model.parameters())
        total_size_gb = total_params * 4 / (1024**3)  # 4 bytes per float32 parameter
        self._rprint(f"Model has {total_params:,} parameters ({total_size_gb:.2f} GB)")

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Generate random input data
            inputs = torch.randn(batch_size, input_size).to(self.rank)
            labels = torch.randn(batch_size, output_size).to(self.rank)

            # Forward pass
            outputs = ddp_model(inputs)
            loss = loss_fn(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                self._rprint(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self._rprint(f"Finished running basic DDP example with huge model")


async def main():
    """Main function that executes the SPMD DDP demo with huge model"""

    # Spawn a process mesh
    local_proc_mesh = await proc_mesh(
        gpus=WORLD_SIZE,
        env={
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12355",
        },
    )

    # Spawn our actor mesh on top of the process mesh
    ddp_actor = await local_proc_mesh.spawn("ddp_actor", DDPActor)

    # Setup torch Distributed
    await ddp_actor.setup.call()

    # Run the demo with extended training
    await ddp_actor.demo_basic.call(
        num_epochs=150,
        batch_size=128,
        input_size=256,
        hidden_size=512 * 32,
        output_size=128,
    )

    # Cleanup
    await ddp_actor.cleanup.call()


if __name__ == "__main__":
    asyncio.run(main())
