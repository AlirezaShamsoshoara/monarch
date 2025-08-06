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
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


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
    async def demo_basic(self):
        self._rprint("Running basic DDP example")
        # setup(rank, world_size)

        # create model and move it to GPU with id rank
        model = ToyModel().to(self.rank)
        ddp_model = DDP(model, device_ids=[self.rank])

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(self.rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()

        # cleanup()
        print(f"{self.rank=} Finished running basic DDP example")


async def main():
    """Main function that executes the SPMD DDP demo"""

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

    # Run the demo
    await ddp_actor.demo_basic.call()

    # Cleanup
    await ddp_actor.cleanup.call()


if __name__ == "__main__":
    asyncio.run(main())
