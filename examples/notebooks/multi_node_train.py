import asyncio
import os
import sys

# Add the current directory to Python path if needed
# sys.path.append(os.getcwd())

from mmt_spmd_ddp import run_multi_node_training

if __name__ == "__main__":
    asyncio.run(run_multi_node_training(gpus_per_node=8))
