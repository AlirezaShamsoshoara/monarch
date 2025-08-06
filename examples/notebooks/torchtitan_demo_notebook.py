#!/usr/bin/env -S grimaldi --kernel conda_torchtitan_conda_prod-cd41765
# fmt: off

""":md
# What is Monarch?
Monarch is python-first language for describing distributed workloads. In this example, we use TorchTitan to demonstrate how Monarch can be used to run and orhestrate any workload.

N.B. This notebook is under construction! Check back soon for examples including:
1. Code synchronization APIs
1. Remote breakpoints
1. Improved Observability workloads


## Step 1: Launch the job on Mast

First, set up your local enviornment. Run the following commands to install TorchTitan/Monarch locally.

1. Get your base conda env:

    ```source /data/users/$USER/fbsource/genai/xlformers/dev/xl_conda.sh activate torchtitan_conda_prod:latest_conveyor_build```

1. Install Titan:

     `with-proxy pip install pyre-extensions torchao cloudpickle tyro datasets torchtitan`
1. Install Monarch:

    `with-proxy pip install --force-reinstall $(buck2 build @fbcode//mode/opt --show-full-simple-output 'fbcode//monarch/python/monarch:monarch_no_torch.whl')`
1. Select your kernel from the dop down.

"""

""":py"""
"""
Steps to set up your conda env


"""

import getpass
import os
import tempfile

from monarch.tools import commands
from monarch.tools.components.meta import hyperactor
from monarch.tools.config import Config
from torchtitan.tools.logging import init_logger, logger
from torchx.specs.fb.component_helpers import Packages


def get_config(temp_dir) -> Config:
    packages = Packages()
    packages.add_package("oil.oilfs:stable")
    packages.add_package("manifold.manifoldfs")

    config = Config(
        scheduler="mast_conda",
        scheduler_args={
            # NOTE: replace with your own values
            "hpcIdentity": "pytorch_distributed",
            "hpcJobOncall": "monarch",
            "hpcClusterUuid": "MastProdCluster",
            "rmAttribution": "pytorch4all_clients_approved",
        },
        appdef=hyperactor.host_mesh_conda(
            meshes=["mesh0:1:gtt_any"],  # mesh_name:num_hosts:host_type
            additional_packages=packages,
        ),
        workspace=temp_dir,
    )
    return config


jobname: str = f"monarch-{getpass.getuser()}"
temp_dir = tempfile.mkdtemp()
config = get_config(temp_dir)

# NOTE: There's currently an issue with the MAST scheduler which sometimes causes "pending" jobs to fail at the following command.
# If this happens, you can re-run the following command to connect to the job once it's running.
await commands.get_or_create(jobname, config)
os.rmdir(temp_dir)

""":md
## Step 2: Launch Training

Below, we wrap TorchTitanss train method in Monarch a Monarch Actor
"""

""":py"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import getpass
import socket
import subprocess
from typing import Optional

import torch
from monarch._rust_bindings.monarch_hyperactor.alloc import AllocConstraints, AllocSpec
from monarch._src.actor.actor_mesh import Actor, current_rank
from monarch._src.actor.endpoint import endpoint
from monarch._src.actor.meta.allocator import MastAllocator, MastAllocatorConfig
from monarch._src.actor.proc_mesh import ProcMesh
from monarch.meta.utils import setup_env_for_distributed

from torchtitan.config_manager import ConfigManager, JobConfig

from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer


tokenizer_file_path = "/tmp/monarch_titan_tokenizer_tokenizer.model"
mount_file_path = "/tmp/mount_fs.sh"


def env_setup() -> None:
    os.environ["NCCL_DEBUG"] = "INFO,WARN"
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    os.environ["TORCH_ADDR2LINE_BINARY"] = "/packages/folly.symbolizer/folly-addr2line"
    os.environ["FUSE_DST"] = "/mnt/mffuse"
    os.environ["MANIFOLDFS_BUCKET"] = "torchtrain_datasets"
    # --- WS-Airstore configuration
    os.environ["ENABLE_AIRSTORE"] = ""
    os.environ["DISABLE_OILFS"] = "1"
    os.environ["AIRSTORE_DECRYPT_SERVER_AFFINITY"] = "parent"
    os.environ["AIRSTORE_DECRYPT_SERVER_PATH"] = (
        "/packages/ws_airstore.client/decrypt_server"
    )
    os.environ["AIRSTORE_LOCAL_MOUNT_ROOT"] = "/mnt/airstore"
    # WS-AIRStore caches the shuffling, sharding information to enable fast startups
    os.environ["AIRSTORE_INTERVAL_CACHE_DIR"] = "/mnt/airstore/airstore_metadata_cache"
    # For long running llamma4 production training jobs, please
    #  set AIRSTORE_FBPKG_ID env var to ws_airstore.client:prod
    os.environ["AIRSTORE_FBPKG_ID"] = "ws_airstore.client:prod"
    # --- OilFS
    os.environ["WS_SSCV2_THRIFT_CONN_POOL_SIZE"] = "250000"
    # Only used for pretraining jobs. Perf tweaks for 8k+ gpu jobs
    os.environ["OILFS_PROFILE"] = "pretraining"


async def create_mast_proc_mesh(
    task_group: str,
    job_name: str,
    monarch_port: int = 26600,
    num_hosts: int = 1,  # TODO: get the task_group size from MAST
    num_gpus: int = 8,
) -> ProcMesh:
    """Create a process mesh using MAST allocation.

    Args:
        task_group: The task group name
        job_name: The MAST job name
        monarch_port: The monarch port to use

    Returns:
        The created process mesh
    """
    allocator = MastAllocator(
        MastAllocatorConfig(
            job_name=job_name,
            remote_allocator_port=monarch_port,
        ),
    )
    spec = AllocSpec(
        AllocConstraints({MastAllocator.ALLOC_LABEL_TASK_GROUP: task_group}),
        hosts=num_hosts,
        gpus=num_gpus,
    )
    allocation = await allocator.allocate(spec)
    hyperactor_mesh = await ProcMesh.from_alloc(allocation, env_setup)

    return hyperactor_mesh


def _get_hostname():
    hostname = socket.gethostname()
    return hostname


class TrainerActorWrapper(Actor):
    def __init__(
        self, job_config: JobConfig, tokenizer_content: str, mount: str, env_to_merge={}
    ):
        self.job_config = job_config
        self.rank = current_rank().rank
        hostname = socket.gethostname()
        print(f"Initializing actor: {self.rank} {hostname=} {current_rank()=}")
        os.environ.update(env_to_merge)

        if self.rank % 8 == 0:
            # just use one actor to do the work
            print(
                f"writing tokenizer to {tokenizer_file_path}; content size: {len(tokenizer_content)}"
            )
            with open(tokenizer_file_path, "w") as tmp_file:
                tmp_file.write(tokenizer_content)
            print(f"writing mount.sh to {mount_file_path}; content size: {len(mount)}")
            with open(mount_file_path, "w") as tmp_file:
                tmp_file.write(mount)
            os.chmod(mount_file_path, 0o777)
            subprocess.run(
                [mount_file_path], capture_output=True, text=True, check=True
            )

    @endpoint
    def get_hostname(self):
        return _get_hostname()

    @endpoint
    def train(self):
        print("Starting training")
        logger.info("magicword: INFO: Starting training")
        logger.error("magicword: ERROR: Starting training")
        logger.critical("magicword: CRITICAL: Starting training")
        config = self.job_config
        trainer: Optional[Trainer] = None

        try:
            trainer = Trainer(config)
            trainer.train()

            if config.checkpoint.create_seed_checkpoint:
                assert (
                    int(os.environ["WORLD_SIZE"]) == 1
                ), "Must create seed checkpoint using a single device, to disable sharding."
                assert (
                    config.checkpoint.enable_checkpoint
                ), "Must enable checkpointing when creating a seed checkpoint."
                trainer.checkpointer.save(curr_step=0, force=True)
                logger.info("Created seed checkpoint")
            else:
                trainer.train()
        finally:
            if trainer:
                trainer.close()

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                logger.info("Process group destroyed.")
        print("Done training")


async def async_main(job_config: JobConfig):
    torch.use_deterministic_algorithms(True)
    job_name = f"monarch-{getpass.getuser()}"

    print(f"Spawning meshes on {job_name}")
    mast_proc_mesh = await create_mast_proc_mesh(
        task_group="mesh0",  # default TG for how we launched the job
        job_name=job_name,
        num_hosts=1,
        num_gpus=8,  # this really only controls the number of workers each actor spawns
    )
    await setup_env_for_distributed(mast_proc_mesh)

    await mast_proc_mesh.logging_option(stream_to_client=True, aggregate_window_sec=3)

    # Read and ship tokenizer file to remote
    file_path = os.path.expanduser(
        f"/home/{getpass.getuser()}/local/fbsource/fbcode/monarch/meta/torchtitan/workspace/test_tiktoken_tokenizer.model"
    )

    mount_path = os.path.expanduser(
        f"/home/{getpass.getuser()}/local/fbsource/fbcode/ai_codesign/oss_infra_launch/torchtitan/mount.sh"
    )

    with open(file_path, "r") as file:
        content = file.read()
    with open(mount_path, "r") as file:
        mount = file.read()

    print(job_config)
    print(f"Spawning meshes on {job_name}")
    trainer_actor = await mast_proc_mesh.spawn(
        "trainer_actor", TrainerActorWrapper, job_config, content, mount, {}
    )
    await trainer_actor.train.call()

    # print(f"trainer_actor: {trainer_actor=}")
    # print("trainer_actor: waiting for rank0_hostname")
    # hostnames = await trainer_actor.get_hostname.call()
    # print("got all hostnames")
    # rank0_hostname = hostnames.item(hosts=0, gpus=0)
    # print(f"{rank0_hostname=}", flush=True)
    # await trainer_actor.train.call(rank0_hostname)


if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()

    mast_job_name = f"monarch-{getpass.getuser()}"
    os.environ["MAST_HPC_JOB_NAME"] = mast_job_name
    manual_args = [
        "--job.config_file",
        os.path.expanduser("~/fbsource/fbcode/monarch/meta/torchtitan/llama3_8b.toml"),
        "--model.tokenizer-path",
        tokenizer_file_path,
        "--training.steps",
        "5",
        "--training.dataset_path",
        "/mnt/mffuse/c4",
        "--job.dump_folder",
        "/mnt/mffuse/outputs/" + mast_job_name,
    ]
    config = config_manager.parse_args(manual_args)
    await async_main(config)
    print("All Done")

""":py"""
