# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple workspace sync example — runs entirely on one machine.

Uses ProcessJob to spawn local subprocesses that simulate remote hosts.
rsync syncs files from a "local workspace" to a "remote workspace" directory.

Usage:
    python examples/workspace_sync/single_laptop_sync.py
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

from monarch._src.job.process import ProcessJob
from monarch.tools.config.workspace import Workspace


async def main():
    # 1. Create temp directories to simulate local and remote workspaces
    tmpdir = Path(tempfile.mkdtemp(prefix="sync_demo_"))
    local_workspace = tmpdir / "local" / "my_project"
    local_workspace.mkdir(parents=True)

    remote_workspace_root = tmpdir / "remote" / "workspace"

    print(f"Local workspace:  {local_workspace}")
    print(f"Remote workspace: {remote_workspace_root}")

    # 2. Create some files in the local workspace
    (local_workspace / "train.py").write_text("print('training!')\n")
    (local_workspace / "model.py").write_text("class MyModel: pass\n")
    subdir = local_workspace / "configs"
    subdir.mkdir()
    (subdir / "config.yaml").write_text("lr: 0.001\nepochs: 10\n")

    print("\nLocal files before sync:")
    for f in sorted(local_workspace.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(local_workspace)}")

    # 3. Create a Workspace config pointing to the local dir
    workspace = Workspace(dirs=[local_workspace])

    # 4. Start a ProcessJob (local subprocesses simulating remote hosts)
    job = ProcessJob(
        {"hosts": 1},
        env={"WORKSPACE_DIR": str(remote_workspace_root)},
    )
    host = job.state(cached_path=None).hosts

    # 5. Sync!
    print("\nSyncing workspace...")
    await host.sync_workspace(workspace)
    print("Sync complete!")

    # 6. Verify files appeared in the remote workspace
    remote_project = remote_workspace_root / "my_project"
    print("\nRemote files after sync:")
    for f in sorted(remote_project.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(remote_project)} -> {f.read_text().strip()}")

    # 7. Modify a file locally and re-sync
    (local_workspace / "train.py").write_text("print('training v2!')\n")
    print("\nModified train.py locally, re-syncing...")
    await host.sync_workspace(workspace)

    print(f"Remote train.py now: {(remote_project / 'train.py').read_text().strip()}")

    # Cleanup
    host.shutdown().get()
    shutil.rmtree(tmpdir)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
