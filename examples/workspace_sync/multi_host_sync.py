# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Workspace sync with multiple simulated hosts — runs on one machine.

Uses ProcessJob with 2 hosts to simulate a multi-host scenario.
Both "remote" hosts receive the synced workspace files.

Usage:
    python examples/workspace_sync/multi_host_sync.py
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

from monarch._src.job.process import ProcessJob
from monarch.tools.config.workspace import Workspace


async def main():
    # 1. Create temp directories
    tmpdir = Path(tempfile.mkdtemp(prefix="sync_multi_"))
    local_workspace = tmpdir / "local" / "my_project"
    local_workspace.mkdir(parents=True)

    remote_workspace_root = tmpdir / "remote" / "workspace"

    print(f"Local workspace:  {local_workspace}")
    print(f"Remote workspace: {remote_workspace_root}")

    # 2. Create project files
    (local_workspace / "train.py").write_text(
        "import torch\nprint('training on', torch.cuda.device_count(), 'GPUs')\n"
    )
    (local_workspace / "model.py").write_text("class MyModel: pass\n")
    (local_workspace / "data.py").write_text("def load_data(): return [1, 2, 3]\n")

    print("\nLocal files:")
    for f in sorted(local_workspace.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(local_workspace)}")

    # 3. Create workspace config
    workspace = Workspace(dirs=[local_workspace])

    # 4. Start ProcessJob with 2 hosts (both are local subprocesses)
    job = ProcessJob(
        {"hosts": 2},
        env={"WORKSPACE_DIR": str(remote_workspace_root)},
    )
    hosts = job.state(cached_path=None).hosts

    # 5. Sync to all hosts
    print("\nSyncing workspace to 2 hosts...")
    await hosts.sync_workspace(workspace)
    print("Sync complete!")

    # 6. Verify files on the remote side
    remote_project = remote_workspace_root / "my_project"
    if remote_project.exists():
        print("\nRemote files after sync:")
        for f in sorted(remote_project.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(remote_project)}")

    # 7. Add a new file and re-sync
    (local_workspace / "utils.py").write_text("def helper(): return 42\n")
    print("\nAdded utils.py locally, re-syncing...")
    await hosts.sync_workspace(workspace)

    if remote_project.exists():
        print("\nRemote files after second sync:")
        for f in sorted(remote_project.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(remote_project)}")

    # Cleanup
    hosts.shutdown().get()
    shutil.rmtree(tmpdir)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
