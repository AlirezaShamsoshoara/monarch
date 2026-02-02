import os
# need to set before importing monarch
os.environ["MONARCH_FILE_LOG"] = "debug"
os.environ["HYPERACTOR_MESH_ENABLE_LOG_FORWARDING"] = "true"

import socket
import subprocess
import sys
import time

from utils import get_host_ip_addr, bootstrap_addr
from monarch.actor import Actor, enable_transport, endpoint
from monarch._src.actor.bootstrap import attach_to_workers


client_port = 26600
#  worker and client can use the same port if they are on different hosts.
worker_port = 26601

host_ip_addr = get_host_ip_addr(addr_type="public")
enable_transport(f"tcp://{host_ip_addr}:{client_port}@tcp://0.0.0.0:{client_port}")


class Hello(Actor):
    @endpoint
    def print_hello(self):
        print("HELLO!")


python_command = f'import socket; from utils import get_host_ip_addr; ip = get_host_ip_addr(); print(f"from subprocess {{ip}}")'

proc = subprocess.Popen(
    [
        sys.executable,
        "-c",
        python_command,
    ],
    env={},
)


print(f'Hello, Lightning World! Your public IP is {get_host_ip_addr()}; private IP is {get_host_ip_addr("private")}')

python_command = f'from utils import bootstrap; bootstrap({worker_port}, "public")'
worker_addr = bootstrap_addr(get_host_ip_addr(), worker_port)
print(f"{worker_addr}")

proc = subprocess.Popen(
    [
        sys.executable,
        "-c",
        python_command,
    ],
    env={
        "MONARCH_FILE_LOG": "debug",
    },
    start_new_session=True,
)

print(f"a worker host is running on pid {proc.pid}")

host_mesh = attach_to_workers(
    name="host_mesh", ca="trust_all_connections", workers=[worker_addr]
)

proc_mesh = host_mesh.spawn_procs()

hello = proc_mesh.spawn("hello", Hello)
hello.print_hello.call().get()
time.sleep(2)
hello.print_hello.call().get()

proc_mesh.stop().get()
host_mesh.shutdown().get()
print("done")

# top -p $(pgrep -d',' python)

# MONARCH_FILE_LOG=debug python main.py
