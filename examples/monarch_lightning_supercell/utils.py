import os
import subprocess
import urllib.request

os.environ["XDG_RUNTIME_DIR"] = "/tmp"

from monarch.actor import run_worker_loop_forever


def get_host_ip_addr(addr_type: str = "public") -> str:
    """Get IP address using curl (works on Lightning AI worker nodes)."""
    try:
        result = subprocess.run(
            ["curl", "-4", "ifconfig.me"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            public_ip = result.stdout.strip()
            if "." in public_ip and len(public_ip.split(".")) == 4:
                return public_ip
        return None
    except Exception as e:
        print(f"Error getting public IP with curl: {e}")
        return None


def bootstrap_addr(ip: str | None, port: int) -> str:
    if ip is not None:
        return f"tcp://{ip}:{port}@tcp://0.0.0.0:{port}"
    return ""


def bootstrap(port: int, addr_type: str = "public") -> None:
    """Called on each MMT worker node to start the Monarch worker loop.

    Uses the alias address format (tcp://PUBLIC_IP:PORT@tcp://0.0.0.0:PORT):
      - dial_to = public IP (how the controller reaches this worker)
      - bind_to = 0.0.0.0 (AWS doesn't allow binding to public IPs)

    Requires torchmonarch<=0.4.1. The alias listen path is broken in 0.5.0
    (regression in listen_with_prebound, introduced by commit a5ac9528d).
    """
    ip = get_host_ip_addr(addr_type)
    address = bootstrap_addr(ip, port)
    print(f"Worker bootstrap starting at {address}")
    run_worker_loop_forever(address=address, ca="trust_all_connections")
