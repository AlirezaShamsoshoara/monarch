import subprocess
import urllib.request

from monarch.actor import run_worker_loop_forever


# This is provider specific because they have different metadata servers.
def get_host_ip_addr_customized(addr_type: str = "public") -> str:
    """
    See AWS doc for details.
    # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
    """
    METADATA_URL = "http://169.254.169.254/latest"

    # Step 1: Get a session token (IMDSv2)
    token = (
        urllib.request.urlopen(
            urllib.request.Request(
                f"{METADATA_URL}/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "30"},
                method="PUT",
            ),
            timeout=2,
        )
        .read()
        .decode()
    )
    # Step 2: Use the token to query metadata
    if addr_type == "public":
        command = "public-ipv4"
    elif addr_type == "private":
        command = "local-ipv4"
    else:
        raise ValueError(f"unsupported: {addr_type}")

    public_ip = (
        urllib.request.urlopen(
            urllib.request.Request(
                f"{METADATA_URL}/meta-data/{command}",
                headers={"X-aws-ec2-metadata-token": token},
            ),
            timeout=2,
        )
        .read()
        .decode()
    )
    return public_ip


def get_host_ip_addr(addr_type: str = "public"):
    """Get the public IP address using curl command (simpler approach)"""
    try:
        result = subprocess.run(
            ["curl", "-4", "ifconfig.me"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            public_ip = result.stdout.strip()
            # Basic validation that we got an IP address
            if "." in public_ip and len(public_ip.split(".")) == 4:
                return public_ip

        return None

    except Exception as e:
        print(f"Error getting public IP with curl: {e}")
        return None


def bootstrap_addr(ip: str | None, port: int) -> str:
    # bound to INADDR_ANY
    if ip != None:
        return f"tcp://{ip}:{port}@tcp://0.0.0.0:{port}"
    return ""


def bootstrap(port: int, addr_type: str = "public") -> None:
    ip = get_host_ip_addr(addr_type)
    address = bootstrap_addr(ip, port)
    run_worker_loop_forever(address=address, ca="trust_all_connections")
