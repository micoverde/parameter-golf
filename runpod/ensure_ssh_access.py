#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap SSH access on a RunPod parameter-golf pod using the Jupyter terminal path."
    )
    parser.add_argument("--pod-id", required=True, help="RunPod pod id, e.g. p3bo73p2x1prxp")
    parser.add_argument(
        "--public-key-file",
        default=str(Path.home() / ".ssh" / "id_rsa.pub"),
        help="Public key to install into /root/.ssh/authorized_keys",
    )
    parser.add_argument(
        "--private-key-file",
        default=str(Path.home() / ".ssh" / "id_rsa"),
        help="Private key used to verify SSH after bootstrap",
    )
    parser.add_argument(
        "--jupyter-password",
        default="",
        help="Override the Jupyter password instead of reading it from pod env",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout for remote bootstrap and SSH verification",
    )
    return parser.parse_args()


def runpod_api_post(query: str) -> dict:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is required")
    response = requests.post(
        f"https://api.runpod.io/graphql?api_key={api_key}",
        json={"query": query},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if "errors" in payload:
        raise SystemExit(json.dumps(payload["errors"], indent=2))
    return payload


def load_pod(pod_id: str) -> dict:
    payload = runpod_api_post(
        """
        query PodsWithMore {
          myself {
            pods {
              id
              name
              desiredStatus
              env
              runtime {
                ports {
                  ip
                  isIpPublic
                  privatePort
                  publicPort
                  type
                }
              }
            }
          }
        }
        """
    )
    for pod in payload["data"]["myself"]["pods"]:
        if pod["id"] == pod_id:
            return pod
    raise SystemExit(f"pod not found: {pod_id}")


def find_jupyter_password(pod: dict, override: str) -> str:
    if override:
        return override
    for item in pod.get("env") or []:
        if item.startswith("JUPYTER_PASSWORD="):
            return item.split("=", 1)[1]
    raise SystemExit(f"JUPYTER_PASSWORD not found for pod {pod['id']}")


def find_ssh_endpoint(pod: dict) -> tuple[str, int]:
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if port.get("isIpPublic") and port.get("privatePort") == 22:
            return port["ip"], int(port["publicPort"])
    raise SystemExit(f"public SSH endpoint not found for pod {pod['id']}")


def build_base_url(pod_id: str) -> str:
    return f"https://{pod_id}-8888.proxy.runpod.net"


def load_public_key(path: str) -> str:
    key = Path(path).read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit(f"public key file is empty: {path}")
    return key


def run_local(cmd: list[str], timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )


def main() -> int:
    args = parse_args()
    pod = load_pod(args.pod_id)
    password = find_jupyter_password(pod, args.jupyter_password)
    public_ip, public_port = find_ssh_endpoint(pod)
    public_key = load_public_key(args.public_key_file)
    base_url = build_base_url(args.pod_id)

    remote_command = f"""set -euo pipefail
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
ssh-keygen -A
cat > /root/.ssh/authorized_keys <<'KEY'
{public_key}
KEY
chmod 600 /root/.ssh/authorized_keys
pkill sshd || true
/usr/sbin/sshd
ss -tlnp | grep ':22'
"""

    jupyter_exec = [
        sys.executable,
        str(Path(__file__).with_name("jupyter_terminal_exec.py")),
        "--base-url",
        base_url,
        "--password",
        password,
        "--command",
        remote_command,
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    remote = run_local(jupyter_exec, args.timeout_seconds + 30)
    if remote.returncode != 0:
        sys.stdout.write(remote.stdout)
        return remote.returncode

    ssh_cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        args.private_key_file,
        "-p",
        str(public_port),
        f"root@{public_ip}",
        "hostname",
    ]
    ssh = run_local(ssh_cmd, min(args.timeout_seconds, 30))
    if ssh.returncode != 0:
        sys.stdout.write(ssh.stdout)
        return ssh.returncode

    print(
        json.dumps(
            {
                "pod_id": pod["id"],
                "pod_name": pod.get("name", ""),
                "base_url": base_url,
                "ssh_host": public_ip,
                "ssh_port": public_port,
                "hostname": ssh.stdout.strip(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
