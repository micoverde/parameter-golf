#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import uuid
from urllib.parse import urlparse, urlunparse

import requests
import websocket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a shell command through a Jupyter terminal websocket.")
    parser.add_argument("--base-url", required=True, help="Base Jupyter URL, for example https://<pod>-8888.proxy.runpod.net")
    parser.add_argument("--password", required=True, help="Jupyter password for the template")
    parser.add_argument("--command", required=True, help="Shell command to execute remotely")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0, help="Read timeout for terminal output")
    return parser.parse_args()


def websocket_url(base_url: str, terminal_name: str) -> str:
    parsed = urlparse(base_url.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, f"/terminals/websocket/{terminal_name}", "", "", ""))


def login(session: requests.Session, base_url: str, password: str) -> None:
    response = session.get(f"{base_url.rstrip('/')}/", timeout=30)
    response.raise_for_status()
    xsrf = session.cookies.get("_xsrf")
    if not xsrf:
        raise SystemExit("missing _xsrf cookie from Jupyter login page")

    login_response = session.post(
        f"{base_url.rstrip('/')}/login?next=%2Flab",
        data={"_xsrf": xsrf, "password": password},
        headers={"X-XSRFToken": xsrf},
        timeout=30,
        allow_redirects=True,
    )
    login_response.raise_for_status()

    me = session.get(f"{base_url.rstrip('/')}/api/me", timeout=30)
    me.raise_for_status()


def create_terminal(session: requests.Session, base_url: str) -> str:
    xsrf = session.cookies.get("_xsrf", "")
    response = session.post(
        f"{base_url.rstrip('/')}/api/terminals",
        headers={"X-XSRFToken": xsrf},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["name"]


def delete_terminal(session: requests.Session, base_url: str, terminal_name: str) -> None:
    xsrf = session.cookies.get("_xsrf", "")
    session.delete(
        f"{base_url.rstrip('/')}/api/terminals/{terminal_name}",
        headers={"X-XSRFToken": xsrf},
        timeout=30,
    )


def main() -> int:
    args = parse_args()
    session = requests.Session()
    login(session, args.base_url, args.password)
    terminal_name = create_terminal(session, args.base_url)
    marker = f"__CODEX_DONE_{uuid.uuid4().hex}__"

    command = f"{args.command}; printf '\\n{marker} %s\\n' $?\n"
    quoted = shlex.quote(command)
    cookie_header = "; ".join(f"{key}={value}" for key, value in session.cookies.get_dict().items())
    xsrf = session.cookies.get("_xsrf", "")
    ws = websocket.create_connection(
        websocket_url(args.base_url, terminal_name),
        header=[f"Cookie: {cookie_header}", f"X-XSRFToken: {xsrf}"],
        timeout=args.timeout_seconds,
    )

    exit_code = None
    try:
        ws.send(json.dumps(["stdin", f"bash -lc {quoted}\n"]))
        while True:
            raw_message = ws.recv()
            channel, data = json.loads(raw_message)
            if channel == "stdout":
                sys.stdout.write(data)
                sys.stdout.flush()
                match = re.search(rf"{re.escape(marker)}\s+(\d+)", data)
                if match:
                    exit_code = int(match.group(1))
                    break
            elif channel == "stderr":
                sys.stderr.write(data)
                sys.stderr.flush()
            elif channel == "disconnect":
                break
    finally:
        try:
            ws.close()
        finally:
            delete_terminal(session, args.base_url, terminal_name)

    if exit_code is None:
        raise SystemExit("remote command finished without exit marker")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
