"""Network-disabled worker for the feasible-progress clock study."""

from __future__ import annotations

import argparse
import json
import os
import socket


class _NetworkDisabledSocket(socket.socket):
    def connect(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("network access is disabled in the local laboratory")

    def connect_ex(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("network access is disabled in the local laboratory")

    def sendto(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("network access is disabled in the local laboratory")


def _deny_network(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("network access is disabled in the local laboratory")


def _disable_network() -> None:
    socket.socket = _NetworkDisabledSocket
    socket.create_connection = _deny_network
    socket.getaddrinfo = _deny_network
    socket.gethostbyaddr = _deny_network
    socket.gethostbyname = _deny_network
    socket.gethostbyname_ex = _deny_network
    os.environ["LEARN2DESIGN_LOCAL_LAB_NETWORK"] = "disabled"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "feasible-progress-clock-trace",
            "feasible-progress-clock-v1",
        ],
        required=True,
    )
    args = parser.parse_args()

    _disable_network()
    from experiments.local_lab.feasible_progress_clock import (
        isolated_worker_trace,
        run_study,
    )

    result = (
        isolated_worker_trace()
        if args.mode == "feasible-progress-clock-trace"
        else run_study(include_process_isolation=True)
    )
    print(
        json.dumps(
            result,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
