"""Network-disabled subprocess entry point for bounded local-lab studies."""

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
        choices=["anchor-lane-stability-v1", "exact-trace", "policy-probe"],
        required=True,
    )
    args = parser.parse_args()

    _disable_network()
    if args.mode == "policy-probe":
        try:
            with socket.socket() as probe:
                probe.connect(("127.0.0.1", 1))
        except RuntimeError:
            network_blocked = True
        else:
            network_blocked = False
        sensitive_fragments = (
            "AWS_",
            "AZURE_",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "KEY",
            "OPENAI",
            "PASSWORD",
            "RUNPOD",
            "SECRET",
            "TOKEN",
        )
        result = {
            "network_blocked": network_blocked,
            "sensitive_environment_names": sorted(
                name
                for name in os.environ
                if any(fragment in name.upper() for fragment in sensitive_fragments)
            ),
        }
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return

    from experiments.local_lab.anchor_lane_stability import (
        isolated_worker_trace,
        run_study,
    )

    if args.mode == "exact-trace":
        result = isolated_worker_trace()
    else:
        result = run_study(include_process_isolation=True)
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
