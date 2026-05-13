from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import f6_flowgen_train_official_reopen_experimental as reopen


TARGET_RUN_ID = "flowgen_tpv1_c2_reopenexp_o07_now1x_prior_ramp_seed2468_v1"
SUMMARY_ROOT = reopen.SUMMARY_ROOT / "rescue_o07"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Single-run rescue wrapper for the official FlowGen reopen policy "
            "O07_now1x_prior_ramp. It reuses the existing reopen runner, tries "
            "one or more devices in sequence, and leaves the original policy "
            "untouched."
        )
    )
    ap.add_argument(
        "--devices",
        default="auto,cpu",
        help=(
            "Comma-separated device fallback chain. Default: auto,cpu. "
            "The script stops on the first successful attempt."
        ),
    )
    ap.add_argument("--summary-only", action="store_true", help="Print the rescue plan without running it.")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def _campaign_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _resolve_attempt_devices(raw_devices: str) -> list[str]:
    devices = [item.strip() for item in raw_devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("No rescue devices were provided.")
    return devices


def _run_status() -> str:
    return reopen._run_materialization_status(TARGET_RUN_ID)


def _build_child_cmd(*, device: str, quiet: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(reopen.__file__).resolve()),
        "--run-one",
        TARGET_RUN_ID,
        "--device",
        device,
    ]
    if quiet:
        cmd.append("--quiet")
    return cmd


def _attempt_payload(*, device: str, returncode: int | None = None, status: str, note: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "device": device,
        "status": status,
    }
    if returncode is not None:
        payload["returncode"] = int(returncode)
    if note:
        payload["note"] = note
    return payload


def main() -> int:
    args = _parse_args()
    devices = _resolve_attempt_devices(args.devices)
    initial_status = _run_status()

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    campaign_id = f"flowgen_official_reopen_rescue_o07_{_campaign_timestamp()}"
    summary_path = SUMMARY_ROOT / f"{campaign_id}.json"

    print("\n" + "=" * 100)
    print("Official FlowGen reopen rescue | O07_now1x_prior_ramp")
    print(f"Target run: {TARGET_RUN_ID}")
    print(f"Current status: {initial_status}")
    print(f"Device attempts: {', '.join(devices)}")
    print(f"Summary JSON: {summary_path}")

    if args.summary_only:
        _write_json(
            summary_path,
            {
                "campaign_id": campaign_id,
                "run_id": TARGET_RUN_ID,
                "initial_status": initial_status,
                "devices": devices,
                "attempts": [],
                "final_status": initial_status,
                "summary_only": True,
            },
        )
        print("\nSummary-only mode enabled. No rescue attempt started.")
        return 0

    if initial_status == "complete":
        _write_json(
            summary_path,
            {
                "campaign_id": campaign_id,
                "run_id": TARGET_RUN_ID,
                "initial_status": initial_status,
                "devices": devices,
                "attempts": [],
                "final_status": initial_status,
                "note": "Run already complete; rescue skipped.",
            },
        )
        print("\nRun already complete. Nothing to rescue.")
        return 0

    attempts: list[dict[str, Any]] = []
    success = False
    final_status = initial_status

    for idx, device in enumerate(devices, start=1):
        print("\n" + "-" * 100)
        print(f"[{idx}/{len(devices)}] Rescue attempt on device={device}")
        child_cmd = _build_child_cmd(device=device, quiet=args.quiet)
        child = subprocess.run(child_cmd, cwd=str(ROOT))
        final_status = _run_status()

        if child.returncode == 0 and final_status == "complete":
            attempts.append(
                _attempt_payload(
                    device=device,
                    returncode=child.returncode,
                    status="completed",
                    note="Run completed successfully on this device.",
                )
            )
            success = True
            print(f"Completed O07 rescue on device={device}.")
            break

        attempts.append(
            _attempt_payload(
                device=device,
                returncode=child.returncode,
                status="failed",
                note=(
                    f"Child attempt finished with returncode={child.returncode}; "
                    f"post-attempt status={final_status}."
                ),
            )
        )
        print(
            f"Attempt failed on device={device} "
            f"(returncode={child.returncode}, post-status={final_status})."
        )

    final_status = _run_status()
    payload = {
        "campaign_id": campaign_id,
        "run_id": TARGET_RUN_ID,
        "initial_status": initial_status,
        "devices": devices,
        "attempts": attempts,
        "final_status": final_status,
        "success": success,
    }
    _write_json(summary_path, payload)

    print("\n" + "=" * 100)
    print("Official FlowGen reopen rescue finished")
    print(f"Success: {success}")
    print(f"Final status: {final_status}")
    print(f"Summary JSON: {summary_path}")
    if final_status == "complete":
        print(f"Run dir: {reopen._run_dir_for(TARGET_RUN_ID)}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
