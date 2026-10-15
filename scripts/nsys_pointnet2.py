"""Optional Nsight Systems availability check / bounded PointNet++ launcher."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def nsight_status():
    executable = shutil.which("nsys")
    if not executable:
        return {
            "available": False,
            "reason": "nsys not installed; use torch profiler instead",
        }
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return {
            "available": result.returncode == 0,
            "path": executable,
            "version": (result.stdout + result.stderr).strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="Otherwise only check availability"
    )
    parser.add_argument(
        "--report", default="pointnet2_nsys", help="Nsight output prefix"
    )
    parser.add_argument("workload_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    status = nsight_status()
    print(json.dumps(status, indent=2), flush=True)
    if not status["available"] or not args.run:
        return
    workload = args.workload_args
    if workload[:1] == ["--"]:
        workload = workload[1:]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).with_name("profile_pointnet2.py")
    command = [
        status["path"],
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "-o",
        args.report,
        sys.executable,
        str(helper),
        *workload,
        "--warmup-batches",
        "1",
        "--profile-batches",
        "2",
        "--nvtx-only",
    ]
    result = subprocess.run(command, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
