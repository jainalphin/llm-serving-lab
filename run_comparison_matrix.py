"""Run reproducible comparison_benchmark.py scenarios in isolated processes."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROFILES = {
    "quick": {
        "scenarios": ((128, 32), (900, 64)),
        "rates": (1, 4, 16, "inf"),
        "num_requests": 30,
    },
    "extreme": {
        "scenarios": ((16, 64), (128, 64), (512, 64), (900, 64)),
        "rates": (1, 2, 4, 8, 16, 32, "inf"),
        "num_requests": 100,
    },
}

ENGINE_CONFIGS = (
    ("hf", None),
    ("pagedserve", "orca"),
    ("pagedserve", "sarathi"),
    ("vllm", None),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--dtype", action="append", choices=("float32", "float16"))
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/comparison"))
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--e2e-slo-ms", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    profile = PROFILES[args.profile]
    dtypes = args.dtype or ["float32"]
    num_requests = args.num_requests or profile["num_requests"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "dtypes": dtypes,
        "num_requests": num_requests,
        "scenarios": [],
    }

    for dtype in dtypes:
        for input_length, output_length in profile["scenarios"]:
            for engine, strategy in ENGINE_CONFIGS:
                label = engine if strategy is None else f"{engine}_{strategy}"
                output_path = args.output_dir / (
                    f"{label}_{dtype}_in{input_length}_out{output_length}.json"
                )
                log_path = output_path.with_suffix(".log")
                scenario = {
                    "engine": engine,
                    "strategy": strategy,
                    "dtype": dtype,
                    "input_length": input_length,
                    "output_length": output_length,
                    "result": str(output_path),
                    "log": str(log_path),
                }
                manifest["scenarios"].append(scenario)
                if output_path.exists() and not args.overwrite:
                    scenario["status"] = "skipped_existing"
                    print(f"Skipping existing {output_path}")
                    continue

                command = [
                    sys.executable,
                    "comparison_benchmark.py",
                    "--engine",
                    engine,
                    "--dtype",
                    dtype,
                    "--input-length",
                    str(input_length),
                    "--output-length",
                    str(output_length),
                    "--num-requests",
                    str(num_requests),
                    "--json-output",
                    str(output_path),
                ]
                for rate in profile["rates"]:
                    command.extend(("--request-rate", str(rate)))
                if strategy is not None:
                    command.extend(("--pagedserve-strategy", strategy))
                for option, value in (
                    ("--ttft-slo-ms", args.ttft_slo_ms),
                    ("--tpot-slo-ms", args.tpot_slo_ms),
                    ("--e2e-slo-ms", args.e2e_slo_ms),
                ):
                    if value is not None:
                        command.extend((option, str(value)))

                print("Running:", " ".join(command), flush=True)
                with log_path.open("w") as log_file:
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log_file.write(line)
                    return_code = process.wait()
                scenario["return_code"] = return_code
                scenario["status"] = "complete" if return_code == 0 else "failed"

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
