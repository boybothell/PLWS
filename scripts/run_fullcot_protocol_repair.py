#!/usr/bin/env python3
"""Run the five canonical-protocol repair lanes and merge successful rows."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPAIR_ROOT = ROOT / "results" / "runs" / "plws_protocol_repair_v1"
PYTHON = Path(
    os.environ.get(
        "PLWS_PYTHON",
        "/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm/.venv/bin/python",
    )
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def status(state: str, **fields: object) -> None:
    path = REPAIR_ROOT / "queue_status.json"
    path.write_text(
        json.dumps(
            {"state": state, "updated_at": timestamp(), **fields},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    plan = json.loads((REPAIR_ROOT / "plan.json").read_text(encoding="utf-8"))
    logs = REPAIR_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    cuda_libs = ":".join(
        sorted(
            str(path)
            for path in (
                Path("/mnt/d/lsj/visual-latent-tts/repos/okay-budget-vllm")
                / ".venv"
                / "lib"
            ).glob("**/nvidia/*/lib")
            if path.is_dir()
        )
    )
    for lane in plan:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(lane["gpu"])
        env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
        env["LD_LIBRARY_PATH"] = (
            f"{cuda_libs}:{env.get('LD_LIBRARY_PATH', '')}"
        )
        command = [
            str(PYTHON),
            str(ROOT / "scripts" / "score_leftover_suppress.py"),
            "--mode",
            "suppress",
            "--model-tag",
            lane["model"],
            "--run-kind",
            "repair",
            "--jobs",
            lane["jobs_path"],
            "--out",
            lane["output"],
            "--isolated-output",
            "--lexicon",
            "core",
            "--k",
            "4",
            "--max-context",
            "37888",
            "--think-tokens",
            "0",
            "--answer-tokens",
            "2048",
            "--sampling-seed",
            "20260904",
            "--batch-size",
            "64",
        ]
        log_path = logs / f"lane_{lane['lane']}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((lane, process, log_handle))
    status("running", lanes=plan)

    failures = []
    for lane, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append(
                {
                    "lane": lane["lane"],
                    "model": lane["model"],
                    "return_code": return_code,
                }
            )
    if failures:
        status("failed", failures=failures)
        raise SystemExit(f"repair lanes failed: {failures}")

    subprocess.run(
        [str(PYTHON), str(ROOT / "scripts" / "finalize_fullcot_protocol_v1.py")],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}"},
        check=True,
    )
    for report_script in (
        "report_fullcot_puma_plws.py",
        "report_fullcot_puma_plws_deer.py",
    ):
        subprocess.run(
            [str(PYTHON), str(ROOT / "scripts" / report_script)],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}"},
            check=True,
        )
    status("succeeded", lanes=plan)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status("failed", error=repr(error))
        raise
