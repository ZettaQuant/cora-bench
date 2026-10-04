"""Resume reader generation, native scoring, matched latency, and reporting."""

import json
import subprocess
import threading
import time
import traceback

from .api import OUT, PROTOCOL, save
from .generate_answers import ReaderRun, sync_cloud
from .score_answers import run as judge


def export(folder):
    if not PROTOCOL["gcs_prefix"]:
        return
    subprocess.run(
        [
            "gcloud",
            "storage",
            "rsync",
            str(OUT / folder),
            PROTOCOL["gcs_prefix"] + "/" + folder,
            "--recursive",
            "--exclude",
            ".*sqlite.*",
            "--quiet",
        ],
        check=True,
    )


def main():
    run = ReaderRun("main")
    run.cloud = bool(PROTOCOL["gcs_prefix"])
    try:
        # On resume, finish scoring saved answers before accepting another reader wave.
        judge("main")
        export("judges")
        while True:
            if PROTOCOL["gcs_prefix"]:
                sync_cloud()
            counts = run.batch()
            judge("main")
            export("judges")
            complete = all(
                (OUT / "selection" / k / "complete.json").exists() for k in ["cpu", "gpu"]
            )
            remaining = run.tasks()
            if complete and not remaining:
                break
            if counts["pending_or_error"] and counts["complete"] == 0:
                raise RuntimeError(
                    "Unresolved reader attempts need inspection; not silently retried"
                )
            time.sleep(45)
        save(run.root / "complete.json", {"complete": True, "finished_unix": time.time()})
        export("reader/main")
        latency = ReaderRun("latency")
        latency.cloud = bool(PROTOCOL["gcs_prefix"])
        latency.batch()
        export("reader/latency")
        from .summarize_results import run as report

        report()
        export("report")
        if not json.loads((OUT / "report/summary.json").read_text())["complete"]:
            raise RuntimeError(
                "Incomplete scoring coverage; see report and judges/main/unresolved. Not marking the experiment complete."
            )
        save(
            OUT / "complete.json",
            {"complete": True, "finished_unix": time.time(), "ledger": run.ledger.totals()},
        )
        if PROTOCOL["gcs_prefix"]:
            subprocess.run(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    str(OUT / "complete.json"),
                    PROTOCOL["gcs_prefix"] + "/complete.json",
                    "--quiet",
                ],
                check=True,
            )
    except BaseException as e:
        save(
            OUT / "supervisor_error.json",
            {"type": type(e).__name__, "error": str(e)[:1000], "time": time.time()},
        )
        if PROTOCOL["gcs_prefix"]:
            subprocess.run(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    str(OUT / "supervisor_error.json"),
                    PROTOCOL["gcs_prefix"] + "/supervisor_error.json",
                    "--quiet",
                ]
            )
        raise


if __name__ == "__main__":
    main()
