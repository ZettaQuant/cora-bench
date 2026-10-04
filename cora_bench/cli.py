"""Command-line entry points for data preparation and benchmark evaluation."""

import argparse
import runpy
import sys

COMMANDS = {
    "prepare-babilong": ("scripts.prepare_babilong", "Generate the frozen BABILong cohort"),
    "prepare-loong": ("scripts.prepare_loong", "Assemble LOONG financial documents"),
    "prepare-nolima": ("scripts.prepare_nolima", "Generate the custom NoLiMa diagnostic"),
    "prepare-longmemeval": ("scripts.prepare_longmemeval", "Assemble LongMemEval histories"),
    "verify-data": ("scripts.verify_data", "Verify input, context, and evidence hashes"),
    "smoke": ("scripts.smoke", "Run an offline BM25 example"),
    "select": ("experiment.select_context", "Select contexts on CPU or GPU"),
    "answer": ("experiment.generate_answers", "Generate reader answers (paid API)"),
    "score": ("experiment.score_answers", "Apply native scorers and judges (paid API)"),
    "audit": ("experiment.audit_results", "Audit saved requests, scores, and contexts"),
    "report": ("experiment.summarize_results", "Build confidence intervals, tables, and plots"),
    "run": ("experiment.run_pipeline", "Resume answers, judgments, latency, and reporting"),
    "evidence": ("analysis.dataset_report", "Score saved evidence-only selections"),
    "select-evidence": ("scripts.select_evidence", "Save selections for the evidence experiment"),
}
NO_ARGUMENTS = {"smoke", "audit", "report", "run"}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    description = "CoRA-Bench: evidence preservation and answer quality under context budgets."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("command", choices=COMMANDS, help="benchmark stage")
    parser.epilog = "\n".join(f"{name:22} {detail}" for name, (_, detail) in COMMANDS.items())
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        return
    command = parser.parse_args(args[:1]).command
    module, description = COMMANDS[command]
    if command in NO_ARGUMENTS:
        argparse.ArgumentParser(prog=f"cora-bench {command}", description=description).parse_args(
            args[1:]
        )
    sys.argv = [f"cora-bench {command}", *args[1:]]
    runpy.run_module("cora_bench." + module, run_name="__main__")
