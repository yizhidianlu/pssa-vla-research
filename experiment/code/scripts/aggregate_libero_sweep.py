"""Aggregate per-task JSONs from a libero sweep into one summary.

Usage:
    python aggregate_libero_sweep.py <sweep_dir>

Writes <sweep_dir>/summary.json and prints a table to stdout.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    sweep = Path(sys.argv[1])
    task_files = sorted(sweep.glob("task_*.json"))
    if not task_files:
        print(f"no task_*.json under {sweep}"); sys.exit(1)

    rows = []
    all_steps_to_success: list[int] = []
    total_succ = 0
    total_rollouts = 0
    for f in task_files:
        d = json.loads(f.read_text())
        sr = d["success_rate"]
        n = d["n_rollouts"]
        succ = d["n_success"]
        steps_succ = [r["steps"] for r in d["rollouts"] if r["success"]]
        steps_fail = [r["steps"] for r in d["rollouts"] if not r["success"]]
        rows.append({
            "task_id": d["task_id"],
            "task_name": d["task_name"],
            "task_language": d["task_language"],
            "n_rollouts": n,
            "n_success": succ,
            "success_rate": sr,
            "mean_steps_succ": statistics.mean(steps_succ) if steps_succ else None,
            "mean_steps_fail": statistics.mean(steps_fail) if steps_fail else None,
            "step_ms_avg": statistics.mean(r["step_ms_avg"] for r in d["rollouts"]),
        })
        total_succ += succ
        total_rollouts += n
        all_steps_to_success.extend(steps_succ)

    overall_sr = total_succ / total_rollouts if total_rollouts else 0
    summary = {
        "suite": "libero_spatial",
        "model_id": json.loads(task_files[0].read_text())["model_id"],
        "n_tasks": len(rows),
        "rollouts_per_task": rows[0]["n_rollouts"] if rows else 0,
        "overall_success_rate": overall_sr,
        "n_success_total": total_succ,
        "n_rollouts_total": total_rollouts,
        "mean_steps_to_success_overall": (
            statistics.mean(all_steps_to_success) if all_steps_to_success else None
        ),
        "tasks": rows,
    }
    out_path = sweep / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*72}\nLIBERO-SPATIAL sweep summary — model: {summary['model_id']}\n{'='*72}")
    print(f"{'tid':>3} {'SR':>7} {'n_succ':>7} {'mean_steps_OK':>14} {'mean_steps_F':>13} {'step_ms':>8}")
    for r in rows:
        ms_ok = f"{r['mean_steps_succ']:.0f}" if r['mean_steps_succ'] is not None else "—"
        ms_f = f"{r['mean_steps_fail']:.0f}" if r['mean_steps_fail'] is not None else "—"
        print(f"{r['task_id']:>3} {r['success_rate']*100:>6.1f}% {r['n_success']:>3}/{r['n_rollouts']:<3} {ms_ok:>14} {ms_f:>13} {r['step_ms_avg']:>7.1f}")
    print("─" * 72)
    print(f"{'ALL':>3} {overall_sr*100:>6.1f}% {total_succ:>3}/{total_rollouts:<3} {statistics.mean(all_steps_to_success):>14.0f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
