"""Prepare WISDM smartwatch runs and replay sessions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swn.config import PROCESSED_DIR, RESULT_DIR  # noqa: E402
from swn.data.sessions import build_sessions, dataset_stats, split_subjects  # noqa: E402
from swn.data.wisdm import extract_runs, load_runs, save_runs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-parse raw txt files")
    args = ap.parse_args()

    npz = PROCESSED_DIR / "wisdm_watch_runs.npz"
    if npz.exists() and not args.force:
        print(f"[data] reusing {npz}")
        runs = load_runs(npz)
    else:
        print("[data] parsing WISDM raw watch files ...")
        runs = extract_runs()
        npz = save_runs(runs)
        print(f"[data] wrote {npz} ({npz.stat().st_size / 1e6:.1f} MB)")

    sessions = build_sessions(runs)
    subjects = sorted({r["subject"] for r in runs})
    split = split_subjects(subjects)
    stats = dataset_stats(sessions)
    stats["subjects"] = split
    stats["n_runs"] = len(runs)

    out = RESULT_DIR / "dataset_stats.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in stats.items() if k != "subjects"}, indent=2, ensure_ascii=False))
    print("split:", {k: len(v) for k, v in split.items()})

    # cache sessions as npz for fast reload
    sess_path = PROCESSED_DIR / "sessions.npz"
    import numpy as np

    np.savez_compressed(
        sess_path,
        ids=np.array([s.session_id for s in sessions]),
        subjects=np.array([s.subject for s in sessions]),
        acc=np.concatenate([s.acc for s in sessions]),
        gyro=np.concatenate([s.gyro for s in sessions]),
        labels=np.concatenate([s.labels for s in sessions]),
        lengths=np.array([len(s.labels) for s in sessions]),
        tr_idx=np.concatenate([np.array([i] * len(sessions[i].transitions), dtype=np.int32) for i in range(len(sessions))] or [np.array([], dtype=np.int32)]),
        tr_t=np.concatenate([np.array([t for t, _, _ in s.transitions], dtype=np.float32) for s in sessions] or [np.array([], dtype=np.float32)]),
    )
    print(f"[data] sessions cached -> {sess_path}")


if __name__ == "__main__":
    main()
