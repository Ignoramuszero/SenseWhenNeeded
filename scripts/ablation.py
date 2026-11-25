"""Ablations: sampling regime, detector variant, vote-window trade-off."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swn.config import DEFAULT_POLICY, FS, IDLE_PERIOD_S, RESULT_DIR  # noqa: E402
from swn.data.sessions import load_sessions, split_subjects  # noqa: E402
from swn.evaluation.metrics import (  # noqa: E402
    event_metrics,
    merge_event_metrics,
    tracking_metrics,
    transition_latency,
)
from swn.features import burst_event_features, burst_stats  # noqa: F401
from swn.models.activity_classifier import ActivityClassifier  # noqa: E402
from swn.models.event_detector import EventDetector, build_event_dataset  # noqa: E402
from swn.runtime import run_session  # noqa: E402


def summarise(sessions, clf, bclf, det, policy):
    acc, duty, lat, ms, en = [], [], [], [], []
    for s in sessions:
        r = run_session(s, "event_driven", clf, det, policy=policy, run_llm=False, burst_classifier=bclf)
        ms.append(event_metrics([t for t in r.triggers if not t.get("bootstrap")], r.true_transitions))
        m = tracking_metrics(r.true, r.pred)
        acc.append((m["accuracy"], m["n"]))
        duty.append(r.counters["duty_cycle"])
        en.append(r.counters["energy_mj"] / r.duration * 3600 / 1000)
        lat += [v["latency_s"] for v in transition_latency(r.true, r.pred, r.true_transitions)
                if v["latency_s"] is not None]
    agg = merge_event_metrics(ms)
    return {
        "accuracy": float(sum(a * n for a, n in acc) / sum(n for _, n in acc)),
        "event_f1": agg["f1"],
        "event_recall": agg["recall"],
        "event_precision": agg["precision"],
        "duty_cycle": float(np.mean(duty)),
        "energy_j_per_hour": float(np.mean(en)),
        "median_switch_latency_s": float(np.median(lat)) if lat else None,
    }


def main() -> None:
    sessions = load_sessions()
    split = split_subjects(sorted({s.subject for s in sessions}))
    va = [s for s in sessions if s.subject in split["val"]]
    te = [s for s in sessions if s.subject in split["test"]]
    clf = ActivityClassifier.load(ROOT / "artifacts/models/activity_classifier.pkl")
    bclf = ActivityClassifier.load(ROOT / "artifacts/models/burst_classifier.pkl")
    det = EventDetector.load(ROOT / "artifacts/models/event_detector.pkl")

    out: dict = {}

    # --- B: cheap-classifier vote window (latency vs accuracy) ---------------
    out["vote_window"] = {}
    for vote in (1, 2, 3):
        pol = replace(DEFAULT_POLICY, burst_vote_window=vote)
        out["vote_window"][vote] = summarise(te, clf, bclf, det, pol)
        print(f"[ablation] vote={vote}", out["vote_window"][vote], flush=True)

    # --- C: detector variant -------------------------------------------------
    out["detector_mode"] = {}
    for mode in ("rule", "rf", "hybrid"):
        det.mode = mode
        out["detector_mode"][mode] = summarise(te, clf, bclf, det, DEFAULT_POLICY)
        print(f"[ablation] mode={mode}", out["detector_mode"][mode], flush=True)
    det.mode = "hybrid"

    # --- A: 1 Hz sampling vs 1 s bursts (window-level ranking quality) -------
    def dataset_1hz(sessions_, history=4):
        X, y = [], []
        for s in sessions_:
            tr = np.array([t for t, _, _ in s.transitions], dtype=np.float64)
            step = int(round(1.0 * FS))
            stats = []
            for b in range(1, len(s.acc) // step):
                end = b * step
                stats.append(burst_stats(s.acc[end - 1 : end], FS))
                hist = np.stack(stats[-history:])
                feat = burst_event_features(hist)
                pos = bool(len(tr) and np.min(np.abs(tr - end / FS)) <= 2.5)
                X.append(feat)
                y.append(1 if pos else 0)
        return np.asarray(X, np.float32), np.asarray(y)

    from swn.features import burst_event_features  # noqa: F401
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score

    ap = {}
    X1_tr, y1_tr = dataset_1hz([s for s in sessions if s.subject in split["train"]])
    X1_te, y1_te = dataset_1hz(te)
    Xtr_b, ytr_b = build_event_dataset(
        [s for s in sessions if s.subject in split["train"]]
    )
    Xte_b, yte_b = build_event_dataset(te)
    rf1 = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                 class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    rf1.fit(*_bal(X1_tr, y1_tr))
    ap["sampling_1hz_test_AP"] = round(float(average_precision_score(y1_te, rf1.predict_proba(X1_te)[:, 1])), 3)

    rfb = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                 class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    rfb.fit(Xtr_b, ytr_b)
    ap["sampling_burst_test_AP"] = round(float(average_precision_score(yte_b, rfb.predict_proba(Xte_b)[:, 1])), 3)
    ap["burst_base_rate"] = round(float(yte_b.mean()), 4)
    ap["hz1_base_rate"] = round(float(y1_te.mean()), 4)
    out["sampling_regime"] = ap
    print("[ablation] sampling", ap, flush=True)

    (RESULT_DIR / "ablation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[ablation] wrote", RESULT_DIR / "ablation.json")


def _bal(X, y, ratio=8.0, seed=0):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep = min(len(neg), int(len(pos) * ratio))
    rng = np.random.default_rng(seed)
    if keep < len(neg):
        neg = rng.choice(neg, keep, replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    return X[idx], y[idx]


if __name__ == "__main__":
    main()
