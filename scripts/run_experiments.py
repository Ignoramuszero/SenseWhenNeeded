"""Run the full comparison experiment on held-out subjects."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swn.config import RESULT_DIR, config_snapshot  # noqa: E402
from swn.data.sessions import load_sessions, split_subjects  # noqa: E402
from swn.evaluation.runner import compare_strategies  # noqa: E402
from swn.models.activity_classifier import ActivityClassifier  # noqa: E402
from swn.models.event_detector import EventDetector  # noqa: E402


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "offline"
    sessions = load_sessions()
    split = split_subjects(sorted({s.subject for s in sessions}))
    test = [s for s in sessions if s.subject in split["test"]]
    clf = ActivityClassifier.load(ROOT / "artifacts/models/activity_classifier.pkl")
    det = EventDetector.load(ROOT / "artifacts/models/event_detector.pkl")
    burst_clf = ActivityClassifier.load(ROOT / "artifacts/models/burst_classifier.pkl")
    print(f"[eval] {len(test)} held-out sessions, LLM backend={backend}, thr={det.threshold}")

    comparison = compare_strategies(test, clf, det, backend=backend, burst_classifier=burst_clf)
    comparison["config"] = config_snapshot()
    comparison["split"] = {k: [int(v) for v in vs] for k, vs in split.items()}
    (RESULT_DIR / "experiments.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for name, s in comparison["strategies"].items():
        c = s["counters"]
        print(
            f"{name:13s} acc={s['accuracy']:.3f} macroF1={s['macro_f1']:.3f} "
            f"duty={c['duty_cycle']:.3f} energy={c['energy_j_per_hour']:.1f}J/h "
            f"gps={c['gps_on_s']:.0f}s/h llm={c['llm_calls_per_hour']:.1f}/h "
            f"lat_med={s['switch_latency_s']['median']}"
        )
        if "events" in s:
            e = s["events"]
            print(
                f"              events P={e['precision']:.2f} R={e['recall']:.2f} F1={e['f1']:.2f} "
                f"median_latency={e['median_latency_s']}s"
            )


if __name__ == "__main__":
    main()
