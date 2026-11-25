#!/usr/bin/env python3
"""Convert the trained sklearn models to Apple Core ML (.mlmodel).

Run this ON THE MAC (or any machine with coremltools) before integrating
the models into the iOS app:

    pip install coremltools joblib numpy scikit-learn
    python swn/coreml_export.py

Outputs into artifacts/coreml/:
    ActivityClassifier.mlmodel   3 s window, 5 activities
    BurstClassifier.mlmodel      1 s window, 5 activities (low-power monitoring)
    EventDetector.mlmodel        burst history -> change / no_change (binary)

The Swift side must compute the *same* features in the *same* order as
swn/features.py — see MACOS_INTEGRATION.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swn.config import MODEL_DIR  # noqa: E402
from swn.models.activity_classifier import ActivityClassifier  # noqa: E402
from swn.models.event_detector import EventDetector  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "coreml"


def export_sklearn(model, feature_names: list[str], out_path: Path, description: str):
    import coremltools as ct

    input_features = [(name, float) for name in feature_names]
    coreml = ct.converters.sklearn.convert(
        model,
        input_features,
        output_feature_name="output",
    )
    coreml.author = "SenseWhenNeeded"
    coreml.short_description = description
    coreml.save(str(out_path))
    print(f"[export] {out_path.name}  ({len(feature_names)} features)")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # 1) main activity classifier (3 s window @ 20 Hz, 5 activities)
    clf = ActivityClassifier.load(MODEL_DIR / "activity_classifier.pkl")
    export_sklearn(
        clf.model,
        clf.feature_names,
        OUT_DIR / "ActivityClassifier.mlmodel",
        "5-activity recognition from a 3 s IMU window (WISDM-trained).",
    )

    # 2) cheap burst classifier (1 s window @ 20 Hz, 5 activities)
    burst = ActivityClassifier.load(MODEL_DIR / "burst_classifier.pkl")
    export_sklearn(
        burst.model,
        burst.feature_names,
        OUT_DIR / "BurstClassifier.mlmodel",
        "5-activity recognition from a 1 s IMU burst (low-power monitoring).",
    )

    # 3) event detector (burst-history features -> binary change/no_change)
    det = EventDetector.load(MODEL_DIR / "event_detector.pkl")
    export_sklearn(
        det.model,
        det.feature_names,
        OUT_DIR / "EventDetector.mlmodel",
        "Binary change-point detector over a history of 1 s burst statistics.",
    )

    print(f"[done] models written to {OUT_DIR}")


if __name__ == "__main__":
    main()
