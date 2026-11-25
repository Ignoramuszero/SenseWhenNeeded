# Sense When Needed: An Event-Driven Adaptive Multimodal Activity Understanding System

**Course Project Report** — Data: WISDM Smartwatch (UCI ML Repository, id 507) + self-collected smartphone recordings (Sensor Logger)

---

## 1. Problem and Motivation

Conventional ubiquitous sensing systems keep every sensor sampling and every model running all the time, regardless of whether the information is actually needed at this moment. An IMU streaming at 20 Hz, a GPS radio polling every second, and a large language model summarizing every window waste most of their energy and tokens on **redundant minutes** — the user has been sitting still for half an hour, and nothing has changed.

This project answers the ML question posed in the original proposal:

> **How can a system use low-cost sensors to decide *when* more information is needed, and dynamically activate expensive sensors (GPS) and LLM calls accordingly, so that activity recognition capability is preserved while resource consumption drops?**

The problem is interesting for three reasons:

1. **It is a decision problem, not a classification problem.** The core challenge is not recognizing activities (that is largely solved) but *valuing information*: when does a new sample change a decision worth paying for?
2. **Real sensor streams make the trade-offs measurable.** We can quantify exactly how much recognition accuracy each percentage point of duty cycle buys, and whether an LLM call at every window (3,584/h) or only at confirmed transitions (10.8/h) changes anything a user would notice.
3. **Deployment is the interesting part.** Models trained on lab datasets (WISDM wrist-watch) meet a different distribution on a real phone in a pocket. Our project therefore closes the loop from simulation to a working iOS app to personalized on-device models — and the failure modes we found (e.g., walking vs. stairs) are themselves informative.

**One-sentence result:** the event-driven policy reproduces always-on recognition accuracy (0.798 vs. 0.801) with **24.2% of the IMU duty cycle** (3.9× lower IMU energy, 0.94 vs. 3.67 J/h), calls the LLM only 10.8×/hour at genuine activity transitions, and opens GPS only 16 s/h — with event detection F1 0.44 and median trigger latency 6.1 s.

---

## 2. Approach and Implementation

### 2.1 System design

The system is a three-state machine driven by a hybrid change-point detector:

```
IDLE  : 1 s @ 20 Hz micro-burst every 6 s (16.7% duty cycle)
        └─ cheap 1 s-window classifier (majority vote) keeps a live estimate
        └─ burst statistics history (last 4 bursts = 24 s context)
        └─ EVENT DETECTION (every 6 s, dual branch):
             rule branch: max robust z-score of level/energy change
             learned branch: Random Forest P(change) ≥ 0.5
             fire if EITHER branch is confident (independent operating points)
ACTIVE: continuous 20 Hz; every 3 s a 59-feature window is classified by the
        main Random Forest; commit when confidence ≥ 0.85 (or ≥ 0.60 consistent)
CONTEXT: on a committed label change → structured JSON → LLM summary;
         on movement onset → GPS for 5 s (GPS ≈ 30× IMU power)
back to IDLE: reference history rebuilt + 8 s cooldown (anti-flapping)
```

Key design decisions, each driven by a measurement (see §3):

- **Duty-cycled bursts instead of literal 1 Hz sampling.** The proposal assumed 1 Hz idle sampling; we measured severe aliasing — a ~2 Hz gait signal sampled at 1 Hz degenerates into phase-drifting slow modulation, and event-ranking quality collapsed to chance (AP 0.033). Burst sampling (20 Hz × 1 s / 6 s) keeps full gait-cycle information at nearly the same energy.
- **Multi-fidelity classification.** A 1 s-window Random Forest loses only 1.2 pp accuracy vs. the 3 s window, so the idle stage can output a *usable* estimate instead of waiting silently.
- **Two independent detector operating points.** A single shared threshold silently disabled one branch (a real bug we hit): the rule branch provides recall, the RF branch precision; firing on max(p_rf, rule) with RF ≥ 0.5 and z ≥ 4.0 beat either alone.

### 2.2 Models

| Model | Input | Architecture | Test accuracy (WISDM, subject-split) |
|---|---|---|---|
| Main classifier | 3 s @ 20 Hz accel+gyro → 59 features (time stats, magnitude spectrum: dominant freq / band energy / entropy, axis correlations) | Random Forest, 300 trees | 0.805 (macro-F1 0.801) |
| Cheap classifier | 1 s @ 20 Hz | Random Forest, 200 trees | 0.793 (macro-F1 0.790) |
| Event detector | (4, 6) burst-statistics history → robust z, ratio, spread | Random Forest, 300 trees + rule branch | event F1 0.44 (P 0.33 / R 0.67) |

### 2.3 What was actually implemented

1. **Full Python codebase** (`swn/` + `scripts/`): data pipeline for WISDM (47 subjects, 94 replay sessions, 698.6 min, 139 real transitions), training, a **pay-per-sample replay simulator** (the only component that can touch the full 20 Hz recording; policies differ *only* in when they "pay"), three-strategy closed-loop evaluation, ablations, figures, and an interactive dashboard.
2. **A working iOS app** (fork of an open-source Core Motion collector) with:
   - the full SWN module in Swift: numerically identical feature extraction (verified against the Python pipeline), Core ML inference, and the same three-state policy running on a dedicated 20 Hz `DeviceMotion` stream;
   - a **life journal**: photo + "log this moment" entries, persisted locally, timestamps aligned with the sensor CSV;
   - **LLM weekly report**: last 7 days of journal + activity timeline → lifestyle summary and health suggestions via any OpenAI-compatible API (user-supplied key);
   - activity changes are **auto-journaled**, so the weekly report gets an activity timeline for free.
3. **Real-device data collection** with Sensor Logger (100 Hz motion export) and an offline pipeline (`process_recording.py`) that converts units (g → m/s²), resamples to 20 Hz, and replays the *same* policy on any recording.
4. **Personalization pipeline** (`personalize.py`): retrains all three models on WISDM train subjects + ground-truth-annotated personal windows, with an explicit no-regression check on untouched WISDM test subjects.

---

## 3. Experimental Results

### 3.1 Main comparison (10 held-out subjects, ~2.4 h closed-loop)

| Metric | Always-on | Periodic 3s/10s | **Event-driven (ours)** |
|---|---|---|---|
| Tracking accuracy | **0.801** | 0.789 | 0.798 |
| Macro F1 | **0.823** | 0.811 | 0.811 |
| IMU duty cycle | 100% | 29.8% | **24.2%** |
| IMU energy proxy (J/h) | 3.67 | 1.15 | **0.94** (3.9× ↓) |
| Total energy incl. GPS (J/h) | 3.67 | 1.15 | 1.42 (61% ↓) |
| LLM calls (/h) | — | — | **10.8** |
| Label-switch latency (median) | 3.1 s | 10.2 s | 12.1 s (6.1 s w/ vote=1) |

**Findings.** The event-driven policy buys back 99.7% of always-on accuracy with 24% of the sampling; it also beats the periodic baseline by 0.9 pp accuracy while sampling 19% *less* — because samples are concentrated at moments of change. LLM usage is decoupled from window count by design: 10.8 meaningful narrated transitions/hour instead of thousands of redundant calls.

### 3.2 Event detection decomposed by transition type (30 test transitions)

| Transition type | Count | Recall | Note |
|---|---|---|---|
| static→moving | 8 | **1.000** | energy step is unmistakable |
| moving→moving | 12 | 0.833 | gait-pattern change |
| moving→static | 1 | 1.000 | |
| static→static (sit↔stand) | 9 | **0.111** | physically invisible to wrist IMU |

Overall event F1 0.44 (P 0.33 / R 0.67), median latency 6.1 s. Since 9 of 30 transitions are sit↔stand (near-zero wrist acceleration difference, < 0.3 m/s²), the 0.67 recall is close to the *detectable upper bound* (~21/30 ≈ 0.70). We report this blind spot honestly rather than hiding it in an aggregate.

### 3.3 Ablations

- **A. Idle sampling regime:** 1 Hz point sampling AP = 0.033 (≈ chance) vs. 1s-burst/6s AP = 0.137 (≈ 8× base rate) — quantified the aliasing problem that motivated duty-cycled bursts.
- **B. Vote window (latency–accuracy knob):** vote=1 → latency 6.1 s, accuracy −1.4 pp; vote=3 → 12.1 s, +1.4 pp. Event F1 unchanged (the detector consumes burst statistics, not labels).
- **C. Detector branches:** rule-only F1 0.437 (recall-heavy), RF-only 0.211 (precision-heavy, misses most transitions), hybrid **0.444** — the branches are complementary, and each needs its own operating point.

### 3.4 From simulation to a real phone: domain shift and personalization (preliminary)

We recorded 259.5 s of ground-truthed daily activity with a smartphone (Sensor Logger, 100 Hz): 0–35 s static, 35–95 s descending stairs, 95–180 s walking on flat ground, 180–259 s sitting. Applying the WISDM-trained models directly:

- The policy **locked onto the wrong label**: 85/85 main-classifier windows in the walking segment predicted *stairs*, and it stayed wrong through the entire sitting segment.
- On the personal recording, macro-F1 was **0.462**.

Diagnosis (measured, not guessed):

1. **The confusion predates the domain shift.** On WISDM's own held-out subjects, **50.5% of walking windows (893/1770) are classified as stairs** — walking is the worst class (F1 0.618) even in-domain. Wrist-worn IMU fundamentally cannot observe *altitude change*, and descending stairs produces gait oscillations nearly identical to flat walking: in our recording, stair-descending vs. flat-walking features almost coincide (dominant frequency 1.67 vs. 1.73 Hz; magnitude std 2.94 vs. 2.56 m/s²).
2. **Phone placement erases the residual cues**, pushing the decision boundary entirely to stairs.

**Personalization fix** (WISDM train subjects + 216 ground-truth personal windows ×5, retrain all three models):

| Metric | Before | After |
|---|---|---|
| Personal recording macro-F1 (3 s / 1 s models) | 0.462 / 0.406 | **1.000 / 1.000** |
| WISDM test subjects macro-F1 (no-regression check) | 0.801 | **0.827** (improved) |
| Event detector at the 3 real transitions | borderline miss (p 0.479 vs 0.50) | all fire, p_rf 0.98–0.99 |
| End-to-end segmentation vs. ground truth | 2 segments, walking swallowed | **4 segments, all boundaries within 1 s**; the detector fired exactly at the 96 s stairs→walking transition |

This exercise also surfaced and fixed a real design inconsistency: the burst history was being sampled every 1 s while the detector had been trained on 6 s-spaced bursts; aligning the cadence to the training distribution restored detector firing.

---

## 4. Live Demo (Planned)

Everything is built; the in-class demo runs on an iPhone with the personalized Core ML models:

1. **Live on-device demo (primary):** open the app → enable "Activity Recognition" → perform sit → walk → stairs → sit. The card shows the live activity estimate with confidence; transitions trigger the event detector, an immediate re-classification, and an auto-journaled entry; the energy/duty-cycle counters show the 1s/6s idle cadence.
2. **Journal + weekly report (second phone screen):** open Daily Log to show auto-logged activity entries and photos; tap the sparkles icon for the LLM weekly report summarizing the week's activity timeline with health suggestions.
3. **Recorded backup (2–3 min):** screen recording of the above, plus the interactive replay dashboard (`artifacts/dashboard.html`) showing the sensor state machine, event probabilities, GPS purchases, and LLM narration stream on WISDM sessions — including one honest failure case (a sit→stand transition undetected for ~3 min).
4. **Offline pipeline on stage data:** run `process_recording.py` on a recording made minutes earlier and show the labeled timeline HTML.

Fallbacks: if the venue has poor GPS/motion conditions, the dashboard replay is fully self-contained; all demo assets ship in one zip.

---

## 5. Discussion

**What worked.**

- The core thesis held quantitatively: **when-to-sample matters more than how-much**. Concentrating 24% of sampling at change moments reproduced always-on accuracy and beat uniform periodic sampling at lower duty cycle.
- The multi-fidelity design (1 s cheap + 3 s main) and the pay-per-sample simulator made every trade-off (latency vs. accuracy, GPS/LLM discipline) measurable and tunable, with ablations to back each knob.
- Shipping the policy into a real iOS app forced engineering honesty: feature parity (verified numerically against Python), unit conversions, and inference cadence all became explicit, testable contracts.
- **Personalization is cheap and effective**: 4 minutes of labeled self-recording recovered from F1 0.46 to 1.00 *without* hurting held-out WISDM performance (0.801 → 0.827). This is a practical recipe for on-device adaptation.

**What did not work.**

- Literal 1 Hz idle sampling (aliasing; replaced by duty-cycled bursts — a measurement-driven deviation from the proposal).
- A single shared detector threshold (silently disabled the RF branch; fixed with independent operating points).
- Cross-device transfer of the classifier: WISDM wrist models lock onto wrong labels on a pocketed phone. And more fundamentally, **walking vs. stairs is near-invisible to IMU alone** — 50% confusion even in-domain, because altitude change is simply not in the signal.

**What could be improved.**

1. **Add a barometer / GNSS vertical velocity** as a *purchased* modality: a few meters of altitude change within seconds is the strongest possible stairs cue, and this fits the project thesis perfectly — the system should buy the expensive sensor exactly when IMU uncertainty is high (e.g., after a locomotion trigger with ambiguous gait features).
2. **Simplify the label taxonomy for cross-device use** (walking/stairs → locomotion) when fine-grained accuracy is not actionable.
3. **More personalization data and subjects**: current personalization is one subject, one session, in-sample; a leave-one-session-out protocol and a few more recordings would make the claim rigorous.
4. **static→static detection** needs a non-IMU modality (pressure, vision, or watch posture) — the detector's quantitative blind spot (recall 0.111) motivates exactly the multi-modal, pay-as-you-need extension this project advocates.
5. **Real energy measurement** on-device (Energy Gauge) to replace the power-model proxy.

**Artifacts.** Full code (`swn/`, `scripts/`), trained + personalized models, all result JSONs, figures, the interactive dashboard, the iOS app with the SWN module, and the offline/personalization pipelines — packaged for reproduction; random seeds and subject splits are fixed and documented.
