# 在 MacBook 上把 SenseWhenNeeded 模型集成进 core-motion-data-collector

本压缩包含 **已训练好的三个模型**（WISDM 腕部 IMU，5 类活动：walking / jogging / stairs / sitting / standing）以及推理所需的全部代码。目标：转成 Core ML 后作为模块加入 core-motion-data-collector，实现"边采集边识别活动"。

## 包内容

| 文件 | 说明 |
|---|---|
| `artifacts/models/activity_classifier.pkl` | 主分类器：3s 窗 @20Hz（accel+gyro）→ 5 类活动（约 56 MB） |
| `artifacts/models/burst_classifier.pkl` | 轻量分类器：1s 窗 @20Hz → 5 类活动（约 49 MB） |
| `artifacts/models/event_detector.pkl` | 事件检测器：突发统计历史 → change / no_change（4 MB） |
| `swn/` | 完整 Python 包（特征提取 + 模型封装 + 推理接口） |
| `swn/coreml_export.py` | **Mac 上运行**：sklearn → Core ML 转换脚本 |
| `artifacts/results/*.json` | 训练指标、三策略实验结果（供参考） |
| `MACOS_INTEGRATION.md` | 本文档 |

**注意**：`.pkl` 是 scikit-learn 模型，iOS 不能直接加载。必须先转 Core ML（见下）。

## 第一步：转 Core ML（Mac 上一次性操作）

```bash
cd SenseWhenNeeded
pip install coremltools joblib numpy scikit-learn
python swn/coreml_export.py
```

生成 `artifacts/coreml/` 下三个 `.mlmodel`：
- `ActivityClassifier.mlmodel` — 3s 窗、5 类、输出 activity + 概率
- `BurstClassifier.mlmodel` — 1s 窗、5 类（低功耗阶段用）
- `EventDetector.mlmodel` — 二分类 change / no_change

Xcode 把 `.mlmodel` 拖进工程即自动生成 Swift 类（如 `ActivityClassifier(input:)` → `.activityLabel`, `.activityProbability`）。

## 第二步：在 Swift 中复现特征提取（集成的核心工作量）

Core ML 模型的输入是 **固定顺序的特征向量**，不是原始传感器数据。Swift 端必须按 `swn/features.py` **完全一致地**计算特征。三个入口：

| 模型 | 输入构造 | 特征函数（Python 源） |
|---|---|---|
| ActivityClassifier | 3s × 20Hz 的 acc+gyro → `window_features(acc, gyro, fs=20)` | `swn/features.py` 的 `window_features` |
| BurstClassifier | 1s × 20Hz 的 acc+gyro → `window_features(acc, gyro, fs=20)` | 同上（窗口不同而已） |
| EventDetector | 最近 4 个 1s 突发的统计堆叠成 (4,6) 矩阵 → `burst_event_features(history)` | `swn/features.py` 的 `burst_stats` + `burst_event_features` |

关键约定（务必核对 `swn/features.py`）：
- 采样率 20 Hz；陀螺仪单位 rad/s；加速度 m/s²
- 特征顺序 = `.mlmodel` 输入参数顺序（coremltools 按 feature_names 命名了每个输入，Xcode 生成的 Swift 初始化器会直接显示名字，照名字对即可）
- EventDetector 的历史缓冲：`IDLE_HISTORY=4` 个突发，每突发 6 个统计量（见 `burst_stats`），突发每 `IDLE_PERIOD_S=6s` 采 1 次
- 活动标签顺序固定：`["walking", "jogging", "stairs", "sitting", "standing"]`（索引即输出类别）

建议移植方式：Swift 写一个 `FeatureExtractor`，单元测试用 Python 跑同一小段数据对比输出（`swn/features.py` 的函数可直接调用生成参考值）。

## 第三步：接入 core-motion-data-collector

推荐挂在 iOS 端（手表端采集频率更高但改动面大）：

1. **缓冲层**：订阅 `MotionDataCollector` 的 `DeviceMotion` 回调，把 `userAcceleration` + `rotationRate` 追加进环形缓冲（注意：特征用的加速度建议 `acceleration = userAcceleration + gravity`，与训练一致，见 `MotionManager.swift` 中 `accelerationX` 的定义）
2. **持续推理**：每 1s 用 `BurstClassifier` 更新当前活动估计；每 3s 用 `ActivityClassifier` 做高保真确认
3. **事件检测**：维护 4 个 1s 突发统计的历史 → `EventDetector`；输出 change 时提高采样频率/触发 UI 提示
4. **和日记联动**：检测到活动变化时自动在 LifeLog 里打点（时间戳天然与传感器流对齐），周报功能即可吃到"活动时间线"

一个诚实的提醒：两个分类器是 300 棵深树的 RandomForest，Core ML 转换后单个模型可能仍有 20-50 MB、推理在手机端为毫秒级。如果嫌大，可以（a）只用 BurstClassifier + EventDetector；（b）回 Python 用更少的树重训（改 `n_estimators` 重训 `scripts/train.py`）。

## 推理参考（不转 Core ML，纯 Python 验证用）

```python
import numpy as np, sys; sys.path.insert(0, ".")
from swn.models.activity_classifier import ActivityClassifier
clf = ActivityClassifier.load("artifacts/models/burst_classifier.pkl")
label, conf, probs = clf.predict_window(acc_1s_20hz, gyro_1s_20hz)  # label: 0..4
```

模型训练与评测细节见 SenseWhenNeeded 项目的 `REPORT.md`（未随包附带，可另行索取）。
