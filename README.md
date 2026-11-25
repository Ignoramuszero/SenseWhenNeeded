# Sense When Needed

**Event-Driven Adaptive Multimodal Sensing for Context-Aware Activity Understanding**

一个可直接运行的课程项目实现：系统用低成本传感器持续监测，在检测到活动状态变化时才提高采样率、激活 GPS 并按需调用 LLM，从而在保持活动识别能力的同时降低资源消耗。

> 详细的方法说明、实验结果与分析见 [REPORT.md](REPORT.md)。

## 核心思路

```
传统系统:   Collect everything → Process everything → Understand everything
本系统:     Monitor cheaply → Detect events → Sense selectively → Understand when needed
```

- **低功耗监测**：1 s / 20 Hz 微突发，每 6 s 一次（16.7% IMU 占空比），轻量 1 s 窗分类器持续刷新估计
- **事件检测**：突发统计量（robust z-score 规则）+ RandomForest 变化检测器，双分支触发
- **主动感知**：触发后切换 20 Hz，用 3 s 窗高保真模型 + 置信度确认（0.85），带 cooldown 防抖
- **按需 GPS**：只在"由静入动"时开启 5 s（30 mW 的大功耗模组不能随便开）
- **按需 LLM**：只在真实活动转换时把结构化上下文发给 LLM 生成摘要；支持自然语言问答

## 快速开始

```bash
# 0) 依赖
pip install numpy pandas scikit-learn matplotlib

# 1) 数据：把 WISDM 原始数据放到 data/raw/
#    UCI: https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset
#    解压后需要 wisdm-dataset/raw/watch/{accel,gyro}/data_*_watch.txt
#    （仓库自带的 data/raw/wisdm.zip 解压脚本见 scripts/prepare_data.py 的调用方式）
python main.py prepare

# 2) 训练（活动识别 + 事件检测 + 阈值调优）
python main.py train

# 3) 三策略对比实验（留出受试者）
python main.py experiments

# 4) 图表 + 交互式看板
python main.py figures
python main.py dashboard
```

产物：

| 文件 | 内容 |
| --- | --- |
| `artifacts/dashboard.html` | **交互式回放看板**（三种策略切换、播放/快进、LLM 摘要流） |
| `artifacts/figures/fig1..4.png` | 四张核心实验图 |
| `artifacts/results/experiments.json` | 全部实验指标 |
| `artifacts/results/strategy_comparison.csv` | 策略对比表 |
| `artifacts/results/training_metrics.json` | 训练/调参记录 |

## 接入真实 LLM

默认使用确定性离线模板（保证实验可复现），配置环境变量即可切换到任意 OpenAI 兼容 API：

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # 可选
export SWN_LLM_MODEL=gpt-4o-mini                    # 可选
python main.py experiments openai
```

## 用自己的 Apple Watch 数据

`MotionCollector` 等应用导出的 CSV（accel/gyro 三轴 + 时间戳）只要整理成
`subject, activity, timestamp_ns, x, y, z` 的 WISDM 行格式，即可直接进入同一管线，
用于"真实设备小规模验证"（对应两周计划的 Day 11）。

## 重要说明（诚实性）

- **GPS 是仿真模态**：WISDM 无 GPS。仿真器用活动相关的潜藏速度生成 GPS 输出，
  策略只能"付费"获取它，不能直接读取 —— 能耗记账因此是公平的。
- **能耗是代理指标**：功率模型（20 Hz IMU 1.0 mW / 待机 0.02 mW / GPS 30 mW）×
  实际开机时间，不是实测电池续航。CPU/内存为进程内实测。
- **所有实验数字由实际运行产生**，报告中没有任何预设的"节省百分比"。

## 目录结构

```
SenseWhenNeeded/
├── swn/
│   ├── config.py                  # 采样/策略/功率模型全部参数
│   ├── features.py                # 3s 窗特征 + 突发统计量与变化特征
│   ├── runtime.py                 # 三种策略的回放引擎（核心）
│   ├── data/wisdm.py, sessions.py # WISDM 解析、活动段、回放会话
│   ├── models/                    # 活动分类器、事件检测器
│   ├── sensing/sensor_simulator.py# 只"发放"被付费样本的模拟器
│   ├── llm/                       # 结构化上下文 + 按需 LLM（离线/OpenAI）
│   └── evaluation/                # 指标与实验编排
├── scripts/                       # prepare / train / experiments / figures / dashboard
├── artifacts/                     # 结果、图表、看板
└── main.py
```

## 仓库内容说明

- `reports/`：课程提交报告（英文版 + 中文版，按 Problem / Approach / Experimental Results / Live Demo / Discussion 五段式组织）
- `artifacts/`：训练好的模型（`models/` WISDM 基线 + `models_personalized/` 真机个性化）、实验结果 JSON、图表、交互式看板
- `data/processed/`：由 WISDM 构建的特征窗口缓存（18 MB × 2），可直接复现训练与实验，无需下载原始数据集
- `data/raw/` 未包含在仓库中（979 MB，且 WISDM 数据集需在 UCI 页面同意使用条款后下载）；获取方式见上方"快速开始"
