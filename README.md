# Forecasting for Optimized EV Charging

This repository implements a complete machine learning pipeline to forecast:
1) EV plug-in availability (binary classification)
2) Energy-to-full (regression, defined as 100 - SOC)

These forecasts are intended for downstream optimization (for example, MPC) to schedule charging during low-cost or high-solar periods.

---

## Repository Structure



---

## Project Structure
```
ev_charging_mlop_demo/
├── models/
│ ├── lgb_classifier.pkl
│ └── lgb_regressor.pkl
├── notebooks/
│ ├── data_exploration.ipynb
│ └── evaluation.ipynb
├── src/
│ ├── init.py
│ ├── config.py
│ ├── feature_engineering.py
│ ├── generate_data.py
│ ├── preprocessing.py
│ ├── train.py
│ └── predict.py
├── tests/
│ ├── init.py
│ ├── test_feature_engineering.py
│ └── test_preprocessing.py
├── .gitignore
├── requirements.txt
└── README.md
└── .gitignore
```

---

## Problem Overview

Residential microgrids and building energy systems often need to schedule EV charging in a cost-efficient way while accounting for PV generation, grid constraints, and user behavior.

To support a charging optimizer, we produce two forecasts:

### 1. Plug-in Availability (Classification)
Predict whether the EV will be plugged in at time t + 15 minutes.  
This determines when charging is physically possible.

### 2. Energy-to-Full (Regression)
Predict energy-to-full at time t + 15 minutes, defined as 100 - SOC_future.  
This determines how much energy needs to be scheduled across a forward horizon, for example 48 hours using roll-forward MPC.

---

## Data Characteristics and Key Findings

EV telemetry is irregular. Most vehicles report at roughly 1 to 10 minute intervals, but the data can contain long gaps ranging from hours to days.

Key implications:
- We do not assume uniform sampling.
- We detect long gaps and mask SOC deltas and derived features across these discontinuities to avoid false charging or discharging signals.
- Across long gaps, SOC often drifts slightly (idle behavior). This supports gap masking rather than treating the delta as meaningful consumption.

Synthetic data generation is included to preserve the statistical characteristics of this behavior without sharing any proprietary data.

---

## Pipeline Stages

### Stage 1: Data Generation
`src/generate_data.py` creates a synthetic telemetry dataset that includes:
- Irregular sampling (1 to 10 minutes typical)
- Structured long gaps (hours to days)
- SOC dynamics with session-based charging behavior
- Gap flags to support safe feature masking

Output:
- `synthetic_ev_data.csv`

### Stage 2: Preprocessing
`src/preprocessing.py` performs:
- Timestamp parsing and sorting
- Per-vehicle time difference computation
- Gap flagging (default threshold 240 minutes)
- Filtering out very short traces

### Stage 3: Feature Engineering
`src/feature_engineering.py` performs:
- SOC delta computation with masking when gap_flag=True
- Time features (hour, day_of_week)
- Lag features (soc_lag_1, soc_lag_2, soc_lag_3)
- Target creation for both models using a configurable horizon:
  - plug_future
  - energy_to_full_future = 100 - soc_future

### Stage 4: Model Training
`src/train.py` trains two LightGBM models:
- LGBMClassifier for plug_future
- LGBMRegressor for energy_to_full_future

Artifacts:
- `models/plug_model.pkl`
- `models/energy_model.pkl`
- `models/metrics.json`

### Stage 5: Inference
`src/predict.py` loads the two models and outputs predictions:
- plug_probability
- predicted_energy_to_full

Output:
- `predictions.csv`

---

## How to Run Locally

### Install dependencies
```bash
pip install -r requirements.txt

```

### Generate data
```bash
python src/generate_data.py

```

### Train models
```bash
python src/train.py

```

### Run inference
```bash
python src/predict.py

```

### Unit Tests
```bash
pytest -q

```

---

# Production MLOps Architecture

## 1. End-to-End MLOps Flow

The system can be deployed in a hybrid architecture combining AWS cloud infrastructure with on-premise edge devices.

### Stage 1: Data Ingestion

Telemetry, pricing signals, and contextual data are ingested via:

- AWS IoT Core or Kafka
- Stored in Amazon S3 as raw landing data
- Metadata stored in AWS Glue Catalog

For edge-first deployments, telemetry is buffered locally and periodically synchronized to the cloud.

### Stage 2: Data Processing

Batch or scheduled pipelines perform:

- Timestamp normalization
- Gap detection and masking
- Feature engineering
- Target generation

Orchestration can be implemented using:

- Apache Airflow on Amazon MWAA
- AWS Step Functions for lighter workflows

Outputs are stored as feature datasets in S3 or a feature store such as Amazon SageMaker Feature Store.

### Stage 3: Model Training

Training is performed in:

- Amazon SageMaker Training Jobs
- Containerized EC2 workloads

Models trained:

- LightGBMClassifier for plug availability
- LightGBMRegressor for energy-to-full

Artifacts stored in:

- S3 model registry bucket
- SageMaker Model Registry (optional)

Metrics are logged for governance and retraining decisions.

### Stage 4: Model Compression and Edge Deployment

For on-premise or embedded edge systems, model footprint and latency are critical.

After training, models can be:

1. Converted to ONNX
2. Quantized to reduce size and inference cost

Quantization options:

- Dynamic quantization
- Post-training integer quantization
- 8-bit weight compression

Benefits:

- Reduced model size
- Lower memory footprint
- Faster inference
- Reduced CPU load on embedded hardware

Compressed models are packaged into Docker images or deployed as standalone inference services.

### Stage 5: Inference Layer

#### Cloud Mode

- REST API via SageMaker Endpoint or AWS Lambda
- Batch inference triggered by scheduler
- Results stored in DynamoDB or S3

#### Edge Mode

- Lightweight inference service
- Runs locally in Docker
- Consumes telemetry in near real time
- Outputs forecasts to local optimizer

Edge-first inference ensures:

- Low latency
- Resilience to network outages
- Reduced cloud dependency

### Stage 6: Monitoring and Retraining

Continuous monitoring includes:

- Data drift detection
- Distribution shift in SOC and plug behavior
- Performance degradation in MAE or AUC

Retraining can be triggered if:

- Drift threshold exceeded
- Performance falls below SLA
- Seasonal change detected

## Hybrid Cloud and Edge Architecture

A realistic deployment pattern for EV charging optimization:

1. Edge device performs:
   - Telemetry buffering
   - Local inference
   - MPC optimization
   - Immediate charging decisions

2. Cloud performs:
   - Model training
   - Model validation
   - Model compression
   - Version control
   - Fleet-wide rollout

Updated quantized models are pushed back to edge devices.

## Why Quantization Matters for Energy Systems

In building energy management systems:

- Hardware often has limited compute
- Real-time response is required
- Deterministic execution is preferred

Quantized LightGBM models:

- Reduce memory footprint
- Improve inference speed
- Maintain acceptable predictive performance
- Lower CPU utilization on edge devices

## Alignment with EV Charging Optimization

The two-model structure maps cleanly into an MPC framework:

1. Plug probability determines feasibility constraints.
2. Energy-to-full determines energy scheduling bounds.

Forecasts are rolled forward across a 48-hour horizon and fed into:

- Linear programming
- Mixed integer programming
- Stochastic MPC

The system can shift charging into:

- Low price windows
- High PV generation windows
- Low grid congestion periods

---

## Contact

This project presents a production-ready forecasting pipeline for real-world grid and EV control systems.

For technical discussion, deployment architecture, or further details, please contact:

Prabhakaran Vasantharajan  
Email: prabhakaranv@proton.me
