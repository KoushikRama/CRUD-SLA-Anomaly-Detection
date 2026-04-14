# 🚨 E-Commerce SLA Anomaly Detection (End-to-End ML System)

---
## 🌐 Live Demo

🚀 **Access the deployed dashboard:**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecom-sla-anomaly-detection-dashboard-5lvhxfr87rf9xu349caae3.streamlit.app/)

---

## 📖 About This Project

In high-scale e-commerce environments, static thresholds for Service Level Agreements (SLAs) fail to account for seasonal traffic patterns and shifting system baselines. This project implements a **Predictive Anomaly Detection System** that moves beyond static rules.

By leveraging **XGBoost Regressors** to model "normal" system behavior, the pipeline calculates real-time residuals to distinguish between expected fluctuations (like peak hour spikes) and genuine system failures. The result is a robust, production-ready deployment utilizing a **Serverless SageMaker** architecture, designed to minimize operational overhead while maximizing detection accuracy.

---

## 🏗️ Architecture

![System Architecture](https://github.com/KoushikRama/CRUD-SLA-Anomaly-Detection/blob/main/images/Architecture.png) 

### Deployment Flow:
1. **Streamlit UI:** Sends SLA metric data via HTTPS.
2. **API Gateway:** Securely routes requests to the model endpoint.
3. **SageMaker Serverless Endpoint:** Runs the XGBoost inference container.
4. **S3 Bucket:** Stores model artifacts, feature scalers, and calculated thresholds.

---

## 🧠 System Overview

This project is a **complete end-to-end ML pipeline**, covering:

- Synthetic data generation (Hourly patterns & noise)
- Feature engineering (Time-series cyclical encoding)
- Model training (XGBoost Regressors per metric)
- Threshold computation (Residual-based)
- SageMaker Serverless deployment
- API Gateway integration
- Streamlit dashboard for real-time visualization

---

## 📁 Project Structure

```text
.
├── config/             # Configuration files for v1/v2 pipelines
├── deploy/             # AWS SageMaker deployment scripts
├── notebooks/          # Exploratory Data Analysis & Validation
├── sagemaker_tar/      # Inference code for SageMaker container
├── src/
│   ├── api/            # FastAPI/Flask application logic
│   ├── common/         # Shared utilities (S3, feature engineering)
│   ├── data/           # Synthetic data generation scripts
│   └── xgboost/        # Training, Inference, and Threshold logic
├── tests/              # Integration tests for AWS services
├── ui/                 # Streamlit dashboard source
├── run_pipeline.py     # Main orchestrator
└── Dockerfile          # Containerization for inference
```
---

## ⚙️ Feature Engineering

### 🕒 Time Features

- `hour`
- `hour_sin`, `hour_cos`  
→ Captures cyclic hourly patterns

---

### 🎯 SLA Metrics

- `success_vol`
- `fail_vol`
- `success_rt_avg`
- `fail_rt_avg`

---

## 📡 API Format

### Request

```json
[
  {
    "timestamp": "2025-04-01 10:00:00",
    "operation": "browse_products",
    "success_vol": 8000,
    "fail_vol": 300,
    "success_rt_avg": 150,
    "fail_rt_avg": 120
  }
]
```

### Response
```json
[
  {
    "operation": "browse_products",
    "hour": 10,
    "success_vol": 8000,
    "fail_vol": 300,
    "success_rt_avg": 150,
    "fail_rt_avg": 120,

    "pred_success_rt_avg": 169.92,
    "pred_fail_rt_avg": 145.48,
    "pred_success_vol": 82533.85,
    "pred_fail_vol": 273.93,

    "Status": "Anomaly",
    "Root_Cause": "success_vol",
    "All_Causes": "success_vol:2.89",
    "Severity": 2.893,
    "Severity_Label": "🚨 Critical"
  }
]
```

--- 

## 🤖 Model (XGBoost Residual-Based)

---

### 🔍 Approach

- Train regression models for each SLA metric
- Predict expected values
- Compute residuals (actual vs predicted)
- Apply threshold-based anomaly detection

---

### ⚙️ Pipeline

```
Features → XGBoost → Prediction → Residual → Threshold → Alert
```
---

## ⚙️ Setup & Installation

This section covers the environment setup and installation required to run the pipeline locally and connect it to AWS services.

---

### 1. Prerequisites

Ensure the following tools are installed and configured:

- **Python 3.9+**
- **AWS CLI** (configured via `aws configure`)
- **Streamlit** (for UI)

#### Required AWS Permissions

Your IAM user/role must have access to:

- `AmazonS3FullAccess`
- `AmazonSageMakerFullAccess`
- `AmazonAPIGatewayAdministrator`

---

### 2. Installation Steps

Clone the repository and install dependencies:

```
# Clone repository
git clone https://github.com/KoushikRama/CRUD-SLA-Anomaly-Detection.git
cd CRUD-SLA-Anomaly-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
---

### 3. Model Training

Run the full pipeline:

```bash
python run_pipeline.py
```

This step performs:

- Data generation
- Feature engineering
- Model training (XGBoost)
- Threshold computation
- Model artifact preparation

---

### 4. Infrastructure Setup

**Step 1: Create S3 Bucket**

```bash
aws s3 mb s3://your-unique-s3-bucket-name
```
This bucket is used to store:

- Training data
- Model artifacts

**Step 2: Model Packaging**

The model must be packaged into a .tar.gz file for SageMaker.

This includes:

This includes:
- Trained model
- `inference.py`
- `feature_engineering.py`

```
cd sagemaker_tar/
tar -czvf model.tar.gz *
```

**Step 3: Upload to S3 Bucket**

```bash
aws s3 cp model.tar.gz s3://your-unique-s3-bucket-name/
```

---

### 5. Deployment (SageMaker)

Run the deployment script:

```Bash
python deploy/deploy_sagemaker.py
```

This will:

- Create SageMaker Model
- Create Serverless Endpoint Configuration
- Deploy Endpoint

---

### 6. Verify the sagemaker endpoint
```
python tests/test_sagemaker.py
```

---

### 7. API Gateway Setup

To expose the SageMaker endpoint as a REST API:

1. Open AWS API Gateway Console
2. Create a REST API
3. Add a POST method
4. Set:
    - Integration Type → AWS Service
    - Service → SageMaker
    - Action → InvokeEndpoint
5. Provide your Endpoint Name

📌 After deployment, replace the API URL in the UI config with your API Gateway endpoint.

---

### 8. Verify the API Gateway URL
```
python tests/test_apigate.py
```
---

### 9. Run UI Dashboard

```
streamlit run ui/app.py
```
---

### 📈 Strengths

- Detects minor deviations precisely
- Provides root cause (metric-level)
- Interpretable outputs

---

### ⚠️ Limitations

- Requires threshold tuning
- Depends on consistent input schema

---

### 🔁 Thresholding Strategy

- Residual-based detection
- Combination of:
    - percentage deviation
    - absolute deviation

---

### 🧪 Data Simulation

Synthetic SLA data includes:
- Realistic traffic patterns (peak & off-peak)
- Load-based latency increases
- Failure rate correlation with load

---

## 🌐 Deployment

**Frontend**

- Streamlit Cloud

**Backend**

- AWS API Gateway
- SageMaker Serverless Endpoint

---

## 📊 Dashboard Features

- 📈 SLA metric visualization
- 🚨 Anomaly detection display
- 🔍 Root cause identification
- 📊 Severity analysis

---

## 🧠 Key Capabilities

- Detect subtle SLA anomalies
- Provide metric-level root cause
- Enable interactive exploration
- Fully serverless ML inference

---

## 🔮 Future Improvements

- Real-time streaming data
- Drift detection
- Auto threshold tuning
- Alerting system (Slack / Email)
- Hybrid anomaly models

---

## 💡 Summary
This project provides a production-ready anomaly detection system for SLA monitoring by combining:

- ML precision (XGBoost residual model)
- Cloud scalability (Serverless SageMaker)
- Interactive UI (Streamlit)

---

## 👨‍💻 Author
### Koushik Rama


