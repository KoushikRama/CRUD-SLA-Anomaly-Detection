import sys
from pathlib import Path

# Fix import path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

from src.data.generate_test_data import generate_test_data
from src.xgboost.inference.infer import run_inference
from src.common.load_main_config import load_data_config
from src.xgboost.evaluation.evaluate import evaluate
import requests

# =========================================
# GRAPH FUNCTION (SEPARATE)
# =========================================
def plot_graph(results, selected_op, selected_metric):

    df_filtered = results[results["operation"] == selected_op]

    df_plot = df_filtered.groupby("timestamp").agg({
        selected_metric: "max",
        f"pred_{selected_metric}": "max"
    }).reset_index()

    # Smooth
    df_plot["actual_smooth"] = df_plot[selected_metric].rolling(3, center=True).mean()
    df_plot["pred_smooth"] = df_plot[f"pred_{selected_metric}"].rolling(3, center=True).mean()

    # 🔥 Get anomaly timestamps
    # Get anomaly rows from original filtered data
    df_anomaly = df_filtered[(df_filtered["Status"] != "Normal ✅") & (df_filtered["Root_Cause"].fillna("") == selected_metric)]

    # Aggregate anomaly severity per timestamp
    df_anomaly = df_anomaly.groupby("timestamp").agg({
        "Severity": "max"   # take worst severity
    }).reset_index()

    # Merge with plot dataframe
    df_anomaly = df_plot.merge(df_anomaly, on="timestamp", how="inner")

    # Plot
    fig = go.Figure()

    # Actual curve
    fig.add_trace(go.Scatter(
        x=df_plot["timestamp"],
        y=df_plot["actual_smooth"],
        mode="lines",
        name="Actual"
    ))

    # Predicted curve
    fig.add_trace(go.Scatter(
        x=df_plot["timestamp"],
        y=df_plot["pred_smooth"],
        mode="lines",
        name="Predicted",
        line=dict(dash="dash")
    ))


    # 🚨 Anomaly markers
    sev = df_anomaly["Severity"]

    # optional clipping to avoid extreme values
    sev = sev.clip(0, 5)

    fig.add_trace(go.Scatter(
        x=df_anomaly["timestamp"],
        y=df_anomaly["actual_smooth"],
        mode="markers",
        name="Anomaly",
        marker=dict(
            size=10,
            color=sev,
            colorscale="Reds",
            cmin=0,
            cmax=5,
            showscale=True,
            colorbar=dict(title="Severity", thickness = 20, len=0.5)
        )
    ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=selected_metric,
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="SLA Dashboard", layout="wide")
st.title("🚨 SLA Anomaly Detection Dashboard")


# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("Controls")

hours = st.sidebar.slider("Test Duration (hours)", 24, 168, 48)
run_button = st.sidebar.button("Run Pipeline")


# =========================================
# SESSION STATE (IMPORTANT)
# =========================================
if "results" not in st.session_state:
    st.session_state.results = None


# =========================================
# RUN PIPELINE (ONLY ON BUTTON)
# =========================================
if run_button:

    st.info("Generating test data...")
    config = load_data_config()

    df_test = generate_test_data(
        start_date=datetime(2025, 4, 1),
        hours=hours,
        config=config
    )

    st.success("Test data generated")

    st.write(df_test.columns)
    st.dataframe(df_test.head())

    st.info("Running inference...")

    API_URL = "http://localhost:8000/predict"

    payload = df_test.copy()
    payload["timestamp"] = payload["timestamp"].astype(str)

    response = requests.post(
        API_URL,
        json=payload.to_dict(orient="records")
    )

    if response.status_code != 200:
        st.error("API failed")

    results = pd.DataFrame(response.json())

    print(results.columns)
    print(results.head())

    st.success("Inference completed")

    results["timestamp"] = df_test["timestamp"]

    # store in session
    st.session_state.results = results


# =========================================
# IF RESULTS EXIST → SHOW DASHBOARD
# =========================================
if st.session_state.results is not None:

    results = st.session_state.results

    # =========================================
    # TABLES (TOP)
    # =========================================
    st.subheader("📊 Sample Data")
    st.dataframe(results.head())

    st.subheader("🚨 Detected Anomalies")
    anomalies = results[results["Status"] != "Normal ✅"]
    st.metric("Total Alerts", len(anomalies))
    st.dataframe(anomalies)

    # ----------------------------
    # METRICS
    # ----------------------------
    st.subheader("📈 Evaluation Metrics")

    metrics = evaluate(results)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Precision", f"{metrics['precision']:.3f}")
    col2.metric("Recall", f"{metrics['recall']:.3f}")
    col3.metric("F1 Score", f"{metrics['f1']:.3f}")
    col4.metric("Alert Rate", f"{metrics['alert_rate']:.3f}")

    # ----------------------------
    # ANOMALY TYPE BREAKDOWN
    # ----------------------------
    st.subheader("🔍 Anomaly Breakdown")

    if "anomaly_type" in results.columns:
        breakdown = results[results["is_anomaly"]]["anomaly_type"].value_counts()
        st.bar_chart(breakdown)

    # =========================================
    # FILTERS (MAIN PANEL)
    # =========================================
    st.subheader("🔍 Filters")

    col1, col2 = st.columns(2)

    with col1:
        operations = results["operation"].unique()
        selected_op = st.selectbox("Select Operation", operations)

    with col2:
        metrics = [
            "success_vol",
            "fail_vol",
            "success_rt_avg",
            "fail_rt_avg"
        ]
        selected_metric = st.selectbox("Select Metric", metrics)

    # =========================================
    # GRAPH (ONLY THIS UPDATES)
    # =========================================
    st.subheader("📈 Trend Analysis")

    plot_graph(results, selected_op, selected_metric)

else:
    st.info("Use the sidebar to run the pipeline")