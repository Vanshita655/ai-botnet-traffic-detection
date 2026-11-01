"""
Streamlit demo: upload a CSV of network flows (CIC-IDS2017 format) and see which
flows the trained Random Forest marks as Normal or Suspicious.

Run (after training):  streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import botnet_detection as bd  # noqa: E402

st.set_page_config(page_title="Botnet Traffic Detection", layout="wide")
st.title("AI-Based Botnet Traffic Detection")
st.caption("Computer Networks mini project - learning prototype. Alerts need human review.")

model_path = ROOT / "models" / "random_forest.joblib"
features_path = ROOT / "models" / "features.json"
if not model_path.exists():
    st.error("No trained model found. Run `python src/train.py --data <path>` first.")
    st.stop()

model = joblib.load(model_path)
features = json.loads(features_path.read_text())

uploaded = st.file_uploader("Upload a CSV of network flows", type="csv")
if uploaded is not None:
    flows = pd.read_csv(uploaded, low_memory=False, encoding="latin-1")
    try:
        result = bd.predict_flows(model, features, flows)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    counts = result["Prediction"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Flows analysed", f"{len(result):,}")
    c2.metric("Normal", f"{counts.get('Normal', 0):,}")
    c3.metric("Suspicious", f"{counts.get('Suspicious', 0):,}")

    st.subheader("Flagged flows (highest bot probability first)")
    flagged = result[result["Prediction"] == "Suspicious"].sort_values(
        "Bot_probability", ascending=False
    )
    st.dataframe(flagged.head(200), use_container_width=True)

    st.download_button(
        "Download all predictions (CSV)",
        result.to_csv(index=False).encode(),
        "predictions.csv",
        "text/csv",
    )

results_dir = ROOT / "results"
if (results_dir / "metrics.csv").exists():
    st.subheader("Model performance on the held-out test set")
    st.dataframe(pd.read_csv(results_dir / "metrics.csv", index_col=0))
    cm = results_dir / "confusion_matrix_random_forest.png"
    if cm.exists():
        st.image(str(cm), width=420)
