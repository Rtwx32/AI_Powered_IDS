import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="AI-IDS Security Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ AI-IDS Security Dashboard")
st.caption("لوحة مراقبة تجريبية — تعرض alerts ناتجة من EnsembleIDS")


@st.cache_data
def load_demo_alerts(n=200):
    rng = np.random.default_rng(7)
    severities = rng.choice(["low", "medium", "high"], n, p=[0.5, 0.3, 0.2])
    attack_types = rng.choice(["dos", "probe", "r2l", "u2r", "normal"], n,
                               p=[0.25, 0.2, 0.15, 0.1, 0.3])
    times = pd.date_range("2026-06-01", periods=n, freq="h")
    return pd.DataFrame({
        "time": times,
        "severity": severities,
        "attack_type": attack_types,
        "probability": rng.random(n).round(3),
    })


df = load_demo_alerts()

st.sidebar.header("🔎 الفلاتر")
severity_filter = st.sidebar.multiselect(
    "Severity", options=df["severity"].unique(), default=list(df["severity"].unique())
)
df_filtered = df[df["severity"].isin(severity_filter)]

col1, col2, col3 = st.columns(3)
col1.metric("إجمالي Alerts", len(df_filtered))
col2.metric("High severity", len(df_filtered[df_filtered["severity"] == "high"]))
col3.metric("متوسط الاحتمالية", round(df_filtered["probability"].mean(), 3))

st.subheader("📋 آخر التنبيهات")
st.dataframe(df_filtered.sort_values("time", ascending=False), use_container_width=True)

st.subheader("📊 توزيع أنواع الهجمات")
attack_counts = df_filtered["attack_type"].value_counts().reset_index()
attack_counts.columns = ["attack_type", "count"]
fig_pie = px.pie(attack_counts, values="count", names="attack_type",
                  title="Attack Types Distribution")
st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("📈 Timeline")
timeline = df_filtered.groupby(df_filtered["time"].dt.date).size().reset_index(name="alerts")
timeline.columns = ["date", "alerts"]
fig_line = px.line(timeline, x="date", y="alerts", title="Alerts Over Time")
st.plotly_chart(fig_line, use_container_width=True)
