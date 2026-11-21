from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import data, store, ui

# Edit names and roles here.
TEAM = [
    ("Team member 1", "Data engineering", "API client, data collection, validation, cleaning, exploratory analysis"),
    ("Team member 2", "Machine learning", "Features, model training and tuning, evaluation, weather data"),
    ("Team member 3", "Product & interface", "Web application, forecast storage and verification, visualisation"),
]

ui.page_header("About", "AirCast Astana",
               "Interactive machine-learning system for next-day PM2.5 forecasting using the OpenWeatherMap API.")

for i, (c, (name, role, work)) in enumerate(zip(st.columns(3), TEAM), start=1):
    c.markdown(ui.card(f"{i:02d}", name, f"<b>{role}</b><br>{work}"), unsafe_allow_html=True)

st.write("")
st.markdown("### System overview")
fc = store.forecasts_frame()
rows = [
    ("Models", "Ridge, Random Forest, XGBoost, SVR, ensemble; persistence baseline", "Leaderboard"),
    ("Application", "Live forecasts, history, automatic verification", "Live Forecast"),
    ("Data", "OpenWeatherMap + Open-Meteo, audit, data dictionary", "Data"),
    ("Preprocessing", "Gap handling, scaling, feature selection", "Data · Experiments"),
    ("Validation", "Chronological 2021–23 / 2024 / 2025, TimeSeriesSplit CV", "Leaderboard"),
    ("Comparisons", "5 feature sets × 5 models, default vs tuned", "Experiments"),
    ("Formulas", "18 formulas linked to code", "Methodology"),
    ("Literature", f"{len(data.literature_table())} studies 2022–2026, 18 APA references", "Methodology"),
    ("Results", "Hypotheses H1–H5 evaluated from stored metrics", "Experiments"),
    ("Interpretability", "Permutation and tree importance, coefficients, error analysis", "Feature Impact"),
    ("Source code", "`src/air_quality/models.py`, code shown per model", "Leaderboard"),
    ("Storage", f"SQLite, {len(fc)} forecasts stored", "Track Record"),
]
ui.table(pd.DataFrame(rows, columns=["Component", "Implementation", "Page"]), height=458)

st.write("")
a, b = st.columns(2, gap="large")
with a:
    st.markdown("### Stack")
    st.markdown(
        "- Python, pandas, scikit-learn, XGBoost\n"
        "- OpenWeatherMap Air Pollution API, Open-Meteo\n"
        "- Streamlit, Plotly\n"
        "- SQLite\n"
        "- pytest with leakage tests, fixed seeds, `config.yaml`")
with b:
    st.markdown("### Run locally")
    st.code("pip install -r requirements.txt\npip install -e .\nstreamlit run app/app.py", language="bash")
    st.caption("API key in `.env`: `OPENWEATHER_API_KEY=...`")

ui.footer()
