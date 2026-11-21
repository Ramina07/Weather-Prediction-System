"""AirCast Astana: next-day PM2.5 forecasting web application.

Run from the project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import os

import streamlit as st

from lib import paths  # noqa: F401  (makes the backend package importable)
from lib import modeling, store, ui

st.set_page_config(page_title="AirCast · Astana PM2.5 forecast", page_icon=str(paths.ASSETS / "favicon.png"),
                   layout="wide", initial_sidebar_state="expanded")
ui.setup_page()

# On Streamlit Community Cloud the API key comes from the app's Secrets instead of a local .env file.
try:
    if not os.environ.get("OPENWEATHER_API_KEY") and "OPENWEATHER_API_KEY" in st.secrets:
        os.environ["OPENWEATHER_API_KEY"] = str(st.secrets["OPENWEATHER_API_KEY"])
except Exception:  # noqa: BLE001  (no secrets file locally)
    pass

# One-time initialisation per browser session: create the database and import CLI forecasts.
if "initialised" not in st.session_state:
    try:
        store.import_cli_history()
    except Exception:  # noqa: BLE001
        pass
    st.session_state["initialised"] = True

V = "views/"
pages = {
    "Forecast": [
        st.Page(V + "home.py", title="Overview", icon=":material/space_dashboard:", default=True),
        st.Page(V + "live_forecast.py", title="Live Forecast", icon=":material/air:"),
        st.Page(V + "track_record.py", title="Track Record", icon=":material/verified:"),
    ],
    "Models": [
        st.Page(V + "model_arena.py", title="Leaderboard", icon=":material/leaderboard:"),
        st.Page(V + "time_machine.py", title="Replay 2025", icon=":material/history:"),
        st.Page(V + "what_if.py", title="Scenario Lab", icon=":material/tune:"),
        st.Page(V + "explainability.py", title="Feature Impact", icon=":material/insights:"),
    ],
    "Research": [
        st.Page(V + "data_explorer.py", title="Data", icon=":material/database:"),
        st.Page(V + "experiments.py", title="Experiments", icon=":material/science:"),
        st.Page(V + "methods.py", title="Methodology", icon=":material/menu_book:"),
        st.Page(V + "about.py", title="About", icon=":material/info:"),
    ],
}
nav = st.navigation(pages)
try:
    st.logo(str(paths.ASSETS / "logo.png"), size="large")
except TypeError:  # older Streamlit without the size argument
    st.logo(str(paths.ASSETS / "logo.png"))

with st.sidebar:
    status = modeling.backend_status()

    def _row(ok: bool, label: str) -> str:
        color = "#14B8A6" if ok else "#F87171"
        return f"<span class='dot' style='background:{color}'></span>{label}<br>"

    st.markdown("<div class='sys'>" + _row(status.get("backend"), "Forecast engine")
                + _row(status.get("benchmark_model"), "Trained models")
                + _row(status.get("xgboost"), "XGBoost runtime")
                + _row(status.get("api_key"), "OpenWeatherMap key") + "</div>", unsafe_allow_html=True)

nav.run()
