from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import streamlit as st

from lib import charts, data, ui

ui.page_header("Research", "Methodology",
               "Research design, formulas, literature and limitations.")


def strip_title(md: str) -> str:
    """Drop the document's own H1 (the tab already has a title)."""
    return re.sub(r"\A#\s[^\n]*\n", "", md.strip())


tabs = st.tabs(["Design", "Formulas", "Literature", "References", "Limitations"])

with tabs[0]:
    a, b, c = st.columns(3)
    a.markdown(ui.card("AIM", "Next-day PM2.5", "Daily mean for Astana from data available at 18:00."),
               unsafe_allow_html=True)
    b.markdown(ui.card("DESIGN", "Chronological split", "Train 2021–23, select on 2024, test once on 2025."),
               unsafe_allow_html=True)
    c.markdown(ui.card("INTEGRITY", "No leakage", "Records up to 17:00 only; preprocessing fitted inside "
                                                  "training folds; leakage tests."), unsafe_allow_html=True)
    st.write("")
    st.markdown(strip_title(data.doc_text("research_methodology.md")))

with tabs[1]:
    st.markdown(strip_title(data.doc_text("mathematical_formulations.md")))

with tabs[2]:
    lit = data.literature_table()
    if lit.empty:
        st.caption("`docs/literature_comparison.csv` not found.")
    else:
        n = len(lit)
        yrs = pd.to_numeric(lit["Publication year"], errors="coerce")
        ui.kpi_row([("Studies", str(n), f"{int(yrs.min())}–{int(yrs.max())}"),
                    ("Day-ahead / similar", str(int(lit["Directly comparable with ours?"].fillna("")
                                                      .str.startswith("Partly").sum())),
                     "partly comparable"),
                    ("In Kazakhstan", str(int(lit["Country / study location"].str.contains("Kazakhstan").sum())),
                     "Almaty and Astana")])
        st.write("")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            q = st.text_input("Search", placeholder="Title, author, model, country")
        with c2:
            comp = st.multiselect("Comparability", sorted(lit["Directly comparable with ours?"]
                                                                           .dropna().str.split(" ").str[0]
                                                                           .str.strip(";,.").unique()))
        view = lit.copy()
        if q:
            mask = view.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)
            view = view[mask]
        if comp:
            view = view[view["Directly comparable with ours?"].fillna("").str.split(" ").str[0]
                        .str.strip(";,.").isin(comp)]
        cols = ["Reference ID", "Authors", "Publication year", "Exact article title", "Country / study location",
                "Models used", "Best-performing model", "MAE", "RMSE", "R²", "Forecast horizon",
                "Directly comparable with ours?", "Source URL"]
        cols = [c for c in cols if c in view.columns]
        ui.table(view[cols], height=420, column_config={
            "Source URL": st.column_config.LinkColumn("Link", display_text="open"),
            "Authors": st.column_config.TextColumn(width="medium"),
            "Exact article title": st.column_config.TextColumn("Title", width="large")})
        st.caption(f"{len(view)} of {n} studies")
        a, b = st.columns(2)
        with a:
            yc = yrs.value_counts().sort_index()
            fig = px.bar(x=yc.index.astype(int).astype(str), y=yc.values, labels={"x": "Year", "y": "Studies"},
                         color_discrete_sequence=["#0F766E"])
            ui.plot(charts.style(fig, 280, legend=False, title="Publication year"))
        with b:
            fam = {"XGBoost / boosting": r"XGB|boost|LightGBM|CatBoost|GBM", "Random Forest": r"Random Forest|RF\b",
                   "SVR / SVM": r"SVR|SVM|Support Vector", "Linear / Ridge": r"Linear|Ridge|Lasso|MLR",
                   "Neural nets (LSTM, MLP)": r"LSTM|MLP|ANN|neural|CNN|GRU|Transformer"}
            cnt = {k: int(lit["Models used"].fillna("").str.contains(v, case=False, regex=True).sum())
                   for k, v in fam.items()}
            fig = px.bar(x=list(cnt.values()), y=list(cnt.keys()), orientation="h",
                         labels={"x": "Studies using it", "y": ""}, color_discrete_sequence=["#7C3AED"])
            ui.plot(charts.style(fig, 280, legend=False, title="Model families"))
        with st.expander("Full literature review"):
            st.markdown(strip_title(data.doc_text("literature_review.md")))

with tabs[3]:
    st.markdown(strip_title(data.doc_text("references_apa.md")))

with tabs[4]:
    st.markdown(strip_title(data.doc_text("limitations.md")))

ui.footer()
