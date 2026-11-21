from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from lib import charts, data, ui

ui.page_header("Models", "Feature impact",
               "Permutation importance on the 2025 test set, plus tree importances and Ridge coefficients.")

perm = data.metric_table("E6_permutation_importance_test.csv")
intr = data.metric_table("E6_intrinsic_importance.csv")
if perm.empty:
    st.error("E6 results not found. Run `python -m air_quality.cli explain`.")
    st.stop()

perm["group"] = perm["feature"].map(data.feature_group)
perm["label"] = perm["feature"].map(data.feature_text)
models = [m for m in data.MODEL_ORDER if m in set(perm["model"])]

c1, c2 = st.columns([1.2, 1])
with c1:
    m = st.radio("Model", models, horizontal=True, format_func=lambda x: data.MODEL_SHORT[x],
                 index=models.index("ensemble") if "ensemble" in models else 0)
with c2:
    n = st.slider("Show top features", 5, 30, 15)

d = perm[perm["model"] == m].sort_values("mae_increase_mean", ascending=False)
a, b = st.columns([1.6, 1])
with a:
    top = d.head(n)
    ui.plot(charts.importance_bars(top, "mae_increase_mean", "label", "group", data.FEATURE_GROUPS,
                                   error_col="mae_increase_std", height=max(360, 26 * n),
                                   title=f"Permutation importance · {data.MODEL_SHORT[m]}"))
    st.caption("Increase in test MAE (µg/m³) when the feature is shuffled, ±1 std.")
with b:
    g = d.assign(pos=d["mae_increase_mean"].clip(lower=0)).groupby("group")["pos"].sum().reset_index()
    g = g[g["pos"] > 0]
    fig = px.pie(g, values="pos", names="group", hole=0.55, color="group", color_discrete_map=data.FEATURE_GROUPS)
    fig.update_traces(textinfo="percent+label", textposition="outside")
    ui.plot(charts.style(fig, 380, legend=False, title="By feature family"))
    top3 = ", ".join(d.head(3)["label"])
    ui.verdict(f"Top inputs: <b>{top3}</b>.")

st.divider()
st.markdown("#### All models")
pv = perm.pivot_table(index="feature", columns="model", values="mae_increase_mean")
pv = pv[[x for x in models if x in pv.columns]]
keep = pv.max(axis=1).sort_values(ascending=False).head(15).index
z = pv.loc[keep]
z.index = [data.feature_text(i) for i in z.index]
z.columns = [data.MODEL_SHORT[c] for c in z.columns]
ui.plot(charts.heatmap(z, height=520, colorscale="Teal", fmt=".2f"))
st.caption("Top 15 features by maximum importance. Models rely on different signals, which helps the ensemble.")

if not intr.empty:
    st.divider()
    st.markdown("#### Model internals")
    t1, t2 = st.tabs(["Tree importance", "Ridge coefficients"])
    with t1:
        tm = st.radio("Tree model", [x for x in ["random_forest", "xgboost"] if x in set(intr["model"])],
                      horizontal=True, format_func=lambda x: data.MODEL_SHORT[x])
        ti = intr[intr["model"] == tm].sort_values("value", ascending=False).head(15).copy()
        ti["label"] = ti["feature"].map(data.feature_text)
        ti["group"] = ti["feature"].map(data.feature_group)
        ui.plot(charts.importance_bars(ti, "value", "label", "group", data.FEATURE_GROUPS, height=440))
        st.caption("Impurity-based importance; biased toward correlated continuous features.")
    with t2:
        rc = intr[intr["model"] == "ridge"].copy()
        if rc.empty:
            st.caption("No Ridge coefficients stored.")
        else:
            rc["abs"] = rc["value"].abs()
            rc = rc.sort_values("abs", ascending=False).head(15)
            rc["label"] = rc["feature"].map(data.feature_text)
            rc["direction"] = rc["value"].map(lambda v: "raises forecast" if v > 0 else "lowers forecast")
            fig = px.bar(rc.sort_values("value"), x="value", y="label", orientation="h", color="direction",
                         color_discrete_map={"raises forecast": "#DC2626", "lowers forecast": "#2563EB"})
            fig.update_layout(yaxis_title=None, xaxis_title="Standardised coefficient", legend_title_text=None)
            ui.plot(charts.style(fig, 460))
            st.caption("Change in forecast (µg/m³) per one standard deviation of the feature.")

with st.expander("Feature glossary"):
    feats = data.model_card().get("features", [])
    if feats:
        ui.table(pd.DataFrame({"Feature": feats, "Meaning": [data.feature_text(x) for x in feats],
                               "Family": [data.feature_group(x) for x in feats]}), height=420)

ui.footer()
