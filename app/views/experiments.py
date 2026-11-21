from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import charts, data, paths, ui

ui.page_header("Research", "Experiments",
               "Seven experiments, five hypotheses. Verdicts are computed from the stored results.")

mc = data.model_comparison()
base4 = ["ridge", "random_forest", "xgboost", "svr"]
ORDER = [m for m in data.MODEL_ORDER if m in set(mc["model"])] if not mc.empty else []


def grouped(df: pd.DataFrame, x: str, y: str, group: str, groups: list, labels: dict, colors: dict,
            ytitle: str, height: int = 380) -> go.Figure:
    fig = go.Figure()
    for g in groups:
        d = df[df[group] == g]
        fig.add_trace(go.Bar(x=d[x], y=d[y], name=labels.get(g, g), marker_color=colors.get(g),
                             text=d[y].round(2), textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=ytitle)
    return charts.style(fig, height)


def hyp(text: str) -> None:
    st.markdown(f"<div style='color:#5B6476;font-size:.9rem;margin:2px 0 10px 0'>{text}</div>", unsafe_allow_html=True)


tabs = st.tabs(["E1 · Models", "E2 · Feature engineering", "E3 · Feature selection", "E4 · Tuning",
                "E5 · High-pollution days", "E6 · Importance", "E7 · Seasons", "Ablations"])

# ---------------------------------------------------------------- E1
with tabs[0]:
    hyp("<b>H1</b> A tree ensemble has the lowest validation MAE. <b>H3</b> The selected model beats persistence.")
    m4 = mc[mc["model"].isin(base4)]
    best_val = m4.sort_values("validation_mae").iloc[0]["model"]
    sel = data.model_selection().get("best", {}).get("model", "ensemble")
    r_sel = mc[mc["model"] == sel].iloc[0]
    r_p = mc[mc["model"] == "persistence"].iloc[0]
    d = mc.melt(id_vars="model", value_vars=["validation_mae", "test_mae"], var_name="stage", value_name="mae")
    d["model_label"] = d["model"].map(data.MODEL_SHORT)
    ui.plot(grouped(d, "model_label", "mae", "stage", ["validation_mae", "test_mae"],
                    {"validation_mae": "Validation 2024", "test_mae": "Test 2025"},
                    {"validation_mae": "#99F6E4", "test_mae": "#0F766E"}, "MAE (µg/m³)"))
    ui.verdict(f"<b>H1 {'supported' if best_val in ('random_forest', 'xgboost') else 'not supported'}.</b> "
               f"Best base model: {data.MODEL_SHORT[best_val]} "
               f"({m4.set_index('model').loc[best_val, 'validation_mae']:.3f}); the ensemble is lower still.",
               "yes" if best_val in ("random_forest", "xgboost") else "partial")
    ui.verdict(f"<b>H3 {'supported' if r_sel['test_mae'] < r_p['test_mae'] else 'not supported'}.</b> "
               f"{r_sel['test_mae']:.2f} vs {r_p['test_mae']:.2f} µg/m³, skill "
               f"{r_sel['test_mae_improvement_pct']:.1f}%. Linear Ridge does not beat persistence "
               f"({mc.set_index('model').loc['ridge', 'test_mae']:.2f}).",
               "yes" if r_sel["test_mae"] < r_p["test_mae"] else "no")

# ---------------------------------------------------------------- E2
with tabs[1]:
    hyp("<b>H2</b> Lag and rolling features reduce MAE versus basic features (weather excluded).")
    e2 = data.metric_table("E2_feature_engineering.csv")
    if not e2.empty:
        stage = st.segmented_control("Stage", ["validation_mae", "test_mae"], default="validation_mae",
                                     format_func=lambda s: "Validation 2024" if s.startswith("val") else "Test 2025",
                                     key="e2s")
        stage = stage or "validation_mae"
        e2 = e2.assign(model_label=e2["model"].map(data.MODEL_SHORT))
        cols = {"basic": "#CBD5E1", "lag": "#5EEAD4", "full": "#0F766E"}
        ui.plot(grouped(e2, "model_label", stage, "feature_config", ["basic", "lag", "full"], data.CONFIG_LABELS,
                        cols, "MAE (µg/m³)"))
        pv = e2.pivot_table(index="model", columns="feature_config", values=stage)
        better = int((pv[["lag", "full"]].min(axis=1) < pv["basic"]).sum())
        tot = len(pv)
        kind = "yes" if better == tot else ("partial" if better else "no")
        ui.verdict(f"<b>H2 {'supported' if kind == 'yes' else 'partially supported' if kind == 'partial' else 'not supported'}"
                   f"</b> on {'validation' if stage.startswith('val') else 'test'}: lag or full features beat basic "
                   f"for {better} of {tot} models; gains are small next to weather features.", kind)

# ---------------------------------------------------------------- E3
with tabs[2]:
    hyp("Full feature set versus a filtered subset selected on training folds.")
    e3 = data.metric_table("E3_feature_selection.csv")
    if not e3.empty:
        v = e3[e3["stage"] == "validation"].assign(model_label=lambda x: x["model"].map(data.MODEL_SHORT))
        a, b = st.columns([1.4, 1])
        with a:
            ui.plot(grouped(v, "model_label", "mae", "feature_config", ["full", "full_selected"], data.CONFIG_LABELS,
                            {"full": "#0F766E", "full_selected": "#A78BFA"}, "Validation MAE (µg/m³)"))
        with b:
            t = e3.pivot_table(index="model", columns=["feature_config", "stage"], values="mae")
            nf = e3.groupby(["model", "feature_config"])["n_features_used"].max().unstack()
            out = pd.DataFrame({"Model": [data.MODEL_SHORT[m] for m in nf.index],
                                "# features full": nf.get("full"), "# features selected": nf.get("full_selected"),
                                "Δ val MAE": (t[("full_selected", "validation")] - t[("full", "validation")]).values,
                                "Δ test MAE": (t[("full_selected", "test")] - t[("full", "test")]).values})
            ui.table(out, column_config={"Δ val MAE": st.column_config.NumberColumn(format="%+.3f"),
                                         "Δ test MAE": st.column_config.NumberColumn(format="%+.3f")})
            st.caption("Negative Δ = selection helped.")

# ---------------------------------------------------------------- E4
with tabs[3]:
    hyp("Library defaults versus RandomizedSearchCV with TimeSeriesSplit.")
    e4 = data.metric_table("E4_hyperparameter_optimization.csv")
    if not e4.empty:
        stage = st.segmented_control("Stage ", ["validation", "test"], default="validation", key="e4s") or "validation"
        z = e4[e4["stage"] == stage].pivot_table(index="model", columns="feature_config",
                                                  values="mae_change_tuned_minus_default")
        z = z.reindex([m for m in data.MODEL_ORDER if m in z.index])[[c for c in data.CONFIG_ORDER if c in z.columns]]
        z.index = [data.MODEL_SHORT[i] for i in z.index]
        z.columns = [data.CONFIG_LABELS[c].split(" (")[0] for c in z.columns]
        ui.plot(charts.heatmap(z, height=340, colorscale="RdBu", zmid=0, fmt="+.2f",
                               title="Change in MAE after tuning (negative = tuning helped)"))
        share = (e4[e4["stage"] == stage]["mae_change_tuned_minus_default"] < 0).mean() * 100
        ui.verdict(f"Tuning lowered {stage} MAE in <b>{share:.0f}%</b> of combinations, most for SVR and Ridge.",
                   "yes" if share >= 60 else "partial")

# ---------------------------------------------------------------- E5
with tabs[4]:
    hyp("<b>H5</b> Errors are much larger on elevated days, which are under-predicted.")
    e5 = data.metric_table("E5_elevated_pollution.csv")
    if not e5.empty:
        thr = st.segmented_control("Threshold", list(e5["threshold_name"].unique()), default=e5["threshold_name"].iloc[0],
                                   format_func=lambda t: {"train_q90": "Training 90th percentile",
                                                          "WHO_24h_15": "WHO 15 µg/m³"}.get(t, t)) \
            or e5["threshold_name"].iloc[0]
        d = e5[e5["threshold_name"] == thr].set_index("model").reindex([m for m in data.MODEL_ORDER
                                                                       if m in set(e5["model"])]).reset_index()
        fig = go.Figure()
        lab = d["model"].map(data.MODEL_SHORT)
        fig.add_trace(go.Bar(x=lab, y=d["mae_non_elevated"], name="Normal days", marker_color="#99F6E4"))
        fig.add_trace(go.Bar(x=lab, y=d["mae_elevated"], name="Elevated days", marker_color="#DC2626"))
        fig.add_trace(go.Scatter(x=lab, y=d["bias_elevated"], name="Bias on elevated days", mode="markers+lines",
                                 marker=dict(size=10, color="#0F172A"), yaxis="y2"))
        fig.update_layout(barmode="group", yaxis2=dict(overlaying="y", side="right", title="Bias (µg/m³)",
                                                       showgrid=False))
        fig.update_yaxes(title_text="MAE (µg/m³)")
        ui.plot(charts.style(fig, 400))
        n_el = int(d["n_elevated_days"].iloc[0])
        ratio = (d["mae_elevated"] / d["mae_non_elevated"]).mean()
        neg = bool((d["bias_elevated"] < 0).all())
        ui.verdict(f"<b>H5 {'supported' if neg and ratio > 2 else 'partially supported'}.</b> {n_el} elevated days: "
                   f"errors {ratio:.1f}× larger, bias {'negative for every model' if neg else 'mostly negative'}.",
                   "yes" if neg and ratio > 2 else "partial")
        f15 = paths.FIGURES / "15_elevated_pollution_predictions.png"
        if f15.exists():
            with st.expander("Figure 15"):
                st.image(str(f15))

# ---------------------------------------------------------------- E6
with tabs[5]:
    hyp("<b>H4</b> Recent PM2.5 history dominates; calendar features rank next.")
    perm = data.metric_table("E6_permutation_importance_test.csv")
    if not perm.empty:
        sel = data.model_selection().get("best", {}).get("model", "ensemble")
        d = perm[perm["model"] == sel].sort_values("mae_increase_mean", ascending=False)
        d["group"] = d["feature"].map(data.feature_group)
        top10 = d.head(10)
        g = top10["group"].value_counts()
        ui.plot(charts.importance_bars(top10.assign(label=top10["feature"].map(data.feature_text)),
                                       "mae_increase_mean", "label", "group", data.FEATURE_GROUPS, height=400,
                                       title=f"Top-10 permutation importance · {data.MODEL_SHORT[sel]}"))
        lead = g.index[0]
        ui.verdict(f"<b>H4 {'supported' if lead == 'PM2.5 history' else 'partially supported'}.</b> Top input: "
                   f"{data.feature_text(d.iloc[0]['feature'])}. Top 10: " + ", ".join(f"{k} {v}" for k, v in g.items())
                   + ".", "yes" if lead == "PM2.5 history" else "partial")
        st.page_link("views/explainability.py", label="Feature impact", icon=":material/arrow_forward:")

# ---------------------------------------------------------------- E7
with tabs[6]:
    hyp("Monthly, seasonal and heating-season errors on 2025.")
    mo = data.metric_table("E7_monthly_errors.csv")
    se = data.metric_table("E7_seasonal_errors.csv")
    he = data.metric_table("E7_heating_errors.csv")
    if not mo.empty:
        pick = st.pills("Models", [m for m in data.MODEL_ORDER if m in set(mo["model"])],
                        default=["ensemble", "persistence"], selection_mode="multi",
                        format_func=lambda m: data.MODEL_SHORT[m], key="e7p")
        fig = go.Figure()
        for m in pick or []:
            d = mo[mo["model"] == m]
            fig.add_trace(go.Scatter(x=[pd.Timestamp(2025, int(i), 1).strftime("%b") for i in d["month"]], y=d["mae"],
                                     name=data.MODEL_SHORT[m], mode="lines+markers",
                                     line=dict(color=data.MODEL_COLORS[m], width=2.5)))
        fig.update_yaxes(title_text="MAE (µg/m³)")
        ui.plot(charts.style(fig, 360, title="Monthly MAE, 2025"))
    a, b = st.columns(2)
    if not se.empty:
        with a:
            z = se.pivot_table(index="model", columns="season", values="mae")
            z = z.reindex([m for m in data.MODEL_ORDER if m in z.index])[[s for s in ["Winter", "Spring", "Summer",
                                                                                     "Autumn"] if s in z.columns]]
            z.index = [data.MODEL_SHORT[i] for i in z.index]
            ui.plot(charts.heatmap(z, height=320, colorscale="OrRd", title="MAE by season"))
    if not he.empty:
        with b:
            z = he.pivot_table(index="model", columns="heating", values="mae")
            z = z.reindex([m for m in data.MODEL_ORDER if m in z.index])
            z.index = [data.MODEL_SHORT[i] for i in z.index]
            ui.plot(charts.heatmap(z, height=320, colorscale="OrRd", title="MAE: heating vs non-heating season"))
    if not he.empty:
        e = he[he["model"] == "ensemble"].set_index("heating")
        if {"Heating", "Non-heating"} <= set(e.index):
            ui.verdict(f"Heating-season errors are <b>{e.loc['Heating', 'mae'] / e.loc['Non-heating', 'mae']:.1f}× "
                       f"larger</b> ({e.loc['Heating', 'mae']:.2f} vs {e.loc['Non-heating', 'mae']:.2f} µg/m³).", "partial")

# ---------------------------------------------------------------- Ablations
with tabs[7]:
    hyp("Evaluated for insight only, not used for model selection.")
    ar = data.all_results()
    t = ar[(ar["variant"] == "tuned") & ar["feature_config"].isin(["full", "full_aqi", "full_weather_past",
                                                                   "full_weather"])]
    stage = st.segmented_control("Stage  ", ["validation", "test"], default="test", key="abs") or "test"
    d = t[t["stage"] == stage].assign(model_label=lambda x: x["model"].map(data.MODEL_SHORT))
    d = d[d["model"] != "persistence"]
    ui.plot(grouped(d, "model_label", "mae", "feature_config", ["full", "full_aqi", "full_weather_past", "full_weather"],
                    data.CONFIG_LABELS, {"full": "#CBD5E1", "full_aqi": "#F9A8D4", "full_weather_past": "#93C5FD",
                                         "full_weather": "#0F766E"}, "MAE (µg/m³)", 420))
    st.caption("AQI adds almost nothing. Most of the weather gain comes from target-day weather. The benchmark "
               "uses reanalysis for target-day weather; live forecasts use the Open-Meteo forecast.")

ui.footer()
