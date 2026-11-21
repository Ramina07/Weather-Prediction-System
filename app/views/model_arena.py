from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import charts, data, ui

ui.page_header("Models", "Leaderboard",
               "Tuned on 2021–2023, selected on 2024, tested once on 2025.")

mc = data.model_comparison()
if mc.empty:
    st.error("`artifacts/metrics/model_comparison.csv` not found. Run the training pipeline first.")
    st.stop()

sel = data.model_selection().get("best", {})
best = sel.get("model", "ensemble")
order = [m for m in data.MODEL_ORDER if m in set(mc["model"])]
mc = mc.set_index("model").loc[order].reset_index()

# ------------------------------------------------------------------ model cards
cols = st.columns(len(order))
for c, m in zip(cols, order):
    r = mc[mc["model"] == m].iloc[0]
    tag = "<span class='tag'>SELECTED</span>" if m == best else ""
    color = data.MODEL_COLORS[m]
    c.markdown(f"""<div class='kpi{" accent" if m == best else ""}'>
        <div class='label' style='display:flex;align-items:center;gap:6px'>
          <span class='dot' style='background:{color};margin:0'></span>{data.MODEL_SHORT[m]}</div>
        <div class='value'>{r['test_mae']:.2f}{tag}</div>
        <div class='sub'>MAE · {data.MODEL_FAMILY[m].lower()}</div></div>""", unsafe_allow_html=True)

st.write("")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ranking", "Accuracy", "Feature sets", "Tuning", "Model guide"])

with tab1:
    metric = st.segmented_control("Metric", ["MAE", "RMSE", "R²", "sMAPE"], default="MAE")
    col = {"MAE": "mae", "RMSE": "rmse", "R²": "r2", "sMAPE": "smape_pct"}[metric or "MAE"]
    fig = go.Figure()
    labels = [data.MODEL_SHORT[m] for m in order]
    if col != "smape_pct":
        fig.add_trace(go.Bar(x=labels, y=mc[f"validation_{col}"], name="Validation 2024", marker_color="#99F6E4",
                             text=mc[f"validation_{col}"].round(2), textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=mc[f"test_{col}"], name="Test 2025",
                         marker_color=[data.MODEL_COLORS[m] for m in order],
                         text=mc[f"test_{col}"].round(2), textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=metric + ("" if col in ("r2", "smape_pct") else " (µg/m³)") +
                     (" (%)" if col == "smape_pct" else ""))
    ui.plot(charts.style(fig, 400))

    lb = pd.DataFrame({
        "Model": [data.MODEL_LABELS[m] + (" (selected)" if m == best else "") for m in order],
        "Features": [data.CONFIG_LABELS.get(c, c) for c in mc["feature_config"]],
        "CV MAE": mc["cv_mae"], "Val MAE": mc["validation_mae"], "Val R²": mc["validation_r2"],
        "Test MAE": mc["test_mae"], "Test RMSE": mc["test_rmse"], "Test R²": mc["test_r2"],
        "Bias": mc["test_bias"], "Skill vs persistence": mc["test_mae_improvement_pct"],
        "Tuning time (s)": mc["tuning_seconds"], "# features": mc["n_features_used"],
    })
    num = lambda f="%.3f": st.column_config.NumberColumn(format=f)  # noqa: E731
    ui.table(lb, column_config={
        "CV MAE": num(), "Val MAE": num(), "Val R²": num(), "Test MAE": num(), "Test RMSE": num(),
        "Test R²": num(), "Bias": num(), "Tuning time (s)": num("%.1f"),
        "Skill vs persistence": st.column_config.ProgressColumn(format="%.1f %%", min_value=-20, max_value=30)})
    shift = data.metric_table("eda_split_distribution_shift.csv").set_index("split")
    shift_txt = (f" (mean {shift.loc['test', 'mean']:.1f} vs {shift.loc['train', 'mean']:.1f} µg/m³ in training)"
                 if {"test", "train"} <= set(shift.index) else "")
    p = mc[mc["model"] == "persistence"].iloc[0]
    b = mc[mc["model"] == best].iloc[0]
    ui.verdict(f"Test MAE <b>{p['test_mae']:.2f} → {b['test_mae']:.2f} µg/m³</b> "
               f"(−{b['test_mae_improvement_pct']:.1f}% vs persistence), R² {p['test_r2']:.2f} → {b['test_r2']:.2f}. "
               f"The validation–test gap reflects a cleaner 2025{shift_txt}.")

with tab2:
    cfg_opts = [c for c in data.CONFIG_ORDER if c in set(data.test_predictions()["feature_config"])]
    cfg = st.selectbox("Feature configuration", cfg_opts,
                       index=cfg_opts.index("full_weather") if "full_weather" in cfg_opts else 0,
                       format_func=lambda c: data.CONFIG_LABELS.get(c, c))
    wide = data.selected_test_predictions(cfg)
    models_avail = [m for m in order if m in wide.columns]
    m = st.radio("Model", models_avail, horizontal=True, format_func=lambda x: data.MODEL_SHORT[x])
    a, b_ = st.columns(2)
    with a:
        st.markdown("**Predicted vs observed, 2025**")
        ui.plot(charts.scatter_actual_pred(wide["actual"], wide[m], data.MODEL_COLORS[m], data.MODEL_SHORT[m]))
    with b_:
        st.markdown("**Error distribution**")
        ui.plot(charts.violin_errors(wide, models_avail))
    st.markdown("**2025 timeline**")
    show = st.multiselect("Lines", models_avail, default=[m, "persistence"] if m != "persistence" else [m],
                          format_func=lambda x: data.MODEL_SHORT[x])
    ui.plot(charts.timeseries(wide, ["actual"] + show, rangeslider=True, who=True, height=430))

with tab3:
    ar = data.all_results()
    stage = st.segmented_control("Data", ["validation", "test"], default="validation",
                                 format_func=lambda s: {"validation": "Validation 2024", "test": "Test 2025"}[s])
    t = ar[(ar["variant"] == "tuned") & (ar["stage"] == (stage or "validation"))]
    z = t.pivot_table(index="model", columns="feature_config", values="mae")
    z = z.reindex([x for x in order if x in z.index])[[c for c in data.CONFIG_ORDER if c in z.columns]]
    z.index = [data.MODEL_SHORT[i] for i in z.index]
    z.columns = [data.CONFIG_LABELS[c].split(" (")[0] for c in z.columns]
    ui.plot(charts.heatmap(z, height=360, colorscale="Tealrose"))
    st.caption("MAE, µg/m³. Weather features give the largest gain for every model. AQI and past-weather sets "
               "are ablations, not used for selection.")

with tab4:
    cv = data.metric_table("cv_results.csv")
    if not cv.empty:
        cv["cv_mae"] = -cv["mean_test_score"]
        cv = cv[cv["cv_mae"] < cv["cv_mae"].quantile(0.98)]
        fcfg = st.selectbox("Feature configuration ", sorted(cv["feature_config"].unique()),
                            index=sorted(cv["feature_config"].unique()).index("full_weather")
                            if "full_weather" in set(cv["feature_config"]) else 0,
                            format_func=lambda c: data.CONFIG_LABELS.get(c, c))
        d = cv[cv["feature_config"] == fcfg]
        fig = go.Figure()
        for mm in [x for x in order if x in set(d["model"])]:
            dd = d[d["model"] == mm]
            fig.add_trace(go.Box(y=dd["cv_mae"], name=data.MODEL_SHORT[mm], marker_color=data.MODEL_COLORS[mm],
                                 boxpoints="all", jitter=0.4, pointpos=0))
        fig.update_yaxes(title_text="Cross-validation MAE of each candidate (µg/m³)")
        ui.plot(charts.style(fig, 400, legend=False))
        st.caption("Each point is one RandomizedSearchCV candidate (5-fold TimeSeriesSplit, 1-day gap). "
                   "The lowest point was kept.")
    e4 = data.metric_table("E4_hyperparameter_optimization.csv")
    if not e4.empty:
        e4v = e4[(e4["stage"] == "validation") & (e4["feature_config"] == "full_weather")]
        st.markdown("**Default vs tuned, validation MAE**")
        ui.table(e4v.assign(model=e4v["model"].map(data.MODEL_SHORT))[["model", "default", "tuned",
                                                                       "mae_change_tuned_minus_default"]],
                 column_config={"model": "Model", "default": st.column_config.NumberColumn("Default", format="%.3f"),
                                "tuned": st.column_config.NumberColumn("Tuned", format="%.3f"),
                                "mae_change_tuned_minus_default": st.column_config.NumberColumn(
                                    "Change", format="%+.3f")})
    st.markdown("**Selected hyperparameters**")
    for mm in order:
        if mm == "persistence":
            continue
        hp = mc[mc["model"] == mm].iloc[0]["best_hyperparameters"]
        try:
            hp = json.loads(hp)
        except (TypeError, ValueError):
            pass
        with st.expander(data.MODEL_LABELS[mm]):
            st.json({k.replace("model__", ""): v for k, v in hp.items()} if isinstance(hp, dict) else hp)

with tab5:
    for mm in order:
        with st.container(border=True):
            a, b_ = st.columns([1, 2.2])
            a.markdown(f"<div style='font-weight:800;font-size:1.1rem;color:{data.MODEL_COLORS[mm]}'>"
                       f"{data.MODEL_LABELS[mm]}</div><div style='color:#64748B'>{data.MODEL_FAMILY[mm]}</div>",
                       unsafe_allow_html=True)
            b_.markdown(data.MODEL_IDEA[mm])
            b_.code(data.CODE_MAP[mm], language="python")


ui.footer()
