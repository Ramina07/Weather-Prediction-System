"""Exploratory data analysis: statistics tables and figures 1-7 (+ supplements).

Every figure has a descriptive title, labelled axes with units and a legend
where needed. Short interpretations are written in the EDA notebook from the
generated statistics (``artifacts/metrics/eda_*.csv/json``).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Config, get_logger
from .data_validation import missing_gap_lengths
from .evaluation import UNIT, _plt, _save
from .feature_engineering import is_heating_season
from .preprocessing import load_clean

POLS = ["pm2_5", "pm10", "no2", "co", "o3", "so2", "nh3", "no"]
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Autocorrelation function with pairwise-complete observations (NaN-safe)."""
    x = np.asarray(x, float)
    out = [1.0]
    for k in range(1, nlags + 1):
        a, b = x[:-k], x[k:]
        m = np.isfinite(a) & np.isfinite(b)
        out.append(float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan)
    return np.array(out)


def run_eda(cfg: Config) -> dict:
    log = get_logger()
    plt = _plt()
    import seaborn as sns
    fdir, mdir = cfg.path("figures_dir"), cfg.path("metrics_dir")
    g = load_clean(cfg.path("clean_hourly_csv"), cfg.tz)
    obs = g[g["is_observed"]].copy()
    obs["month"] = obs["timestamp_local"].dt.month
    obs["year"] = obs["timestamp_local"].dt.year
    obs["dow"] = obs["timestamp_local"].dt.dayofweek
    obs["season"] = obs["month"].map(SEASONS)
    hs = cfg["evaluation"]["heating_season"]
    obs["heating"] = obs["timestamp_local"].map(lambda t: "Heating" if is_heating_season(t, hs["start"], hs["end"]) else "Non-heating")
    feats = pd.read_csv(cfg.path("features_csv"))
    daily = feats[["target_date", "target_pm25", "target_valid_hours", "split", "eligible"]].copy()
    daily["target_date"] = pd.to_datetime(daily["target_date"])
    out: dict = {}

    # descriptive statistics
    desc = obs[POLS + ["aqi"]].describe(percentiles=[.05, .25, .5, .75, .95, .99]).T
    desc["skew"] = obs[POLS + ["aqi"]].skew()
    desc["missing"] = g.loc[g["is_observed"], POLS + ["aqi"]].isna().sum()
    desc.to_csv(mdir / "eda_descriptive_statistics.csv")
    out["descriptive"] = desc.round(3).to_dict()

    # Fig 1 hourly + daily series
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(obs["timestamp_local"], obs["pm2_5"], lw=0.3, color="#9DB4D3", label="Hourly")
    dm = obs.groupby(obs["timestamp_local"].dt.date)["pm2_5"].mean()
    ax.plot(pd.to_datetime(dm.index), dm.values, lw=0.8, color="#1F3B73", label="Daily mean")
    ax.set_ylabel(f"PM2.5 ({UNIT})"); ax.set_xlabel("Local date (Asia/Almaty)")
    ax.set_title("Fig. 1. Historical PM2.5 for Astana (OpenWeatherMap API-provided values)")
    ax.legend()
    _save(fig, fdir / "01_pm25_timeseries.png")

    # Fig 2 missing values / completeness
    comp = g.groupby(g["timestamp_local"].dt.tz_localize(None).dt.to_period("M"))["is_observed"].mean() * 100
    pmv = g.groupby(g["timestamp_local"].dt.tz_localize(None).dt.to_period("M"))["pm2_5"].apply(lambda s: s.notna().mean() * 100)
    fig, ax = plt.subplots(figsize=(13, 3.8))
    x = comp.index.to_timestamp()
    ax.bar(x, comp.values, width=25, color="#4C72B0", label="Hours with a record")
    ax.plot(x, pmv.values, color="#C44E52", marker=".", label="Hours with valid PM2.5")
    ax.set_ylim(0, 105); ax.set_ylabel("Completeness (%)"); ax.set_xlabel("Month")
    ax.set_title("Fig. 2. Monthly data completeness of the hourly grid")
    ax.legend()
    _save(fig, fdir / "02_missing_values_completeness.png")
    gaps = missing_gap_lengths(g)
    out["gaps"] = {"n_gaps": int(len(gaps)), "max_gap_hours": int(gaps.max()) if len(gaps) else 0,
                   "total_missing_hours": int((~g["is_observed"]).sum())}
    comp.rename("pct_hours_observed").to_frame().assign(pct_valid_pm25=pmv.values).to_csv(mdir / "eda_monthly_completeness.csv")

    # Fig 3 distributions
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, p in zip(axes.ravel(), POLS):
        v = obs[p].dropna()
        ax.hist(v, bins=60, color="#4C72B0", alpha=0.85)
        ax.set_title(p.upper().replace("_", "."))
        ax.set_xlabel(f"{p} ({UNIT})"); ax.set_ylabel("Hours")
        ax.set_yscale("log")
    fig.suptitle("Fig. 3. Distributions of hourly pollutant concentrations (log-scaled counts)")
    _save(fig, fdir / "03_pollutant_distributions.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(obs["pm2_5"].dropna(), bins=80, color="#1F3B73")
    ax[0].set_xlabel(f"Hourly PM2.5 ({UNIT})"); ax[0].set_ylabel("Hours"); ax[0].set_title("Hourly PM2.5")
    dv = daily[daily["target_pm25"].notna()]["target_pm25"]
    ax[1].hist(dv, bins=60, color="#C44E52")
    ax[1].axvline(cfg["evaluation"]["who_24h_guideline"], color="k", ls="--", label="WHO 24-h AQG (15)")
    ax[1].set_xlabel(f"Daily mean PM2.5 ({UNIT})"); ax[1].set_ylabel("Days"); ax[1].set_title("Daily-mean PM2.5 (targets)")
    ax[1].legend()
    fig.suptitle("Fig. 3b. Distribution of PM2.5")
    _save(fig, fdir / "03b_pm25_distribution.png")

    # Fig 4 seasonal patterns
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    mo = obs.groupby("month")["pm2_5"].agg(["mean", "median"])
    axes[0].plot(mo.index, mo["mean"], marker="o", label="Mean"); axes[0].plot(mo.index, mo["median"], marker="s", label="Median")
    axes[0].set_xticks(range(1, 13)); axes[0].set_xlabel("Month"); axes[0].set_ylabel(f"PM2.5 ({UNIT})")
    axes[0].set_title("Monthly"); axes[0].legend()
    sns.boxplot(data=obs, x="season", y="pm2_5", order=["Winter", "Spring", "Summer", "Autumn"], ax=axes[1],
                showfliers=False, color="#9DB4D3")
    axes[1].set_xlabel("Season"); axes[1].set_ylabel(f"PM2.5 ({UNIT})"); axes[1].set_title("Seasonal (outliers hidden)")
    hr = obs.groupby(["heating", obs["timestamp_local"].dt.hour])["pm2_5"].mean().unstack(0)
    for c in hr.columns:
        axes[2].plot(hr.index, hr[c], marker=".", label=c)
    axes[2].set_xlabel("Local hour of day"); axes[2].set_ylabel(f"Mean PM2.5 ({UNIT})"); axes[2].set_title("Hour of day")
    axes[2].legend()
    dw = obs.groupby("dow")["pm2_5"].mean()
    axes[3].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dw.values, color="#55A868")
    axes[3].set_xlabel("Day of week"); axes[3].set_ylabel(f"Mean PM2.5 ({UNIT})"); axes[3].set_title("Day of week")
    fig.suptitle(f"Fig. 4. Temporal patterns of PM2.5 (heating season: {hs['start']} to {hs['end']})")
    _save(fig, fdir / "04_seasonal_patterns.png")
    out["monthly_mean"] = mo["mean"].round(3).to_dict()
    out["seasonal_mean"] = obs.groupby("season")["pm2_5"].mean().round(3).to_dict()
    out["heating_mean"] = obs.groupby("heating")["pm2_5"].agg(["mean", "median", "count"]).round(3).to_dict()
    out["dow_mean"] = dw.round(3).to_dict()
    yr = obs.groupby(["year", "month"])["pm2_5"].mean().unstack(0)
    yr.to_csv(mdir / "eda_monthly_mean_by_year.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in yr.columns:
        ax.plot(yr.index, yr[c], marker="o", label=str(c))
    ax.set_xticks(range(1, 13)); ax.set_xlabel("Month"); ax.set_ylabel(f"Mean PM2.5 ({UNIT})")
    ax.set_title("Fig. 4b. Monthly mean PM2.5 by year"); ax.legend(title="Year")
    _save(fig, fdir / "04b_monthly_trend_by_year.png")

    # Fig 5 correlation
    corr = obs[POLS + ["aqi"]].corr(method="spearman")
    corr.to_csv(mdir / "eda_spearman_correlation.csv")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Spearman ρ"})
    ax.set_title("Fig. 5. Spearman correlation between hourly pollutants")
    _save(fig, fdir / "05_correlation_heatmap.png")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    smp = obs.sample(min(len(obs), 8000), random_state=cfg.seed)
    for ax, p in zip(axes, ["pm10", "co", "no2", "o3"]):
        ax.scatter(smp[p], smp["pm2_5"], s=3, alpha=0.3)
        ax.set_xlabel(f"{p} ({UNIT})"); ax.set_ylabel(f"PM2.5 ({UNIT})")
        ax.set_title(f"PM2.5 vs {p} (ρ={corr.loc['pm2_5', p]:.2f})")
    fig.suptitle("Fig. 5b. PM2.5 versus other pollutants (random sample of hours)")
    _save(fig, fdir / "05b_pm25_vs_pollutants.png")

    # Fig 6 autocorrelation
    hourly_acf = acf(g["pm2_5"].to_numpy(), 168)
    dser = daily.set_index("target_date")["target_pm25"].asfreq("D")
    daily_acf = acf(dser.to_numpy(), 30)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    axes[0].bar(range(len(hourly_acf)), hourly_acf, width=1.0, color="#4C72B0")
    axes[0].set_xlabel("Lag (hours)"); axes[0].set_ylabel("Autocorrelation"); axes[0].set_title("Hourly PM2.5")
    axes[1].bar(range(len(daily_acf)), daily_acf, color="#C44E52")
    ci = 1.96 / np.sqrt(dser.notna().sum())
    axes[1].axhline(ci, ls="--", color="k", lw=0.8); axes[1].axhline(-ci, ls="--", color="k", lw=0.8)
    axes[1].set_xlabel("Lag (days)"); axes[1].set_ylabel("Autocorrelation"); axes[1].set_title("Daily-mean PM2.5 (±95% white-noise band)")
    fig.suptitle("Fig. 6. Autocorrelation and time dependence of PM2.5")
    _save(fig, fdir / "06_autocorrelation.png")
    out["acf_hourly"] = {str(k): round(float(hourly_acf[k]), 4) for k in (1, 6, 12, 24, 48, 72, 168)}
    out["acf_daily"] = {str(k): round(float(daily_acf[k]), 4) for k in (1, 2, 3, 7, 14, 30)}
    try:
        from statsmodels.tsa.stattools import adfuller
        s = dser.dropna()
        stat, p, *_ = adfuller(s.values, autolag="AIC")
        out["adf_daily"] = {"statistic": float(stat), "p_value": float(p), "n": int(len(s)),
                            "note": "ADF on daily means with gaps dropped (approximate)"}
    except Exception as exc:  # statsmodels optional at runtime
        out["adf_daily"] = f"skipped: {exc}"

    # Fig 7 split timeline + distribution shift
    el = daily[daily["eligible"]]
    fig, ax = plt.subplots(figsize=(13, 3.8))
    colors = {"train": "#4C72B0", "validation": "#DD8452", "test": "#55A868", "unused": "#BBBBBB"}
    for s, gdf in el.groupby("split"):
        ax.scatter(gdf["target_date"], gdf["target_pm25"], s=5, color=colors.get(s), label=f"{s} (n={len(gdf)})")
    ax.set_ylabel(f"Next-day mean PM2.5 ({UNIT})"); ax.set_xlabel("Target date")
    ax.set_title("Fig. 7. Chronological train (2021-2023) / validation (2024) / test (2025) split of eligible samples")
    ax.legend()
    _save(fig, fdir / "07_split_timeline.png")
    shift = el.groupby("split")["target_pm25"].describe()
    shift.to_csv(mdir / "eda_split_distribution_shift.csv")
    out["split_shift"] = shift.round(3).to_dict()
    try:
        from scipy.stats import ks_2samp
        tr = el[el.split == "train"]["target_pm25"]
        out["ks_train_vs"] = {s: {"statistic": float(r.statistic), "p_value": float(r.pvalue)} for s in ("validation", "test")
                              for r in [ks_2samp(tr, el[el.split == s]["target_pm25"])]}
    except Exception as exc:  # pragma: no cover
        out["ks_train_vs"] = f"skipped: {exc}"
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in ["train", "validation", "test"]:
        v = el[el.split == s]["target_pm25"]
        if len(v):
            ax.hist(v, bins=40, alpha=0.45, density=True, label=s, color=colors[s])
    ax.set_xlabel(f"Daily mean PM2.5 ({UNIT})"); ax.set_ylabel("Density")
    ax.set_title("Fig. 7b. Target distribution shift across splits"); ax.legend()
    _save(fig, fdir / "07b_split_distribution_shift.png")

    # outliers / extremes
    q99 = obs["pm2_5"].quantile(0.99)
    ext = obs[obs["pm2_5"] > q99]
    out["extremes"] = {"hourly_q99": float(q99), "n_hours_above_q99": int(len(ext)),
                       "share_in_heating_season_pct": float(100 * (ext["heating"] == "Heating").mean()) if len(ext) else None,
                       "n_flagged_robust_z": int(g["pm25_extreme_flag"].sum()) if "pm25_extreme_flag" in g else None,
                       "daily_days_above_who_15": int((dv > cfg["evaluation"]["who_24h_guideline"]).sum()),
                       "n_daily_targets": int(len(dv))}
    (mdir / "eda_summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info("EDA figures written to %s", fdir)
    return out
