# Mathematical formulations

All formulas are written in LaTeX (copy into Word via *Insert → Equation → LaTeX*, or into any LaTeX editor).
Each formula matches the implemented code; the "Code" line gives the function that computes it.

**Notation.** $h$ indexes hourly records labelled by the **start** of their hour (OpenWeatherMap `dt`).
$x^{(p)}_h$ is the API-provided concentration of pollutant $p$ (µg/m³) in hour $h$. $d$ is a local calendar date
(Asia/Almaty). The issue time for date $d$ is $\tau_d = d\;18{:}00$ local. A record labelled $h$ is available at
time $h + 1\,\mathrm{h}$; the set of records usable at $\tau_d$ is therefore $\{h : h + 1\,\mathrm{h} \le \tau_d\}$,
and the newest usable record is $h^{*}_d = \tau_d - 1\,\mathrm{h}$ (17:00).

---

## 1. Next-day PM2.5 target

$$
y_d \;=\; \bar{x}^{(\mathrm{PM2.5})}_{d+1}
\qquad\text{(defined only if } |H_{d+1}| \ge N_{\min}\text{)}
$$

* $y_d$: target for a forecast issued at $\tau_d$; $N_{\min}=20$ (configurable `forecast.target_min_valid_hours`).
* **Why:** a genuine day-ahead quantity: nothing from day $d+1$ is known at $\tau_d$.
* **Code:** `feature_engineering.daily_targets`, `build_dataset`.

## 2. Daily average aggregation

$$
\bar{x}^{(\mathrm{PM2.5})}_{D} \;=\; \frac{1}{|H_D|}\sum_{h \in H_D} x^{(\mathrm{PM2.5})}_h ,
\qquad H_D = \{h : \mathrm{date}_{\mathrm{local}}(h) = D,\; x^{(\mathrm{PM2.5})}_h \text{ observed}\}
$$

* $|H_D|$ is the number of valid hourly observations on local date $D$ (usually 24; 25 on 2024-02-29 when
  Kazakhstan moved from UTC+6 to UTC+5). Stored as `target_valid_hours`.
* **Code:** `feature_engineering.daily_targets`.

## 3. Missing-value handling

Missing hours are **not** interpolated. A windowed statistic is computed only when the share of valid hours
reaches a coverage threshold $c$ (= 0.75, `forecast.rolling_min_coverage`); otherwise the feature is missing.
Missing feature values are then imputed inside each model pipeline with the training-fold median:

$$
\tilde{z}_{ij} =
\begin{cases}
z_{ij}, & z_{ij} \text{ observed} \\
\operatorname{median}\{z_{kj} : k \in \mathcal{T},\; z_{kj} \text{ observed}\}, & \text{otherwise}
\end{cases}
$$

* $\mathcal{T}$: indices of the **training** samples of the current fold. Targets are never imputed.
* **Code:** `feature_engineering._cov_ok`; `sklearn.impute.SimpleImputer(strategy="median")` in `models.build_pipeline`.

## 4. Standardization (Ridge and SVR only)

$$
z'_{ij} = \frac{\tilde{z}_{ij} - \mu_j}{\sigma_j},
\qquad \mu_j = \frac{1}{|\mathcal{T}|}\sum_{k\in\mathcal{T}} \tilde{z}_{kj},
\quad \sigma_j = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{k\in\mathcal{T}} (\tilde{z}_{kj}-\mu_j)^2}
$$

* $\mu_j,\sigma_j$ are estimated on the training fold only and reused for validation/test rows.
* **Code:** `sklearn.preprocessing.StandardScaler` in `models.build_pipeline`.

## 5. Lag features

$$
\mathrm{lag}^{(p)}_{L}(d) = x^{(p)}_{\,h^{*}_d - L}, \qquad L \in \{24, 48, 72\}\ \text{hours}
$$

and the last available value $\mathrm{last}^{(p)}(d) = x^{(p)}_{h^{*}_d - a}$ with the smallest age
$a \in \{0,1,2\}$ for which the record is observed (`last_value_max_age_hours` = 3).

* **Code:** `feature_engineering.compute_features` (`at`, `_last_valid`).

## 6. Rolling means

$$
\mathrm{mean}^{(p)}_{w}(d) = \frac{1}{|V_w|}\sum_{h \in V_w} x^{(p)}_h,
\qquad V_w = \{h : h^{*}_d - w < h \le h^{*}_d,\; x^{(p)}_h \text{ observed}\},
\qquad \text{if } \tfrac{|V_w|}{w} \ge c
$$

* $w \in \{6,12,24,48,72\}$ h for PM2.5; $w=24$ h for the other pollutants. Every window ends at $h^{*}_d$, i.e.
  strictly before the issue time.
* **Code:** `feature_engineering._mean`.

## 7. Rolling standard deviation (and min/max, trend)

$$
\mathrm{sd}_{24}(d) = \sqrt{\frac{1}{|V_{24}|-1}\sum_{h \in V_{24}} \left(x_h - \mathrm{mean}_{24}(d)\right)^2}
$$

Trend: ordinary-least-squares slope of PM2.5 against hour index $t$ over the last 24 h,
$\beta_{24}(d) = \dfrac{\sum (t - \bar t)(x_t - \bar x)}{\sum (t-\bar t)^2}$ (µg/m³ per hour).

* **Code:** `feature_engineering._std`, `_min`, `_max`, `_slope`.

## 8. Ridge Regression

$$
\hat{\boldsymbol\beta} = \arg\min_{\beta_0,\boldsymbol\beta}\;
\sum_{i=1}^{n}\left(y_i - \beta_0 - \mathbf{z}_i^{\top}\boldsymbol\beta\right)^2 + \alpha\,\lVert\boldsymbol\beta\rVert_2^2 ,
\qquad \hat{y} = \hat\beta_0 + \mathbf{z}^{\top}\hat{\boldsymbol\beta}
$$

* $\alpha > 0$ controls L2 shrinkage (tuned on a log scale in $[10^{-3},10^{3}]$). Closed form:
  $\hat{\boldsymbol\beta} = (Z^{\top}Z + \alpha I)^{-1} Z^{\top}\mathbf{y}$ for centred data.
* **Why:** stable coefficients under the strong collinearity of lag/rolling features (Hoerl & Kennard, 1970).
* **Code:** `models.make_estimator("ridge")`.

## 9. Random Forest prediction aggregation

$$
\hat{y}(\mathbf{z}) = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{z};\,\Theta_b)
$$

* $T_b$: regression tree grown on a bootstrap sample (drawn with replacement) of the training data, with a random
  subset of `max_features` candidate features at each split; each leaf predicts the mean target of its samples;
  splits minimise the within-node squared error. $B$ = `n_estimators` (Breiman, 2001).
* **Code:** `models.make_estimator("random_forest")`.

## 10. XGBoost additive boosting

$$
\hat{y}^{(K)}_i = \sum_{k=1}^{K} \eta\, f_k(\mathbf{z}_i),
\qquad
\mathcal{L} = \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2 + \sum_{k=1}^{K}\Omega(f_k),
\qquad
\Omega(f) = \gamma T + \tfrac{1}{2}\lambda \lVert \mathbf{w} \rVert_2^2
$$

At iteration $k$ the new tree minimises the second-order approximation
$\sum_i \left[g_i f_k(\mathbf{z}_i) + \tfrac12 h_i f_k(\mathbf{z}_i)^2\right] + \Omega(f_k)$ with
$g_i = \partial_{\hat y}\ell$, $h_i = \partial^2_{\hat y}\ell$; the optimal leaf weight is
$w_j^{*} = -\dfrac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i + \lambda}$.

* $\eta$ = `learning_rate`, $T$ = number of leaves, $\lambda$ = `reg_lambda`, $\mathbf{w}$ = leaf weights
  (Chen & Guestrin, 2016). The base score (initial prediction) is omitted for brevity.
* **Code:** `models.make_estimator("xgboost")` (objective `reg:squarederror`).

## 11. SVR ε-insensitive loss

$$
\min_{\mathbf{w},b,\boldsymbol\xi,\boldsymbol\xi^{*}}\;
\tfrac12\lVert\mathbf{w}\rVert^2 + C\sum_{i=1}^{n}(\xi_i + \xi_i^{*})
\quad\text{s.t.}\quad
\begin{cases}
y_i - \mathbf{w}^{\top}\phi(\mathbf{z}_i) - b \le \varepsilon + \xi_i\\
\mathbf{w}^{\top}\phi(\mathbf{z}_i) + b - y_i \le \varepsilon + \xi_i^{*}\\
\xi_i,\xi_i^{*} \ge 0
\end{cases}
$$

Equivalently the loss is $\ell_\varepsilon(r) = \max(0, |r| - \varepsilon)$. Prediction:
$\hat y(\mathbf{z}) = \sum_i (\alpha_i - \alpha_i^{*}) K(\mathbf{z}_i,\mathbf{z}) + b$ with the RBF kernel
$K(\mathbf{z},\mathbf{z}') = \exp(-\gamma\lVert\mathbf{z}-\mathbf{z}'\rVert^2)$ (Smola & Schölkopf, 2004).

* $C$: regularization trade-off; $\varepsilon$: width of the no-penalty tube (µg/m³); $\gamma$: kernel bandwidth.
* **Code:** `models.make_estimator("svr")`.

## 12. MAE

$$\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n} \lvert y_i - \hat y_i \rvert \quad [\mu g/m^3]$$

**Code:** `evaluation.mae` (primary model-selection metric).

## 13. RMSE

$$\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat y_i)^2} \quad [\mu g/m^3]$$

**Code:** `evaluation.rmse`.

## 14. Coefficient of determination

$$R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}$$

$\bar y$ is the mean of the evaluated targets. $R^2$ can be negative and is **not** a percentage of correct
predictions. **Code:** `evaluation.r2`.

## 15. sMAPE (and guarded MAPE)

$$
\mathrm{sMAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\frac{2\,\lvert y_i-\hat y_i\rvert}{\lvert y_i\rvert+\lvert\hat y_i\rvert+\epsilon},
\qquad
\mathrm{MAPE}_{\ge 1} = \frac{100\%}{|S|}\sum_{i\in S}\frac{\lvert y_i-\hat y_i\rvert}{\lvert y_i\rvert},\;
S=\{i: \lvert y_i\rvert \ge 1\,\mu g/m^3\}
$$

$\epsilon = 10^{-6}$. MAPE is restricted to targets ≥ 1 µg/m³ because it explodes near zero.
**Code:** `evaluation.smape`, `evaluation.mape`.

## 16. Persistence baseline

$$\hat y^{\mathrm{pers}}_d = \mathrm{mean}^{(\mathrm{PM2.5})}_{24}(d) = \frac{1}{|V_{24}|}\sum_{h=h^{*}_d-23}^{h^{*}_d} x^{(\mathrm{PM2.5})}_h$$

The trailing 24-h mean ending at 17:00 on day $d$ — fully available at 18:00 (unlike the calendar-day mean of $d$).
**Code:** `models.PersistenceForecaster`.

## 17. Relative improvement against persistence

$$
S_{\mathrm{MAE}} = 1 - \frac{\mathrm{MAE}_{\mathrm{model}}}{\mathrm{MAE}_{\mathrm{pers}}},
\qquad \Delta_{\%} = 100\, S_{\mathrm{MAE}}
$$

$S>0$: the model beats persistence on the same dates; $S<0$: persistence is better.
**Code:** `evaluation.skill_vs_persistence`, column `test_mae_improvement_pct` in `model_comparison.csv`.

## 18. Empirical prediction interval (optional output)

$$
\left[\hat y + q_{0.05}(r),\; \hat y + q_{0.95}(r)\right],\qquad r = y - \hat y \text{ on the 2024 validation year}
$$

Residuals $r$ come from the train-only fit; the interval is approximate (assumes residuals in operation resemble
validation residuals) and is clipped at 0. **Code:** `training._bundle`, `prediction.make_forecast`.
