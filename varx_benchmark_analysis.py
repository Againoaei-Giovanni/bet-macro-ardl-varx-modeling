#!/usr/bin/env python3
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss

# Keep output clean from benign numerical warnings during matrix ops.
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\.linalg\._linalg")
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module=r"statsmodels\.tsa\.vector_ar\.var_model"
)


INPUT_FILE = Path("Date licenta actualizare.xlsx")
OUTPUT_EXCEL = Path("rezultate_pasi_1_6_varx_all_endog_fx_control.xlsx")
PLOTS_DIR = Path("plots")
DEPENDENT_VAR = "r_BET"
VAR_LAG_MIN = 1
VAR_LAG_MAX = 12
VAR_LAG_SELECTION_CRITERION = "bic"
ENABLE_CISS = True
CISS_FILE_GLOB = "ECB Data Portal long_*.xlsx"
CISS_SHEET_NAME = "DATA(CISS)"
CISS_TRANSFORMED_COL = "CISS"
ENABLE_VAR_SHOCK_DUMMIES = False
VAR_SHOCK_DATES = [
    "2008-09-30",
    "2008-10-31",
    "2008-11-30",
    "2020-03-31",
    "2020-04-30",
    "2020-05-31",
    "2022-02-28",
    "2022-03-31",
]
VAR_MODEL_MODE = "varx"  # options: "var", "varx"
VAR_EXCLUDE_ENDOG = []  # used in VAR mode
VARX_ENDOG_COLS = [
    "r_BET",
    "r_Masa monetara",
    "INF",
    "UNEMP",
    "Y10- Romania",
    "r_MSCI",
    "CISS",
]
VARX_EXOG_COLS = ["r_FX"]


def build_shock_dummies(date_series: pd.Series, shock_dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    exog = pd.DataFrame(index=date_series.index)
    for d in shock_dates:
        dt = pd.Timestamp(d)
        exog[f"D_{dt.strftime('%Y%m%d')}"] = (date_series == dt).astype(float)

    meta = pd.DataFrame(
        {
            "Metric": ["ShockDummiesEnabled", "ShockDummyCount", "ShockDates"],
            "Value": [
                True,
                int(len(shock_dates)),
                ", ".join(shock_dates),
            ],
        }
    )
    return exog, meta


def find_latest_ciss_file(base_dir: Path) -> Path | None:
    candidates = sorted(
        base_dir.glob(CISS_FILE_GLOB),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def prepare_ciss_for_merge(base_dates: pd.Series, ciss_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ciss_raw = pd.read_excel(ciss_file, sheet_name=CISS_SHEET_NAME)
    required_cols = {"DATE", "OBS.VALUE"}
    if not required_cols.issubset(set(ciss_raw.columns)):
        raise ValueError(
            "Fisierul CISS nu are coloanele necesare DATE si OBS.VALUE in foaia DATA(CISS)."
        )

    ciss_raw["DATE"] = pd.to_datetime(ciss_raw["DATE"], errors="coerce")
    ciss_raw["OBS.VALUE"] = pd.to_numeric(ciss_raw["OBS.VALUE"], errors="coerce")
    ciss_raw = ciss_raw.dropna(subset=["DATE", "OBS.VALUE"]).sort_values("DATE")

    ciss_raw["Month"] = ciss_raw["DATE"].dt.to_period("M")
    monthly = ciss_raw.groupby("Month", as_index=False)["OBS.VALUE"].last()
    monthly["Date"] = monthly["Month"].dt.to_timestamp("M")
    monthly = monthly.rename(columns={"OBS.VALUE": "CISS_level"})[["Date", "CISS_level"]]

    # Make CISS comparable to existing differenced/return-like transformed series.
    monthly[CISS_TRANSFORMED_COL] = monthly["CISS_level"].diff()

    ciss_aligned = pd.DataFrame({"Date": base_dates}).merge(monthly, on="Date", how="left")
    ciss_for_merge = ciss_aligned[["Date", CISS_TRANSFORMED_COL]].copy()

    ciss_meta = pd.DataFrame(
        {
            "Metric": [
                "CISS_SourceFile",
                "CISS_Sheet",
                "CISS_Transformation",
                "CISS_BaseRows",
                "CISS_MissingAfterMerge",
                "CISS_AlignedDateMin",
                "CISS_AlignedDateMax",
            ],
            "Value": [
                str(ciss_file.name),
                CISS_SHEET_NAME,
                "Monthly last observation, then first difference",
                int(len(ciss_aligned)),
                int(ciss_aligned[CISS_TRANSFORMED_COL].isna().sum()),
                ciss_aligned["Date"][ciss_aligned[CISS_TRANSFORMED_COL].notna()].min().date()
                if ciss_aligned[CISS_TRANSFORMED_COL].notna().any()
                else pd.NaT,
                ciss_aligned["Date"][ciss_aligned[CISS_TRANSFORMED_COL].notna()].max().date()
                if ciss_aligned[CISS_TRANSFORMED_COL].notna().any()
                else pd.NaT,
            ],
        }
    )
    return ciss_for_merge, ciss_meta


def load_and_prepare_data(path: Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    df = pd.read_excel(path)

    if "Date" not in df.columns:
        raise ValueError("Coloana 'Date' nu exista in fisierul Excel.")

    # Datele sunt stocate ca serial Excel; le convertim explicit in datetime.
    if pd.api.types.is_numeric_dtype(df["Date"]):
        base = pd.Timestamp("1899-12-30")
        df["Date"] = base + pd.to_timedelta(df["Date"], unit="D")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)
    ciss_meta = pd.DataFrame({"Metric": ["CISS_Status"], "Value": ["Not requested"]})

    if ENABLE_CISS:
        ciss_file = find_latest_ciss_file(path.parent)
        if ciss_file is not None:
            ciss_for_merge, ciss_meta = prepare_ciss_for_merge(df["Date"], ciss_file)
            df = df.merge(ciss_for_merge, on="Date", how="left")
        else:
            ciss_meta = pd.DataFrame(
                {
                    "Metric": ["CISS_Status", "CISS_FilePattern"],
                    "Value": ["No CISS file found", CISS_FILE_GLOB],
                }
            )

    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = [c for c in df.columns if c != "Date"]
    return df, numeric_cols, ciss_meta


def build_data_qa(
    df: pd.DataFrame, numeric_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date_series = df["Date"].dropna()
    month_index = date_series.dt.year * 12 + date_series.dt.month
    month_step = month_index.diff()
    irregular_gap_mask = month_step.ne(1) & month_step.notna()
    irregular_gaps = pd.DataFrame(
        {
            "Date": date_series[irregular_gap_mask].values,
            "MonthStepFromPrevious": month_step[irregular_gap_mask].values,
        }
    )

    qa_summary = pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns",
                "DateMin",
                "DateMax",
                "DuplicateDates",
                "DateIsSortedAscending",
                "TotalMissingValues",
                "IrregularMonthlyGapsCount",
            ],
            "Value": [
                int(df.shape[0]),
                int(df.shape[1]),
                date_series.min().date() if not date_series.empty else pd.NaT,
                date_series.max().date() if not date_series.empty else pd.NaT,
                int(df["Date"].duplicated().sum()),
                bool(df["Date"].is_monotonic_increasing),
                int(df.isna().sum().sum()),
                int(irregular_gaps.shape[0]),
            ],
        }
    )

    missing_by_col = (
        df.isna()
        .sum()
        .rename("MissingCount")
        .to_frame()
        .assign(MissingPct=lambda x: (x["MissingCount"] / len(df)) * 100)
        .reset_index()
        .rename(columns={"index": "Column"})
    )

    outlier_summary_rows: list[dict[str, float | int | str]] = []
    outlier_points_rows: list[dict[str, object]] = []
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            outlier_summary_rows.append(
                {
                    "Variable": col,
                    "IQR_Outliers": np.nan,
                    "IQR_Lower": np.nan,
                    "IQR_Upper": np.nan,
                }
            )
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        count_outliers = int(mask.sum())

        outlier_summary_rows.append(
            {
                "Variable": col,
                "IQR_Outliers": count_outliers,
                "IQR_Lower": float(lower),
                "IQR_Upper": float(upper),
            }
        )

        if count_outliers > 0:
            points = df.loc[mask, ["Date", col]].copy()
            points["Variable"] = col
            points = points.rename(columns={col: "Value"})
            outlier_points_rows.extend(points.to_dict(orient="records"))

    outlier_summary = pd.DataFrame(outlier_summary_rows)
    outlier_points = pd.DataFrame(outlier_points_rows)

    return qa_summary, missing_by_col, irregular_gaps, outlier_summary, outlier_points


def build_descriptive_stats(
    df: pd.DataFrame, numeric_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = pd.DataFrame(index=numeric_cols)
    for col in numeric_cols:
        s = df[col].dropna()
        stats.loc[col, "count"] = s.count()
        stats.loc[col, "mean"] = s.mean()
        stats.loc[col, "std"] = s.std(ddof=1)
        stats.loc[col, "min"] = s.min()
        stats.loc[col, "q25"] = s.quantile(0.25)
        stats.loc[col, "median"] = s.median()
        stats.loc[col, "q75"] = s.quantile(0.75)
        stats.loc[col, "max"] = s.max()
        stats.loc[col, "skewness"] = s.skew()
        stats.loc[col, "kurtosis"] = s.kurt()
    stats = stats.reset_index().rename(columns={"index": "Variable"})

    corr = df[numeric_cols].corr()
    corr = corr.reset_index().rename(columns={"index": "Variable"})
    return stats, corr


def run_stationarity_tests(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in numeric_cols:
        s = df[col].dropna()
        row: dict[str, object] = {"Variable": col}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_stat, adf_p, adf_lags, adf_nobs, adf_cv, adf_ic = adfuller(
                s, regression="c", autolag="AIC"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, kpss_lags, kpss_cv = kpss(
                s, regression="c", nlags="auto"
            )

        adf_stationary = adf_p < 0.05
        kpss_stationary = kpss_p > 0.05

        if adf_stationary and kpss_stationary:
            decision = "Stationary (ADF+KPSS)"
        elif (not adf_stationary) and (not kpss_stationary):
            decision = "Non-stationary (ADF+KPSS)"
        else:
            decision = "Mixed evidence"

        row.update(
            {
                "ADF_stat": adf_stat,
                "ADF_pvalue": adf_p,
                "ADF_lags": adf_lags,
                "ADF_nobs": adf_nobs,
                "ADF_cv_1pct": adf_cv["1%"],
                "ADF_cv_5pct": adf_cv["5%"],
                "ADF_cv_10pct": adf_cv["10%"],
                "KPSS_stat": kpss_stat,
                "KPSS_pvalue": kpss_p,
                "KPSS_lags": kpss_lags,
                "KPSS_cv_10pct": kpss_cv["10%"],
                "KPSS_cv_5pct": kpss_cv["5%"],
                "KPSS_cv_2_5pct": kpss_cv["2.5%"],
                "KPSS_cv_1pct": kpss_cv["1%"],
                "Decision_5pct": decision,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def newey_west_lags(nobs: int) -> int:
    # Rule-of-thumb bandwidth used often in monthly macro-finance settings.
    return max(1, int(np.floor(4 * (nobs / 100.0) ** (2.0 / 9.0))))


def result_to_coef_table(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    model_name: str,
    cov_type: str,
    hac_maxlags: int | None = None,
) -> pd.DataFrame:
    exog_names = result.model.exog_names
    conf_int = np.asarray(result.conf_int())

    out = pd.DataFrame(
        {
            "Model": model_name,
            "CovType": cov_type,
            "HAC_maxlags": hac_maxlags if hac_maxlags is not None else np.nan,
            "Variable": exog_names,
            "Coef": np.asarray(result.params),
            "StdErr": np.asarray(result.bse),
            "t_or_z": np.asarray(result.tvalues),
            "PValue": np.asarray(result.pvalues),
            "CI_Lower_95": conf_int[:, 0],
            "CI_Upper_95": conf_int[:, 1],
        }
    )
    return out


def build_vif_table(x: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, col in enumerate(x.columns):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                try:
                    vif = variance_inflation_factor(x.values, i)
                except Exception:
                    vif = np.nan
        rows.append({"Model": model_name, "Variable": col, "VIF": vif})
    return pd.DataFrame(rows)


def run_ols_benchmark(
    df: pd.DataFrame, dependent_var: str
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    predictors = [c for c in df.columns if c not in ["Date", dependent_var]]

    # Static benchmark: contemporaneous regressors.
    static_data = df[["Date", dependent_var] + predictors].dropna().copy()
    x_static = sm.add_constant(static_data[predictors], has_constant="add")
    y_static = static_data[dependent_var]
    ols_static = sm.OLS(y_static, x_static).fit()
    nw_static = newey_west_lags(int(ols_static.nobs))
    hac_static = ols_static.get_robustcov_results(cov_type="HAC", maxlags=nw_static)

    # Dynamic benchmark: add 1 lag for dependent and each predictor.
    dyn = df[["Date", dependent_var] + predictors].copy()
    dep_lag = f"{dependent_var}_l1"
    dyn[dep_lag] = dyn[dependent_var].shift(1)
    lagged_predictors: list[str] = []
    for col in predictors:
        lag_col = f"{col}_l1"
        dyn[lag_col] = dyn[col].shift(1)
        lagged_predictors.append(lag_col)

    dynamic_predictors = [dep_lag] + predictors + lagged_predictors
    dynamic_data = dyn[["Date", dependent_var] + dynamic_predictors].dropna().copy()
    x_dynamic = sm.add_constant(dynamic_data[dynamic_predictors], has_constant="add")
    y_dynamic = dynamic_data[dependent_var]
    ols_dynamic = sm.OLS(y_dynamic, x_dynamic).fit()
    nw_dynamic = newey_west_lags(int(ols_dynamic.nobs))
    hac_dynamic = ols_dynamic.get_robustcov_results(cov_type="HAC", maxlags=nw_dynamic)

    coef_static_ols = result_to_coef_table(ols_static, "OLS_Static", "nonrobust")
    coef_static_hac = result_to_coef_table(
        hac_static, "OLS_Static", "HAC", hac_maxlags=nw_static
    )
    coef_dynamic_ols = result_to_coef_table(ols_dynamic, "OLS_Dynamic_L1", "nonrobust")
    coef_dynamic_hac = result_to_coef_table(
        hac_dynamic, "OLS_Dynamic_L1", "HAC", hac_maxlags=nw_dynamic
    )

    def diagnostics_row(
        model_name: str,
        model_ols: sm.regression.linear_model.RegressionResultsWrapper,
        hac_lags: int,
    ) -> dict[str, object]:
        resid = model_ols.resid
        bg_lags = max(1, min(12, int(model_ols.nobs // 10)))
        bg_lm_stat, bg_lm_pvalue, bg_f_stat, bg_f_pvalue = acorr_breusch_godfrey(
            model_ols, nlags=bg_lags
        )
        white_lm_stat, white_lm_pvalue, white_f_stat, white_f_pvalue = het_white(
            resid, model_ols.model.exog
        )
        jb_stat, jb_pvalue, jb_skew, jb_kurt = jarque_bera(resid)

        return {
            "Model": model_name,
            "NObs": int(model_ols.nobs),
            "R2": model_ols.rsquared,
            "Adj_R2": model_ols.rsquared_adj,
            "AIC": model_ols.aic,
            "BIC": model_ols.bic,
            "HAC_maxlags": hac_lags,
            "DurbinWatson": durbin_watson(resid),
            "BG_Lags": bg_lags,
            "BG_LM_Stat": bg_lm_stat,
            "BG_LM_PValue": bg_lm_pvalue,
            "BG_F_Stat": bg_f_stat,
            "BG_F_PValue": bg_f_pvalue,
            "White_LM_Stat": white_lm_stat,
            "White_LM_PValue": white_lm_pvalue,
            "White_F_Stat": white_f_stat,
            "White_F_PValue": white_f_pvalue,
            "JB_Stat": jb_stat,
            "JB_PValue": jb_pvalue,
            "JB_Skew": jb_skew,
            "JB_Kurtosis": jb_kurt,
        }

    diagnostics = pd.DataFrame(
        [
            diagnostics_row("OLS_Static", ols_static, nw_static),
            diagnostics_row("OLS_Dynamic_L1", ols_dynamic, nw_dynamic),
        ]
    )

    fitted_static = pd.DataFrame(
        {
            "Date": static_data["Date"].values,
            "Actual": y_static.values,
            "Fitted": ols_static.fittedvalues.values,
            "Residual": ols_static.resid.values,
        }
    )
    fitted_dynamic = pd.DataFrame(
        {
            "Date": dynamic_data["Date"].values,
            "Actual": y_dynamic.values,
            "Fitted": ols_dynamic.fittedvalues.values,
            "Residual": ols_dynamic.resid.values,
        }
    )

    vif_static = build_vif_table(static_data[predictors], "OLS_Static")
    vif_dynamic = build_vif_table(dynamic_data[dynamic_predictors], "OLS_Dynamic_L1")

    model_specs = pd.DataFrame(
        {
            "Model": ["OLS_Static", "OLS_Dynamic_L1"],
            "DependentVariable": [dependent_var, dependent_var],
            "Specification": [
                (
                    f"{dependent_var}_t = c + "
                    + " + ".join([f"{x}_t" for x in predictors])
                    + " + e_t"
                ),
                (
                    f"{dependent_var}_t = c + {dependent_var}_t-1 + "
                    + " + ".join([f"{x}_t" for x in predictors])
                    + " + "
                    + " + ".join([f"{x}_t-1" for x in predictors])
                    + " + e_t"
                ),
            ],
        }
    )

    return (
        model_specs,
        coef_static_ols,
        coef_static_hac,
        coef_dynamic_ols,
        coef_dynamic_hac,
        diagnostics,
        fitted_static,
        fitted_dynamic,
        vif_static,
        vif_dynamic,
    )


def run_var_workflow(
    df: pd.DataFrame,
    endog_cols: list[str],
    exog: pd.DataFrame | None = None,
    lag_min: int = VAR_LAG_MIN,
    lag_max: int = VAR_LAG_MAX,
    selection_criterion: str = VAR_LAG_SELECTION_CRITERION,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    working = df[["Date"] + endog_cols].copy()
    exog_cols: list[str] = []
    if exog is not None and not exog.empty:
        exog_cols = list(exog.columns)
        working = working.join(exog)
    working = working.dropna().copy()

    dates = working["Date"].copy()
    data = working[endog_cols].copy()
    exog_used = working[exog_cols].copy() if exog_cols else pd.DataFrame(index=working.index)

    lag_rows: list[dict[str, object]] = []

    for lag in range(lag_min, lag_max + 1):
        try:
            fit = VAR(data, exog=exog_used if exog_cols else None).fit(lag, trend="c")
        except Exception:
            continue

        whiten_nlags = min(24, max(12, lag + 1), int(fit.nobs - 1))
        if whiten_nlags <= lag:
            white_stat = np.nan
            white_p = np.nan
            white_df = np.nan
        else:
            try:
                white_test = fit.test_whiteness(nlags=whiten_nlags)
                white_stat = white_test.test_statistic
                white_p = white_test.pvalue
                white_df = white_test.df
            except Exception:
                white_stat = np.nan
                white_p = np.nan
                white_df = np.nan

        lag_rows.append(
            {
                "Lag": lag,
                "NObs": int(fit.nobs),
                "AIC": fit.aic,
                "BIC": fit.bic,
                "HQIC": fit.hqic,
                "FPE": fit.fpe,
                "LogLik": fit.llf,
                "Whiteness_nlags": whiten_nlags,
                "Whiteness_Stat": white_stat,
                "Whiteness_PValue": white_p,
                "Whiteness_df": white_df,
            }
        )

    lag_criteria = pd.DataFrame(lag_rows).sort_values("Lag").reset_index(drop=True)
    if lag_criteria.empty:
        raise RuntimeError("Nu s-a putut estima niciun model VAR pe intervalul de lag setat.")

    criterion_map = {"aic": "AIC", "bic": "BIC", "hqic": "HQIC", "fpe": "FPE"}
    selected_lags = {
        key: int(lag_criteria.loc[lag_criteria[col].idxmin(), "Lag"])
        for key, col in criterion_map.items()
    }

    criterion_key = selection_criterion.lower()
    if criterion_key not in selected_lags:
        criterion_key = "bic"
    selected_lag = selected_lags[criterion_key]
    var_res = VAR(data, exog=exog_used if exog_cols else None).fit(selected_lag, trend="c")

    eq_names = list(var_res.params.columns)
    reg_names = list(var_res.params.index)
    coef_rows: list[dict[str, object]] = []
    for reg in reg_names:
        for eq in eq_names:
            coef_rows.append(
                {
                    "Equation": eq,
                    "Regressor": reg,
                    "Coef": float(var_res.params.loc[reg, eq]),
                    "StdErr": float(var_res.stderr.loc[reg, eq]),
                    "tStat": float(var_res.tvalues.loc[reg, eq]),
                    "PValue": float(var_res.pvalues.loc[reg, eq]),
                }
            )
    var_coefficients = pd.DataFrame(coef_rows)

    roots = var_res.roots
    roots_table = pd.DataFrame(
        {
            "RootIndex": np.arange(1, len(roots) + 1),
            "RealPart": np.real(roots),
            "ImagPart": np.imag(roots),
            "Modulus": np.abs(roots),
            "OutsideUnitCircle": np.abs(roots) > 1,
        }
    )

    normality = var_res.test_normality()
    whiten_nlags_final = min(24, max(12, selected_lag + 1), int(var_res.nobs - 1))
    if whiten_nlags_final > selected_lag:
        whiteness = var_res.test_whiteness(nlags=whiten_nlags_final)
        whiten_stat = whiteness.test_statistic
        whiten_p = whiteness.pvalue
        whiten_df = whiteness.df
    else:
        whiten_stat = np.nan
        whiten_p = np.nan
        whiten_df = np.nan

    # Equation-wise serial correlation diagnostics:
    # 1) LM (Breusch-Godfrey) at lags 1..12
    # 2) Ljung-Box at lags 1..12
    lm_max_lag = max(1, min(12, int(var_res.nobs - 1)))
    lb_max_lag = max(1, min(12, int(var_res.nobs - 1)))
    y_all = var_res.model.endog[var_res.k_ar :]
    x_all = var_res.endog_lagged
    lm_rows: list[dict[str, object]] = []
    lb_rows: list[dict[str, object]] = []
    for i, eq in enumerate(eq_names):
        eq_ols = sm.OLS(y_all[:, i], x_all).fit()
        for lag in range(1, lm_max_lag + 1):
            try:
                lm_stat, lm_p, f_stat, f_p = acorr_breusch_godfrey(eq_ols, nlags=lag)
            except Exception:
                lm_stat, lm_p, f_stat, f_p = np.nan, np.nan, np.nan, np.nan
            lm_rows.append(
                {
                    "Equation": eq,
                    "Lag": lag,
                    "LM_Stat": lm_stat,
                    "LM_PValue": lm_p,
                    "F_Stat": f_stat,
                    "F_PValue": f_p,
                }
            )

        lb = acorr_ljungbox(var_res.resid[eq], lags=range(1, lb_max_lag + 1), return_df=True)
        for lag, row in lb.iterrows():
            lb_rows.append(
                {
                    "Equation": eq,
                    "Lag": int(lag),
                    "LB_Stat": float(row["lb_stat"]),
                    "LB_PValue": float(row["lb_pvalue"]),
                }
            )

    lm_results = pd.DataFrame(lm_rows)
    lb_results = pd.DataFrame(lb_rows)

    lm_final = lm_results[lm_results["Lag"] == lm_max_lag]
    lb_final = lb_results[lb_results["Lag"] == lb_max_lag]
    lm_min_p = float(lm_final["LM_PValue"].min()) if not lm_final.empty else np.nan
    lb_min_p = float(lb_final["LB_PValue"].min()) if not lb_final.empty else np.nan

    diagnostics = pd.DataFrame(
        [
            {
                "Test": "Stability (all roots outside unit circle)",
                "Statistic": np.nan,
                "PValue": np.nan,
                "DF": np.nan,
                "Decision_5pct": "Pass" if var_res.is_stable() else "Fail",
            },
            {
                "Test": f"Whiteness Portmanteau (nlags={whiten_nlags_final})",
                "Statistic": whiten_stat,
                "PValue": whiten_p,
                "DF": whiten_df,
                "Decision_5pct": "Pass" if pd.notna(whiten_p) and whiten_p > 0.05 else "Fail",
            },
            {
                "Test": "Normality (Jarque-Bera, system)",
                "Statistic": normality.test_statistic,
                "PValue": normality.pvalue,
                "DF": normality.df,
                "Decision_5pct": "Pass" if normality.pvalue > 0.05 else "Fail",
            },
            {
                "Test": f"LM serial corr (equation-wise, lag={lm_max_lag}, min p)",
                "Statistic": np.nan,
                "PValue": lm_min_p,
                "DF": np.nan,
                "Decision_5pct": "Pass" if pd.notna(lm_min_p) and lm_min_p > 0.05 else "Fail",
            },
            {
                "Test": f"Ljung-Box (equation-wise, lag={lb_max_lag}, min p)",
                "Statistic": np.nan,
                "PValue": lb_min_p,
                "DF": np.nan,
                "Decision_5pct": "Pass" if pd.notna(lb_min_p) and lb_min_p > 0.05 else "Fail",
            },
        ]
    )

    selected_row = lag_criteria.loc[lag_criteria["Lag"] == selected_lag].iloc[0]
    info = pd.DataFrame(
        {
            "Metric": [
                "EndogenousVariables",
                "SelectedLag",
                "SelectionCriterionUsed",
                "SelectedByAIC",
                "SelectedByBIC",
                "SelectedByHQIC",
                "SelectedByFPE",
                "NObsUsed",
                "LogLik",
                "AIC",
                "BIC",
                "HQIC",
                "FPE",
                "IsStable",
                "ExogenousVariablesCount",
                "ExogenousVariables",
            ],
            "Value": [
                ", ".join(endog_cols),
                selected_lag,
                criterion_key.upper(),
                selected_lags["aic"],
                selected_lags["bic"],
                selected_lags["hqic"],
                selected_lags["fpe"],
                int(var_res.nobs),
                selected_row["LogLik"],
                selected_row["AIC"],
                selected_row["BIC"],
                selected_row["HQIC"],
                selected_row["FPE"],
                bool(var_res.is_stable()),
                int(len(exog_cols)),
                ", ".join(exog_cols) if exog_cols else "(none)",
            ],
        }
    )

    resid = var_res.resid.copy()
    resid.insert(0, "Date", dates.loc[resid.index].values)

    dw_vals = durbin_watson(var_res.resid.values)
    resid_summary = pd.DataFrame(
        {
            "Variable": endog_cols,
            "ResidualMean": var_res.resid.mean().values,
            "ResidualStd": var_res.resid.std(ddof=1).values,
            "DurbinWatson": dw_vals,
        }
    )

    fitted = var_res.fittedvalues.copy()
    bet_fit = pd.DataFrame(
        {
            "Date": dates.loc[fitted.index].values,
            "Actual": data.loc[fitted.index, DEPENDENT_VAR].values,
            "Fitted": fitted[DEPENDENT_VAR].values,
            "Residual": data.loc[fitted.index, DEPENDENT_VAR].values
            - fitted[DEPENDENT_VAR].values,
        }
    )

    var_exog_used_out = pd.DataFrame({"Date": dates.values})
    if exog_cols:
        var_exog_used_out = pd.concat(
            [var_exog_used_out.reset_index(drop=True), exog_used.reset_index(drop=True)],
            axis=1,
        )

    var_exog_meta = pd.DataFrame(
        {
            "Metric": [
                "ExogEnabled",
                "ExogCount",
                "ExogColumns",
            ],
            "Value": [
                bool(exog_cols),
                int(len(exog_cols)),
                ", ".join(exog_cols) if exog_cols else "(none)",
            ],
        }
    )

    return (
        lag_criteria,
        info,
        var_coefficients,
        diagnostics,
        roots_table,
        resid_summary,
        resid,
        bet_fit,
        var_exog_used_out,
        var_exog_meta,
        lm_results,
        lb_results,
    )


def save_plots(df: pd.DataFrame, numeric_cols: list[str], plots_dir: Path) -> list[str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    created_files: list[str] = []

    # 1) Time-series panel
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(14, 2.2 * len(numeric_cols)), sharex=True)
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.plot(df["Date"], df[col], linewidth=1.2)
        ax.set_title(col)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    f1 = plots_dir / "01_timeseries_panel.png"
    fig.savefig(f1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    created_files.append(f1.name)

    # 2) Histogram panel
    n = len(numeric_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for i, col in enumerate(numeric_cols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.hist(df[col].dropna(), bins=25, alpha=0.8)
        ax.set_title(col)
        ax.grid(alpha=0.2)
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.tight_layout()
    f2 = plots_dir / "02_histograms.png"
    fig.savefig(f2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    created_files.append(f2.name)

    # 3) Correlation heatmap
    corr = df[numeric_cols].corr().values
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(numeric_cols)))
    ax.set_yticks(np.arange(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    for r in range(corr.shape[0]):
        for c in range(corr.shape[1]):
            ax.text(c, r, f"{corr[r, c]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    f3 = plots_dir / "03_correlation_heatmap.png"
    fig.savefig(f3, dpi=160, bbox_inches="tight")
    plt.close(fig)
    created_files.append(f3.name)

    # 4) Boxplots for outlier view
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot(
        [df[c].dropna().values for c in numeric_cols],
        tick_labels=numeric_cols,
        showfliers=True,
    )
    ax.set_title("Boxplots (Outlier Check)")
    ax.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    f4 = plots_dir / "04_boxplots.png"
    fig.savefig(f4, dpi=160, bbox_inches="tight")
    plt.close(fig)
    created_files.append(f4.name)

    return created_files


def save_ols_plots(
    fitted_static: pd.DataFrame, fitted_dynamic: pd.DataFrame, plots_dir: Path
) -> list[str]:
    created_files: list[str] = []

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    axes[0].plot(fitted_static["Date"], fitted_static["Actual"], label="Actual", linewidth=1.2)
    axes[0].plot(fitted_static["Date"], fitted_static["Fitted"], label="Fitted", linewidth=1.2)
    axes[0].set_title("OLS Static: Actual vs Fitted (r_BET)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        fitted_dynamic["Date"], fitted_dynamic["Actual"], label="Actual", linewidth=1.2
    )
    axes[1].plot(
        fitted_dynamic["Date"], fitted_dynamic["Fitted"], label="Fitted", linewidth=1.2
    )
    axes[1].set_title("OLS Dynamic L1: Actual vs Fitted (r_BET)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    f5 = plots_dir / "05_ols_actual_vs_fitted.png"
    fig.savefig(f5, dpi=160, bbox_inches="tight")
    plt.close(fig)
    created_files.append(f5.name)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plot_acf(
        fitted_static["Residual"],
        lags=24,
        ax=axes[0],
        title="Residual ACF - OLS Static",
        zero=False,
    )
    plot_acf(
        fitted_dynamic["Residual"],
        lags=24,
        ax=axes[1],
        title="Residual ACF - OLS Dynamic L1",
        zero=False,
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    f6 = plots_dir / "06_ols_residual_acf.png"
    fig.savefig(f6, dpi=160, bbox_inches="tight")
    plt.close(fig)
    created_files.append(f6.name)

    return created_files


def save_var_plots(
    lag_criteria: pd.DataFrame,
    var_bet_fitted: pd.DataFrame,
    var_residuals: pd.DataFrame,
    plots_dir: Path,
) -> list[str]:
    created_files: list[str] = []

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(lag_criteria["Lag"], lag_criteria["AIC"], marker="o", label="AIC")
    axes[0].plot(lag_criteria["Lag"], lag_criteria["BIC"], marker="o", label="BIC")
    axes[0].plot(lag_criteria["Lag"], lag_criteria["HQIC"], marker="o", label="HQIC")
    axes[0].set_title("VAR Lag Criteria (AIC/BIC/HQIC)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(lag_criteria["Lag"], lag_criteria["FPE"], marker="o", color="#cc5500")
    axes[1].set_yscale("log")
    axes[1].set_title("VAR Lag Criterion (FPE, log scale)")
    axes[1].set_xlabel("Lag")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    f7 = plots_dir / "07_var_lag_criteria.png"
    fig.savefig(f7, dpi=160, bbox_inches="tight")
    plt.close(fig)
    created_files.append(f7.name)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(var_bet_fitted["Date"], var_bet_fitted["Actual"], label="Actual", linewidth=1.2)
    ax.plot(var_bet_fitted["Date"], var_bet_fitted["Fitted"], label="Fitted", linewidth=1.2)
    ax.set_title("VAR: Actual vs Fitted (r_BET)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    f8 = plots_dir / "08_var_bet_actual_vs_fitted.png"
    fig.savefig(f8, dpi=160, bbox_inches="tight")
    plt.close(fig)
    created_files.append(f8.name)

    resid_cols = [c for c in var_residuals.columns if c != "Date"]
    fig, axes = plt.subplots(
        len(resid_cols), 1, figsize=(14, 2.0 * len(resid_cols)), sharex=True
    )
    if len(resid_cols) == 1:
        axes = [axes]
    for i, col in enumerate(resid_cols):
        ax = axes[i]
        ax.plot(var_residuals["Date"], var_residuals[col], linewidth=1.0)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
        ax.set_title(f"VAR Residual: {col}")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    f9 = plots_dir / "09_var_residuals_panel.png"
    fig.savefig(f9, dpi=160, bbox_inches="tight")
    plt.close(fig)
    created_files.append(f9.name)

    return created_files


def write_excel_output(
    output_path: Path,
    data_clean: pd.DataFrame,
    ciss_meta: pd.DataFrame,
    shock_dummy_meta: pd.DataFrame,
    qa_summary: pd.DataFrame,
    missing_by_col: pd.DataFrame,
    irregular_gaps: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    outlier_points: pd.DataFrame,
    descriptive_stats: pd.DataFrame,
    corr: pd.DataFrame,
    stationarity: pd.DataFrame,
    ols_specs: pd.DataFrame,
    ols_coef_static_ols: pd.DataFrame,
    ols_coef_static_hac: pd.DataFrame,
    ols_coef_dynamic_ols: pd.DataFrame,
    ols_coef_dynamic_hac: pd.DataFrame,
    ols_diagnostics: pd.DataFrame,
    ols_fitted_static: pd.DataFrame,
    ols_fitted_dynamic: pd.DataFrame,
    ols_vif_static: pd.DataFrame,
    ols_vif_dynamic: pd.DataFrame,
    var_lag_criteria: pd.DataFrame,
    var_info: pd.DataFrame,
    var_coefficients: pd.DataFrame,
    var_diagnostics: pd.DataFrame,
    var_roots: pd.DataFrame,
    var_resid_summary: pd.DataFrame,
    var_residuals: pd.DataFrame,
    var_bet_fitted: pd.DataFrame,
    var_exog_used: pd.DataFrame,
    var_exog_meta: pd.DataFrame,
    var_lm_results: pd.DataFrame,
    var_ljungbox_results: pd.DataFrame,
    plot_files: list[str],
) -> None:
    with pd.ExcelWriter(output_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        data_clean.to_excel(writer, sheet_name="data_clean", index=False)
        ciss_meta.to_excel(writer, sheet_name="ciss_meta", index=False)
        shock_dummy_meta.to_excel(writer, sheet_name="shock_dummies_meta", index=False)
        qa_summary.to_excel(writer, sheet_name="qa_summary", index=False)
        missing_by_col.to_excel(writer, sheet_name="qa_missing", index=False)
        irregular_gaps.to_excel(writer, sheet_name="qa_irregular_gaps", index=False)
        outlier_summary.to_excel(writer, sheet_name="qa_outliers", index=False)
        outlier_points.to_excel(writer, sheet_name="qa_outlier_points", index=False)
        descriptive_stats.to_excel(writer, sheet_name="descriptive_stats", index=False)
        corr.to_excel(writer, sheet_name="correlation_matrix", index=False)
        stationarity.to_excel(writer, sheet_name="stationarity", index=False)
        ols_specs.to_excel(writer, sheet_name="ols_specs", index=False)
        ols_coef_static_ols.to_excel(writer, sheet_name="ols_static_coef_ols", index=False)
        ols_coef_static_hac.to_excel(writer, sheet_name="ols_static_coef_hac", index=False)
        ols_coef_dynamic_ols.to_excel(writer, sheet_name="ols_dynamic_coef_ols", index=False)
        ols_coef_dynamic_hac.to_excel(writer, sheet_name="ols_dynamic_coef_hac", index=False)
        ols_diagnostics.to_excel(writer, sheet_name="ols_diagnostics", index=False)
        ols_fitted_static.to_excel(writer, sheet_name="ols_static_fitted", index=False)
        ols_fitted_dynamic.to_excel(writer, sheet_name="ols_dynamic_fitted", index=False)
        ols_vif_static.to_excel(writer, sheet_name="ols_static_vif", index=False)
        ols_vif_dynamic.to_excel(writer, sheet_name="ols_dynamic_vif", index=False)
        var_lag_criteria.to_excel(writer, sheet_name="var_lag_criteria", index=False)
        var_info.to_excel(writer, sheet_name="var_info", index=False)
        var_coefficients.to_excel(writer, sheet_name="var_coefficients", index=False)
        var_diagnostics.to_excel(writer, sheet_name="var_diagnostics", index=False)
        var_roots.to_excel(writer, sheet_name="var_roots", index=False)
        var_resid_summary.to_excel(writer, sheet_name="var_resid_summary", index=False)
        var_residuals.to_excel(writer, sheet_name="var_residuals", index=False)
        var_bet_fitted.to_excel(writer, sheet_name="var_bet_fitted", index=False)
        var_exog_used.to_excel(writer, sheet_name="var_exog_used", index=False)
        var_exog_meta.to_excel(writer, sheet_name="var_exog_meta", index=False)
        var_lm_results.to_excel(writer, sheet_name="var_lm_serial", index=False)
        var_ljungbox_results.to_excel(writer, sheet_name="var_ljungbox", index=False)
        pd.DataFrame({"PlotFile": plot_files}).to_excel(
            writer, sheet_name="plot_files", index=False
        )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Nu gasesc fisierul: {INPUT_FILE.resolve()}")

    df, numeric_cols, ciss_meta = load_and_prepare_data(INPUT_FILE)

    qa_summary, missing_by_col, irregular_gaps, outlier_summary, outlier_points = build_data_qa(
        df, numeric_cols
    )
    descriptive_stats, corr = build_descriptive_stats(df, numeric_cols)
    stationarity = run_stationarity_tests(df, numeric_cols)
    (
        ols_specs,
        ols_coef_static_ols,
        ols_coef_static_hac,
        ols_coef_dynamic_ols,
        ols_coef_dynamic_hac,
        ols_diagnostics,
        ols_fitted_static,
        ols_fitted_dynamic,
        ols_vif_static,
        ols_vif_dynamic,
    ) = run_ols_benchmark(df, DEPENDENT_VAR)

    if VAR_MODEL_MODE.lower() == "varx":
        var_endog_cols = [c for c in VARX_ENDOG_COLS if c in numeric_cols]
        var_base_exog_cols = [
            c for c in VARX_EXOG_COLS if c in df.columns and c not in var_endog_cols
        ]
    else:
        var_endog_cols = [c for c in numeric_cols if c not in VAR_EXCLUDE_ENDOG]
        var_base_exog_cols = []

    var_exog_input = pd.DataFrame(index=df.index)
    if var_base_exog_cols:
        var_exog_input = pd.concat(
            [var_exog_input, df[var_base_exog_cols].reset_index(drop=True)], axis=1
        )

    if ENABLE_VAR_SHOCK_DUMMIES:
        shock_dummies, shock_dummy_meta = build_shock_dummies(df["Date"], VAR_SHOCK_DATES)
        var_exog_input = pd.concat(
            [var_exog_input.reset_index(drop=True), shock_dummies.reset_index(drop=True)],
            axis=1,
        )
    else:
        shock_dummy_meta = pd.DataFrame(
            {"Metric": ["ShockDummiesEnabled", "ShockDummyCount"], "Value": [False, 0]}
        )

    (
        var_lag_criteria,
        var_info,
        var_coefficients,
        var_diagnostics,
        var_roots,
        var_resid_summary,
        var_residuals,
        var_bet_fitted,
        var_exog_used,
        var_exog_meta,
        var_lm_results,
        var_ljungbox_results,
    ) = run_var_workflow(
        df=df,
        endog_cols=var_endog_cols,
        exog=var_exog_input,
        lag_min=VAR_LAG_MIN,
        lag_max=VAR_LAG_MAX,
        selection_criterion=VAR_LAG_SELECTION_CRITERION,
    )

    plot_files = save_plots(df, numeric_cols, PLOTS_DIR)
    plot_files.extend(save_ols_plots(ols_fitted_static, ols_fitted_dynamic, PLOTS_DIR))
    plot_files.extend(save_var_plots(var_lag_criteria, var_bet_fitted, var_residuals, PLOTS_DIR))

    write_excel_output(
        output_path=OUTPUT_EXCEL,
        data_clean=df,
        ciss_meta=ciss_meta,
        shock_dummy_meta=shock_dummy_meta,
        qa_summary=qa_summary,
        missing_by_col=missing_by_col,
        irregular_gaps=irregular_gaps,
        outlier_summary=outlier_summary,
        outlier_points=outlier_points,
        descriptive_stats=descriptive_stats,
        corr=corr,
        stationarity=stationarity,
        ols_specs=ols_specs,
        ols_coef_static_ols=ols_coef_static_ols,
        ols_coef_static_hac=ols_coef_static_hac,
        ols_coef_dynamic_ols=ols_coef_dynamic_ols,
        ols_coef_dynamic_hac=ols_coef_dynamic_hac,
        ols_diagnostics=ols_diagnostics,
        ols_fitted_static=ols_fitted_static,
        ols_fitted_dynamic=ols_fitted_dynamic,
        ols_vif_static=ols_vif_static,
        ols_vif_dynamic=ols_vif_dynamic,
        var_lag_criteria=var_lag_criteria,
        var_info=var_info,
        var_coefficients=var_coefficients,
        var_diagnostics=var_diagnostics,
        var_roots=var_roots,
        var_resid_summary=var_resid_summary,
        var_residuals=var_residuals,
        var_bet_fitted=var_bet_fitted,
        var_exog_used=var_exog_used,
        var_exog_meta=var_exog_meta,
        var_lm_results=var_lm_results,
        var_ljungbox_results=var_ljungbox_results,
        plot_files=plot_files,
    )

    print(f"Done. Rezultate numerice: {OUTPUT_EXCEL.resolve()}")
    print(f"Done. Ploturi: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
