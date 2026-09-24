#!/usr/bin/env python3
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.ardl import ARDL, UECM
from statsmodels.tsa.stattools import adfuller, kpss

# Keep output cleaner from benign warnings in matrix ops.
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\.linalg\._linalg")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"statsmodels\.tsa\.ardl\.model")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"statsmodels\.tsa\.ar_model")


INPUT_FILE = Path("Date licenta actualizare.xlsx")
OUTPUT_EXCEL = Path("rezultate_ardl.xlsx")
PLOTS_DIR = Path("plots_ardl")

DEPENDENT_VAR = "r_BET"
DEFAULT_REGRESSORS = [
    "r_Masa monetara",
    "INF",
    "UNEMP",
    "Y10- Romania",
    "r_MSCI",
    "r_FX",
]

# Optional CISS merge. Keep False for baseline ARDL unless requested.
ENABLE_CISS = False
CISS_FILE_GLOB = "ECB Data Portal long_*.xlsx"
CISS_SHEET_NAME = "DATA(CISS)"
CISS_TRANSFORMED_COL = "CISS"

# Monthly data: AIC/BIC/HQIC select lag lengths only (all regressors are forced in model).
ARDL_MIN_LAG_ENDOG = 1
ARDL_MAX_LAG_ENDOG = 3
ARDL_MIN_ORDER_EXOG = 1
ARDL_MAX_ORDER_EXOG = 3
ARDL_SELECTION_CRITERION = "bic"  # options: aic, bic, hqic
ARDL_TREND = "c"  # constant
ARDL_CAUSAL = False  # include lag 0 for exogenous vars
TOP_MODELS_TO_EXPORT = 25

# PSS bounds test case:
# 1: no intercept, no trend
# 2: restricted intercept, no trend
# 3: unrestricted intercept, no trend (most common)
# 4: unrestricted intercept, restricted trend
# 5: unrestricted intercept and trend
UECM_BOUNDS_CASE = 3


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
            "Fisierul CISS nu are coloanele DATE si OBS.VALUE in foaia DATA(CISS)."
        )

    ciss_raw["DATE"] = pd.to_datetime(ciss_raw["DATE"], errors="coerce")
    ciss_raw["OBS.VALUE"] = pd.to_numeric(ciss_raw["OBS.VALUE"], errors="coerce")
    ciss_raw = ciss_raw.dropna(subset=["DATE", "OBS.VALUE"]).sort_values("DATE")

    ciss_raw["Month"] = ciss_raw["DATE"].dt.to_period("M")
    monthly = ciss_raw.groupby("Month", as_index=False)["OBS.VALUE"].last()
    monthly["Date"] = monthly["Month"].dt.to_timestamp("M")
    monthly = monthly.rename(columns={"OBS.VALUE": "CISS_level"})[["Date", "CISS_level"]]
    monthly[CISS_TRANSFORMED_COL] = monthly["CISS_level"].diff()

    ciss_aligned = pd.DataFrame({"Date": base_dates}).merge(monthly, on="Date", how="left")
    ciss_for_merge = ciss_aligned[["Date", CISS_TRANSFORMED_COL]].copy()

    ciss_meta = pd.DataFrame(
        {
            "Metric": [
                "CISS_SourceFile",
                "CISS_Sheet",
                "CISS_Transformation",
                "CISS_MissingAfterMerge",
            ],
            "Value": [
                ciss_file.name,
                CISS_SHEET_NAME,
                "Monthly last observation, then first difference",
                int(ciss_for_merge[CISS_TRANSFORMED_COL].isna().sum()),
            ],
        }
    )
    return ciss_for_merge, ciss_meta


def load_and_prepare_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(path)
    if "Date" not in df.columns:
        raise ValueError("Coloana 'Date' nu exista in fisier.")

    # Excel serial dates support (if needed).
    if pd.api.types.is_numeric_dtype(df["Date"]):
        base = pd.Timestamp("1899-12-30")
        df["Date"] = base + pd.to_timedelta(df["Date"], unit="D")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)
    ciss_meta = pd.DataFrame({"Metric": ["CISS_Status"], "Value": ["Not requested"]})

    if ENABLE_CISS:
        ciss_file = find_latest_ciss_file(path.parent)
        if ciss_file is None:
            ciss_meta = pd.DataFrame(
                {"Metric": ["CISS_Status", "CISS_FilePattern"], "Value": ["Not found", CISS_FILE_GLOB]}
            )
        else:
            ciss_for_merge, ciss_meta = prepare_ciss_for_merge(df["Date"], ciss_file)
            df = df.merge(ciss_for_merge, on="Date", how="left")

    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, ciss_meta


def safe_adf(series: pd.Series) -> tuple[float, float]:
    s = series.dropna()
    if len(s) < 20:
        return np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval, *_ = adfuller(s, regression="c", autolag="AIC")
        return float(stat), float(pval)
    except Exception:
        return np.nan, np.nan


def safe_kpss(series: pd.Series) -> tuple[float, float]:
    s = series.dropna()
    if len(s) < 20:
        return np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval, *_ = kpss(s, regression="c", nlags="auto")
        return float(stat), float(pval)
    except Exception:
        return np.nan, np.nan


def stationary_flag(adf_p: float, kpss_p: float) -> bool:
    return pd.notna(adf_p) and pd.notna(kpss_p) and (adf_p < 0.05) and (kpss_p > 0.05)


def integration_diagnostics(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for col in cols:
        s0 = df[col]
        s1 = df[col].diff()
        s2 = s1.diff()

        adf0, adf0_p = safe_adf(s0)
        kpss0, kpss0_p = safe_kpss(s0)
        adf1, adf1_p = safe_adf(s1)
        kpss1, kpss1_p = safe_kpss(s1)
        adf2, adf2_p = safe_adf(s2)
        kpss2, kpss2_p = safe_kpss(s2)

        is_i0 = stationary_flag(adf0_p, kpss0_p)
        is_i1 = (not is_i0) and stationary_flag(adf1_p, kpss1_p)
        is_i2_plus = (not is_i0) and (not is_i1) and stationary_flag(adf2_p, kpss2_p)

        if is_i0:
            integration_order = "I(0)"
        elif is_i1:
            integration_order = "I(1)"
        elif is_i2_plus:
            integration_order = "I(2)+"
        else:
            integration_order = "Unclear/Mixed"

        rows.append(
            {
                "Variable": col,
                "ADF_Level_p": adf0_p,
                "KPSS_Level_p": kpss0_p,
                "ADF_D1_p": adf1_p,
                "KPSS_D1_p": kpss1_p,
                "ADF_D2_p": adf2_p,
                "KPSS_D2_p": kpss2_p,
                "IntegrationOrder": integration_order,
                "ARDL_Allowed": integration_order in {"I(0)", "I(1)", "Unclear/Mixed"},
            }
        )

    out = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "Metric": [
                "VariablesChecked",
                "I0_Count",
                "I1_Count",
                "I2plus_Count",
                "Unclear_Count",
                "Has_I2plus",
                "ARDL_Proceed_Recommended",
            ],
            "Value": [
                int(len(out)),
                int((out["IntegrationOrder"] == "I(0)").sum()),
                int((out["IntegrationOrder"] == "I(1)").sum()),
                int((out["IntegrationOrder"] == "I(2)+").sum()),
                int((out["IntegrationOrder"] == "Unclear/Mixed").sum()),
                bool((out["IntegrationOrder"] == "I(2)+").any()),
                bool(not (out["IntegrationOrder"] == "I(2)+").any()),
            ],
        }
    )
    return out, summary


def newey_west_lags(nobs: int) -> int:
    return max(1, int(np.floor(4 * (nobs / 100.0) ** (2.0 / 9.0))))


def result_to_coef_table(
    params: pd.Series,
    bse: pd.Series,
    tvals: pd.Series,
    pvals: pd.Series,
    conf_int: pd.DataFrame,
    model_name: str,
    cov_type: str,
    hac_maxlags: int | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "Model": model_name,
            "CovType": cov_type,
            "HAC_maxlags": hac_maxlags if hac_maxlags is not None else np.nan,
            "Variable": params.index,
            "Coef": params.values,
            "StdErr": bse.values,
            "t_or_z": tvals.values,
            "PValue": pvals.values,
            "CI_Lower_95": conf_int.iloc[:, 0].values,
            "CI_Upper_95": conf_int.iloc[:, 1].values,
        }
    )
    return out


def fmt_lags(lags: object) -> str:
    if lags is None:
        return "(none)"
    if isinstance(lags, (list, tuple, np.ndarray)):
        if len(lags) == 0:
            return "(none)"
        return ",".join(str(int(x)) for x in lags)
    try:
        return str(int(lags))
    except Exception:
        return str(lags)


def fmt_dl_lags(dl_lags: dict[str, object]) -> str:
    if not dl_lags:
        return "(none)"
    parts = [f"{k}:{fmt_lags(v)}" for k, v in dl_lags.items()]
    return " | ".join(parts)


def lag_grid_rank_table(
    lag_grid: pd.DataFrame, criterion_col: str, top_n: int
) -> pd.DataFrame:
    ranked = lag_grid.sort_values(criterion_col).head(top_n).reset_index(drop=True).copy()
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    ranked.insert(1, "Criterion", criterion_col)
    return ranked[
        [
            "Rank",
            "Criterion",
            "AR_Lag_p",
            "Exog_Lag_q",
            "AIC",
            "BIC",
            "HQIC",
            "LogLik",
            "NObs",
        ]
    ]


def save_ardl_plots(
    fitted_df: pd.DataFrame,
    residuals: pd.Series,
    ci_residuals: pd.DataFrame,
    plots_dir: Path,
) -> list[str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(fitted_df["Date"], fitted_df["Actual"], label="Actual", linewidth=1.2)
    ax.plot(fitted_df["Date"], fitted_df["Fitted"], label="Fitted", linewidth=1.2)
    ax.set_title("ARDL: Actual vs Fitted (r_BET)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    f1 = plots_dir / "01_ardl_actual_vs_fitted.png"
    fig.savefig(f1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    files.append(f1.name)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].plot(fitted_df["Date"], fitted_df["Residual"], linewidth=1.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    axes[0].set_title("ARDL Residuals")
    axes[0].grid(alpha=0.2)

    lags = min(24, max(1, int(len(residuals.dropna()) // 3)))
    plot_acf(
        residuals.dropna(),
        lags=lags,
        ax=axes[1],
        title=f"Residual ACF (lags={lags})",
        zero=False,
    )
    axes[1].grid(alpha=0.2)

    axes[2].hist(residuals.dropna(), bins=25, alpha=0.8)
    axes[2].set_title("Residual Histogram")
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    f2 = plots_dir / "02_ardl_residual_diagnostics.png"
    fig.savefig(f2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    files.append(f2.name)

    if not ci_residuals.empty:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(ci_residuals["Date"], ci_residuals["CI_Residual"], linewidth=1.0)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
        ax.set_title("UECM Cointegrating Residual")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        f3 = plots_dir / "03_uecm_ci_residual.png"
        fig.savefig(f3, dpi=160, bbox_inches="tight")
        plt.close(fig)
        files.append(f3.name)

    return files


def run_ardl_workflow(
    df: pd.DataFrame,
    dependent_var: str,
    regressor_cols: list[str],
) -> dict[str, pd.DataFrame]:
    model_data = df[["Date", dependent_var] + regressor_cols].dropna().copy()
    model_data = model_data.sort_values("Date").reset_index(drop=True)
    model_data = model_data.set_index("Date").asfreq("ME").dropna()
    y = model_data[dependent_var]
    x = model_data[regressor_cols]

    hold_back = max(ARDL_MAX_LAG_ENDOG, ARDL_MAX_ORDER_EXOG)
    lag_rows: list[dict[str, object]] = []
    for p in range(ARDL_MIN_LAG_ENDOG, ARDL_MAX_LAG_ENDOG + 1):
        for q in range(ARDL_MIN_ORDER_EXOG, ARDL_MAX_ORDER_EXOG + 1):
            order = {col: q for col in regressor_cols}
            try:
                model = ARDL(
                    endog=y,
                    lags=p,
                    exog=x,
                    order=order,
                    trend=ARDL_TREND,
                    causal=ARDL_CAUSAL,
                    hold_back=hold_back,
                    missing="drop",
                )
                res = model.fit()
            except Exception:
                continue

            lag_rows.append(
                {
                    "AR_Lag_p": int(p),
                    "Exog_Lag_q": int(q),
                    "NObs": int(res.nobs),
                    "LogLik": float(res.llf),
                    "AIC": float(res.aic),
                    "BIC": float(res.bic),
                    "HQIC": float(res.hqic),
                }
            )

    lag_grid = pd.DataFrame(lag_rows)
    if lag_grid.empty:
        raise RuntimeError("Nu s-a putut estima niciun model ARDL in intervalul de laguri setat.")

    criterion_map = {"aic": "AIC", "bic": "BIC", "hqic": "HQIC"}
    selected_specs_rows: list[dict[str, object]] = []
    for key, col in criterion_map.items():
        row = lag_grid.loc[lag_grid[col].idxmin()]
        selected_specs_rows.append(
            {
                "Criterion": key.upper(),
                "Selected_AR_Lag_p": int(row["AR_Lag_p"]),
                "Selected_Exog_Lag_q": int(row["Exog_Lag_q"]),
                "CriterionValue": float(row[col]),
            }
        )

    selected_ic = ARDL_SELECTION_CRITERION.lower()
    if selected_ic not in criterion_map:
        selected_ic = "bic"
    selected_col = criterion_map[selected_ic]
    selected_row = lag_grid.loc[lag_grid[selected_col].idxmin()]
    selected_p = int(selected_row["AR_Lag_p"])
    selected_q = int(selected_row["Exog_Lag_q"])
    selected_dl_spec = {col: selected_q for col in regressor_cols}

    ardl_model = ARDL(
        endog=y,
        lags=selected_p,
        exog=x,
        order=selected_dl_spec,
        trend=ARDL_TREND,
        causal=ARDL_CAUSAL,
        hold_back=hold_back,
        missing="drop",
    )

    ardl_res = ardl_model.fit()
    nw_lags = newey_west_lags(int(ardl_res.nobs))
    ardl_hac = ardl_model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

    coef_nonrobust = result_to_coef_table(
        params=ardl_res.params,
        bse=ardl_res.bse,
        tvals=ardl_res.tvalues,
        pvals=ardl_res.pvalues,
        conf_int=ardl_res.conf_int(),
        model_name=f"ARDL_{selected_ic.upper()}_p{selected_p}_q{selected_q}",
        cov_type="nonrobust",
    )
    coef_hac = result_to_coef_table(
        params=ardl_hac.params,
        bse=ardl_hac.bse,
        tvals=ardl_hac.tvalues,
        pvals=ardl_hac.pvalues,
        conf_int=ardl_hac.conf_int(),
        model_name=f"ARDL_{selected_ic.upper()}_p{selected_p}_q{selected_q}",
        cov_type="HAC",
        hac_maxlags=nw_lags,
    )

    serial = ardl_res.test_serial_correlation(lags=12).reset_index().rename(columns={"index": "Lag"})
    arch = ardl_res.test_heteroskedasticity(lags=12).reset_index().rename(columns={"index": "Lag"})
    normality = (
        ardl_res.test_normality()
        .rename_axis("Metric")
        .reset_index(name="Value")
    )

    lb_fallback = acorr_ljungbox(ardl_res.resid.dropna(), lags=range(1, 13), return_df=True)
    lb_fallback = lb_fallback.reset_index().rename(
        columns={"index": "Lag", "lb_stat": "LB_Stat", "lb_pvalue": "LB_PValue"}
    )

    roots = np.asarray(ardl_res.roots)
    roots_table = pd.DataFrame(
        {
            "RootIndex": np.arange(1, len(roots) + 1),
            "RealPart": np.real(roots),
            "ImagPart": np.imag(roots),
            "Modulus": np.abs(roots),
            "OutsideUnitCircle": np.abs(roots) > 1,
        }
    )

    resid = ardl_res.resid.copy()
    fitted = ardl_res.fittedvalues.copy()
    actual = y.loc[fitted.index]
    fitted_df = pd.DataFrame(
        {
            "Date": fitted.index,
            "Actual": actual.values,
            "Fitted": fitted.values,
            "Residual": resid.loc[fitted.index].values,
        }
    )

    serial_p_min = np.nan
    if "LB P-value" in serial.columns and serial["LB P-value"].notna().any():
        serial_p_min = float(serial["LB P-value"].min())
    elif lb_fallback["LB_PValue"].notna().any():
        serial_p_min = float(lb_fallback["LB_PValue"].min())

    arch_p_min = (
        float(arch["P-value"].min())
        if ("P-value" in arch.columns and arch["P-value"].notna().any())
        else np.nan
    )
    jb_p = np.nan
    if not normality.empty and (normality["Metric"] == "P-value").any():
        jb_p = float(normality.loc[normality["Metric"] == "P-value", "Value"].iloc[0])

    info = pd.DataFrame(
        {
            "Metric": [
                "DependentVariable",
                "RegressorsForcedInAllModels",
                "LagSearchRange_p",
                "LagSearchRange_q",
                "SelectedCriterion",
                "Selected_AR_Lag_p",
                "Selected_Exog_Lag_q",
                "Selected_OrderByVariable",
                "AllRegressorsIncluded",
                "NObsUsed",
                "LogLik",
                "AIC",
                "BIC",
                "HQIC",
                "HAC_Maxlags",
                "DurbinWatson",
                "RootsOutsideUnitCircle_All",
            ],
            "Value": [
                dependent_var,
                ", ".join(regressor_cols),
                f"{ARDL_MIN_LAG_ENDOG}..{ARDL_MAX_LAG_ENDOG}",
                f"{ARDL_MIN_ORDER_EXOG}..{ARDL_MAX_ORDER_EXOG}",
                selected_ic.upper(),
                int(selected_p),
                int(selected_q),
                fmt_dl_lags(selected_dl_spec),
                True,
                int(ardl_res.nobs),
                float(ardl_res.llf),
                float(ardl_res.aic),
                float(ardl_res.bic),
                float(ardl_res.hqic),
                int(nw_lags),
                float(durbin_watson(ardl_res.resid)),
                bool((np.abs(roots) > 1).all()),
            ],
        }
    )

    diagnostics = pd.DataFrame(
        [
            {
                "Test": "Residual serial correlation (Ljung-Box min p across lags 1..12)",
                "PValue": serial_p_min,
                "Decision_5pct": "Pass" if pd.notna(serial_p_min) and serial_p_min > 0.05 else "Fail",
            },
            {
                "Test": "ARCH heteroskedasticity (min p across lags 1..12)",
                "PValue": arch_p_min,
                "Decision_5pct": "Pass" if pd.notna(arch_p_min) and arch_p_min > 0.05 else "Fail",
            },
            {
                "Test": "Residual normality (Jarque-Bera)",
                "PValue": jb_p,
                "Decision_5pct": "Pass" if pd.notna(jb_p) and jb_p > 0.05 else "Fail",
            },
            {
                "Test": "Dynamic stability (all inverse roots outside unit circle)",
                "PValue": np.nan,
                "Decision_5pct": "Pass" if (np.abs(roots) > 1).all() else "Fail",
            },
        ]
    )

    # UECM + bounds test + long-run relation
    uecm_status = pd.DataFrame({"Metric": ["UECM_Status"], "Value": ["Not attempted"]})
    uecm_coef = pd.DataFrame()
    bounds_main = pd.DataFrame()
    bounds_crit = pd.DataFrame()
    ci_params = pd.DataFrame()
    ci_residuals = pd.DataFrame()
    ecm_speed = pd.DataFrame()

    try:
        uecm_model = UECM.from_ardl(ardl_model)
        uecm_res = uecm_model.fit()
        bt = uecm_res.bounds_test(case=UECM_BOUNDS_CASE)

        uecm_coef = result_to_coef_table(
            params=uecm_res.params,
            bse=uecm_res.bse,
            tvals=uecm_res.tvalues,
            pvals=uecm_res.pvalues,
            conf_int=uecm_res.conf_int(),
            model_name=f"UECM_case{UECM_BOUNDS_CASE}",
            cov_type="nonrobust",
        )

        bounds_main = pd.DataFrame(
            {
                "Metric": [
                    "BoundsCase",
                    "F_stat",
                    "Lower_pvalue_I0",
                    "Upper_pvalue_I1",
                    "Null",
                    "Alternative",
                ],
                "Value": [
                    int(UECM_BOUNDS_CASE),
                    float(bt.stat),
                    float(bt.p_values["lower"]),
                    float(bt.p_values["upper"]),
                    bt.null,
                    bt.alternative,
                ],
            }
        )

        bounds_crit = bt.crit_vals.reset_index().rename(
            columns={"percentile": "Percentile", "lower": "I0_Lower", "upper": "I1_Upper"}
        )

        ci_params = uecm_res.ci_params.rename_axis("Variable").reset_index(name="Coef")
        ci_residuals = uecm_res.ci_resids.reset_index()
        ci_residuals.columns = ["Date", "CI_Residual"]

        dep_l1_name = None
        for name in uecm_res.params.index:
            if name.startswith(f"{dependent_var}.L1"):
                dep_l1_name = name
                break
        if dep_l1_name is not None:
            ecm_speed = pd.DataFrame(
                {
                    "Metric": ["ECT_Coefficient", "ECT_PValue", "ECT_NegativeAndSignificant_5pct"],
                    "Value": [
                        float(uecm_res.params[dep_l1_name]),
                        float(uecm_res.pvalues[dep_l1_name]),
                        bool(
                            (uecm_res.params[dep_l1_name] < 0.0)
                            and (uecm_res.pvalues[dep_l1_name] < 0.05)
                        ),
                    ],
                }
            )
        else:
            ecm_speed = pd.DataFrame(
                {"Metric": ["ECT_Status"], "Value": ["Dependent L1 term not found in UECM params"]}
            )

        uecm_status = pd.DataFrame({"Metric": ["UECM_Status"], "Value": ["OK"]})

    except Exception as exc:
        uecm_status = pd.DataFrame(
            {
                "Metric": ["UECM_Status", "UECM_Message"],
                "Value": ["Failed", str(exc)],
            }
        )

    plot_files = save_ardl_plots(
        fitted_df=fitted_df,
        residuals=resid,
        ci_residuals=ci_residuals,
        plots_dir=PLOTS_DIR,
    )

    selected_specs = pd.DataFrame(selected_specs_rows)
    return {
        "data_model_used": model_data.reset_index(),
        "lag_grid_all": lag_grid.sort_values(["AR_Lag_p", "Exog_Lag_q"]).reset_index(drop=True),
        "selected_specs": selected_specs,
        "rank_aic": lag_grid_rank_table(lag_grid, "AIC", TOP_MODELS_TO_EXPORT),
        "rank_bic": lag_grid_rank_table(lag_grid, "BIC", TOP_MODELS_TO_EXPORT),
        "rank_hqic": lag_grid_rank_table(lag_grid, "HQIC", TOP_MODELS_TO_EXPORT),
        "ardl_info": info,
        "ardl_coef_nonrobust": coef_nonrobust,
        "ardl_coef_hac": coef_hac,
        "ardl_diag": diagnostics,
        "ardl_serial": serial,
        "ardl_arch": arch,
        "ardl_normality": normality,
        "ardl_ljungbox_fallback": lb_fallback,
        "ardl_roots": roots_table,
        "ardl_fitted": fitted_df,
        "uecm_status": uecm_status,
        "uecm_coef": uecm_coef,
        "bounds_main": bounds_main,
        "bounds_crit": bounds_crit,
        "ci_params": ci_params,
        "ci_residuals": ci_residuals,
        "ecm_speed": ecm_speed,
        "plot_files": pd.DataFrame({"PlotFile": plot_files}),
    }


def write_excel(output_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(output_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        for sheet_name, table in tables.items():
            safe_sheet = sheet_name[:31]
            if table is None:
                continue
            if table.empty:
                pd.DataFrame({"Info": ["(empty)"]}).to_excel(writer, sheet_name=safe_sheet, index=False)
            else:
                table.to_excel(writer, sheet_name=safe_sheet, index=False)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Nu gasesc fisierul: {INPUT_FILE.resolve()}")

    df, ciss_meta = load_and_prepare_data(INPUT_FILE)

    candidate_regressors = [c for c in DEFAULT_REGRESSORS if c in df.columns and c != DEPENDENT_VAR]
    if ENABLE_CISS and (CISS_TRANSFORMED_COL in df.columns):
        candidate_regressors.append(CISS_TRANSFORMED_COL)

    if DEPENDENT_VAR not in df.columns:
        raise ValueError(f"Dependent variable lipseste din fisier: {DEPENDENT_VAR}")
    if not candidate_regressors:
        raise ValueError("Nu exista regresori disponibili pentru ARDL.")

    integration_tbl, integration_summary = integration_diagnostics(
        df=df,
        cols=[DEPENDENT_VAR] + candidate_regressors,
    )

    ardl_tables = run_ardl_workflow(
        df=df,
        dependent_var=DEPENDENT_VAR,
        regressor_cols=candidate_regressors,
    )

    output_tables: dict[str, pd.DataFrame] = {
        "ciss_meta": ciss_meta,
        "integration_orders": integration_tbl,
        "integration_summary": integration_summary,
    }
    output_tables.update(ardl_tables)

    write_excel(OUTPUT_EXCEL, output_tables)

    print(f"Done. Rezultate ARDL: {OUTPUT_EXCEL.resolve()}")
    print(f"Done. Ploturi ARDL: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
