# Quantitative Macro-Financial Modeling: BET Index Dynamics (2007–2025)

An empirical econometrics pipeline evaluating the transmission channels of domestic macroeconomic fundamentals versus external financial factors and systemic stress on the Bucharest Stock Exchange (BET Index).

## Research Scope & Dataset
* **Sample Horizon:** March 2007 – June 2025 (Monthly frequency, $N = 220$ observations) covering major market regimes: the 2008 Global Financial Crisis, European Sovereign Debt Crisis, COVID-19 shock, and the 2022–2023 inflationary cycle.
* **Target Indicator:** Log returns of the Bucharest Stock Exchange benchmark (`r_BET`).
* **Domestic Macro Fundamentals:** Broad Money Growth (`r_Masa monetara`), Inflation (`INF`), Unemployment Rate (`UNEMP`), 10-Year Romanian Government Bond Yields (`Y10- Romania`).
* **External Financial & Risk Drivers:** Global Equity Benchmark (`r_MSCI`), Exchange Rate dynamics (`r_FX` - EUR/RON), and the ECB Composite Indicator of Systemic Stress (`CISS`).

---

## Methodological Framework

1. **Stationarity & Data Diagnostics:**
   - Evaluated integration orders via dual unit root testing: Augmented Dickey-Fuller (ADF, drift, AIC lag selection) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS, automatic bandwidth).
   - Confirmed all transformed series are stationary $I(0)$ at the 5% significance level.
   - Identified significant excess kurtosis (7.98) and negative skewness (-1.37) in BET returns, indicating fat tails and downside asymmetric vulnerability.

2. **Benchmark Dynamic OLS:**
   - Formulated static and dynamic ($L1$) benchmark specifications with HAC Newey-West standard errors ($m = 4 \times (n/100)^{2/9}$).
   - Achieved $R^2 = 57.91\%$ ($R^2_{adj} = 54.80\%$), passing Breusch-Godfrey autocorrelation diagnostics ($p = 0.413$, $DW = 2.013$).

3. **Multivariate Vector Autoregression with Exogenous Inputs (VARX):**
   - Endogenous block: `r_BET`, `r_Masa monetara`, `INF`, `UNEMP`, `Y10- Romania`, `r_MSCI`, `CISS`.
   - Exogenous control: `r_FX` (EUR/RON pass-through).
   - Selected optimal lag $p = 1$ based on the Bayesian Information Criterion (BIC = -63.66).
   - Estimated Generalized Impulse Response Functions (GIRF; Koop, Pesaran & Potter, 1996) with 1,000 non-parametric bootstrap iterations (95% CI) over a 12-month horizon.

4. **Autoregressive Distributed Lag (ARDL-UECM):**
   - Unrestricted Error Correction Model grid search ($p \in [1, 3]$, $q \in [1, 3]$) under BIC.
   - Pesaran, Shin & Smith (PSS) Bounds Testing to evaluate level cointegration and dynamic equilibrium speed of adjustment.

---

## Key Empirical Findings

* **External Factors Dominate Domestic Fundamentals:**
  - **Global Equity Pass-Through (`r_MSCI`):** Robust positive determinant across all specifications ($\beta_{OLS, contemporary} = 0.615, p < 0.001$; $\beta_{VARX, L1} = 0.569, p < 0.001$; $\beta_{ARDL, L1} = 1.184, p < 0.001$).
  - **Foreign Exchange Risk (`r_FX`):** Domestic currency depreciation severely contracts equity returns, dominating lagged dynamics ($\beta_{OLS, L1} = -1.041, p < 0.001$; $\beta_{VARX} = -0.710, p = 0.006$; $\beta_{ARDL, L1} = -1.330, p < 0.001$). A 1% monthly depreciation in EUR/RON translates into an immediate 0.7%–1.3% decline in BET returns.
* **Domestic Macro Insulation:**
  - Domestic variables (`INF`, `UNEMP`, `Y10- Romania`) demonstrated no statistically systematic explanatory power on monthly stock returns, confirming that fundamental economic signals act indirectly over longer macro horizons.
  - Money supply expansion (`r_Masa monetara`) exhibited a lagged negative effect in VARX ($\beta = -0.449, p = 0.079$), signaling market concerns over upcoming inflationary pressures and liquidity pooling in the banking sector rather than equity capital.
* **Transient Systemic Stress Transmission:**
  - GIRF analysis revealed that shocks to the ECB systemic stress index (`CISS`) trigger a statistically significant 4-month oscillating adjustment process before dissipating, reflecting liquidity illiquidity absorption in frontier/emerging markets.
* **Cointegration & Adjustment Dynamics:**
  - The F-Bounds test rejected the null of no level relationship ($F = 44.443, p < 0.001$). The error-correction speed of adjustment term ($ECT_{-1} = -0.981, p < 0.001$) confirms near-instantaneous mean reversion consistent with stationary financial returns.

---

## Project Structure
* `varx_benchmark_analysis.py`: Data ingestion, ADF/KPSS testing, dynamic OLS benchmarks, VARX system estimation, and GIRF bootstrap generation.
* `ardl_bounds_testing.py`: ARDL grid search, UECM formulation, and Pesaran F-Bounds test pipeline.
* `plots/`: Exported figures (time-series distributions, correlation matrices, residual ACF, GIRF curves).

## Requirements
* Python 3.10+
* `statsmodels`, `pandas`, `numpy`, `matplotlib`, `xlsxwriter`
