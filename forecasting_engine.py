"""
forecasting_engine.py
=====================
Production-ready ML forecasting engine for Bankruptcy GenBI.

Models used:
  - Linear Regression          (baseline trend)
  - Polynomial Regression      (non-linear trend capture)
  - Exponential Smoothing      (pure numpy, respects recency)
  - Ensemble Average           (blend all three for robustness)

Returns structured prediction objects consumed directly by app.py.
"""

from __future__ import annotations
import sqlite3
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger("bankruptcy_genbi.forecast")

# ---------------------------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------------------------

def _get_table() -> str:
    try:
        conn = sqlite3.connect("data.db")
        cur  = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND (name='uploaded_data' OR name='bankruptcy4')"
        )
        rows = cur.fetchall()
        conn.close()
        names = [r[0] for r in rows]
        return "uploaded_data" if "uploaded_data" in names else (names[0] if names else "uploaded_data")
    except Exception:
        return "uploaded_data"


def load_monthly_series(
    chapter_filter: Optional[int] = None,
    state_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    prose_filter: Optional[str] = None,
    client_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return a monthly time-series DataFrame:
      Month (YYYY-MM) | Filing_Count | Avg_Risk | Chapter_7 | Chapter_11 | Chapter_13
    Gaps are filled with 0 so the series is contiguous.
    """
    table = _get_table()
    try:
        conn = sqlite3.connect("data.db")
        cur  = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = {r[1] for r in cur.fetchall()}

        date_col  = "Open_date" if "Open_date" in cols else ("date_filed" if "date_filed" in cols else None)
        score_col = "match_score" if "match_score" in cols else None

        if not date_col:
            conn.close()
            return pd.DataFrame()

        where = []
        if chapter_filter and "chapter" in cols:
            where.append(f"chapter = {chapter_filter}")
        if state_filter and "State" in cols:
            where.append(f"State = '{state_filter}'")
        if status_filter and "status" in cols:
            where.append(f"status = '{status_filter}'")
        if prose_filter and "prose_indicator" in cols:
            if prose_filter in ["1", "Y", "Yes"]:
                where.append("(prose_indicator = '1' OR prose_indicator = 'Y' OR prose_indicator = 'Yes')")
            elif prose_filter in ["0", "N", "No"]:
                where.append("(prose_indicator = '0' OR prose_indicator = 'N' OR prose_indicator = 'No')")
            else:
                where.append(f"prose_indicator = '{prose_filter}'")

        if client_filter and "client" in cols:
            where.append(f"client = '{client_filter}'")

        base = " AND ".join(where) if where else "1=1"
        avg_score = f"ROUND(AVG({score_col}), 2)" if score_col else "NULL"

        # Base monthly aggregation
        df = pd.read_sql_query(
            f"""
            SELECT substr({date_col},1,7) AS Month,
                   COUNT(*) AS Filing_Count,
                   {avg_score} AS Avg_Risk
            FROM {table}
            WHERE {date_col} IS NOT NULL AND {date_col} != ''
                  AND {date_col} LIKE '20%'
                  AND ({base})
            GROUP BY Month
            ORDER BY Month
            """,
            conn,
        )

        # Per-chapter monthly volumes (pivot later)
        df_chap = pd.DataFrame()
        if "chapter" in cols:
            df_chap = pd.read_sql_query(
                f"""
                SELECT substr({date_col},1,7) AS Month,
                       chapter,
                       COUNT(*) AS cnt
                FROM {table}
                WHERE {date_col} IS NOT NULL AND {date_col} != ''
                      AND {date_col} LIKE '20%'
                      AND chapter IS NOT NULL
                      AND ({base})
                GROUP BY Month, chapter
                ORDER BY Month
                """,
                conn,
            )

        conn.close()

        if df.empty:
            return pd.DataFrame()

        # Fill gaps in time series
        df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
        df = df.set_index("Month")
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="MS")
        df = df.reindex(full_idx, fill_value=0)
        df.index.name = "Month"
        df["Avg_Risk"] = df["Avg_Risk"].replace(0, np.nan).ffill().bfill()

        # Merge chapter pivot
        if not df_chap.empty:
            df_chap["Month"] = pd.to_datetime(df_chap["Month"], format="%Y-%m")
            pivot = df_chap.pivot_table(
                index="Month", columns="chapter", values="cnt", aggfunc="sum", fill_value=0
            )
            pivot.columns = [f"Ch_{int(c)}" for c in pivot.columns]
            df = df.join(pivot, how="left").fillna(0)

        df = df.reset_index()
        df["t"] = np.arange(len(df))          # numeric time index
        return df

    except Exception as e:
        logger.error(f"load_monthly_series error: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# MODEL HELPERS
# ---------------------------------------------------------------------------

def _linear_forecast(t_hist: np.ndarray, y_hist: np.ndarray,
                     t_future: np.ndarray) -> np.ndarray:
    """OLS linear regression via numpy."""
    A = np.vstack([t_hist, np.ones_like(t_hist)]).T
    coef, *_ = np.linalg.lstsq(A, y_hist, rcond=None)
    return np.polyval([coef[0], coef[1]], t_future)


def _poly_forecast(t_hist: np.ndarray, y_hist: np.ndarray,
                   t_future: np.ndarray, deg: int = 2) -> np.ndarray:
    """Polynomial regression (degree 2 by default)."""
    try:
        coef = np.polyfit(t_hist, y_hist, deg)
        return np.polyval(coef, t_future).clip(0)
    except Exception:
        return _linear_forecast(t_hist, y_hist, t_future)


def _exp_smooth_forecast(y_hist: np.ndarray, horizon: int,
                         alpha: float = 0.3) -> np.ndarray:
    """
    Double exponential smoothing (Holt's method) - numpy only.
    Returns `horizon` future values.
    """
    if len(y_hist) < 2:
        return np.full(horizon, y_hist[-1] if len(y_hist) else 0.0)

    # Level and trend initialisation
    level  = float(y_hist[0])
    trend  = float(y_hist[1] - y_hist[0])
    beta   = 0.1

    for v in y_hist[1:]:
        prev_level = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend

    return np.array([max(0.0, level + i * trend) for i in range(1, horizon + 1)])


def _sklearn_forecast(t_hist: np.ndarray, y_hist: np.ndarray,
                      t_future: np.ndarray) -> Optional[np.ndarray]:
    """
    Gradient Boosting or Ridge regression via sklearn (if available).
    Falls back to None if sklearn not installed.
    """
    try:
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1.0))
        model.fit(t_hist.reshape(-1, 1), y_hist)
        return model.predict(t_future.reshape(-1, 1)).clip(0)
    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"sklearn forecast error: {e}")
        return None


# ---------------------------------------------------------------------------
# PRIMARY FORECAST FUNCTION
# ---------------------------------------------------------------------------

def run_filing_forecast(
    df_series: pd.DataFrame,
    horizon_months: int = 12,
    target_col: str = "Filing_Count",
) -> dict:
    """
    Run ensemble forecast on the monthly filing series.

    Returns
    -------
    {
      "history":    pd.DataFrame with Month + Filing_Count + Avg_Risk,
      "forecast":   pd.DataFrame with Month, Linear, Poly, ExpSmooth, [Sklearn], Ensemble,
      "metrics":    dict of model accuracy metrics (MAPE, MAE on last 12 months),
      "insights":   list[str] of business-level insight strings,
      "trend":      "increasing" | "decreasing" | "stable",
      "trend_pct":  float,
    }
    """
    if df_series.empty or target_col not in df_series.columns:
        return {}

    y = df_series[target_col].values.astype(float)
    t = df_series["t"].values.astype(float)

    if len(y) < 6:
        return {}

    # --- Hold-out validation (last 6 months) ---
    split = max(6, len(y) - 6)
    t_train, y_train = t[:split], y[:split]
    t_val,   y_val   = t[split:], y[split:]

    # --- Future time index ---
    t_future = np.arange(t[-1] + 1, t[-1] + horizon_months + 1)
    future_months = pd.date_range(
        df_series["Month"].iloc[-1] + pd.DateOffset(months=1),
        periods=horizon_months,
        freq="MS",
    )

    # --- Build forecasts ---
    lin_future  = _linear_forecast(t_train, y_train, t_future)
    poly_future = _poly_forecast(t_train, y_train, t_future, deg=2)
    exp_future  = _exp_smooth_forecast(y_train, horizon_months, alpha=0.3)
    skl_future  = _sklearn_forecast(t_train, y_train, t_future)

    preds = [lin_future, poly_future, exp_future]
    names = ["Linear Regression", "Polynomial (deg-2)", "Exp. Smoothing"]
    if skl_future is not None:
        preds.append(skl_future)
        names.append("Ridge Poly (sklearn)")

    ensemble = np.mean(np.vstack(preds), axis=0).clip(0)

    forecast_df = pd.DataFrame({"Month": future_months})
    for name, vals in zip(names, preds):
        forecast_df[name] = np.round(vals, 1)
    forecast_df["Ensemble Forecast"] = np.round(ensemble, 1)

    # --- Validation metrics (on hold-out) ---
    val_lin   = _linear_forecast(t_train, y_train, t_val)
    val_poly  = _poly_forecast(t_train, y_train, t_val, deg=2)
    val_exp   = _exp_smooth_forecast(y_train, len(t_val), alpha=0.3)
    val_preds = [val_lin, val_poly, val_exp]
    val_skl   = _sklearn_forecast(t_train, y_train, t_val)
    if val_skl is not None:
        val_preds.append(val_skl)
    val_ensemble = np.mean(np.vstack(val_preds), axis=0).clip(0)

    mape = _mape(y_val, val_ensemble)
    mae  = float(np.mean(np.abs(y_val - val_ensemble)))
    rmse = float(np.sqrt(np.mean((y_val - val_ensemble) ** 2)))

    # --- Trend analysis ---
    recent = y[-12:] if len(y) >= 12 else y
    first_half  = recent[:len(recent)//2].mean()
    second_half = recent[len(recent)//2:].mean()
    trend_pct   = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0.0

    if   trend_pct >  5:  trend = "increasing"
    elif trend_pct < -5:  trend = "decreasing"
    else:                 trend = "stable"

    # --- Business insights ---
    next_q_total = int(round(ensemble[:3].sum()))
    next_y_total = int(round(ensemble.sum()))
    peak_month   = future_months[int(np.argmax(ensemble))].strftime("%B %Y")
    low_month    = future_months[int(np.argmin(ensemble))].strftime("%B %Y")
    avg_hist     = float(np.mean(y[-12:])) if len(y) >= 12 else float(np.mean(y))

    insights = _build_insights(
        trend=trend, trend_pct=trend_pct, avg_hist=avg_hist,
        next_q_total=next_q_total, next_y_total=next_y_total,
        peak_month=peak_month, low_month=low_month,
        mape=mape, ensemble=ensemble, y=y,
        df_series=df_series,
    )

    return {
        "history":   df_series[["Month", target_col, "Avg_Risk"]].copy(),
        "forecast":  forecast_df,
        "metrics":   {"MAPE (%)": round(mape, 1), "MAE": round(mae, 1), "RMSE": round(rmse, 1)},
        "insights":  insights,
        "trend":     trend,
        "trend_pct": round(trend_pct, 1),
        "model_names": names + ["Ensemble Forecast"],
        "next_q":    next_q_total,
        "next_year": next_y_total,
        "peak_month": peak_month,
    }


# ---------------------------------------------------------------------------
# CHAPTER-LEVEL BREAKDOWN FORECAST
# ---------------------------------------------------------------------------

def run_chapter_forecast(df_series: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    """
    Forecast each chapter (Ch_7, Ch_11, Ch_13) independently.
    Returns a single DataFrame with Month + Ch_7_forecast + Ch_11_forecast + Ch_13_forecast.
    """
    results = {}
    future_months = pd.date_range(
        df_series["Month"].iloc[-1] + pd.DateOffset(months=1),
        periods=horizon_months, freq="MS",
    )
    results["Month"] = future_months

    for col in [c for c in df_series.columns if c.startswith("Ch_")]:
        y = df_series[col].values.astype(float)
        t = df_series["t"].values.astype(float)
        if y.sum() == 0:
            continue
        t_future = np.arange(t[-1] + 1, t[-1] + horizon_months + 1)
        lin  = _linear_forecast(t, y, t_future)
        poly = _poly_forecast(t, y, t_future)
        exp  = _exp_smooth_forecast(y, horizon_months)
        ens  = np.mean([lin, poly, exp], axis=0).clip(0)
        results[col.replace("Ch_", "Chapter ")] = np.round(ens, 1)

    return pd.DataFrame(results) if len(results) > 1 else pd.DataFrame()


# ---------------------------------------------------------------------------
# RISK SCORE TREND FORECAST
# ---------------------------------------------------------------------------

def run_risk_score_forecast(df_series: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    """Forecast average match risk score trajectory."""
    if "Avg_Risk" not in df_series.columns or df_series["Avg_Risk"].isna().all():
        return pd.DataFrame()

    df_valid = df_series.dropna(subset=["Avg_Risk"])
    y = df_valid["Avg_Risk"].values.astype(float)
    t = df_valid["t"].values.astype(float)

    if len(y) < 4:
        return pd.DataFrame()

    t_future = np.arange(t[-1] + 1, t[-1] + horizon_months + 1)
    future_months = pd.date_range(
        df_series["Month"].iloc[-1] + pd.DateOffset(months=1),
        periods=horizon_months, freq="MS",
    )

    poly = _poly_forecast(t, y, t_future, deg=2).clip(0, 100)
    exp  = _exp_smooth_forecast(y, horizon_months, alpha=0.25).clip(0, 100)
    ens  = np.mean([poly, exp], axis=0)

    return pd.DataFrame({
        "Month": future_months,
        "Forecasted Avg Risk Score": np.round(ens, 1),
        "Upper Bound": np.round(np.minimum(ens + ens.std() * 1.5, 100), 1),
        "Lower Bound": np.round(np.maximum(ens - ens.std() * 1.5, 0),  1),
    })


# ---------------------------------------------------------------------------
# STATE-LEVEL ANOMALY DETECTION
# ---------------------------------------------------------------------------

def detect_state_anomalies(top_n: int = 10) -> pd.DataFrame:
    """
    Detect states whose recent 12-month filing trend diverges
    significantly from their historical mean (z-score > 1.5).
    """
    table = _get_table()
    try:
        conn = sqlite3.connect("data.db")
        cur  = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = {r[1] for r in cur.fetchall()}
        if "State" not in cols or "Open_date" not in cols:
            conn.close()
            return pd.DataFrame()

        df = pd.read_sql_query(
            f"""
            SELECT State,
                   substr(Open_date,1,4) AS Year,
                   COUNT(*) AS Filings
            FROM {table}
            WHERE Open_date IS NOT NULL AND Open_date LIKE '20%'
                  AND State IS NOT NULL AND State != ''
            GROUP BY State, Year
            ORDER BY State, Year
            """,
            conn,
        )
        conn.close()

        if df.empty:
            return pd.DataFrame()

        results = []
        for state, grp in df.groupby("State"):
            grp = grp.sort_values("Year")
            vals = grp["Filings"].values.astype(float)
            if len(vals) < 3:
                continue
            mu, sigma = vals[:-1].mean(), vals[:-1].std()
            last = vals[-1]
            z = (last - mu) / sigma if sigma > 0 else 0.0
            trend = " Spike" if z > 1.5 else (" Drop" if z < -1.5 else " Normal")
            results.append({
                "State": state,
                "Historical Avg (Filings/yr)": round(mu, 1),
                "Latest Year Filings": int(last),
                "Z-Score": round(z, 2),
                "Anomaly Status": trend,
            })

        out = pd.DataFrame(results).sort_values("Z-Score", key=abs, ascending=False)
        return out.head(top_n)

    except Exception as e:
        logger.error(f"detect_state_anomalies error: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# RISK CONCENTRATION HEATMAP DATA
# ---------------------------------------------------------------------------

def get_risk_heatmap_data() -> pd.DataFrame:
    """
    Return a pivot of avg match_score by (State  Chapter)
    suitable for a Plotly heatmap.
    """
    table = _get_table()
    try:
        conn = sqlite3.connect("data.db")
        cur  = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = {r[1] for r in cur.fetchall()}
        if not all(c in cols for c in ["State", "chapter", "match_score"]):
            conn.close()
            return pd.DataFrame()

        df = pd.read_sql_query(
            f"""
            SELECT State,
                   chapter,
                   ROUND(AVG(match_score), 1) AS Avg_Risk,
                   COUNT(*) AS Case_Count
            FROM {table}
            WHERE State IS NOT NULL AND State != ''
                  AND chapter IS NOT NULL
                  AND match_score IS NOT NULL
            GROUP BY State, chapter
            HAVING COUNT(*) >= 2
            """,
            conn,
        )
        conn.close()
        if df.empty:
            return pd.DataFrame()

        pivot = df.pivot_table(
            index="State", columns="chapter", values="Avg_Risk", aggfunc="mean"
        )
        pivot.columns = [f"Ch {int(c)}" for c in pivot.columns]
        return pivot.round(1)

    except Exception as e:
        logger.error(f"get_risk_heatmap_data error: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# HELPER: MAPE
# ---------------------------------------------------------------------------

def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


# ---------------------------------------------------------------------------
# BUSINESS INSIGHT GENERATOR
# ---------------------------------------------------------------------------

def _build_insights(
    trend, trend_pct, avg_hist, next_q_total, next_y_total,
    peak_month, low_month, mape, ensemble, y, df_series,
) -> list[str]:
    ins = []

    # 1. Trend & Caseload impact
    trend_lbl = "growth" if trend == "increasing" else "decline" if trend == "decreasing" else "stable pattern"
    impact_text = (
        "growing caseload that may require additional resource deployment" if trend == "increasing"
        else "shrinking caseload, allowing operational resources to be optimized" if trend == "decreasing"
        else "stable pipeline for operational capacity planning"
    )
    ins.append(
        f"**Caseload Direction:** Filing volume shows an **{trend} trend** (a **{abs(trend_pct):.1f}% {trend_lbl}** "
        f"compared to the prior period). This indicates a {impact_text}."
    )

    # 2. Volume forecast
    monthly_run = round(next_q_total / 3)
    ins.append(
        f"**Short-Term Caseload Pipeline:** A total of **{next_q_total:,} filings** are projected over the "
        f"next 90 days. This represents an average run-rate of **{monthly_run:,} filings per month**, "
        f"which should be factored into immediate operational capacity planning."
    )
    
    annual_historical = avg_hist * 12
    deviation = ((next_y_total - annual_historical) / annual_historical * 100) if annual_historical > 0 else 0.0
    ins.append(
        f"**12-Month Portfolio Volume:** Total projected filings for the coming year are estimated at "
        f"**{next_y_total:,}**. Compared to a historical baseline of **{annual_historical:.0f}** annual filings "
        f"(averaging **{avg_hist:.1f}/month**), this represents a **{deviation:+.1f}% deviation** from historic norms."
    )

    # 3. Seasonal peak / trough
    ins.append(
        f"**Peak Workload Window:** Filing activity is expected to peak in **{peak_month}**. "
        f"Operational managers should prepare for higher processing volume during this period by aligning staffing schedules."
    )
    ins.append(
        f"**Low-Volume Planning Window:** The quietest filing period is projected to occur in **{low_month}**. "
        f"This window is ideal for conducting team audits, system training, or clearing outstanding backlogs."
    )

    # 4. Risk alert
    if "Avg_Risk" in df_series.columns:
        latest_risk = df_series["Avg_Risk"].dropna().iloc[-1] if not df_series["Avg_Risk"].dropna().empty else 0
        if latest_risk >= 90:
            ins.append(
                f"**Portfolio Risk Status:** The latest average match risk score is **{latest_risk:.1f}**, "
                f"which is critically elevated. Immediate compliance triage and audits are recommended to mitigate risk exposure."
            )
        elif latest_risk >= 75:
            ins.append(
                f"**Portfolio Risk Status:** The latest average match risk score stands at **{latest_risk:.1f}** (elevated). "
                f"Proactive portfolio monitoring and escalation protocols should be active."
            )
        else:
            ins.append(
                f"**Portfolio Risk Status:** The latest average match risk score is **{latest_risk:.1f}** (stable). "
                f"All metrics remain within acceptable thresholds; continue standard monitoring cadence."
            )

    # 5. Volatility (Replaces mathematical "coefficient of variation" with business consistency terms)
    cv = (np.std(y) / np.mean(y) * 100) if np.mean(y) > 0 else 0
    if cv > 40:
        ins.append(
            f"**Filing Volatility Alert:** Filings exhibit highly unpredictable patterns (variation rate: **{cv:.1f}%**). "
            f"Localized volume spikes are likely; strategic resource buffers should be maintained."
        )
    elif cv < 15:
        ins.append(
            f"**Stable Filing Pattern:** Low volatility (variation rate: **{cv:.1f}%**) suggests highly consistent "
            f"filing behavior, allowing for highly predictable operational scheduling."
        )

    # 6. Model reliability (Replaces technical terms like MAPE and holdout validations)
    if not np.isnan(mape):
        accuracy = 100 - mape
        confidence = "high" if accuracy >= 85 else "moderate"
        ins.append(
            f"**Forecast Reliability:** Back-testing indicates a **{confidence} forecast confidence** "
            f"(historical model accuracy score: **{accuracy:.1f}%**). "
            f"{'These projections can be directly integrated into strategic plans.' if accuracy >= 85 else 'We recommend utilizing these trends as directional guidelines rather than exact targets.'}"
        )

    return ins
