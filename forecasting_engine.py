"""
forecasting_engine.py
=====================
High-accuracy ML forecasting engine for Bankruptcy GenBI.

Key design decisions:
  1. Seasonal Decomposition (STL-lite) — strips trend+seasonal before fitting,
     allowing every model to work on stationary residuals.
  2. Seasonal Naïve with Drift — highly accurate for monthly data with seasons.
  3. Damped Holt's Trend — prevents runaway trend extrapolation.
  4. Holt-Winters (additive + multiplicative fallback) — full triple smoothing.
  5. Gradient Boosting / Ridge (sklearn, optional) — on feature-engineered data.
  6. Validation-weighted ensemble — best hold-out model gets highest weight.
  7. Consistent in-sample fit — fitted line uses SAME weights as the forecast.
"""

from __future__ import annotations
import sqlite3
import datetime
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger("bankruptcy_genbi.forecast")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

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
    start_date: Optional[datetime.date | str] = None,
    end_date: Optional[datetime.date | str] = None,
) -> pd.DataFrame:
    """
    Return a monthly time-series DataFrame:
      Month (period) | Filing_Count | Avg_Risk | Ch_7 | Ch_11 | Ch_13
    Gaps are forward-filled / zero-filled so the series is contiguous.
    """
    table = _get_table()
    try:
        conn = sqlite3.connect("data.db")
        cur  = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = {r[1] for r in cur.fetchall()}

        # Resolve database column names case-insensitively
        def get_db_col(logical_name):
            for c in cols:
                if c.lower() == logical_name.lower():
                    return c
            # Fallback client mapping
            if logical_name.lower() == 'client':
                for c in cols:
                    if c.lower() == 'client_name':
                        return c
            return None

        date_col  = get_db_col("open_date") or get_db_col("date_filed")
        score_col = get_db_col("match_score")

        if not date_col:
            conn.close()
            return pd.DataFrame()

        where = []
        chapter_col = get_db_col("chapter")
        if chapter_filter and chapter_col:
            where.append(f"{chapter_col} = {chapter_filter}")
        state_col = get_db_col("state")
        if state_filter and state_col:
            where.append(f"{state_col} = '{state_filter}'")
        status_col = get_db_col("status")
        if status_filter and status_col:
            where.append(f"{status_col} = '{status_filter}'")
        prose_col = get_db_col("prose_indicator")
        if prose_filter and prose_col:
            if prose_filter in ["1", "Y", "Yes"]:
                where.append(f"({prose_col} = '1' OR {prose_col} = 'Y' OR {prose_col} = 'Yes' OR {prose_col} = 1)")
            elif prose_filter in ["0", "N", "No"]:
                where.append(f"({prose_col} = '0' OR {prose_col} = 'N' OR {prose_col} = 'No' OR {prose_col} = 0)")
            else:
                where.append(f"{prose_col} = '{prose_filter}'")
        client_col = get_db_col("client_name") or get_db_col("client")
        if client_filter and client_col:
            where.append(f"{client_col} = '{client_filter}'")
        if start_date:
            s = start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date)
            where.append(f"{date_col} >= '{s}'")
        if end_date:
            e = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)
            where.append(f"{date_col} <= '{e}'")

        base      = " AND ".join(where) if where else "1=1"
        avg_score = f"ROUND(AVG({score_col}), 2)" if score_col else "NULL"

        df = pd.read_sql_query(
            f"""
            SELECT substr({date_col},1,7) AS Month,
                   COUNT(*) AS Filing_Count,
                   {avg_score} AS Avg_Risk
            FROM {table}
            WHERE {date_col} IS NOT NULL AND {date_col} != ''
                  AND {date_col} LIKE '20%' AND ({base})
            GROUP BY Month ORDER BY Month
            """, conn,
        )

        df_chap = pd.DataFrame()
        if "chapter" in cols:
            df_chap = pd.read_sql_query(
                f"""
                SELECT substr({date_col},1,7) AS Month, chapter, COUNT(*) AS cnt
                FROM {table}
                WHERE {date_col} IS NOT NULL AND {date_col} != ''
                      AND {date_col} LIKE '20%' AND chapter IS NOT NULL AND ({base})
                GROUP BY Month, chapter ORDER BY Month
                """, conn,
            )
        conn.close()

        if df.empty:
            return pd.DataFrame()

        df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
        df = df.set_index("Month")

        sd_ts = pd.to_datetime(str(start_date)[:7], format="%Y-%m") if start_date else df.index.min()
        ed_ts = pd.to_datetime(str(end_date)[:7],   format="%Y-%m") if end_date   else df.index.max()

        if pd.notna(sd_ts) and pd.notna(ed_ts) and sd_ts <= ed_ts:
            full_idx = pd.date_range(sd_ts, ed_ts, freq="MS")
        else:
            full_idx = pd.date_range(df.index.min(), df.index.max(), freq="MS")

        df = df.reindex(full_idx, fill_value=0)
        df.index.name = "Month"
        df["Avg_Risk"] = df["Avg_Risk"].replace(0, np.nan).ffill().bfill()

        if not df_chap.empty:
            df_chap["Month"] = pd.to_datetime(df_chap["Month"], format="%Y-%m")
            pivot = df_chap.pivot_table(index="Month", columns="chapter", values="cnt", aggfunc="sum", fill_value=0)
            pivot.columns = [f"Ch_{int(c)}" for c in pivot.columns]
            df = df.join(pivot, how="left").fillna(0)

        df = df.reset_index()
        df["t"] = np.arange(len(df))
        return df

    except Exception as e:
        logger.error(f"load_monthly_series error: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _winsorize(y: np.ndarray, p: float = 0.05) -> np.ndarray:
    """Clip extreme values at the p-th and (1-p)-th percentiles."""
    lo, hi = np.nanpercentile(y, p * 100), np.nanpercentile(y, (1 - p) * 100)
    return np.clip(y, lo, hi)


def _seasonal_indices(y: np.ndarray, period: int = 12) -> np.ndarray:
    """
    Compute additive seasonal indices using a centred moving-average baseline.
    Returns an array of length `period` (one index per season position).
    """
    n = len(y)
    if n < period * 2:
        return np.zeros(period)

    # Centred moving average (trend proxy)
    half = period // 2
    trend = np.convolve(y, np.ones(period) / period, mode="valid")
    # Align: if even period, we need a half-shift
    if period % 2 == 0:
        trend = (trend[:-1] + trend[1:]) / 2.0
        offset = half
    else:
        offset = half

    n_trend = len(trend)
    residual = y[offset: offset + n_trend] - trend  # additive residuals

    # Average residuals by season position
    indices = np.zeros(period)
    counts  = np.zeros(period)
    for i, v in enumerate(residual):
        s = (offset + i) % period
        indices[s] += v
        counts[s]  += 1

    with np.errstate(invalid="ignore"):
        indices = np.where(counts > 0, indices / counts, 0.0)

    # Centre so they sum to 0
    indices -= indices.mean()
    return indices


def _deseasonalize(y: np.ndarray, season_idx: np.ndarray, period: int = 12) -> np.ndarray:
    n = len(y)
    s = np.array([season_idx[i % period] for i in range(n)])
    return y - s


def _reseasonalize(vals: np.ndarray, season_idx: np.ndarray,
                   start_pos: int, period: int = 12) -> np.ndarray:
    n = len(vals)
    s = np.array([season_idx[(start_pos + i) % period] for i in range(n)])
    return vals + s


# ─────────────────────────────────────────────────────────────────────────────
# MODEL IMPLEMENTATIONS  (all operate on deseasonalised data)
# ─────────────────────────────────────────────────────────────────────────────

def _linear_fit_predict(t: np.ndarray, y: np.ndarray,
                        t_pred: np.ndarray) -> np.ndarray:
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[0] * t_pred + coef[1]


def _poly_fit_predict(t: np.ndarray, y: np.ndarray,
                      t_pred: np.ndarray, deg: int = 2) -> np.ndarray:
    try:
        coef = np.polyfit(t, y, deg)
        return np.polyval(coef, t_pred)
    except Exception:
        return _linear_fit_predict(t, y, t_pred)


def _damped_holt(y: np.ndarray, horizon: int) -> np.ndarray:
    """
    Damped-trend Holt's exponential smoothing.
    Damps the trend by phi each step to prevent overconfident extrapolation.
    Parameters are chosen via grid search over last 20% of series.
    """
    n = len(y)
    if n < 3:
        return np.full(horizon, float(np.mean(y)))

    val_start = max(2, int(n * 0.8))
    best = (float("inf"), 0.3, 0.1, 0.9)  # (mse, alpha, beta, phi)

    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for beta in [0.01, 0.05, 0.1, 0.15, 0.2]:
            for phi in [0.80, 0.85, 0.90, 0.95, 0.98, 1.0]:
                lvl = float(y[0])
                trd = float(y[1] - y[0]) if n > 1 else 0.0
                for v in y[:val_start]:
                    pl = lvl
                    lvl = alpha * v + (1 - alpha) * (lvl + phi * trd)
                    trd = beta * (lvl - pl) + (1 - beta) * phi * trd
                preds = []
                for h in range(1, n - val_start + 1):
                    cum_phi = sum(phi ** i for i in range(1, h + 1))
                    preds.append(max(0.0, lvl + cum_phi * trd))
                if preds:
                    actual = y[val_start: val_start + len(preds)]
                    mse = float(np.mean((actual - np.array(preds)) ** 2))
                    if mse < best[0]:
                        best = (mse, alpha, beta, phi)

    _, a, b, phi = best
    lvl = float(y[0])
    trd = float(y[1] - y[0]) if n > 1 else 0.0
    for v in y[1:]:
        pl = lvl
        lvl = a * v + (1 - a) * (lvl + phi * trd)
        trd = b * (lvl - pl) + (1 - b) * phi * trd

    result = []
    for h in range(1, horizon + 1):
        cum_phi = sum(phi ** i for i in range(1, h + 1))
        result.append(max(0.0, lvl + cum_phi * trd))
    return np.array(result)


def _damped_holt_fitted(y: np.ndarray,
                         alpha: float = 0.3,
                         beta: float = 0.1,
                         phi: float = 0.9) -> np.ndarray:
    """One-step-ahead fitted values for damped Holt's (whole series)."""
    n = len(y)
    if n < 2:
        return y.copy()
    # Tune first
    best = (float("inf"), alpha, beta, phi)
    val_start = max(2, int(n * 0.8))
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for be in [0.01, 0.05, 0.1, 0.15, 0.2]:
            for ph in [0.80, 0.85, 0.90, 0.95, 0.98, 1.0]:
                lvl = float(y[0]); trd = float(y[1] - y[0]) if n > 1 else 0.0
                for v in y[:val_start]:
                    pl = lvl
                    lvl = a * v + (1 - a) * (lvl + ph * trd)
                    trd = be * (lvl - pl) + (1 - be) * ph * trd
                preds = [max(0.0, lvl + ph * trd)] * max(1, n - val_start)
                actual = y[val_start:val_start + len(preds)]
                mse = float(np.mean((actual - np.array(preds[:len(actual)])) ** 2))
                if mse < best[0]:
                    best = (mse, a, be, ph)
    _, a, be, ph = best

    lvl = float(y[0]); trd = float(y[1] - y[0]) if n > 1 else 0.0
    fitted = [lvl + ph * trd]
    for v in y[1:]:
        pl = lvl
        lvl = a * v + (1 - a) * (lvl + ph * trd)
        trd = be * (lvl - pl) + (1 - be) * ph * trd
        fitted.append(max(0.0, lvl + ph * trd))
    return np.array(fitted[:n]).clip(0)


def _holt_winters(y: np.ndarray, horizon: int, period: int = 12) -> Optional[np.ndarray]:
    """
    Holt-Winters with parameter grid search. Returns None if series too short.
    """
    n = len(y)
    if n < period * 2:
        return None

    val_start = max(period + 2, int(n * 0.8))
    best = (float("inf"), 0.3, 0.05, 0.2)

    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for beta in [0.01, 0.05, 0.1, 0.15]:
            for gamma in [0.1, 0.2, 0.3, 0.4]:
                lvl = float(np.mean(y[:period]))
                trd = float((np.mean(y[period:2*period]) - np.mean(y[:period])) / period)
                sea = [float(y[i]) - lvl for i in range(period)]
                for i in range(val_start):
                    s = i % period; v = float(y[i]); pl = lvl
                    lvl = alpha * (v - sea[s]) + (1 - alpha) * (pl + trd)
                    trd = beta * (lvl - pl) + (1 - beta) * trd
                    sea[s] = gamma * (v - lvl) + (1 - gamma) * sea[s]
                preds = []
                for h in range(1, n - val_start + 1):
                    s = (val_start + h - 1) % period
                    preds.append(max(0.0, lvl + h * trd + sea[s]))
                if preds:
                    actual = y[val_start: val_start + len(preds)]
                    mse = float(np.mean((actual - np.array(preds)) ** 2))
                    if mse < best[0]:
                        best = (mse, alpha, beta, gamma)

    _, alpha, beta, gamma = best
    lvl = float(np.mean(y[:period]))
    trd = float((np.mean(y[period:2*period]) - np.mean(y[:period])) / period)
    sea = [float(y[i]) - lvl for i in range(period)]
    for i in range(n):
        s = i % period; v = float(y[i]); pl = lvl
        lvl = alpha * (v - sea[s]) + (1 - alpha) * (pl + trd)
        trd = beta * (lvl - pl) + (1 - beta) * trd
        sea[s] = gamma * (v - lvl) + (1 - gamma) * sea[s]

    return np.array([max(0.0, lvl + h * trd + sea[(n + h - 1) % period])
                     for h in range(1, horizon + 1)])


def _holt_winters_fitted(y: np.ndarray, period: int = 12) -> Optional[np.ndarray]:
    """In-sample Holt-Winters one-step-ahead fitted values."""
    n = len(y)
    if n < period * 2:
        return None
    # Use default tuned parameters
    hw_preds = _holt_winters(y, horizon=1, period=period)
    if hw_preds is None:
        return None

    # Grid-searched parameters — refit full series for in-sample
    val_start = max(period + 2, int(n * 0.8))
    best = (float("inf"), 0.3, 0.05, 0.2)
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for beta in [0.01, 0.05, 0.1, 0.15]:
            for gamma in [0.1, 0.2, 0.3, 0.4]:
                lvl = float(np.mean(y[:period]))
                trd = float((np.mean(y[period:2*period]) - np.mean(y[:period])) / period)
                sea = [float(y[i]) - lvl for i in range(period)]
                for i in range(val_start):
                    s = i % period; v = float(y[i]); pl = lvl
                    lvl = alpha * (v - sea[s]) + (1 - alpha) * (pl + trd)
                    trd = beta * (lvl - pl) + (1 - beta) * trd
                    sea[s] = gamma * (v - lvl) + (1 - gamma) * sea[s]
                preds = []
                for h in range(1, n - val_start + 1):
                    s = (val_start + h - 1) % period
                    preds.append(max(0.0, lvl + h * trd + sea[s]))
                if preds:
                    actual = y[val_start:val_start + len(preds)]
                    mse = float(np.mean((actual - np.array(preds)) ** 2))
                    if mse < best[0]:
                        best = (mse, alpha, beta, gamma)

    _, alpha, beta, gamma = best
    lvl = float(np.mean(y[:period]))
    trd = float((np.mean(y[period:2*period]) - np.mean(y[:period])) / period)
    sea = [float(y[i]) - lvl for i in range(period)]
    fitted = []
    for i in range(n):
        s = i % period; v = float(y[i]); pl = lvl
        one_step = max(0.0, lvl + trd + sea[s])
        fitted.append(one_step)
        lvl = alpha * (v - sea[s]) + (1 - alpha) * (pl + trd)
        trd = beta * (lvl - pl) + (1 - beta) * trd
        sea[s] = gamma * (v - lvl) + (1 - gamma) * sea[s]
    return np.array(fitted).clip(0)


def _seasonal_naive_drift(y: np.ndarray, horizon: int, period: int = 12) -> Optional[np.ndarray]:
    """
    Seasonal Naïve with linear drift.
    Forecast = same month last year  +  overall drift per step.
    Highly accurate for monthly data with stable seasonality.
    """
    n = len(y)
    if n < period + 1:
        return None
    # Linear drift on the full series
    slope = (y[-1] - y[0]) / (n - 1) if n > 1 else 0.0
    result = []
    for h in range(1, horizon + 1):
        base_idx = n - period + ((h - 1) % period)
        if 0 <= base_idx < n:
            base_val = float(y[base_idx])
        else:
            base_val = float(y[-1])
        result.append(max(0.0, base_val + slope * h))
    return np.array(result)


def _seasonal_naive_drift_fitted(y: np.ndarray, period: int = 12) -> np.ndarray:
    """In-sample one-step-ahead for seasonal naïve with drift."""
    n = len(y)
    if n < period + 1:
        return y.copy()
    slope = (y[-1] - y[0]) / (n - 1) if n > 1 else 0.0
    fitted = []
    for i in range(n):
        prev_season = i - period
        if prev_season >= 0:
            fitted.append(max(0.0, float(y[prev_season]) + slope))
        else:
            fitted.append(float(y[i]))  # no seasonality data yet
    return np.array(fitted).clip(0)


def _sklearn_forecast(t: np.ndarray, y: np.ndarray,
                      t_pred: np.ndarray,
                      months: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Gradient Boosting (preferred) or Ridge with seasonal features (sklearn).
    Includes month-of-year as a feature for seasonal awareness.
    Falls back to None if sklearn not installed.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        # Build feature matrix: [t, t^2, sin(2pi*month/12), cos(2pi*month/12)]
        def _features(t_arr, month_arr=None):
            feats = [t_arr, t_arr ** 2]
            if month_arr is not None:
                feats.append(np.sin(2 * np.pi * month_arr / 12))
                feats.append(np.cos(2 * np.pi * month_arr / 12))
            return np.column_stack(feats)

        if months is not None:
            n_future = len(t_pred)
            # Infer future months from the last known month index
            last_m = int(months[-1]) if len(months) > 0 else 1
            future_months_arr = np.array([(last_m - 1 + i) % 12 + 1 for i in range(1, n_future + 1)])
            X_train = _features(t, months)
            X_pred  = _features(t_pred, future_months_arr)
        else:
            X_train = _features(t)
            X_pred  = _features(t_pred)

        model = make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                min_samples_leaf=2, subsample=0.8, random_state=42
            )
        )
        model.fit(X_train, y)
        return model.predict(X_pred).clip(0), model, months
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"GBM forecast error: {e}")

    # Fallback: Ridge with polynomial + seasonal features
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        def _features_simple(t_arr, month_arr=None):
            feats = [t_arr, t_arr ** 2, t_arr ** 3]
            if month_arr is not None:
                feats.append(np.sin(2 * np.pi * month_arr / 12))
                feats.append(np.cos(2 * np.pi * month_arr / 12))
            return np.column_stack(feats)

        if months is not None:
            last_m = int(months[-1]) if len(months) > 0 else 1
            future_months_arr = np.array([(last_m - 1 + i) % 12 + 1 for i in range(1, len(t_pred) + 1)])
            X_train = _features_simple(t, months)
            X_pred  = _features_simple(t_pred, future_months_arr)
        else:
            X_train = _features_simple(t)
            X_pred  = _features_simple(t_pred)

        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X_train, y)
        return model.predict(X_pred).clip(0), model, months
    except Exception as e:
        logger.warning(f"Ridge forecast error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _weighted_ensemble(preds: list, errors: list) -> np.ndarray:
    """Inverse-error weighted average. NaN errors get near-zero weight."""
    safe = [max(e, 1e-3) if not np.isnan(e) else 1e9 for e in errors]
    inv  = [1.0 / s for s in safe]
    total = sum(inv)
    weights = [w / total for w in inv]
    stack = np.vstack(preds)
    out = np.zeros(stack.shape[1])
    for w, p in zip(weights, stack):
        out += w * p
    return out.clip(0)


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY FORECAST FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_filing_forecast(
    df_series: pd.DataFrame,
    horizon_months: int = 12,
    target_col: str = "Filing_Count",
) -> dict:
    """
    High-accuracy ensemble forecast on the monthly filing time series.

    Returns
    -------
    {
      "history":   DataFrame  Month | Filing_Count | Avg_Risk
      "forecast":  DataFrame  Month | per-model columns | Ensemble Forecast
      "fitted":    DataFrame  Month | Fitted  (in-sample, same weights as forecast)
      "metrics":   dict       MAPE (%) | MAE | RMSE | Accuracy
      "insights":  list[str]
      "trend":     str
      "trend_pct": float
      "next_q":    int
      "next_year": int
      "peak_month": str
    }
    """
    if df_series.empty or target_col not in df_series.columns:
        return {}

    y_raw = df_series[target_col].values.astype(float)
    t     = df_series["t"].values.astype(float)
    months_of_year = df_series["Month"].dt.month.values  # 1–12

    n = len(y_raw)
    if n < 6:
        return {}

    # ── 1. Winsorise extreme outliers (5th-95th percentile) ─────────────────
    y = _winsorize(y_raw, p=0.02)

    # ── 2. Seasonal decomposition ────────────────────────────────────────────
    period = 12
    season_idx = _seasonal_indices(y, period)
    y_deseas   = _deseasonalize(y, season_idx, period)

    # ── 3. Train / validation split ─────────────────────────────────────────
    # Use last min(12, 20%) months as hold-out; keep at least 12 for training
    val_size = min(12, max(3, int(n * 0.2)))
    split    = max(12, n - val_size)
    if split >= n:
        split = n - 3
    t_tr, y_tr = t[:split], y_deseas[:split]
    t_va, y_va = t[split:], y_deseas[split:]
    m_tr = months_of_year[:split]

    # ── 4. Future time / month arrays ────────────────────────────────────────
    t_fut = np.arange(t[-1] + 1, t[-1] + horizon_months + 1)
    last_m = int(months_of_year[-1])
    fut_months_arr = np.array([(last_m - 1 + i) % 12 + 1 for i in range(1, horizon_months + 1)])
    future_dates = pd.date_range(
        df_series["Month"].iloc[-1] + pd.DateOffset(months=1),
        periods=horizon_months, freq="MS",
    )
    fut_season = _reseasonalize(np.zeros(horizon_months), season_idx,
                                start_pos=int(t[-1] + 1) % period)

    def _with_season(vals_deseas: np.ndarray, h_start: int) -> np.ndarray:
        s = np.array([season_idx[(h_start + i) % period] for i in range(len(vals_deseas))])
        return (vals_deseas + s).clip(0)

    fut_start_pos = int(t[-1] + 1) % period
    val_start_pos = int(t_tr[-1] + 1) % period

    # ── 5. Build future forecasts on deseasonalised series ───────────────────
    lin_f   = _linear_fit_predict(t_tr, y_tr, t_fut)
    poly_f  = _poly_fit_predict(t_tr, y_tr, t_fut, deg=2)
    holt_f  = _damped_holt(y_tr, horizon_months)
    hw_f    = _holt_winters(y, horizon_months, period)     # HW gets raw y
    snd_f   = _seasonal_naive_drift(y, horizon_months, period)  # SND gets raw y
    skl_res = _sklearn_forecast(t_tr, y_tr, t_fut, m_tr)
    skl_f   = skl_res[0] if skl_res else None

    # ── 6. Validation predictions for weighting ──────────────────────────────
    val_lin   = _linear_fit_predict(t_tr, y_tr, t_va)
    val_poly  = _poly_fit_predict(t_tr, y_tr, t_va, deg=2)
    val_holt  = _damped_holt(y_tr, len(t_va))

    # For HW and SND validate on raw y
    split_raw   = split
    y_va_raw    = y[split_raw:]
    val_hw      = _holt_winters(y[:split_raw], len(y_va_raw), period)
    val_snd     = _seasonal_naive_drift(y[:split_raw], len(y_va_raw), period)
    val_skl_raw = _sklearn_forecast(t_tr, y_tr, t_va, m_tr)
    val_skl     = val_skl_raw[0] if val_skl_raw else None

    # Reseasonalise deseasonal model val predictions before error calc
    val_lin_r  = _with_season(val_lin,  val_start_pos)
    val_poly_r = _with_season(val_poly, val_start_pos)
    val_holt_r = _with_season(val_holt, val_start_pos)
    val_skl_r  = _with_season(val_skl, val_start_pos) if val_skl is not None else None

    # Future reseasonalised predictions
    lin_final  = _with_season(lin_f,  fut_start_pos)
    poly_final = _with_season(poly_f, fut_start_pos)
    holt_final = _with_season(holt_f, fut_start_pos)
    skl_final  = _with_season(skl_f,  fut_start_pos) if skl_f is not None else None

    # ── 7. Assemble models and errors ────────────────────────────────────────
    models_fut  = [lin_final, poly_final, holt_final]
    models_val  = [val_lin_r, val_poly_r, val_holt_r]
    names       = ["Linear + Season", "Poly (deg-2) + Season", "Damped Holt + Season"]

    if hw_f is not None and val_hw is not None:
        models_fut.append(hw_f)
        models_val.append(val_hw)
        names.append("Holt-Winters")

    if snd_f is not None and val_snd is not None:
        models_fut.append(snd_f)
        models_val.append(val_snd)
        names.append("Seasonal Naïve+Drift")

    if skl_final is not None and val_skl_r is not None:
        models_fut.append(skl_final)
        models_val.append(val_skl_r)
        names.append("Gradient Boosting")

    val_errors = [_mape(y_va_raw, vp) for vp in models_val]
    weights    = _compute_weights(val_errors)

    ensemble     = _weighted_ensemble(models_fut, val_errors)
    val_ensemble = _weighted_ensemble(models_val, val_errors)

    # ── 8. Metrics ───────────────────────────────────────────────────────────
    mape     = _mape(y_va_raw, val_ensemble)
    mae_val  = _mae(y_va_raw, val_ensemble)
    rmse_val = float(np.sqrt(np.mean((y_va_raw - val_ensemble) ** 2)))
    accuracy = max(0.0, min(100.0, 100.0 - mape)) if not np.isnan(mape) else 0.0

    # ── 9. In-sample fitted values (same weights as forecast) ────────────────
    # Compute each model's in-sample prediction, then weighted-blend them
    fit_lin   = _with_season(_linear_fit_predict(t, y_deseas, t), 0)
    fit_poly  = _with_season(_poly_fit_predict(t, y_deseas, t, deg=2), 0)
    fit_holt  = _damped_holt_fitted(y_deseas)
    fit_holt  = _with_season(fit_holt, 0)
    fit_hw    = _holt_winters_fitted(y, period)  # already raw
    fit_snd   = _seasonal_naive_drift_fitted(y, period)  # already raw

    fit_preds = [fit_lin, fit_poly, fit_holt]
    fit_names = ["Linear + Season", "Poly (deg-2) + Season", "Damped Holt + Season"]

    if fit_hw is not None:
        fit_preds.append(fit_hw)
        fit_names.append("Holt-Winters")

    if fit_snd is not None:
        fit_preds.append(fit_snd)
        fit_names.append("Seasonal Naïve+Drift")

    if skl_final is not None:
        try:
            skl_fit_res = _sklearn_forecast(t, y_deseas, t, months_of_year)
            if skl_fit_res is not None:
                fit_skl = _with_season(skl_fit_res[0], 0)
                fit_preds.append(fit_skl)
                fit_names.append("Gradient Boosting")
        except Exception:
            pass

    # Use only models that are in the forecast ensemble, with same weights
    # Match by name to get consistent weights
    fit_weights = []
    fit_matched = []
    for fn, fp in zip(fit_names, fit_preds):
        if fn in names:
            idx = names.index(fn)
            fit_weights.append(weights[idx])
            fit_matched.append(fp)

    if fit_matched:
        total_w = sum(fit_weights)
        norm_w  = [w / total_w for w in fit_weights]
        fitted_ensemble = np.zeros(n)
        for w, fp in zip(norm_w, fit_matched):
            arr = np.array(fp).clip(0)
            fitted_ensemble += w * arr[:n]
        fitted_ensemble = fitted_ensemble.clip(0)
    else:
        fitted_ensemble = np.mean([np.array(fp).clip(0)[:n] for fp in fit_preds], axis=0)

    fitted_df = pd.DataFrame({
        "Month":  df_series["Month"].values,
        "Fitted": np.round(fitted_ensemble, 1),
    })

    # ── 10. Forecast DataFrame ────────────────────────────────────────────────
    forecast_df = pd.DataFrame({"Month": future_dates})
    for nm, vals in zip(names, models_fut):
        forecast_df[nm] = np.round(vals, 1)
    forecast_df["Ensemble Forecast"] = np.round(ensemble, 1)

    # ── 11. Trend analysis (on raw y) ────────────────────────────────────────
    recent = y_raw[-12:] if n >= 12 else y_raw
    half   = len(recent) // 2
    first_h, second_h = recent[:half].mean(), recent[half:].mean()
    trend_pct = ((second_h - first_h) / first_h * 100) if first_h > 0 else 0.0
    trend = "increasing" if trend_pct > 5 else ("decreasing" if trend_pct < -5 else "stable")

    next_q_total = int(round(ensemble[:3].sum()))
    next_y_total = int(round(ensemble.sum()))
    peak_month   = future_dates[int(np.argmax(ensemble))].strftime("%B %Y")
    low_month    = future_dates[int(np.argmin(ensemble))].strftime("%B %Y")
    avg_hist     = float(np.mean(y_raw[-12:])) if n >= 12 else float(np.mean(y_raw))

    insights = _build_insights(
        trend=trend, trend_pct=trend_pct, avg_hist=avg_hist,
        next_q_total=next_q_total, next_y_total=next_y_total,
        peak_month=peak_month, low_month=low_month,
        mape=mape, ensemble=ensemble, y=y_raw,
        df_series=df_series,
    )

    return {
        "history":    df_series[["Month", target_col, "Avg_Risk"]].copy(),
        "forecast":   forecast_df,
        "fitted":     fitted_df,
        "metrics":    {
            "MAPE (%)": round(mape, 1),
            "MAE":      round(mae_val, 1),
            "RMSE":     round(rmse_val, 1),
            "Accuracy": round(accuracy, 1),
        },
        "insights":   insights,
        "trend":      trend,
        "trend_pct":  round(trend_pct, 1),
        "model_names": names + ["Ensemble Forecast"],
        "next_q":     next_q_total,
        "next_year":  next_y_total,
        "peak_month": peak_month,
    }


def _compute_weights(errors: list) -> list:
    safe = [max(e, 1e-3) if not np.isnan(e) else 1e9 for e in errors]
    inv  = [1.0 / s for s in safe]
    total = sum(inv)
    return [w / total for w in inv]


# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER-LEVEL BREAKDOWN FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def run_chapter_forecast(df_series: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    """Forecast each chapter column independently using the best available model."""
    if df_series.empty:
        return pd.DataFrame()

    future_dates = pd.date_range(
        df_series["Month"].iloc[-1] + pd.DateOffset(months=1),
        periods=horizon_months, freq="MS",
    )
    results = {"Month": future_dates}
    period  = 12

    for col in [c for c in df_series.columns if c.startswith("Ch_")]:
        y = df_series[col].values.astype(float)
        if y.sum() == 0:
            continue

        y_w = _winsorize(y, 0.02)
        si  = _seasonal_indices(y_w, period)
        yd  = _deseasonalize(y_w, si, period)
        t   = df_series["t"].values.astype(float)
        t_f = np.arange(t[-1] + 1, t[-1] + horizon_months + 1)
        fp  = int(t[-1] + 1) % period

        def _rs(v):
            return np.array([max(0.0, v[i] + si[(fp + i) % period]) for i in range(len(v))])

        preds = [
            _rs(_linear_fit_predict(t, yd, t_f)),
            _rs(_poly_fit_predict(t, yd, t_f, deg=2)),
            _rs(_damped_holt(yd, horizon_months)),
        ]
        snd = _seasonal_naive_drift(y_w, horizon_months, period)
        if snd is not None:
            preds.append(snd)
        hw = _holt_winters(y_w, horizon_months, period)
        if hw is not None:
            preds.append(hw)

        ens = np.mean(preds, axis=0).clip(0)
        results[col.replace("Ch_", "Chapter ")] = np.round(ens, 1)

    return pd.DataFrame(results) if len(results) > 1 else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORE FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def run_risk_score_forecast(df_series: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    """Forecast average match risk score trajectory."""
    if "Avg_Risk" not in df_series.columns or df_series["Avg_Risk"].isna().all():
        return pd.DataFrame()

    df_v = df_series.dropna(subset=["Avg_Risk"])
    y    = df_v["Avg_Risk"].values.astype(float)
    t    = df_v["t"].values.astype(float)
    n    = len(y)

    if n < 4:
        return pd.DataFrame()

    t_f = np.arange(t[-1] + 1, t[-1] + horizon_months + 1)
    future_dates = pd.date_range(
        df_series["Month"].iloc[-1] + pd.DateOffset(months=1),
        periods=horizon_months, freq="MS",
    )

    poly = _poly_fit_predict(t, y, t_f, deg=2).clip(0, 100)
    holt = _damped_holt(y, horizon_months).clip(0, 100)
    ens  = ((poly + holt) / 2.0)
    std  = float(np.std(y[-12:])) if n >= 12 else float(np.std(y))

    return pd.DataFrame({
        "Month": future_dates,
        "Forecasted Avg Risk Score": np.round(ens, 1),
        "Upper Bound": np.round(np.minimum(ens + std * 1.5, 100), 1),
        "Lower Bound": np.round(np.maximum(ens - std * 1.5, 0), 1),
    })


# ─────────────────────────────────────────────────────────────────────────────
# STATE ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_state_anomalies(top_n: int = 10) -> pd.DataFrame:
    table = _get_table()
    try:
        conn = sqlite3.connect("data.db")
        cur  = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = {r[1] for r in cur.fetchall()}
        def get_db_col(logical_name):
            for c in cols:
                if c.lower() == logical_name.lower():
                    return c
            return None
            
        state_col = get_db_col("state")
        date_col = get_db_col("open_date") or get_db_col("date_filed")
        
        if not (state_col and date_col):
            conn.close()
            return pd.DataFrame()

        df = pd.read_sql_query(
            f"""
            SELECT {state_col} AS State, substr({date_col},1,4) AS Year, COUNT(*) AS Filings
            FROM {table}
            WHERE {date_col} IS NOT NULL AND {date_col} LIKE '20%'
                  AND {state_col} IS NOT NULL AND {state_col} != ''
            GROUP BY {state_col}, Year ORDER BY {state_col}, Year
            """, conn,
        )
        conn.close()
        if df.empty:
            return pd.DataFrame()

        results = []
        for state, grp in df.groupby("State"):
            grp  = grp.sort_values("Year")
            vals = grp["Filings"].values.astype(float)
            if len(vals) < 3:
                continue
            mu, sigma = vals[:-1].mean(), vals[:-1].std()
            last = vals[-1]
            z    = (last - mu) / sigma if sigma > 0 else 0.0
            trend = "⬆ Spike" if z > 1.5 else ("⬇ Drop" if z < -1.5 else "● Normal")
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


# ─────────────────────────────────────────────────────────────────────────────
# RISK CONCENTRATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_heatmap_data() -> pd.DataFrame:
    table = _get_table()
    try:
        conn = sqlite3.connect("data.db")
        cur  = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = {r[1] for r in cur.fetchall()}
        def get_db_col(logical_name):
            for c in cols:
                if c.lower() == logical_name.lower():
                    return c
            return None
            
        state_col = get_db_col("state")
        chapter_col = get_db_col("chapter")
        score_col = get_db_col("match_score")
        
        if not (state_col and chapter_col and score_col):
            conn.close()
            return pd.DataFrame()

        df = pd.read_sql_query(
            f"""
            SELECT {state_col} AS State, {chapter_col} AS chapter,
                   ROUND(AVG({score_col}), 1) AS Avg_Risk,
                   COUNT(*) AS Case_Count
            FROM {table}
            WHERE {state_col} IS NOT NULL AND {state_col} != ''
                  AND {chapter_col} IS NOT NULL AND {score_col} IS NOT NULL
            GROUP BY {state_col}, {chapter_col} HAVING COUNT(*) >= 2
            """, conn,
        )
        conn.close()
        if df.empty:
            return pd.DataFrame()

        pivot = df.pivot_table(index="State", columns="chapter", values="Avg_Risk", aggfunc="mean")
        pivot.columns = [f"Ch {int(c)}" for c in pivot.columns]
        return pivot.round(1)
    except Exception as e:
        logger.error(f"get_risk_heatmap_data error: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS INSIGHT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _build_insights(
    trend, trend_pct, avg_hist, next_q_total, next_y_total,
    peak_month, low_month, mape, ensemble, y, df_series,
) -> list[str]:
    ins = []

    trend_lbl   = "growth" if trend == "increasing" else "decline" if trend == "decreasing" else "stable pattern"
    impact_text = (
        "growing caseload that may require additional resource deployment" if trend == "increasing"
        else "shrinking caseload, allowing operational resources to be optimized" if trend == "decreasing"
        else "stable pipeline for operational capacity planning"
    )
    ins.append(
        f"**Caseload Direction:** Filing volume shows an **{trend} trend** "
        f"(a **{abs(trend_pct):.1f}% {trend_lbl}** compared to the prior period). "
        f"This indicates a {impact_text}."
    )

    monthly_run = round(next_q_total / 3)
    ins.append(
        f"**Short-Term Caseload Pipeline:** A total of **{next_q_total:,} filings** are projected "
        f"over the next 90 days — an average run-rate of **{monthly_run:,} filings per month**."
    )

    annual_hist = avg_hist * 12
    deviation   = ((next_y_total - annual_hist) / annual_hist * 100) if annual_hist > 0 else 0.0
    ins.append(
        f"**12-Month Portfolio Volume:** Projected annual filings: **{next_y_total:,}**. "
        f"Compared to the historical baseline of **{annual_hist:.0f}** filings/year, "
        f"this is a **{deviation:+.1f}% deviation**."
    )

    ins.append(
        f"**Peak Workload Window:** Filing activity is expected to peak in **{peak_month}**. "
        f"Align staffing schedules accordingly."
    )
    ins.append(
        f"**Low-Volume Planning Window:** The quietest period is projected in **{low_month}** — "
        f"ideal for audits, system training, or clearing backlogs."
    )

    if "Avg_Risk" in df_series.columns:
        latest_risk = df_series["Avg_Risk"].dropna().iloc[-1] if not df_series["Avg_Risk"].dropna().empty else 0
        if latest_risk >= 90:
            ins.append(f"**Portfolio Risk Status:** Latest risk score **{latest_risk:.1f}** is critically elevated. Immediate compliance triage recommended.")
        elif latest_risk >= 75:
            ins.append(f"**Portfolio Risk Status:** Latest risk score **{latest_risk:.1f}** is elevated. Proactive monitoring is active.")
        else:
            ins.append(f"**Portfolio Risk Status:** Latest risk score **{latest_risk:.1f}** is stable.")

    cv = (np.std(y) / np.mean(y) * 100) if np.mean(y) > 0 else 0
    if cv > 40:
        ins.append(f"**Filing Volatility Alert:** High variability ({cv:.1f}%). Maintain strategic resource buffers.")
    elif cv < 15:
        ins.append(f"**Stable Filing Pattern:** Low variability ({cv:.1f}%) supports predictable operational scheduling.")

    if not np.isnan(mape):
        accuracy   = max(0.0, 100.0 - mape)
        confidence = "high" if accuracy >= 85 else "moderate" if accuracy >= 70 else "low"
        ins.append(
            f"**Forecast Reliability:** Back-testing shows **{confidence} confidence** "
            f"(model accuracy: **{accuracy:.1f}%**). "
            f"{'Projections can be used directly in strategic plans.' if accuracy >= 85 else 'Use as directional guidelines rather than exact targets.'}"
        )

    return ins
