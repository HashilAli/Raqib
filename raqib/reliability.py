"""FMEA register, CMMS history and Weibull life model.

Owner role: reliability & maintenance engineer.
FMEA scales follow the usual 1-10 convention (S severity, O occurrence,
D detection: 10 = cannot detect). D_after is the detection rating once the
RAQIB detector listed on the row is in service, so the RPN reduction shows
where the platform earns its keep.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def load_fmea() -> pd.DataFrame:
    f = pd.read_csv(DATA / "fmea.csv")
    f["RPN_before"] = f["S"] * f["O"] * f["D_before"]
    f["RPN_after"] = f["S"] * f["O"] * f["D_after"]
    f["RPN_reduction_pct"] = (100 * (1 - f["RPN_after"] / f["RPN_before"])).round(0)
    return f


def recompute_rpn(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    for c in ("S", "O", "D_before", "D_after"):
        f[c] = pd.to_numeric(f[c], errors="coerce").clip(1, 10).fillna(5).astype(int)
    f["RPN_before"] = f["S"] * f["O"] * f["D_before"]
    f["RPN_after"] = f["S"] * f["O"] * f["D_after"]
    f["RPN_reduction_pct"] = (100 * (1 - f["RPN_after"] / f["RPN_before"])).round(0)
    return f


def severity_for(f: pd.DataFrame, risk_class: str) -> int:
    rows = f[f["risk_class"] == risk_class]
    return int(rows["S"].max()) if len(rows) else 5


def load_cmms() -> pd.DataFrame:
    return pd.read_csv(DATA / "cmms_history.csv", parse_dates=["date"])


def fit_weibull(times: np.ndarray) -> tuple[float, float]:
    """Median-rank regression Weibull fit (beta shape, eta scale in hours)."""
    t = np.sort(np.asarray(times, dtype=float))
    n = len(t)
    if n < 2:
        return 2.0, float(t.mean()) if n else 10000.0
    F = (np.arange(1, n + 1) - 0.3) / (n + 0.4)       # Bernard's approximation
    x, y = np.log(t), np.log(-np.log(1 - F))
    beta, c = np.polyfit(x, y, 1)
    return float(beta), float(np.exp(-c / beta))


def conditional_reliability(age_h: float, horizon_h: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """P(survive another horizon_h | survived age_h)."""
    r = lambda t: np.exp(-(np.asarray(t) / eta) ** beta)
    return r(age_h + horizon_h) / r(age_h)


def pump_bearing_weibull() -> dict:
    h = load_cmms()
    vib = h[(h["fm_code"] == "VIB") & h["run_hours_at_failure"].notna()]
    beta, eta = fit_weibull(vib["run_hours_at_failure"].to_numpy())
    return {"beta": beta, "eta": eta, "n": len(vib), "mtbf": eta * math.gamma(1 + 1 / beta)}
