"""RAQIB digital twin in shadow mode.

The twin receives the same measured inputs as the plant and computes what
each key tag *should* read from first principles (mass balance, pump curve,
hydraulic gradient). The gap between measured and expected (the residual)
is the earliest sign of a failing asset or a lying sensor.

Owner roles: process / simulation engineer (models, calibration),
OT data engineer (data quality).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .plant import (DT_MIN, GASOLINE_TANKS, LSHH_LEVEL, STEPS_PER_H, TANK_AREA,
                    VIB_ALARM, VIB_TRIP)

RECONCILE_TOL = 0.05     # m; sensor and model agree -> reconcile model to sensor
LEVEL_RES_ALERT = 0.15   # m


def calibrate(df: pd.DataFrame, hours: float = 24.0) -> dict:
    """Fit twin parameters on an initial healthy window (data reconciliation)."""
    w = df.iloc[: int(hours * STEPS_PER_H)]
    w = w[w["FI-001"] > 50]
    q1, q2 = w["FI-001"].to_numpy(), w["FI-002"].to_numpy()
    dp = (w["PT-001"] - w["PT-002"]).to_numpy()
    kl = float(np.sum(dp * q1 ** 2) / np.sum(q1 ** 4))        # least squares, no-leak model
    run = df.iloc[: int(hours * STEPS_PER_H)]
    run = run[run["PUMP-RUN"] != "-"]
    X = np.c_[np.ones(len(run)), run["FT-LOAD"]]
    vib_coef = np.linalg.lstsq(X, run["P201-VIB"], rcond=None)[0]
    tb_coef = np.linalg.lstsq(np.c_[np.ones(len(run)), run["FT-LOAD"]], run["P201-TBRG"] - run["TT-900"], rcond=None)[0]
    pd_coef = np.linalg.lstsq(np.c_[np.ones(len(run)), run["FT-LOAD"] ** 2], run["P201-PDIS"], rcond=None)[0]
    return {"line_kl": kl, "vib": vib_coef.tolist(), "tbrg": tb_coef.tolist(), "pdis": pd_coef.tolist()}


def shadow(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Return a frame of expected values, residuals and twin-estimated levels."""
    params = params or calibrate(df)
    dt_h = DT_MIN / 60.0
    out = pd.DataFrame({"time": df["time"], "step": df["step"]})

    # --- tank mass balance with reconciliation -------------------------------
    recv = df["TAS-RECV"].to_numpy()
    fin, fload = df["FI-002"].to_numpy(), df["FT-LOAD"].to_numpy()
    W = 2 * STEPS_PER_H        # 2 h balance window
    for tank, col in (("TK-101", "LT-101"), ("TK-103", "LT-103"), ("TK-102", "LT-102")):
        meas = df[col].to_numpy()
        if tank == "TK-102":
            net = (df["FT-102IN"] - df["FT-102OUT"]).to_numpy()
        else:
            other = np.where(recv == GASOLINE_TANKS[0], GASOLINE_TANKS[1], GASOLINE_TANKS[0])
            net = np.where(recv == tank, fin, 0.0) - np.where(other == tank, fload, 0.0)
        cum = np.concatenate([[0.0], np.cumsum(net[1:] * dt_h / TANK_AREA)])
        # windowed balance: level W steps ago + metered net volume since then
        lag = np.maximum(np.arange(len(meas)) - W, 0)
        exp_w = meas[lag] + cum - cum[lag]
        res = meas - exp_w
        # trusted estimate: anchored to the last time sensor and physics agreed
        est = np.empty_like(meas)
        trusted = np.abs(res) < RECONCILE_TOL
        anchor = 0
        for i in range(len(meas)):
            if trusted[i]:
                anchor = i
            est[i] = meas[anchor] + cum[i] - cum[anchor]
        out[f"{tank}_exp"] = exp_w
        out[f"{tank}_res"] = res
        out[f"{tank}_est"] = est
        out[f"{tank}_trusted"] = trusted
        # flatline: a real radar gauge always shows mm-level noise
        out[f"{tank}_flat"] = pd.Series(meas).diff().abs().rolling(STEPS_PER_H, min_periods=3).max().fillna(1).to_numpy() < 1e-6

    # --- pipeline mass balance & leak location --------------------------------
    q1, q2 = df["FI-001"], df["FI-002"]
    imb = (q1 - q2).rolling(STEPS_PER_H, min_periods=1).mean()
    out["line_imbalance"] = imb
    out["line_imbalance_pct"] = 100 * imb / q1.rolling(STEPS_PER_H, min_periods=1).mean().clip(lower=50)
    kl = params["line_kl"]
    dp = (df["PT-001"] - df["PT-002"]).rolling(STEPS_PER_H, min_periods=1).mean()
    q1m, q2m = q1.rolling(STEPS_PER_H, min_periods=1).mean(), q2.rolling(STEPS_PER_H, min_periods=1).mean()
    denom = (q1m ** 2 - q2m ** 2).where(lambda s: s.abs() > 1e3)
    out["leak_pos_frac"] = ((dp / kl - q2m ** 2) / denom).clip(0, 1)
    out["PT-001_exp"] = df["PT-002"] + kl * q1 ** 2
    out["PT-001_res"] = df["PT-001"] - out["PT-001_exp"]

    # --- loading pump ---------------------------------------------------------
    running = df["PUMP-RUN"] == "A"
    b0, b1 = params["vib"]
    out["P201-VIB_exp"] = np.where(running, b0 + b1 * df["FT-LOAD"], np.nan)
    out["P201-VIB_res"] = np.where(running, df["P201-VIB"] - out["P201-VIB_exp"], np.nan)
    t0, t1 = params["tbrg"]
    out["P201-TBRG_exp"] = np.where(running, df["TT-900"] + t0 + t1 * df["FT-LOAD"], np.nan)
    out["P201-TBRG_res"] = np.where(running, df["P201-TBRG"] - out["P201-TBRG_exp"], np.nan)
    p0, p1 = params["pdis"]
    out["P201-PDIS_exp"] = np.where(running, p0 + p1 * df["FT-LOAD"] ** 2, np.nan)
    out["P201-PDIS_res"] = np.where(running, df["P201-PDIS"] - out["P201-PDIS_exp"], np.nan)
    return out


def twin_accuracy(df: pd.DataFrame, sh: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of the twin vs measurement per key variable (acceptance: within +/-2%)."""
    rows = []
    for tag, exp in (("LT-101", "TK-101_exp"), ("LT-103", "TK-103_exp"), ("LT-102", "TK-102_exp"),
                     ("PT-001", "PT-001_exp"), ("P201-TBRG", "P201-TBRG_exp"), ("P201-PDIS", "P201-PDIS_exp")):
        m, e = df[tag], sh[exp]
        ok = e.notna() & (m.abs() > 1e-3)
        if ok.sum() == 0:
            continue
        span = max(float(m[ok].quantile(0.95) - m[ok].quantile(0.05)), 1e-6)
        err = (m[ok] - e[ok]).abs()
        rows.append({"tag": tag, "mape_pct": float(100 * (err / m[ok].abs()).median()),
                     "err_pct_span": float(100 * err.median() / span),
                     "within_2pct": float(100 * (err / m[ok].abs() <= 0.02).mean())})
    return pd.DataFrame(rows)


def lookahead_tank(df: pd.DataFrame, sh: pd.DataFrame, i: int, hours: float = 24.0) -> dict:
    """Project the receiving tank forward with the twin's level estimate and recent flows."""
    rec = df["TAS-RECV"].iat[i]
    lo = max(0, i - STEPS_PER_H)
    q_in = float(df["FI-002"].iloc[lo:i + 1].mean())
    est = float(sh[f"{rec}_est"].iat[i])
    meas = float(df[f"LT-{rec[-3:]}"].iat[i])
    rate = q_in / TANK_AREA                     # m/h
    t_hh = (LSHH_LEVEL - est) / rate if rate > 1e-3 else np.inf
    t_hh_meas = (LSHH_LEVEL - meas) / rate if rate > 1e-3 else np.inf
    hrs = np.arange(0, hours + 1e-9, 0.25)
    return {"tank": rec, "est": est, "meas": meas, "rate_m_h": rate,
            "hours_to_lshh": max(0.0, t_hh), "hours_to_lshh_by_gauge": max(0.0, t_hh_meas),
            "proj_h": hrs, "proj_est": np.minimum(est + rate * hrs, 20.0),
            "proj_meas": np.minimum(meas + rate * hrs, 20.0)}


def pump_limits() -> dict:
    return {"alarm": VIB_ALARM, "trip": VIB_TRIP}
