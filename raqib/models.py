"""Hybrid prediction engine: physics residuals from the twin + machine learning.

* Isolation Forest on twin residuals        -> multivariate anomaly score
* One monotonic gradient-boosted model per -> probability of each incident type
  failure mode (physics-informed features)    within the next HORIZON_H hours
* Exponential degradation fit + bootstrap   -> remaining useful life (RUL) of P-201A
* Occlusion attribution                     -> "why" for every prediction

Owner role: data scientist / ML engineer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from . import plant
from .plant import STEPS_PER_H, TANK_AREA, LSHH_LEVEL, VIB_TRIP
from .twin import shadow

HORIZON_H = 12
CLASSES = ("overfill", "leak", "pump_trip", "power_trip", "vapor", "cyber")
MODEL_VERSION = "v0.4.0"

FEATURES = [
    "tank_res_abs", "tank_flat", "hidden_h_to_lshh", "line_imb_pct", "line_imb_6h", "line_cusum",
    "pt2_drop_1h", "vib_res_1h", "vib_res_slope_24h", "vib_1h", "tbrg_res_1h",
    "sag_count_6h", "vmin_6h", "ambient", "lel", "lel_slope_1h", "modbus_w_1h",
]
# Physics-informed feature set and monotonic direction per failure mode (+1: riskier as it grows).
# Restricting each model to causal signals stops it learning time-of-day or weather coincidences.
CLASS_FEATURES = {
    "overfill": {"tank_res_abs": 1, "tank_flat": 1, "hidden_h_to_lshh": -1},
    "leak": {"line_cusum": 1, "line_imb_pct": 1, "line_imb_6h": 1, "pt2_drop_1h": 1},
    "pump_trip": {"vib_res_1h": 1, "vib_res_slope_24h": 1, "vib_1h": 1, "tbrg_res_1h": 1},
    "power_trip": {"sag_count_6h": 1, "vmin_6h": -1, "ambient": 1},
    "vapor": {"lel": 1, "lel_slope_1h": 1},
    "cyber": {"modbus_w_1h": 1, "tank_res_abs": 1, "tank_flat": -1},
}
ANOMALY_FEATURES = ["tank_res_abs", "line_cusum", "pt2_drop_1h", "vib_res_1h", "tbrg_res_1h",
                    "vmin_6h", "lel", "modbus_w_1h"]


def build_features(df: pd.DataFrame, sh: pd.DataFrame) -> pd.DataFrame:
    H = STEPS_PER_H
    f = pd.DataFrame(index=df.index)
    res = sh[["TK-101_res", "TK-103_res", "TK-102_res"]].abs()
    f["tank_res_abs"] = res.max(axis=1).rolling(3, min_periods=1).mean()
    rec = df["TAS-RECV"]
    flat = np.where(rec == "TK-101", sh["TK-101_flat"], sh["TK-103_flat"])
    f["tank_flat"] = flat.astype(float)
    est = np.where(rec == "TK-101", sh["TK-101_est"], sh["TK-103_est"])
    rate = df["FI-002"].rolling(H, min_periods=1).mean() / TANK_AREA
    h_to = np.clip((LSHH_LEVEL - est) / rate.clip(lower=1e-3), 0, 48)
    # only informative when the gauge is no longer trusted: hours left that the operator cannot see
    f["hidden_h_to_lshh"] = np.where(f["tank_res_abs"] > 0.1, h_to, 48.0)
    f["line_imb_pct"] = sh["line_imbalance_pct"].clip(-20, 20)
    # CUSUM on the instantaneous meter imbalance: catches a small leak within minutes
    inst = (100 * (df["FI-001"] - df["FI-002"]) / df["FI-001"].clip(lower=50)).clip(-20, 20).to_numpy()
    cus, acc = np.zeros(len(df)), 0.0
    for i, v in enumerate(inst):
        acc = min(50.0, max(0.0, acc + v - 0.4))
        cus[i] = acc
    f["line_cusum"] = cus
    f["line_imb_6h"] = f["line_imb_pct"].rolling(6 * H, min_periods=1).mean()
    f["pt2_drop_1h"] = (df["PT-002"].shift(H) - df["PT-002"]).fillna(0)
    vr = pd.Series(sh["P201-VIB_res"]).ffill().fillna(0)
    f["vib_res_1h"] = vr.rolling(H, min_periods=1).mean()
    t = pd.Series(np.arange(len(df)) / H, index=df.index)
    w = 24 * H
    cov = (t * vr).rolling(w, min_periods=H).mean() - t.rolling(w, min_periods=H).mean() * vr.rolling(w, min_periods=H).mean()
    var = t.rolling(w, min_periods=H).var(ddof=0)
    f["vib_res_slope_24h"] = (cov / var).fillna(0) * 24        # mm/s per day
    f["vib_1h"] = df["P201-VIB"].rolling(H, min_periods=1).mean()
    f["tbrg_res_1h"] = pd.Series(sh["P201-TBRG_res"]).ffill().fillna(0).rolling(H, min_periods=1).mean()
    f["sag_count_6h"] = (df["GRID-V"] < 0.95).astype(float).rolling(6 * H, min_periods=1).sum()
    f["vmin_6h"] = df["GRID-V"].rolling(6 * H, min_periods=1).min()
    f["ambient"] = df["TT-900"]
    f["lel"] = df["GD-01"]
    f["lel_slope_1h"] = (df["GD-01"] - df["GD-01"].shift(H)).fillna(0)
    f["modbus_w_1h"] = df["NET-MODBUS-W"].rolling(H, min_periods=1).sum()
    return f[FEATURES]


def fault_onset(df: pd.DataFrame) -> dict:
    """Ground-truth masks of steps where a physical precursor exists (used for scoring only)."""
    lt_dev = np.zeros(len(df), dtype=bool)
    for t in ("101", "102", "103"):
        lt_dev |= (df[f"LT-{t}"] - df[f"true_L{t}"]).abs().to_numpy() > 0.05
    acc = np.maximum.accumulate
    return {"overfill": acc(lt_dev), "leak": (df["true_leak"] > 0).to_numpy(),
            "pump_trip": acc((df["true_vib_add"] > 0.05).to_numpy()),
            "power_trip": acc((df["true_stress"] > 0.05).to_numpy()),
            "vapor": acc((df["true_release"] > 0).to_numpy()),
            "cyber": acc((df["NET-MODBUS-W"] > 0).to_numpy())}


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-label target: incident of each type within the next HORIZON_H hours (or ongoing).

    A step is only labelled once a precursor physically exists, so models learn
    from signals rather than from coincidences in the weather or the schedule.
    """
    n, H = len(df), HORIZON_H * STEPS_PER_H
    on = fault_onset(df)
    lab = pd.DataFrame(False, index=df.index, columns=CLASSES)
    ev = df["event"].replace({"esd_lshh": "overfill"})
    for i in np.flatnonzero(ev.isin(["overfill", "pump_trip", "power_trip"]).to_numpy()):
        c = ev.iat[i]
        m = np.zeros(n, dtype=bool)
        m[max(0, i - H): i + 1] = True
        if c == "overfill":
            m[i: min(n, i + 2 * STEPS_PER_H)] = True      # hazard persists after the event
        lab[c] |= m & on[c]
    lab["leak"] = on["leak"]
    lab["cyber"] = on["cyber"]                             # an intrusion is a hazard while it lasts
    lab["vapor"] = (df["true_release"] > 0.5).to_numpy()  # a release is a hazard while it lasts
    return lab


def prepare(df: pd.DataFrame):
    sh = shadow(df)
    return sh, build_features(df, sh)


@dataclass
class Engine:
    clfs: dict                            # one monotonic gradient-boosted model per failure mode
    iso: IsolationForest
    iso_ref: np.ndarray
    baseline: pd.Series                   # feature medians on healthy data (occlusion reference)
    train_info: dict = field(default_factory=dict)
    feature_ref: pd.DataFrame | None = None

    def proba(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c, clf in self.clfs.items():
            out[c] = clf.predict_proba(X[list(CLASS_FEATURES[c])])[:, 1]
        return out

    def anomaly(self, X: pd.DataFrame) -> np.ndarray:
        s = -self.iso.score_samples(X[ANOMALY_FEATURES])
        return np.clip(np.searchsorted(np.sort(self.iso_ref), s) / len(self.iso_ref), 0, 1)

    def explain(self, x: pd.Series, cls: str, top: int = 5) -> pd.DataFrame:
        """Occlusion attribution: how much P(cls) drops when a feature is reset to normal."""
        feats = list(CLASS_FEATURES[cls])
        X = pd.DataFrame([x[FEATURES]] * (len(feats) + 1)).reset_index(drop=True)
        for j, name in enumerate(feats):
            X.loc[j + 1, name] = self.baseline[name]
        p = self.proba(X)[cls].to_numpy()
        out = pd.DataFrame({"feature": feats, "value": x[feats].to_numpy(),
                            "normal": self.baseline[feats].to_numpy(), "contribution": p[0] - p[1:]})
        return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).head(top)

    def deviations(self, x: pd.Series, cls: str) -> pd.DataFrame:
        """How far each causal signal sits from healthy operation, in healthy standard deviations."""
        feats = list(CLASS_FEATURES[cls])
        sd = self.feature_ref[feats].std().replace(0, np.nan)
        z = ((x[feats] - self.baseline[feats]) / sd).fillna(0) * pd.Series(CLASS_FEATURES[cls])
        return pd.DataFrame({"feature": feats, "value": x[feats].to_numpy(), "z": z.to_numpy()}).sort_values("z", ascending=False)


def _runs(seeds, days):
    out = []
    for sc in plant.SCENARIOS:
        for sd in seeds:
            df = plant.run_scenario(sc, seed=sd, days=days)
            sh, X = prepare(df)
            out.append((sc, sd, df, sh, X, build_labels(df)))
    return out


def train(train_seeds=(11, 12, 13, 14), test_seeds=(21, 22, 23), days: float = 10.0) -> Engine:
    t0 = time.time()
    tr = _runs(train_seeds, days)
    X = pd.concat([r[4] for r in tr], ignore_index=True)
    Y = pd.concat([r[5] for r in tr], ignore_index=True)
    clfs = {}
    for c, mono in CLASS_FEATURES.items():
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                                             min_samples_leaf=80, l2_regularization=1.0,
                                             monotonic_cst=list(mono.values()), random_state=0)
        clfs[c] = clf.fit(X[list(mono)], Y[c].astype(int))
    healthy = pd.concat([r[4] for r in tr if r[0] == "normal"], ignore_index=True)
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(healthy[ANOMALY_FEATURES])
    eng = Engine(clfs=clfs, iso=iso, iso_ref=-iso.score_samples(healthy[ANOMALY_FEATURES]),
                 baseline=healthy.median(), feature_ref=healthy.sample(2000, random_state=0))
    eng.train_info = {"version": MODEL_VERSION, "rows": len(X), "runs": len(tr), "horizon_h": HORIZON_H,
                      "train_seeds": list(train_seeds), "test_seeds": list(test_seeds),
                      "positives": {c: int(Y[c].sum()) for c in CLASSES}}
    eng.train_info["evaluation"] = evaluate(eng, test_seeds, days)
    eng.train_info["train_seconds"] = round(time.time() - t0, 1)
    return eng


def first_alert(p: np.ndarray, thr: float = 0.5, persist: int = 3) -> int | None:
    run = 0
    for i, v in enumerate(p):
        run = run + 1 if v >= thr else 0
        if run >= persist:
            return i - persist + 1
    return None


SCENARIO_TARGET = {"overfill": "overfill", "leak": "leak", "pump": "pump_trip", "power": "power_trip",
                   "cyber": "cyber", "vapor": "vapor"}


def evaluate(eng: Engine, seeds, days) -> dict:
    """Acceptance tests (project doc section 8): each scenario caught before its incident,
    no alerts on healthy runs. Lead time is measured to the first incident; a leak or an
    intrusion has no precursor, so for those it is time-to-detect after onset instead."""
    rows, Ys, Ps = [], [], []
    for sc, sd, df, sh, X, Y in _runs(seeds, days):
        P = eng.proba(X)
        Ys.append(Y.to_numpy()); Ps.append(P[list(CLASSES)].to_numpy())
        if sc == "normal":
            alerts = sum(first_alert(P[c].to_numpy()) is not None for c in CLASSES)
            rows.append({"scenario": sc, "seed": sd, "target": "-", "passed": alerts == 0,
                         "lead_h": np.nan, "detect_after_onset_h": np.nan, "false_alerts": alerts})
            continue
        c = SCENARIO_TARGET[sc]
        onset = int(np.argmax(fault_onset(df)[c]))
        ev = df.index[df["event"].replace({"esd_lshh": "overfill"}) == c]
        fa = first_alert(P[c].to_numpy())
        others = sum(first_alert(P[o].to_numpy()) is not None for o in CLASSES
                     if o != c and not (sc == "cyber" and o == "overfill"))
        lead = (int(ev[0]) - fa) / STEPS_PER_H if (fa is not None and len(ev)) else np.nan
        det = (fa - onset) / STEPS_PER_H if fa is not None else np.nan
        if c in ("leak", "cyber"):
            ok = fa is not None and det <= 1.0
        else:
            ok = fa is not None and (not len(ev) or fa < int(ev[0]))
        rows.append({"scenario": sc, "seed": sd, "target": c, "passed": bool(ok), "lead_h": lead,
                     "detect_after_onset_h": det, "false_alerts": others})
    Y, P = np.vstack(Ys), np.vstack(Ps)
    per_class = {}
    for j, c in enumerate(CLASSES):
        hit, truth = P[:, j] >= 0.5, Y[:, j]
        tp, fp, fn = int((hit & truth).sum()), int((hit & ~truth).sum()), int((~hit & truth).sum())
        per_class[c] = {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "support": int(truth.sum())}
    return {"acceptance": pd.DataFrame(rows), "per_class": pd.DataFrame(per_class).T}


# --- Remaining useful life ------------------------------------------------------
def pump_rul(df: pd.DataFrame, sh: pd.DataFrame, i: int, n_boot: int = 300, seed: int = 0) -> dict:
    """Fit y = c*exp(b t) to the vibration residual of P-201A and extrapolate to the trip limit."""
    H = STEPS_PER_H
    lo = max(0, i - 96 * H)
    seg = sh.iloc[lo:i + 1]
    y = pd.Series(seg["P201-VIB_res"]).rolling(H, min_periods=H // 2).mean()
    t = (seg["step"] - seg["step"].iat[-1]) / H                  # hours, 0 = now
    ok = y.notna() & (y > 0.08)
    base = float(np.nanmean(sh["P201-VIB_exp"].iloc[lo:i + 1])) if np.isfinite(np.nanmean(sh["P201-VIB_exp"].iloc[lo:i + 1])) else 2.2
    target = VIB_TRIP - base
    res = {"rul_p50": np.inf, "rul_p10": np.inf, "rul_p90": np.inf, "p_fail_72h": 0.0,
           "trend": "none", "fit_t": None, "fit_y": None, "base": base, "target": target}
    if ok.sum() < 6 * H:
        return res
    tt, yy = t[ok].to_numpy(), np.log(y[ok].to_numpy())
    rng = np.random.default_rng(seed)
    ruls = []
    # moving-block bootstrap (6 h blocks): residuals are autocorrelated, so plain resampling
    # would make the interval look far more certain than it is
    blocks = [np.flatnonzero((tt >= b) & (tt < b + 6)) for b in np.arange(tt.min(), 0.01, 6)]
    blocks = [b for b in blocks if len(b)]
    for _ in range(n_boot):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        if np.ptp(tt[idx]) < 6:
            continue
        b, a = np.polyfit(tt[idx], yy[idx], 1)
        ruls.append((np.log(target) - a) / b if b > 1e-4 else np.inf)
    ruls = np.clip(np.array(ruls), 0, np.inf)
    b, a = np.polyfit(tt, yy, 1)
    if b <= 1e-4:
        return res
    hh = np.linspace(tt.min(), min(np.percentile(ruls, 50) * 1.2, 24 * 30), 120)
    res.update({"rul_p50": float(np.percentile(ruls, 50)), "rul_p10": float(np.percentile(ruls, 10)),
                "rul_p90": float(np.percentile(ruls, 90)), "p_fail_72h": float(np.mean(ruls < 72)),
                "trend": "exponential", "fit_t": hh, "fit_y": base + np.exp(a + b * hh)})
    return res


# --- Drift ----------------------------------------------------------------------
def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population stability index between training and live feature distributions."""
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    r = np.histogram(np.clip(ref, qs[0], qs[-1]), qs)[0] / len(ref) + 1e-4
    c = np.histogram(np.clip(cur, qs[0], qs[-1]), qs)[0] / max(len(cur), 1) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))
