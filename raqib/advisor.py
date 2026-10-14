"""Live risk index, solution engine (what-if on the twin) and CMMS work orders.

R_i(t) = P_i(fail | t, dt) x C_i x (1 - B_i)        (project doc, section 6)

P comes from the prediction engine, C from FMEA severity, B from the health
of the protection barriers RAQIB can read (SIS bypass register, standby
equipment, generators). The advisor is level L2: it proposes options tested
on the twin; the operator executes them through the DCS. It never writes to
the control or safety systems.

Owner roles: process engineer (what-if), reliability engineer (work orders).
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import plant
from .plant import Actions, DT_MIN, GASOLINE_TANKS, STEPS_PER_H

ASSETS = [
    # id, Arabic name, English name, risk class, x, y on the site sketch (metres)
    ("TK-101", "خزان بنزين TK-101", "Gasoline tank TK-101", "overfill", 120, 260),
    ("TK-103", "خزان بنزين TK-103", "Gasoline tank TK-103", "overfill", 220, 260),
    ("TK-102", "خزان ديزل TK-102", "Diesel tank TK-102", "overfill", 320, 260),
    ("PL-S3", "خط الأنابيب، القطاع S3", "Pipeline segment S3", "leak", 30, 140),
    ("P-201A", "مضخة التحميل P-201A", "Loading pump P-201A", "pump_trip", 200, 160),
    ("LB-01", "رصيف تحميل الصهاريج LB-01", "Truck loading bay LB-01", "vapor", 330, 90),
    ("EL-33KV", "مغذي الكهرباء 33 ك.ف", "33 kV power incomer", "power_trip", 70, 40),
    ("OT-NET", "شبكة التحكم (TAS/DCS)", "Control network (TAS/DCS)", "cyber", 180, 40),
]


@dataclass
class Barriers:
    lshh_in_service: bool = True        # tank high-high level switch (SIS) healthy, not bypassed
    standby_pump_ok: bool = True
    generators_ok: bool = True
    gas_esd_ok: bool = True              # loading bay ESD + gas detection
    data_diode_ok: bool = True           # one-way gateway between OT and IT

    def effectiveness(self, cls: str) -> float:
        return {"overfill": 0.6 if self.lshh_in_service else 0.0,
                "pump_trip": 0.5 if self.standby_pump_ok else 0.0,
                "power_trip": 0.5 if self.generators_ok else 0.0,
                "vapor": 0.4 if self.gas_esd_ok else 0.0,
                "leak": 0.2,
                "cyber": 0.4 if self.data_diode_ok else 0.0}[cls]


def risk_table(P: pd.Series, rul: dict, recv_tank: str, worst_tank: str | None,
               fmea: pd.DataFrame, barriers: Barriers) -> pd.DataFrame:
    from .reliability import severity_for
    rows = []
    for aid, ar, en, cls, x, y in ASSETS:
        p = float(P.get(cls, 0.0))
        if cls == "overfill":
            target = worst_tank or recv_tank
            p = p if aid == target else min(p, 0.02)
            if worst_tank and aid == worst_tank:
                p = max(p, 0.3)               # receiving blind: the gauge is no longer trusted
        if cls == "pump_trip":
            p = max(p, float(rul.get("p_fail_72h", 0.0)) * 0.9)
        c = severity_for(fmea, cls) / 10
        b = barriers.effectiveness(cls)
        rows.append({"asset": aid, "name_ar": ar, "name_en": en, "class": cls, "x": x, "y": y,
                     "P": p, "C": c, "B": b, "risk": 100 * p * c * (1 - b)})
    return pd.DataFrame(rows).sort_values("risk", ascending=False).reset_index(drop=True)


def risk_band(r: float) -> str:
    return "high" if r >= 30 else "medium" if r >= 10 else "low"


# --- Solution engine ----------------------------------------------------------------
def options_for(cls: str, recv_tank: str) -> list[dict]:
    other = GASOLINE_TANKS[1] if recv_tank == GASOLINE_TANKS[0] else GASOLINE_TANKS[0]
    base = {"key": "none", "ar": "الاستمرار دون إجراء", "en": "Continue, no action", "actions": Actions()}
    opts = {
        "overfill": [
            {"key": "divert", "ar": f"تحويل الاستلام إلى {other} والقياس اليدوي", "en": f"Divert receipt to {other} + manual dip", "actions": Actions(divert_to=other)},
            {"key": "reduce", "ar": "خفض معدل الاستلام 50%", "en": "Cut receipt rate by 50%", "actions": Actions(receipt_factor=0.5)},
            {"key": "stop", "ar": "إيقاف الاستلام بالتنسيق مع مركز التحكم بالخط", "en": "Stop receipt with pipeline control centre", "actions": Actions(receipt_factor=0.0)},
        ],
        "cyber": [
            {"key": "manual", "ar": f"تشغيل يدوي: تحويل الاستلام إلى {other} وقياس يدوي", "en": f"Manual ops: divert to {other} + manual dip", "actions": Actions(divert_to=other)},
            {"key": "stop", "ar": "إيقاف الاستلام حتى عزل الشبكة", "en": "Stop receipt until network is isolated", "actions": Actions(receipt_factor=0.0)},
        ],
        "leak": [
            {"key": "isolate", "ar": "عزل القطاع S3 بصمامات الخط", "en": "Isolate segment S3 block valves", "actions": Actions(isolate_line=True)},
            {"key": "reduce", "ar": "خفض ضغط الخط (المعدل 50%)", "en": "Reduce line pressure (50% rate)", "actions": Actions(receipt_factor=0.5)},
        ],
        "pump_trip": [
            {"key": "standby", "ar": "التحويل إلى المضخة الاحتياطية P-201B", "en": "Change over to standby P-201B", "actions": Actions(pump_standby=True)},
            {"key": "slow", "ar": "خفض سرعة P-201A إلى 80%", "en": "Run P-201A at 80% speed", "actions": Actions(pump_speed=0.8)},
        ],
        "power_trip": [
            {"key": "shed", "ar": "فصل الأحمال غير الأساسية", "en": "Shed non-essential load", "actions": Actions(load_shed=True)},
            {"key": "gen", "ar": "تشغيل مولدات الطوارئ مسبقاً", "en": "Pre-start emergency generators", "actions": Actions(gen_prestart=True)},
            {"key": "both", "ar": "الإجراءان معاً", "en": "Both", "actions": Actions(load_shed=True, gen_prestart=True)},
        ],
        "vapor": [
            {"key": "stop", "ar": "إيقاف التحميل وعزل الرصيف (ESD)", "en": "Stop loading and ESD the bay", "actions": Actions(stop_loading=True)},
        ],
    }[cls]
    return [base] + opts


def _outcome(df: pd.DataFrame, horizon_steps: int) -> dict:
    ev = df[df["event"] != ""]
    first = ev.iloc[0] if len(ev) else None
    return {"incident": first["event"] if first is not None else "",
            "hours_to_incident": (first["step"] - df["step"].iat[0]) / STEPS_PER_H if first is not None else np.nan,
            "received_m3": float(df["FI-002"].sum() * DT_MIN / 60),
            "loaded_m3": float(df["FT-LOAD"].sum() * DT_MIN / 60),
            "max_level": float(df[["true_L101", "true_L103"]].max().max()),
            "max_lel": float(df["GD-01"].max()),
            "max_vib": float(df["P201-VIB"].max()),
            "spilled_m3": float(df["true_leak"].sum() * DT_MIN / 60)}


HORIZON_H = {"overfill": 48, "cyber": 48, "leak": 24, "pump_trip": 72, "power_trip": 36, "vapor": 12}
INCIDENT_COST = {"overfill": 1000, "vapor": 900, "leak": 600, "esd_lshh": 300,
                 "power_trip": 250, "pump_trip": 150}


def whatif(cfg: plant.ScenarioConfig, state: plant.PlantState, cls: str, recv_tank: str,
           twin_levels: dict | None = None, hours: float | None = None) -> pd.DataFrame:
    """Branch the twin from the current state once per option and score each future."""
    hours = hours or HORIZON_H[cls]
    rows = []
    base_out = None
    for opt in options_for(cls, recv_tank):
        st = copy.deepcopy(state)
        if twin_levels:                       # start from the twin's reconciled levels, not the raw gauges
            st.levels.update(twin_levels)
        df, _ = plant.simulate(cfg, hours, actions=opt["actions"], state=st)
        out = _outcome(df, int(hours * STEPS_PER_H))
        if opt["key"] == "none":
            base_out = out
        rows.append({"key": opt["key"], "ar": opt["ar"], "en": opt["en"], **out,
                     "actions": asdict(opt["actions"]), "trace": df})
    res = pd.DataFrame(rows)
    res["lost_m3"] = (base_out["received_m3"] - res["received_m3"]).clip(lower=0) + \
                     (base_out["loaded_m3"] - res["loaded_m3"]).clip(lower=0)
    # score: an avoided incident dominates; then product spilled to ground; then lost throughput
    res["score"] = (res["incident"].map(lambda e: INCIDENT_COST.get(e, 0)) + res["spilled_m3"]
                    + res["lost_m3"] / 50)
    res["horizon_h"] = hours
    res["rank"] = res["score"].rank(method="first").astype(int)
    return res.sort_values("rank").reset_index(drop=True)


# --- CMMS (SAP PM style) notification --------------------------------------------
FM_CODE = {"overfill": "AIR", "leak": "ELP", "pump_trip": "VIB", "power_trip": "UST", "vapor": "ELP", "cyber": "AIR"}
FLOC = {"TK-101": "OM-JIF-TF-TK101", "TK-103": "OM-JIF-TF-TK103", "TK-102": "OM-JIF-TF-TK102",
        "PL-S3": "OM-MSPPP-PL-S3", "P-201A": "OM-JIF-PU-P201A", "LB-01": "OM-JIF-LB-01",
        "EL-33KV": "OM-JIF-EL-33KV", "OT-NET": "OM-JIF-OT-NET"}


def work_order(asset: str, cls: str, option: dict, p: float, rul: dict | None, now: pd.Timestamp) -> dict:
    prio = "1-Very high" if p >= 0.7 else "2-High" if p >= 0.4 else "3-Medium"
    wo = {
        "notification_type": "M2" if cls != "pump_trip" else "M1",
        "functional_location": FLOC.get(asset, asset),
        "equipment": asset,
        "priority": prio,
        "damage_code_iso14224": FM_CODE[cls],
        "short_text": f"RAQIB: {cls} risk on {asset} (P={p:.0%})",
        "recommended_action": option.get("en", ""),
        "required_start": str(now),
        "permit_to_work": {"hot_work": False, "confined_space": False, "isolation_required": cls in ("pump_trip", "leak")},
        "source": "RAQIB L2 advisor - created as a draft for planner approval",
        "created": str(pd.Timestamp.now(tz="Asia/Muscat").floor("s")),
    }
    if cls == "pump_trip" and rul and np.isfinite(rul.get("rul_p50", np.inf)):
        wo["rul_hours_p10_p50_p90"] = [round(rul["rul_p10"], 1), round(rul["rul_p50"], 1), round(rul["rul_p90"], 1)]
        wo["spares"] = [{"material": "BRG-6312-C3", "qty": 2}, {"material": "SEAL-P201-MECH", "qty": 1}]
    return wo


def to_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
