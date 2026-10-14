"""OT integration layer, data quality and secure-by-design controls.

Everything RAQIB reads arrives through a one-way gateway (data diode
pattern). The gateway object below has no write path to Level 0-2: any
attempt raises and is logged, which the dashboard demonstrates.

Owner roles: OT data / integration engineer, OT cybersecurity specialist.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .plant import STEPS_PER_H

CONNECTORS = pd.DataFrame([
    # system, Purdue level, protocol, direction, tags in prototype
    ("DCS", "1-2", "OPC UA (Eclipse Milo / asyncua)", "read-only via diode", "PT-001 PT-002 P201-*"),
    ("SIS / ESD", "1", "OPC UA via isolated gateway", "read-only via diode", "LSHH status, bypass register"),
    ("Pipeline SCADA + LDS", "2", "DNP3 / IEC 60870-5-104", "read-only via diode", "FI-001 FI-002"),
    ("TAS / tank gauging", "2", "Modbus TCP", "read-only via diode", "LT-101 LT-102 LT-103 FT-LOAD TAS-RECV"),
    ("Vibration monitoring", "2", "OPC UA", "read-only via diode", "P201-VIB"),
    ("Fire & gas", "1-2", "OPC UA", "read-only via diode", "GD-01"),
    ("Network IDS", "3", "Syslog / REST", "read-only", "NET-MODBUS-W"),
    ("Historian", "3", "REST (Timescale/Influx in prototype)", "read", "all"),
    ("CMMS (SAP PM)", "4", "OData", "read + create notification", "work orders"),
    ("Weather", "external", "REST", "read", "TT-900 WIND-*"),
], columns=["system", "purdue_level", "protocol", "direction", "tags"])

TAG_RANGES = {
    "FI-001": (0, 800), "FI-002": (0, 800), "PT-001": (0, 80), "PT-002": (0, 15),
    "LT-101": (0, 20), "LT-102": (0, 20), "LT-103": (0, 20), "FT-LOAD": (0, 800),
    "P201-VIB": (0, 20), "P201-TBRG": (0, 130), "P201-PDIS": (0, 15), "GRID-V": (0, 1.2),
    "GD-01": (0, 100), "TT-900": (-5, 60),
}


@dataclass
class ReadOnlyGateway:
    """Models the one-way data diode: reads succeed, writes towards control are impossible."""
    audit: list = field(default_factory=list)

    def read(self, df: pd.DataFrame, tag: str, i: int) -> float:
        self.audit.append(("READ", tag, str(df["time"].iat[i])))
        return float(df[tag].iat[i])

    def write(self, tag: str, value: float):
        self.audit.append(("WRITE-BLOCKED", tag, str(value)))
        raise PermissionError(f"Write to {tag} refused: RAQIB has no path to Level 0-2 (one-way gateway).")


def data_quality(df: pd.DataFrame, sh: pd.DataFrame, i: int, window_h: float = 2.0) -> pd.DataFrame:
    """ISA-style tag health: stale/flatline, range, spikes, and physics consistency vs the twin."""
    lo = max(0, i - int(window_h * STEPS_PER_H))
    w = df.iloc[lo:i + 1]
    phys = {"LT-101": "TK-101_res", "LT-102": "TK-102_res", "LT-103": "TK-103_res",
            "P201-VIB": "P201-VIB_res", "PT-001": "PT-001_res"}
    rows = []
    for tag, (a, b) in TAG_RANGES.items():
        s = w[tag].astype(float)
        flat = s.diff().abs().max() < 1e-6 if len(s) > 3 else False
        oor = bool(((s < a) | (s > b)).any())
        z = (s - s.rolling(6, min_periods=3).mean()) / (s.rolling(24, min_periods=6).std() + 1e-9)
        spikes = int((z.abs() > 6).sum())
        res = sh[phys[tag]].iloc[lo:i + 1].abs().mean() if tag in phys else np.nan
        thr = {"P201-VIB": 0.6, "PT-001": 1.0}.get(tag, 0.15)
        inconsistent = bool(res > thr) if np.isfinite(res) else False
        ok = not (flat or oor or inconsistent)
        status = "OK" if ok else "FLATLINE" if flat else "OUT-OF-RANGE" if oor else "PHYSICS-MISMATCH"
        rows.append({"tag": tag, "value": float(s.iat[-1]), "status": status, "flatline": flat,
                     "out_of_range": oor, "spikes": spikes, "twin_residual": res})
    return pd.DataFrame(rows)


def integrity_verdict(sh: pd.DataFrame, df: pd.DataFrame, i: int) -> dict:
    """Separate a broken instrument from a manipulated one using physics + network evidence."""
    lo = max(0, i - STEPS_PER_H)
    res = {t: float(sh[f"{t}_res"].iloc[lo:i + 1].abs().mean()) for t in ("TK-101", "TK-103", "TK-102")}
    tank = max(res, key=res.get)
    if res[tank] < 0.15:
        return {"verdict": "consistent", "tank": None, "residual": res[tank]}
    flat = bool(sh[f"{tank}_flat"].iat[i])
    writes = int(df["NET-MODBUS-W"].iloc[lo:i + 1].sum())
    if flat and writes == 0:
        v = "instrument_failure"
    elif writes > 0:
        v = "suspected_cyber"
    else:
        v = "physics_mismatch"
    return {"verdict": v, "tank": tank, "residual": res[tank], "flatline": flat, "modbus_writes_1h": writes}


IEC62443 = pd.DataFrame([
    ("FR1 Identification & authentication", "SSO + MFA for all users; no shared accounts", "design"),
    ("FR2 Use control", "Role-based access: operator / engineer / admin; least privilege", "prototype: read-only app"),
    ("FR3 System integrity", "Signed model artefacts; twin physics check flags manipulated data", "prototype"),
    ("FR4 Data confidentiality", "On-prem LLM and data; TLS inside DMZ", "design"),
    ("FR5 Restricted data flow", "Hardware data diode L2 -> L3.5; zones & conduits SL-2 (SL-3 for DCS zone)", "prototype: gateway has no write path"),
    ("FR6 Timely response to events", "Alerts to SOC; OT IDS (Nozomi / Claroty / Dragos class)", "prototype: IDS feed simulated"),
    ("FR7 Resource availability", "Edge inference continues if WAN is lost; RAQIB failure cannot affect the plant", "design"),
], columns=["requirement", "control", "status"])

PURDUE_DOT = """
digraph G {
  rankdir=BT; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];
  subgraph cluster_l01 { label="Level 0-1  Field & SIS"; style=dashed; f [label="Sensors, valves\\nSIS / ESD", fillcolor="#fde2e2"]; }
  subgraph cluster_l2 { label="Level 2  Control"; style=dashed; dcs [label="DCS / SCADA / TAS\\nF&G, vibration", fillcolor="#fde2e2"]; }
  diode [label="One-way data diode\\n(no return path)", shape=cds, fillcolor="#ffe8b3"];
  subgraph cluster_l35 { label="Level 3.5  OT DMZ"; style=dashed; ing [label="Ingestion\\nKafka + OPC UA client", fillcolor="#e3f0ff"]; }
  subgraph cluster_l3 { label="Level 3  RAQIB platform (on-prem)"; style=dashed;
     ts [label="Time-series DB\\nTimescale / Influx", fillcolor="#e3f0ff"];
     twin [label="Digital twin\\n(shadow, look-ahead, what-if)", fillcolor="#dff5e3"];
     ml [label="Prediction engine\\nanomaly, risk, RUL", fillcolor="#dff5e3"];
     adv [label="Advisor (L2)\\nranked options", fillcolor="#dff5e3"]; }
  subgraph cluster_l4 { label="Level 4  Business"; style=dashed; cmms [label="CMMS / SAP PM\\n(work-order drafts)", fillcolor="#efe3ff"]; ui [label="Operator & engineer\\ndashboards", fillcolor="#efe3ff"]; }
  f -> dcs; dcs -> diode; diode -> ing; ing -> ts; ts -> twin; twin -> ml; ml -> adv; adv -> ui; adv -> cmms;
  ui -> dcs [style=dotted, label="operator acts\\nmanually via DCS", fontsize=9];
}
"""
