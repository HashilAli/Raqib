"""Simulated product terminal: the stand-in for the real plant.

This module plays the role of the physical site (a products storage and
distribution terminal at the end of a products pipeline). It produces the
"measured" tags a historian would hold, plus hidden ``true_*`` columns used
only for scoring the models. RAQIB itself never reads the ``true_*`` columns.

Owner role: process / simulation engineer.

Units: flow m3/h, level m, pressure bar(g), temperature degC,
vibration mm/s RMS, LEL % of lower flammable limit.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DT_MIN = 5                       # simulation step, minutes
STEPS_PER_H = 60 // DT_MIN
START = pd.Timestamp("2026-07-14 00:00")   # a hot Omani summer week

# Tank geometry (floating-roof tanks, 40 m diameter)
TANK_AREA = 1250.0               # m2
TANK_HEIGHT = 20.0               # m
SWITCH_LEVEL = 17.0              # operating procedure: change receiving tank
LSHH_LEVEL = 19.0                # independent high-high level switch (SIS)
OVERFILL_LEVEL = 19.6            # product reaches the rim -> spill
GASOLINE_TANKS = ("TK-101", "TK-103")

# Pipeline (terminal segment of a products line)
LINE_KM = 60.0
LINE_KL = 35.0 / 330.0 ** 2      # friction coefficient * length, bar/(m3/h)^2
LEAK_POS = 0.55                  # fraction of the segment from the upstream station

# Pump P-201A (gasoline loading pump), ISO 10816-3 style limits
VIB_ALARM = 4.5
VIB_TRIP = 7.1
_DEG_A, _DEG_TAU, _DEG_T0 = 0.0763, 45.0, 12.0

# Gasoline vapour
LFL_G_M3 = 40.5                  # ~1.4 vol% at 30 degC
DETECTOR_DIST = 20.0             # m from loading arm to gas detector GD-01
DETECTOR_BEARING = 300.0         # deg, detector sits downwind of the prevailing (NW-bound) wind

SCENARIOS = ("normal", "overfill", "leak", "pump", "power", "cyber", "vapor")
EVENT_TYPES = ("overfill", "leak", "pump_trip", "power_trip", "vapor")


@dataclass
class Actions:
    """Operator actions that the what-if engine can apply to a copy of the plant."""
    receipt_factor: float = 1.0      # pipeline receipt rate multiplier
    divert_to: str | None = None     # force receipt into this tank
    isolate_line: bool = False       # close segment block valves
    pump_standby: bool = False       # run standby pump P-201B instead of P-201A
    pump_speed: float = 1.0          # VFD speed fraction for the loading pump
    stop_loading: bool = False       # stop all truck loading
    load_shed: bool = False          # shed non-essential electrical load
    gen_prestart: bool = False       # pre-start emergency diesel generators


@dataclass
class PlantState:
    k: int = 0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    levels: dict = field(default_factory=lambda: {"TK-101": 12.0, "TK-102": 10.0, "TK-103": 9.0})
    receiving: str = "TK-101"
    frozen: dict = field(default_factory=dict)          # tank -> stuck reading
    spoofed: str | None = None                          # tank whose gauge is replayed
    spoof_k0: int | None = None
    lt_history: dict = field(default_factory=lambda: {t: [] for t in ("TK-101", "TK-102", "TK-103")})
    pump_age: float = 0.0                               # effective degradation hours
    pump_tripped_until: int = -1
    pump_failed: bool = False
    grid_stress: float = 0.0
    blackout_until: int = -1
    receipt_shut: bool = False                          # ESD or overflow shut the receipt
    release_stopped_at: int | None = None
    events: list = field(default_factory=list)


@dataclass
class ScenarioConfig:
    name: str = "normal"
    seed: int = 0
    fault_h: float = 108.0           # fault onset (hours from start), scenario-specific default
    lshh_in_service: bool = True     # independent high-high switch healthy
    standby_available: bool = True


def default_config(scenario: str, seed: int = 0, **kw) -> ScenarioConfig:
    onset = {"overfill": 72.0, "leak": 120.0, "pump": 12.0, "power": 96.0,
             "cyber": 84.0, "vapor": 139.0, "normal": 1e9}[scenario]
    cfg = ScenarioConfig(name=scenario, seed=seed, fault_h=onset)
    if scenario == "overfill":
        cfg.lshh_in_service = False   # Buncefield pattern: the independent switch was inoperable too
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def plume_conc(q_gs: float, x: float, y: float, u: float, night: bool) -> float:
    """Ground-level Gaussian plume concentration (g/m3) for a ground release.

    Briggs rural dispersion: stability F at night with light wind, D otherwise.
    """
    if x <= 0.5 or q_gs <= 0:
        return 0.0
    u = max(u, 0.5)
    if night and u < 3:
        sy = 0.04 * x / np.sqrt(1 + 0.0001 * x)
        sz = 0.016 * x / (1 + 0.0003 * x)
    else:
        sy = 0.08 * x / np.sqrt(1 + 0.0001 * x)
        sz = 0.06 * x / np.sqrt(1 + 0.0015 * x)
    # low-wind plume meander widens the crosswind spread (sigma_theta ~20 deg at night)
    sy = float(np.hypot(sy, x * (0.35 if night else 0.15)))
    return float(q_gs / (np.pi * sy * sz * u) * np.exp(-y * y / (2 * sy * sy)))


def _release_rate(cfg: ScenarioConfig, h: float, st: PlantState) -> float:
    if cfg.name != "vapor" or h < cfg.fault_h:
        return 0.0
    q = min(400.0, 3.0 * np.exp((h - cfg.fault_h) / 1.5))
    if st.release_stopped_at is not None:
        q *= np.exp(-(st.k - st.release_stopped_at) / 2.0)   # residual pool evaporating
    return q


def simulate(cfg: ScenarioConfig, hours: float, actions: Actions | None = None,
             state: PlantState | None = None) -> tuple[pd.DataFrame, PlantState]:
    """Run the plant for ``hours`` and return (tag history, final state).

    Pass ``state`` (deep-copied by the caller) to continue from a snapshot,
    which is how the what-if engine branches the future.
    """
    a = actions or Actions()
    st = state or PlantState(rng=np.random.default_rng(cfg.seed))
    rng = st.rng
    rows = []
    n = int(round(hours * STEPS_PER_H))
    dt_h = DT_MIN / 60.0

    for _ in range(n):
        k = st.k
        h = k * dt_h
        hod = h % 24
        night = hod < 6 or hod >= 19
        fault = h >= cfg.fault_h
        event = ""

        # --- weather -------------------------------------------------------
        ambient = 36 + 8 * np.sin(2 * np.pi * (hod - 9) / 24) + rng.normal(0, 0.5)
        wind = max(0.3, 3.2 + 2.2 * np.sin(2 * np.pi * (hod - 11) / 24) + rng.normal(0, 0.4))
        if cfg.name == "vapor" and fault:
            wind = max(0.3, 0.8 + rng.normal(0, 0.2))        # calm evening
        wind_dir = (300 + 25 * np.sin(2 * np.pi * h / 30) + rng.normal(0, 5)) % 360  # "towards", deg

        # --- electrical grid ----------------------------------------------
        volt, freq = 1.0 + rng.normal(0, 0.004), 50 + rng.normal(0, 0.02)
        if cfg.name == "power" and fault:
            lam = 0.02 + 0.30 * max(0.0, (ambient - 39) / 5) * min(1.0, (h - cfg.fault_h) / 24)
            if rng.random() < lam:
                depth = rng.uniform(0.08, 0.25)
                volt -= depth
                freq -= 0.6 * depth
                st.grid_stress += depth * (0.5 if a.load_shed else 1.0)
        st.grid_stress *= 0.985
        blackout = k < st.blackout_until
        if not blackout and st.grid_stress > 1.15:
            dur = 0.25 if a.gen_prestart else 2.0
            st.blackout_until = k + int(dur * STEPS_PER_H)
            st.grid_stress = 0.0
            blackout = True
            event = "power_trip"
            st.events.append((k, "power_trip"))
        if blackout:
            volt = rng.normal(0.02, 0.01) if not a.gen_prestart else rng.normal(0.97, 0.01)

        # --- pipeline receipt ---------------------------------------------
        q_src = 330 * (1 + 0.05 * np.sin(2 * np.pi * h / 24 + 1.0)) + rng.normal(0, 5)
        q_src *= a.receipt_factor
        # a manual diversion is managed by hand-dipping: stop receipt when the diverted tank is full
        if a.divert_to and st.levels[a.divert_to] >= SWITCH_LEVEL:
            q_src = 0.0
        if a.isolate_line or st.receipt_shut or blackout:
            q_src = 0.0
        q_leak = 0.0
        if cfg.name == "leak" and fault:
            frac = min(0.04, 0.012 + 0.03 * (h - cfg.fault_h) / 24)
            q_leak = frac * (q_src if q_src > 0 else 330 * 0.25)   # line drains by gravity when stopped
            if a.isolate_line:
                q_leak *= 0.15
        q_term = max(0.0, q_src - q_leak)
        if q_src > 0:
            p2 = 4.0 + 0.004 * q_term - 8 * q_leak / max(q_src, 1)
            p1 = p2 + LINE_KL * (q_src ** 2 * LEAK_POS + q_term ** 2 * (1 - LEAK_POS))
        else:
            p2, p1 = 2.5 - 10 * q_leak / 330, 8.0 - 20 * q_leak / 330
        if cfg.name == "leak" and fault and k == int(cfg.fault_h * STEPS_PER_H):
            event = event or "leak"
            st.events.append((k, "leak"))

        # --- receiving tank selection (TAS logic acts on MEASURED level) ---
        rec = a.divert_to or st.receiving
        draw = GASOLINE_TANKS[1] if rec == GASOLINE_TANKS[0] else GASOLINE_TANKS[0]

        # --- loading demand & pump ----------------------------------------
        base_load = 420 + rng.normal(0, 40) if 6 <= hod < 22 else 150 + rng.normal(0, 20)
        pump_ok = not blackout and k >= st.pump_tripped_until and not a.stop_loading
        running = "-"
        speed = a.pump_speed
        if pump_ok:
            if a.pump_standby or st.pump_failed:
                running = "B" if cfg.standby_available else "-"
            else:
                running = "A"
        q_load = max(0.0, base_load * speed) if running != "-" else 0.0
        if st.levels[draw] < 1.2:
            q_load = 0.0

        if running == "A":
            st.pump_age += dt_h * speed ** 3
        vib_add = 0.0
        if cfg.name == "pump" and st.pump_age > _DEG_T0:
            vib_add = _DEG_A * (np.exp((st.pump_age - _DEG_T0) / _DEG_TAU) - 1)
        run_vib_add = vib_add if running == "A" else 0.0
        vib = (1.6 + 0.0015 * q_load + run_vib_add + rng.normal(0, 0.12)) if running != "-" else abs(rng.normal(0.2, 0.05))
        if running == "A" and 1.6 + 0.0015 * q_load + vib_add >= VIB_TRIP and not st.pump_failed:
            st.pump_failed = True
            st.pump_tripped_until = k + 6       # 30 min until standby is lined up
            event = event or "pump_trip"
            st.events.append((k, "pump_trip"))
        t_brg = ambient + (22 + 0.01 * q_load + 15 * run_vib_add / 4.9 if running != "-" else 2) + rng.normal(0, 0.4)
        p_dis = (9.0 - 2.0e-5 * q_load ** 2 - 0.4 * run_vib_add / 4.9 + rng.normal(0, 0.05)) if running != "-" else 0.3
        amps = (80 + 0.12 * q_load + 2 * run_vib_add / 4.9 + rng.normal(0, 0.8)) if running != "-" else 0.0

        # --- diesel tank (independent batch receipts) ----------------------
        q102_in = 300.0 if (h % 72) < 10 and not blackout else 0.0
        q102_out = (45.0 if 6 <= hod < 22 else 15.0) * (0 if a.stop_loading or blackout else 1)

        # --- tank balances -------------------------------------------------
        st.levels[rec] += q_term * dt_h / TANK_AREA
        st.levels[draw] -= q_load * dt_h / TANK_AREA
        st.levels["TK-102"] += (q102_in - q102_out) * dt_h / TANK_AREA

        if cfg.lshh_in_service and st.levels[rec] >= LSHH_LEVEL and not st.receipt_shut:
            st.receipt_shut = True
            event = event or "esd_lshh"
            st.events.append((k, "esd_lshh"))
        if st.levels[rec] >= OVERFILL_LEVEL:
            if not any(e == "overfill" for _, e in st.events):
                event = "overfill"
                st.events.append((k, "overfill"))
            st.levels[rec] = OVERFILL_LEVEL        # spilling over the rim
            st.receipt_shut = True                 # emergency response closes the line

        # --- instrument faults --------------------------------------------
        if cfg.name == "overfill" and fault and not st.frozen and rec == st.receiving and st.levels[rec] >= 12.0:
            st.frozen[rec] = round(st.levels[rec], 3)
        if cfg.name == "cyber" and fault and st.spoofed is None:
            st.spoofed = rec

        lt = {}
        for t, lvl in st.levels.items():
            v = lvl + rng.normal(0, 0.005)
            if t in st.frozen:
                v = st.frozen[t]
            elif t == st.spoofed:
                # replay attack: a recorded 6 h loop from 30 h earlier is fed to the TAS/DCS
                hist = st.lt_history[t]
                k0 = st.spoof_k0 if st.spoof_k0 is not None else len(hist)
                st.spoof_k0 = k0
                start = max(0, k0 - 30 * STEPS_PER_H)
                v = hist[start + (len(hist) - k0) % (6 * STEPS_PER_H)]
            lt[t] = v
        for t in lt:
            st.lt_history[t].append(lt[t] if t != st.spoofed else st.levels[t] + rng.normal(0, 0.005))

        # TAS changes the receiving tank when the MEASURED level hits the procedure limit
        if a.divert_to is None and lt[st.receiving] >= SWITCH_LEVEL:
            st.receiving = draw

        # --- vapour release at the loading bay -----------------------------
        if a.stop_loading and st.release_stopped_at is None and cfg.name == "vapor" and fault:
            st.release_stopped_at = k
        q_rel = _release_rate(cfg, h, st)
        rel_bearing = (DETECTOR_BEARING - wind_dir + 540) % 360 - 180
        xd = DETECTOR_DIST * np.cos(np.radians(rel_bearing))
        yd = DETECTOR_DIST * np.sin(np.radians(rel_bearing))
        conc = plume_conc(q_rel, xd, yd, wind, night) if xd > 0 else 0.0
        lel = min(100.0, 100 * conc / LFL_G_M3 + abs(rng.normal(0, 0.3)))
        if cfg.name == "vapor" and lel >= 60 and not any(e == "vapor" for _, e in st.events):
            event = event or "vapor"
            st.events.append((k, "vapor"))

        modbus_writes = int(rng.poisson(3)) if (cfg.name == "cyber" and fault) else 0

        rows.append({
            "time": START + pd.Timedelta(minutes=DT_MIN * k),
            "step": k,
            "TT-900": ambient, "WIND-SPD": wind, "WIND-DIR": wind_dir,
            "FI-001": q_src * (1 + rng.normal(0, 0.004)),
            "FI-002": q_term * (1 + rng.normal(0, 0.004)),
            "PT-001": p1 + rng.normal(0, 0.1), "PT-002": p2 + rng.normal(0, 0.05),
            "LT-101": lt["TK-101"], "LT-102": lt["TK-102"], "LT-103": lt["TK-103"],
            "FT-LOAD": q_load * (1 + rng.normal(0, 0.004)),
            "FT-102IN": q102_in, "FT-102OUT": q102_out,
            "P201-VIB": vib, "P201-TBRG": t_brg, "P201-PDIS": p_dis, "P201-AMP": amps,
            "GRID-V": volt, "GRID-F": freq,
            "GD-01": lel, "NET-MODBUS-W": modbus_writes,
            "TAS-RECV": rec, "PUMP-RUN": running,
            "true_L101": st.levels["TK-101"], "true_L102": st.levels["TK-102"], "true_L103": st.levels["TK-103"],
            "true_leak": q_leak, "true_vib_add": vib_add, "true_release": q_rel,
            "true_stress": st.grid_stress, "event": event,
        })
        st.k += 1

    return pd.DataFrame(rows), st


def snapshot(state: PlantState) -> PlantState:
    return copy.deepcopy(state)


def run_scenario(scenario: str, seed: int = 0, days: float = 7.0, **cfg_kw) -> pd.DataFrame:
    df, _ = simulate(default_config(scenario, seed, **cfg_kw), days * 24)
    return df
