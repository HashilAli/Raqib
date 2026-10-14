"""Fast checks of the physics, the twin and the safety guarantees. Run: pytest -q"""
import numpy as np
import pytest

from raqib import advisor, plant, reliability, security
from raqib.models import build_features, build_labels, pump_rul
from raqib.twin import shadow


@pytest.mark.parametrize("scenario", plant.SCENARIOS)
def test_every_scenario_runs(scenario):
    df = plant.run_scenario(scenario, seed=3, days=2)
    assert len(df) == 2 * 24 * plant.STEPS_PER_H
    assert df[["LT-101", "LT-102", "LT-103"]].ge(0).all().all()


def test_twin_catches_stuck_gauge_long_before_overflow():
    df = plant.run_scenario("overfill", seed=1, days=7)
    sh = shadow(df)
    res = sh[["TK-101_res", "TK-103_res"]].abs().max(axis=1)
    first = int(np.argmax(res.to_numpy() > 0.15))
    overflow = int(df.index[df["event"] == "overfill"][0])
    assert first > 0 and (overflow - first) / plant.STEPS_PER_H > 12


def test_normal_run_has_no_twin_residual():
    df = plant.run_scenario("normal", seed=5, days=4)
    sh = shadow(df)
    assert sh[["TK-101_res", "TK-103_res", "TK-102_res"]].abs().max().max() < 0.1


def test_leak_location_within_10_percent():
    df = plant.run_scenario("leak", seed=2, days=7)
    sh = shadow(df)
    est = sh["leak_pos_frac"][df["true_leak"] > 0].iloc[24:].median()
    assert abs(est - plant.LEAK_POS) < 0.10


def test_rul_predicts_pump_failure_window():
    df = plant.run_scenario("pump", seed=1, days=10)
    sh = shadow(df)
    trip = int(df.index[df["event"] == "pump_trip"][0])
    i = trip - 72 * plant.STEPS_PER_H
    r = pump_rul(df, sh, i)
    assert 36 < r["rul_p50"] < 120


def test_labels_start_only_after_precursor():
    df = plant.run_scenario("vapor", seed=4, days=7)
    y = build_labels(df)
    assert not y["vapor"][df["true_release"] == 0].any()


def test_gateway_cannot_write_to_control():
    gw = security.ReadOnlyGateway()
    with pytest.raises(PermissionError):
        gw.write("FIC-002.SP", 0.0)
    assert gw.audit[-1][0] == "WRITE-BLOCKED"


def test_whatif_prefers_option_that_prevents_overfill():
    cfg = plant.default_config("overfill", seed=1)
    df, st = plant.simulate(cfg, 110)
    sh = shadow(df)
    tl = {t: float(sh[f"{t}_est"].iat[-1]) for t in ("TK-101", "TK-103", "TK-102")}
    res = advisor.whatif(cfg, st, "overfill", df["TAS-RECV"].iat[-1], tl)
    assert res.iloc[0]["incident"] == ""
    assert (res["key"] == "none").any() and res.loc[res["key"] == "none", "incident"].iat[0] == "overfill"


def test_fmea_rpn():
    f = reliability.load_fmea()
    assert (f["RPN_before"] == f["S"] * f["O"] * f["D_before"]).all()
    assert (f["RPN_after"] <= f["RPN_before"]).all()


@pytest.mark.slow
def test_models_pass_acceptance_on_unseen_runs():
    from raqib.models import train
    eng = train(train_seeds=(11, 12, 13), test_seeds=(21, 22), days=10)
    acc = eng.train_info["evaluation"]["acceptance"]
    assert acc["passed"].mean() >= 0.9
    assert acc.loc[acc["scenario"] == "normal", "false_alerts"].sum() == 0
