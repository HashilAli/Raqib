"""RAQIB - Refinery AI for Quick Incident Prediction & Bypass.

Open-source prototype dashboard. Run:  streamlit run app.py
Everything runs on synthetic data from a simulated product terminal.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from raqib import advisor, geo, plant, reliability, security
from raqib.assistant import Assistant
from raqib.models import CLASS_FEATURES, CLASSES, Engine, first_alert, prepare, psi, pump_rul, train
from raqib.plant import LSHH_LEVEL, STEPS_PER_H, SWITCH_LEVEL, VIB_ALARM, VIB_TRIP
from raqib.twin import lookahead_tank, twin_accuracy

st.set_page_config(page_title="RAQIB | رقيب", page_icon="🛡️", layout="wide")

# ------------------------------------------------------------------ language
# deep links, e.g. ?lang=en&scenario=pump&page=asset (handy for sharing a specific screen)
if "lang" not in st.session_state:
    qp = st.query_params
    st.session_state.lang = qp.get("lang", "ar") if qp.get("lang") in ("ar", "en") else "ar"
    st.session_state.page0 = qp.get("page", "control")
    st.session_state.scenario0 = qp.get("scenario", "overfill")


def L(ar: str, en: str) -> str:
    return ar if st.session_state.lang == "ar" else en


AR = st.session_state.lang == "ar"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, label, button, input, textarea {{ font-family: 'Tajawal', 'Segoe UI', sans-serif; }}
{'''.stMarkdown, .stAlert, [data-testid="stMetric"], [data-testid="stCaptionContainer"], h1, h2, h3, h4, label, .stRadio, .stCheckbox, [data-testid="stExpander"] summary {direction: rtl; text-align: right;}
section[data-testid="stSidebar"] {direction: rtl;}''' if AR else ''}
.raqib-badge {{display:inline-block;padding:2px 10px;border-radius:12px;font-weight:600;font-size:0.85rem}}
.b-high {{background:#fde2e2;color:#a61b1b}} .b-medium {{background:#fff1cc;color:#8a5a00}} .b-low {{background:#dff5e3;color:#1d6b33}}
</style>
""", unsafe_allow_html=True)

SCENARIO_LABEL = {
    "normal": ("تشغيل طبيعي", "Normal operation"),
    "overfill": ("فيضان خزان: مقياس مستوى عالق (نمط Buncefield)", "Tank overfill: stuck level gauge (Buncefield pattern)"),
    "leak": ("تسرب في خط أنابيب المنتجات", "Products pipeline leak"),
    "pump": ("تدهور محمل مضخة التحميل P-201A", "Loading pump P-201A bearing degradation"),
    "power": ("اضطراب الشبكة الكهربائية في حر الصيف", "Grid disturbance in summer heat"),
    "cyber": ("هجوم سيبراني: إعادة تشغيل قراءات المستوى", "Cyber attack: replayed level readings"),
    "vapor": ("انبعاث أبخرة في رصيف التحميل ليلاً", "Night-time vapour release at the loading bay"),
}
DEFAULT_NOW = {"normal": 150.0, "overfill": 118.0, "leak": 121.5, "pump": 150.0, "power": 106.0,
               "cyber": 100.0, "vapor": 144.5}
CLASS_LABEL = {
    "overfill": ("فيضان خزان", "Tank overfill"), "leak": ("تسرب الخط", "Pipeline leak"),
    "pump_trip": ("فشل المضخة", "Pump failure"), "power_trip": ("انقطاع الكهرباء", "Power loss"),
    "vapor": ("سحابة أبخرة", "Vapour cloud"), "cyber": ("تلاعب سيبراني", "Cyber manipulation"),
    "integrity": ("مقياس غير موثوق", "Untrusted instrument"),
}
FEATURE_LABEL = {
    "tank_res_abs": ("فرق المستوى عن التوأم (م)", "Level residual vs twin (m)"),
    "tank_flat": ("المقياس ثابت بلا تذبذب", "Gauge flatlined"),
    "hidden_h_to_lshh": ("ساعات حتى LSHH بتقدير التوأم", "Hours to LSHH (twin estimate)"),
    "line_cusum": ("CUSUM اختلال الخط", "Line imbalance CUSUM"),
    "line_imb_pct": ("اختلال موازنة الخط %", "Line imbalance %"),
    "line_imb_6h": ("اختلال الخط - متوسط 6 ساعات", "Line imbalance 6 h mean"),
    "pt2_drop_1h": ("هبوط ضغط الوصول (بار/ساعة)", "Arrival pressure drop (bar/h)"),
    "vib_res_1h": ("فرق الاهتزاز عن التوأم", "Vibration residual"),
    "vib_res_slope_24h": ("ميل الاهتزاز (مم/ث/يوم)", "Vibration slope (mm/s/day)"),
    "vib_1h": ("الاهتزاز (مم/ث)", "Vibration (mm/s)"),
    "tbrg_res_1h": ("فرق حرارة المحمل", "Bearing temp residual"),
    "sag_count_6h": ("هبوطات الجهد في 6 ساعات", "Voltage sags in 6 h"),
    "vmin_6h": ("أدنى جهد في 6 ساعات (pu)", "Min voltage 6 h (pu)"),
    "ambient": ("حرارة الجو", "Ambient temperature"),
    "lel": ("تركيز الغاز % LEL", "Gas % LEL"),
    "lel_slope_1h": ("ارتفاع الغاز في ساعة", "Gas rise in 1 h"),
    "modbus_w_1h": ("أوامر Modbus غير معروفة/ساعة", "Unknown Modbus writes / h"),
}


def cl(c):
    return L(*CLASS_LABEL[c])


def fl(f):
    return L(*FEATURE_LABEL.get(f, (f, f)))


# ------------------------------------------------------------------ cached computation
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    return train()


@st.cache_resource(show_spinner=False)
def get_assistant() -> Assistant:
    return Assistant()


@st.cache_data(show_spinner=False)
def get_run(scenario: str, seed: int, lshh: bool, standby: bool, days: float = 7.0):
    cfg = plant.default_config(scenario, seed, lshh_in_service=lshh, standby_available=standby)
    df, _ = plant.simulate(cfg, days * 24)
    sh, X = prepare(df)
    eng = get_engine()
    P = eng.proba(X)
    P["anomaly"] = eng.anomaly(X)
    return df, sh, X, P


@st.cache_data(show_spinner=False)
def get_whatif(scenario: str, seed: int, lshh: bool, standby: bool, i: int, cls: str):
    cfg = plant.default_config(scenario, seed, lshh_in_service=lshh, standby_available=standby)
    _, state = plant.simulate(cfg, (i + 1) / STEPS_PER_H)
    df, sh, _, _ = get_run(scenario, seed, lshh, standby)
    tl = {t: float(sh[f"{t}_est"].iat[i]) for t in ("TK-101", "TK-103", "TK-102")}
    res = advisor.whatif(cfg, state, cls, df["TAS-RECV"].iat[i], tl)
    return res


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🛡️ RAQIB · رقيب")
    lang = st.radio("Language / اللغة", ["العربية", "English"], horizontal=True,
                    index=0 if st.session_state.lang == "ar" else 1)
    new_lang = "ar" if lang == "العربية" else "en"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    PAGES = {
        "control": ("🏭 غرفة القيادة", "🏭 Control room"),
        "twin": ("🔁 التوأم الرقمي", "🔁 Digital twin"),
        "asset": ("⚙️ بطاقة المعدة P-201A", "⚙️ Asset card P-201A"),
        "advisor": ("🧪 ماذا لو والحلول", "🧪 What-if & solutions"),
        "emergency": ("🚨 مركز الطوارئ", "🚨 Emergency centre"),
        "fmea": ("📋 FMEA والموثوقية", "📋 FMEA & reliability"),
        "mlops": ("🧠 النماذج وMLOps", "🧠 Models & MLOps"),
        "security": ("🔐 تكامل OT والأمن", "🔐 OT integration & security"),
        "assistant": ("💬 المساعد", "💬 Assistant"),
        "voice": ("🗣️ صوت المشغل", "🗣️ Operator voice"),
        "about": ("👥 الفريق والمعمارية", "👥 Team & architecture"),
    }
    page = st.radio(L("الشاشة", "Screen"), list(PAGES), format_func=lambda k: L(*PAGES[k]),
                    index=list(PAGES).index(st.session_state.page0) if st.session_state.page0 in PAGES else 0)
    st.divider()
    scenario = st.selectbox(L("السيناريو", "Scenario"), plant.SCENARIOS,
                            index=plant.SCENARIOS.index(st.session_state.scenario0) if st.session_state.scenario0 in plant.SCENARIOS else 1,
                            format_func=lambda s: L(*SCENARIO_LABEL[s]))
    cfg0 = plant.default_config(scenario)
    seed = st.number_input(L("بذرة المحاكاة", "Simulation seed"), 0, 999, 1)
    now_h = st.slider(L("الوقت الحالي (ساعات من بداية المحاكاة)", "Current time (hours since start)"),
                      24.0, 167.75, DEFAULT_NOW[scenario], 0.25, key=f"now_{scenario}")
    st.caption(L("حرّك الوقت لترى متى ينبه رقيب قبل الحادث.", "Move time to see how early RAQIB warns."))
    with st.expander(L("حواجز الحماية (يقرأها رقيب ولا يغيرها)", "Protection barriers (read, never written)")):
        lshh = st.checkbox(L("مفتاح LSHH في الخدمة", "LSHH switch in service"), cfg0.lshh_in_service, key=f"lshh_{scenario}")
        standby = st.checkbox(L("المضخة الاحتياطية متاحة", "Standby pump available"), True)
        gens = st.checkbox(L("مولدات الطوارئ جاهزة", "Emergency generators ready"), True)
        gas_esd = st.checkbox(L("كشف الغاز والإيقاف الطارئ للرصيف سليم", "Bay gas detection & ESD healthy"), True)
    shadow_mode = st.toggle(L("وضع الظل (لا ترسل إنذارات للمشغل)", "Shadow mode (no alerts to operators)"), False)
    st.caption(L("بيانات اصطناعية من محاكي محطة منتجات. ليست بيانات أي جهة حقيقية.",
                 "Synthetic data from a simulated products terminal. Not real site data."))

with st.spinner(L("تدريب النماذج على سيناريوهات المحاكاة (مرة واحدة فقط)...",
                  "Training models on simulated scenarios (first load only)...")):
    eng = get_engine()
    df_all, sh_all, X_all, P_all = get_run(scenario, int(seed), lshh, standby)

i = min(int(now_h * STEPS_PER_H), len(df_all) - 1)
df, sh, X, P = df_all.iloc[: i + 1], sh_all.iloc[: i + 1], X_all.iloc[: i + 1], P_all.iloc[: i + 1]
now = df["time"].iat[-1]
p_now = P.iloc[-1]
rul = pump_rul(df_all, sh_all, i)
fmea = reliability.load_fmea()
barriers = advisor.Barriers(lshh_in_service=lshh, standby_pump_ok=standby, generators_ok=gens, gas_esd_ok=gas_esd)
verdict = security.integrity_verdict(sh_all, df_all, i)
risk = advisor.risk_table(p_now, rul, df["TAS-RECV"].iat[-1], verdict.get("tank"), fmea, barriers)

ALERT_THR = 0.5


def active_alerts() -> list[dict]:
    out = []
    for c in CLASSES:
        if p_now[c] >= ALERT_THR:
            fa = first_alert(P[c].to_numpy(), ALERT_THR)
            since = (i - fa) / STEPS_PER_H if fa is not None else 0.0
            out.append({"class": c, "p": float(p_now[c]), "since_h": since})
    if verdict["verdict"] != "consistent":
        # the twin no longer trusts a gauge: raise it now, long before the 12 h incident window opens
        res = sh["TK-101_res"].abs().combine(sh["TK-103_res"].abs(), max).combine(sh["TK-102_res"].abs(), max)
        fa = first_alert((res > 0.15).astype(float).to_numpy(), 0.5)
        out.append({"class": "integrity", "p": 1.0, "since_h": (i - fa) / STEPS_PER_H if fa is not None else 0.0,
                    "tank": verdict["tank"], "verdict": verdict["verdict"]})
    if rul["p_fail_72h"] >= 0.5 and not any(a["class"] == "pump_trip" for a in out):
        out.append({"class": "pump_trip", "p": rul["p_fail_72h"], "since_h": 0.0, "rul": True})
    return sorted(out, key=lambda a: -a["p"])


alerts = active_alerts()
if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "outbox" not in st.session_state:
    st.session_state.outbox = []
if "gateway" not in st.session_state:
    st.session_state.gateway = security.ReadOnlyGateway()


def badge(band: str) -> str:
    txt = {"high": L("مرتفع", "High"), "medium": L("متوسط", "Medium"), "low": L("منخفض", "Low")}[band]
    return f'<span class="raqib-badge b-{band}">{txt}</span>'


def cesium_map(height: int = 520, **kw):
    rows = [dict(r, name=r["name_ar"] if AR else r["name_en"], band=advisor.risk_band(r["risk"])) for r in risk.to_dict("records")]
    levels = {t: float(sh[f"{t}_est"].iat[-1]) for t in ("TK-101", "TK-103", "TK-102")}
    components.iframe(geo.map_url(geo.map_payload(rows, levels, AR, **kw)), height=height)


def time_axis(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2),
                      hovermode="x unified")
    return fig


# ================================================================== pages
def page_control():
    st.title(L("🏭 غرفة القيادة — رقيب", "🏭 Control room — RAQIB"))
    st.caption(f"{L('محطة منتجات افتراضية', 'Virtual product terminal')} · {now:%Y-%m-%d %H:%M} · {L(*SCENARIO_LABEL[scenario])}")
    acc = twin_accuracy(df, sh)
    dq = security.data_quality(df_all, sh_all, i)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(L("تنبيهات نشطة", "Active alerts"), 0 if shadow_mode else len(alerts))
    c2.metric(L("معدات بخطر مرتفع", "Assets at high risk"), int((risk["risk"] >= 30).sum()))
    c3.metric(L("شذوذ متعدد المتغيرات", "Multivariate anomaly"), f"{p_now['anomaly']:.0%}")
    c4.metric(L("دقة التوأم (ضمن ±2%)", "Twin accuracy (within ±2%)"), f"{acc['within_2pct'].median():.0f}%")
    c5.metric(L("جودة البيانات", "Data quality"), f"{(dq['status'] == 'OK').mean():.0%}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader(L("خريطة الموقع ملونة بمؤشر الخطر الحي", "Site map coloured by the live risk index"))
        cesium_map(440)
        st.caption(L("خريطة ثلاثية الأبعاد (CesiumJS). ارتفاع السائل داخل كل خزان من تقدير التوأم، واللون من مؤشر الخطر. اضغط أي معدة لتفاصيلها. المخطط توضيحي على إحداثيات الموقع التجريبي في وثيقة المشروع.",
                     "3D map (CesiumJS). Liquid height in each tank is the twin's estimate; colour is the risk index. Click any asset for details. Illustrative layout at the pilot coordinates from the project document."))
        st.caption("R = P × C × (1 − B)  · " + L("P احتمال الحادث، C العواقب من FMEA، B فاعلية الحواجز.",
                                              "P incident probability, C consequence from FMEA, B barrier effectiveness."))
    with right:
        st.subheader(L("أهم 5 تنبؤات", "Top 5 predictions"))
        top = risk.head(5).copy()
        for _, r in top.iterrows():
            band = advisor.risk_band(r["risk"])
            st.markdown(f"**{r['asset']}** · {cl(r['class'])} {badge(band)}  \n"
                        f"{L('الخطر', 'Risk')} {r['risk']:.0f} · P {r['P']:.0%}", unsafe_allow_html=True)
        st.subheader(L("التنبيهات", "Alerts"))
        if shadow_mode:
            st.info(L("وضع الظل مفعل: التنبؤات تُسجل للتقييم ولا تظهر للمشغل.",
                      "Shadow mode is on: predictions are logged for evaluation, not shown to operators."))
        elif not alerts:
            st.success(L("لا توجد تنبيهات. كل المعدات ضمن السلوك المتوقع.", "No alerts. Every asset behaves as the twin expects."))
        for a in ([] if shadow_mode else alerts):
            with st.container(border=True):
                if a["class"] == "integrity":
                    la = lookahead_tank(df, sh, len(df) - 1)
                    st.markdown(f"**⚠️ {cl('integrity')}: {a['tank']}**  \n{L('منذ', 'for')} {a['since_h']:.1f} {L('ساعة', 'h')}")
                    g = float(df[f"LT-{a['tank'][-3:]}"].iat[-1])
                    e = float(sh[a["tank"] + "_est"].iat[-1])
                    st.caption(L(f"المقياس يقرأ {g:.2f} م، لكن موازنة الكتلة في التوأم تقدّر المستوى بـ {e:.2f} م. الوقت حتى LSHH حسب التوأم {la['hours_to_lshh']:.1f} ساعة.",
                                 f"The gauge reads {g:.2f} m, but the twin's mass balance puts the level at {e:.2f} m. Hours to LSHH by the twin: {la['hours_to_lshh']:.1f}."))
                    st.caption({"instrument_failure": L("التشخيص: عطل مستشعر (قراءة ثابتة).", "Diagnosis: instrument failure (flat reading)."),
                                "suspected_cyber": L("التشخيص: تلاعب مشتبه به (قراءة طبيعية الشكل تخالف الفيزياء + أوامر Modbus غريبة).", "Diagnosis: suspected manipulation (normal-looking reading that breaks physics + unknown Modbus writes)."),
                                "physics_mismatch": L("التشخيص: تعارض مع الفيزياء.", "Diagnosis: physics mismatch.")}[a["verdict"]])
                    continue
                head = f"⚠️ {cl(a['class'])} — {a['p']:.0%}"
                since = f"  \n{L('منذ', 'for')} {a['since_h']:.1f} {L('ساعة', 'h')}" if a["since_h"] > 0 else ""
                st.markdown(f"**{head}**{since}")
                if a["class"] == "pump_trip" and np.isfinite(rul["rul_p50"]):
                    st.caption(L(f"الزمن المتبقي المقدر {rul['rul_p50']:.0f} ساعة", f"Estimated RUL {rul['rul_p50']:.0f} h"))
                ex = eng.explain(X.iloc[-1], a["class"], top=3)
                why = "، ".join(f"{fl(r.feature)} = {r.value:.2f}" for r in ex.itertuples() if r.contribution > 0.01)
                if not why:
                    dv = eng.deviations(X.iloc[-1], a["class"]).head(2)
                    why = "، ".join(f"{fl(r.feature)} = {r.value:.2f} ({r.z:+.1f}σ)" for r in dv.itertuples())
                if why:
                    st.caption(L("السبب: ", "Why: ") + why)
                b1, b2, b3 = st.columns(3)
                for col, (lab, val) in zip((b1, b2, b3), ((L("✓ صحيح", "✓ True"), "true"), (L("✗ خاطئ", "✗ False"), "false"),
                                                          (L("تم التعامل", "Handled"), "handled"))):
                    if col.button(lab, key=f"fb_{a['class']}_{val}_{i}"):
                        st.session_state.feedback.append({"time": str(now), "scenario": scenario, "class": a["class"],
                                                          "p": round(a["p"], 3), "label": val})
                        st.toast(L("شكراً، سُجل التقييم لتدريب النموذج القادم.", "Thanks, logged for the next model."))

    st.subheader(L("احتمال كل نوع حادث خلال 12 ساعة", "Probability of each incident type within 12 h"))
    fig = go.Figure()
    w = P.iloc[-min(len(P), 72 * STEPS_PER_H):]
    tt = df["time"].iloc[-len(w):]
    for c in CLASSES:
        fig.add_trace(go.Scatter(x=tt, y=w[c], name=cl(c), mode="lines"))
    fig.add_hline(y=ALERT_THR, line_dash="dot", annotation_text=L("حد التنبيه", "alert threshold"))
    ev = df_all[(df_all["event"] != "")]
    for _, e in ev.iterrows():
        if e["time"] <= now:
            fig.add_vline(x=e["time"], line_color="red", line_dash="dash")
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    st.plotly_chart(time_axis(fig), width="stretch")
    future = df_all[(df_all["event"] != "") & (df_all["time"] > now)]
    if len(future):
        e = future.iloc[0]
        st.caption(L(f"للعرض فقط (حقيقة المحاكاة): الحادث «{e['event']}» سيقع في {e['time']:%m-%d %H:%M}، بعد {(e['step'] - i) / STEPS_PER_H:.1f} ساعة من الآن.",
                     f"Demo only (simulation ground truth): incident '{e['event']}' will happen at {e['time']:%m-%d %H:%M}, {(e['step'] - i) / STEPS_PER_H:.1f} h from now."))


def page_twin():
    st.title(L("🔁 التوأم الرقمي في وضع الظل", "🔁 Digital twin in shadow mode"))
    st.markdown(L("التوأم يستقبل نفس مدخلات الموقع ويحسب ما **يجب** أن تقرأه المستشعرات (موازنة الكتلة، منحنى المضخة، انحدار الضغط). الفرق المستمر بين المقيس والمتوقع هو أول علامة على عطل أو مستشعر كاذب.",
                  "The twin takes the same inputs as the site and computes what each sensor **should** read (mass balance, pump curve, hydraulic gradient). A persistent gap between measured and expected is the first sign of a failing asset or a lying sensor."))
    show_truth = st.toggle(L("إظهار الحقيقة المخفية (متاح في المحاكاة فقط)", "Show hidden ground truth (simulation only)"), False)
    group = st.radio(L("المجموعة", "Group"), ["tanks", "pipeline", "pump"], horizontal=True,
                     index={"leak": 1, "pump": 2}.get(scenario, 0), key=f"group_{scenario}",
                     format_func=lambda g: {"tanks": L("الخزانات", "Tanks"), "pipeline": L("خط الأنابيب", "Pipeline"), "pump": L("المضخة", "Pump")}[g])
    W = df.iloc[-min(len(df), 72 * STEPS_PER_H):].index
    t = df.loc[W, "time"]
    if group == "tanks":
        tank = st.selectbox(L("الخزان", "Tank"), ["TK-101", "TK-103", "TK-102"],
                            index=["TK-101", "TK-103", "TK-102"].index(verdict["tank"]) if verdict.get("tank") else 0)
        tag = f"LT-{tank[-3:]}"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=df.loc[W, tag], name=L("المقيس", "Measured") + f" {tag}"))
        fig.add_trace(go.Scatter(x=t, y=sh.loc[W, f"{tank}_est"], name=L("تقدير التوأم", "Twin estimate"), line=dict(dash="dash")))
        if show_truth:
            fig.add_trace(go.Scatter(x=t, y=df.loc[W, f"true_L{tank[-3:]}"], name=L("الحقيقة (محاكاة)", "Truth (sim)"), line=dict(dash="dot", color="gray")))
        fig.add_hline(y=SWITCH_LEVEL, line_dash="dot", annotation_text=L("حد التشغيل 17 م", "operating limit 17 m"))
        fig.add_hline(y=LSHH_LEVEL, line_color="red", annotation_text="LSHH 19 m")
        fig.update_yaxes(title=L("المستوى (م)", "Level (m)"))
        st.plotly_chart(time_axis(fig), width="stretch")
        fig2 = go.Figure(go.Scatter(x=t, y=sh.loc[W, f"{tank}_res"], name="residual", fill="tozeroy"))
        fig2.add_hline(y=0.15, line_dash="dot"); fig2.add_hline(y=-0.15, line_dash="dot")
        fig2.update_yaxes(title=L("الفرق (م)", "Residual (m)"))
        st.plotly_chart(time_axis(fig2, 220), width="stretch")

        st.subheader(L("المحاكاة الاستباقية للخزان المستلم (24 ساعة)", "Look-ahead for the receiving tank (24 h)"))
        la = lookahead_tank(df, sh, len(df) - 1)
        c1, c2, c3 = st.columns(3)
        c1.metric(L("الخزان المستلم", "Receiving tank"), la["tank"])
        c2.metric(L("ساعات حتى LSHH حسب المقياس", "Hours to LSHH by the gauge"), f"{la['hours_to_lshh_by_gauge']:.1f}")
        c3.metric(L("ساعات حتى LSHH حسب التوأم", "Hours to LSHH by the twin"), f"{la['hours_to_lshh']:.1f}",
                  delta=f"{la['hours_to_lshh'] - la['hours_to_lshh_by_gauge']:+.1f} h", delta_color="inverse")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=la["proj_h"], y=la["proj_meas"], name=L("إسقاط من المقياس", "Projected from gauge")))
        fig3.add_trace(go.Scatter(x=la["proj_h"], y=la["proj_est"], name=L("إسقاط من التوأم", "Projected from twin"), line=dict(dash="dash")))
        fig3.add_hline(y=LSHH_LEVEL, line_color="red")
        fig3.update_xaxes(title=L("ساعات من الآن", "hours from now")); fig3.update_yaxes(title="m")
        st.plotly_chart(time_axis(fig3, 260), width="stretch")
    elif group == "pipeline":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=df.loc[W, "FI-001"], name="FI-001 " + L("(محطة البداية)", "(upstream)")))
        fig.add_trace(go.Scatter(x=t, y=df.loc[W, "FI-002"], name="FI-002 " + L("(الوصول)", "(arrival)")))
        fig.update_yaxes(title="m³/h")
        st.plotly_chart(time_axis(fig), width="stretch")
        fig2 = go.Figure(go.Scatter(x=t, y=sh.loc[W, "line_imbalance_pct"], fill="tozeroy", name=L("اختلال الموازنة %", "Imbalance %")))
        fig2.add_trace(go.Scatter(x=t, y=X.loc[W, "line_cusum"] / 10, name="CUSUM / 10"))
        fig2.add_hline(y=1.0, line_dash="dot")
        st.plotly_chart(time_axis(fig2, 240), width="stretch")
        lp = sh["leak_pos_frac"].iloc[-STEPS_PER_H:].median()
        if p_now["leak"] >= ALERT_THR and np.isfinite(lp):
            st.error(L(f"تسرب مرجح عند {lp * 60:.1f} كم من محطة البداية (القطاع S3).", f"Likely leak at {lp * 60:.1f} km from the upstream station (segment S3)."))
            if show_truth:
                st.caption(L(f"الموقع الحقيقي في المحاكاة: {plant.LEAK_POS * 60:.1f} كم", f"True location in the simulation: {plant.LEAK_POS * 60:.1f} km"))
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=df.loc[W, "P201-VIB"], name=L("الاهتزاز المقيس", "Measured vibration")))
        fig.add_trace(go.Scatter(x=t, y=sh.loc[W, "P201-VIB_exp"], name=L("المتوقع من التوأم", "Twin expected"), line=dict(dash="dash")))
        fig.add_hline(y=VIB_ALARM, line_dash="dot", annotation_text="alarm 4.5")
        fig.add_hline(y=VIB_TRIP, line_color="red", annotation_text="trip 7.1")
        fig.update_yaxes(title="mm/s")
        st.plotly_chart(time_axis(fig), width="stretch")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=t, y=df.loc[W, "P201-PDIS"], name=L("ضغط الطرد", "Discharge pressure")))
        fig2.add_trace(go.Scatter(x=t, y=sh.loc[W, "P201-PDIS_exp"], name=L("منحنى المضخة", "Pump curve"), line=dict(dash="dash")))
        st.plotly_chart(time_axis(fig2, 240), width="stretch")
    st.subheader(L("معيار القبول: دقة التوأم ±2%", "Acceptance: twin accuracy ±2%"))
    acc = twin_accuracy(df, sh).rename(columns={"tag": "Tag", "mape_pct": "MAPE %", "err_pct_span": "err % of span", "within_2pct": "% within ±2%"})
    st.dataframe(acc.round(2), width="stretch", hide_index=True)


def page_asset():
    st.title(L("⚙️ بطاقة المعدة: مضخة التحميل P-201A", "⚙️ Asset card: loading pump P-201A"))
    run = df["PUMP-RUN"].iat[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(L("المضخة العاملة", "Running pump"), {"A": "P-201A", "B": "P-201B", "-": L("متوقفة", "stopped")}[run])
    c2.metric(L("الاهتزاز", "Vibration"), f"{df['P201-VIB'].iloc[-STEPS_PER_H:].mean():.2f} mm/s")
    rul_txt = f"{rul['rul_p50']:.0f} h" if np.isfinite(rul["rul_p50"]) else L("لا تدهور", "no degradation")
    c3.metric(L("الزمن المتبقي (P50)", "Remaining life (P50)"), rul_txt)
    c4.metric(L("احتمال الفشل خلال 72 ساعة", "P(failure within 72 h)"), f"{rul['p_fail_72h']:.0%}")
    if np.isfinite(rul["rul_p50"]):
        st.info(L(f"نطاق الزمن المتبقي: {rul['rul_p10']:.0f} إلى {rul['rul_p90']:.0f} ساعة (P10–P90). الموعد المقترح للصيانة قبل {(now + pd.Timedelta(hours=rul['rul_p10'] * 0.7)):%Y-%m-%d %H:%M}.",
                  f"Remaining-life band: {rul['rul_p10']:.0f} to {rul['rul_p90']:.0f} h (P10–P90). Suggested maintenance before {(now + pd.Timedelta(hours=rul['rul_p10'] * 0.7)):%Y-%m-%d %H:%M}."))
    H = STEPS_PER_H
    W = df.iloc[-min(len(df), 96 * H):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=W["time"], y=W["P201-VIB"].rolling(H, min_periods=1).mean(), name=L("الاهتزاز (متوسط ساعة)", "Vibration (1 h mean)")))
    if rul["fit_t"] is not None:
        ft = now + pd.to_timedelta(rul["fit_t"], unit="h")
        fig.add_trace(go.Scatter(x=ft, y=rul["fit_y"], name=L("نموذج التدهور الأسي", "Exponential degradation fit"), line=dict(dash="dash", color="#d62728")))
    fig.add_hline(y=VIB_ALARM, line_dash="dot", annotation_text="ISO 10816 alarm")
    fig.add_hline(y=VIB_TRIP, line_color="red", annotation_text="trip")
    fig.update_yaxes(title="mm/s", range=[0, 8])
    st.plotly_chart(time_axis(fig, 340), width="stretch")

    a, b = st.columns(2)
    with a:
        st.subheader(L("لماذا؟ المؤشرات مقارنة بالتشغيل السليم", "Why? Signals versus healthy operation"))
        dv = eng.deviations(X.iloc[-1], "pump_trip")
        dv["feature"] = dv["feature"].map(fl)
        fig = go.Figure(go.Bar(x=dv["z"], y=dv["feature"], orientation="h",
                               marker_color=np.where(dv["z"] > 3, "#d62728", np.where(dv["z"] > 1.5, "#f0a202", "#2ca02c"))))
        fig.add_vline(x=3, line_dash="dot")
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=L("انحراف عن الطبيعي (σ)", "deviation from normal (σ)"))
        st.plotly_chart(fig, width="stretch")
        st.caption(L("كل عمود يبين كم يبعد المؤشر عن قيمته في التشغيل السليم، مقاساً بالانحراف المعياري.",
                     "Each bar shows how far a signal sits from healthy operation, in standard deviations."))
    with b:
        st.subheader(L("نموذج Weibull من سجل CMMS", "Weibull model from CMMS history"))
        wb = reliability.pump_bearing_weibull()
        cm = reliability.load_cmms()
        last = cm[(cm["equipment"] == "P-201A") & (cm["fm_code"] == "VIB")]["date"].max()
        age = max(0.0, (now.tz_localize(None) - last).total_seconds() / 3600 * 0.7)
        hz = np.linspace(0, 24 * 60, 60)
        R = reliability.conditional_reliability(age, hz, wb["beta"], wb["eta"])
        fig = go.Figure(go.Scatter(x=hz / 24, y=R, name="R(t | age)"))
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=L("أيام", "days"), yaxis_tickformat=".1%")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"β = {wb['beta']:.2f}, η = {wb['eta']:,.0f} h, MTBF ≈ {wb['mtbf']:,.0f} h · "
                   + L(f"عمر التشغيل منذ آخر إصلاح ≈ {age:,.0f} ساعة. Weibull يعطي الخطر المسبق، والاهتزاز يحدّثه لحظياً.",
                       f"run-hours since last repair ≈ {age:,.0f}. Weibull gives the prior; vibration updates it live."))
    st.subheader(L("سجل الصيانة (CMMS)", "Maintenance history (CMMS)"))
    st.dataframe(reliability.load_cmms().sort_values("date", ascending=False), width="stretch", hide_index=True)


def page_advisor():
    st.title(L("🧪 محرك الحلول: ماذا لو؟", "🧪 Solution engine: what if?"))
    st.markdown(L("لكل تنبيه يولد رقيب 2–4 خيارات، **ويجربها على التوأم** من الحالة الحالية، ثم يرتبها حسب: هل تمنع الحادث، كمية المنتج المسكوب، والفاقد الإنتاجي. **المستوى L2 مستشار:** المشغل ينفذ يدوياً عبر DCS.",
                  "For each alert RAQIB generates 2–4 options, **tests each on the twin** from the current state, then ranks them by: does it prevent the incident, product spilled, and throughput lost. **Level L2 advisor:** the operator executes through the DCS."))
    default = next((a["class"] for a in alerts if a["class"] != "integrity"),
                   "cyber" if verdict["verdict"] == "suspected_cyber" else "overfill" if verdict["verdict"] != "consistent" else risk.iloc[0]["class"])
    cls = st.selectbox(L("المشكلة", "Problem"), list(CLASSES), index=list(CLASSES).index(default), format_func=cl)
    with st.spinner(L("تشغيل التوأم لكل خيار...", "Running the twin for each option...")):
        res = get_whatif(scenario, int(seed), lshh, standby, i, cls)
    best = res.iloc[0]
    st.success(L(f"التوصية: {best['ar']}", f"Recommendation: {best['en']}"))
    show = pd.DataFrame({
        L("الترتيب", "Rank"): res["rank"],
        L("الخيار", "Option"): res["ar"] if AR else res["en"],
        L("الحادث خلال الأفق", "Incident in horizon"): res["incident"].replace("", "—"),
        L("ساعات حتى الحادث", "Hours to incident"): res["hours_to_incident"].map(lambda v: "—" if pd.isna(v) else f"{v:.1f}"),
        L("منتج مسكوب م³", "Spilled m³"): res["spilled_m3"].round(1),
        L("فاقد إنتاجي م³", "Throughput lost m³"): res["lost_m3"].round(0),
        L("أعلى مستوى م", "Peak level m"): res["max_level"].round(2),
        L("أعلى LEL %", "Peak LEL %"): res["max_lel"].round(1),
    })
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption(L(f"أفق المحاكاة {int(res['horizon_h'].iat[0])} ساعة، يبدأ من مستويات التوأم المصححة لا من المقاييس.",
                 f"Horizon {int(res['horizon_h'].iat[0])} h, starting from the twin's reconciled levels, not the raw gauges."))
    var = {"overfill": ("true_L101", "true_L103"), "cyber": ("true_L101", "true_L103"), "leak": ("true_leak",),
           "pump_trip": ("P201-VIB",), "power_trip": ("GRID-V",), "vapor": ("GD-01",)}[cls]
    fig = go.Figure()
    for _, r in res.iterrows():
        tr = r["trace"]
        y = tr[list(var)].max(axis=1)
        fig.add_trace(go.Scatter(x=(tr["step"] - tr["step"].iat[0]) / STEPS_PER_H, y=y, name=r["ar"] if AR else r["en"]))
    if cls in ("overfill", "cyber"):
        fig.add_hline(y=LSHH_LEVEL, line_color="red", annotation_text="LSHH")
    fig.update_xaxes(title=L("ساعات من الآن", "hours from now"))
    st.plotly_chart(time_axis(fig, 320), width="stretch")

    st.subheader(L("الموافقة وأمر العمل", "Approval and work order"))
    opt_names = list(res["ar"] if AR else res["en"])
    choice = st.radio(L("الخيار المعتمد", "Approved option"), range(len(res)), format_func=lambda k: opt_names[k], horizontal=False)
    asset = risk[risk["class"] == cls].iloc[0]["asset"]
    wo = advisor.work_order(asset, cls, {"en": res.iloc[choice]["en"]}, float(p_now.get(cls, 0)), rul if cls == "pump_trip" else None, now)
    c1, c2 = st.columns(2)
    if c1.button(L("اعتماد وإنشاء مسودة أمر عمل في CMMS", "Approve and draft a CMMS work order"), type="primary"):
        st.session_state.outbox.append(wo)
        st.toast(L("أُنشئت مسودة أمر العمل. التنفيذ في الموقع يبقى بيد المشغل.", "Work-order draft created. Execution on site stays with the operator."))
    c2.download_button(L("تحميل أمر العمل JSON", "Download work order JSON"), advisor.to_json(wo), file_name=f"WO_{asset}_{now:%Y%m%d%H%M}.json")
    st.code(advisor.to_json(wo), language="json")
    if st.session_state.outbox:
        st.caption(L(f"صندوق CMMS: {len(st.session_state.outbox)} مسودة", f"CMMS outbox: {len(st.session_state.outbox)} drafts"))


def page_emergency():
    st.title(L("🚨 مركز الطوارئ: توأم العواقب", "🚨 Emergency centre: consequence twin"))
    r = df.iloc[-1]
    night = r["time"].hour < 6 or r["time"].hour >= 19
    # back-calculate the source term from the detector reading (inverse plume)
    rel = (plant.DETECTOR_BEARING - r["WIND-DIR"] + 540) % 360 - 180
    xd, yd = plant.DETECTOR_DIST * np.cos(np.radians(rel)), plant.DETECTOR_DIST * np.sin(np.radians(rel))
    unit = plant.plume_conc(1.0, xd, yd, r["WIND-SPD"], night) if xd > 0 else 0
    q_est = (r["GD-01"] / 100 * plant.LFL_G_M3 / unit) if unit > 0 and r["GD-01"] > 3 else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GD-01", f"{r['GD-01']:.1f} % LEL")
    c2.metric(L("الرياح", "Wind"), f"{r['WIND-SPD']:.1f} m/s → {r['WIND-DIR']:.0f}°")
    c3.metric(L("الاستقرار الجوي", "Stability"), "F" if night and r["WIND-SPD"] < 3 else "D")
    c4.metric(L("معدل الانبعاث المقدر", "Estimated release"), f"{q_est:.0f} g/s")
    q = st.slider(L("معدل انبعاث للمحاكاة (غ/ث)، القيمة الافتراضية من الكاشف", "Release rate to simulate (g/s), default from the detector"),
                  0, 500, int(min(500, max(q_est, 50 if q_est == 0 else q_est))))
    src = (330.0, 90.0)
    xs, ys = np.linspace(-30, 400, 150), np.linspace(-20, 310, 120)
    th = np.radians(r["WIND-DIR"])
    gx, gy = np.meshgrid(xs - src[0], ys - src[1])
    xdn, ycr = gx * np.sin(th) + gy * np.cos(th), gx * np.cos(th) - gy * np.sin(th)
    Z = np.vectorize(lambda a, b: 100 * plant.plume_conc(q, a, b, r["WIND-SPD"], night) / plant.LFL_G_M3 if a > 0 else 0.0)(xdn, ycr)
    det = (src[0] + plant.DETECTOR_DIST * np.sin(np.radians(plant.DETECTOR_BEARING)),
           src[1] + plant.DETECTOR_DIST * np.cos(np.radians(plant.DETECTOR_BEARING)))
    up = (src[0] - 90 * np.sin(th), src[1] - 90 * np.cos(th))
    cesium_map(500, overlay=geo.lel_overlay(xs, ys, Z) if q > 0 else None,
               arrows=[{"x0": src[0] - 15 * np.sin(th), "y0": src[1] - 15 * np.cos(th), "x1": up[0], "y1": up[1],
                        "color": "#2ca02c", "label": L("الإخلاء عكس الريح", "Evacuate upwind")}],
               markers=[{"x": det[0], "y": det[1], "label": "GD-01", "color": "#1f77b4"}])
    st.caption(L("الألوان: تركيز الأبخرة % LEL (أصفر 5% حتى أحمر داكن فوق 100%).", "Colours: vapour concentration % LEL (yellow 5% to dark red above 100%)."))
    zone = (Z >= 100).sum() * (xs[1] - xs[0]) * (ys[1] - ys[0])
    st.caption(L(f"مساحة المنطقة القابلة للاشتعال (≥100% LEL): {zone:,.0f} م². نموذج غاوسي مبسط (Briggs) للتوضيح؛ في المشروع الحقيقي PHAST أو CFD.",
                 f"Flammable area (≥100% LEL): {zone:,.0f} m². Simplified Gaussian (Briggs) model for illustration; the real project uses PHAST or CFD."))
    st.subheader(L("قائمة مهام الاستجابة", "Response task list"))
    for t in [L("إيقاف التحميل وتفعيل ESD للرصيف LB-01", "Stop loading and trip ESD on bay LB-01"),
              L("إبعاد السائقين وإيقاف محركات الصهاريج", "Move drivers away and stop truck engines"),
              L("إزالة مصادر الإشعال في اتجاه الريح", "Remove ignition sources downwind"),
              L("الإخلاء إلى نقطة التجمع عكس الريح", "Evacuate to the upwind muster point"),
              L("تجهيز الرغوة وتبريد الخزانات المجاورة", "Ready foam and cool adjacent tanks"),
              L("إبلاغ الدفاع المدني ومركز التحكم", "Notify civil defence and control centre")]:
        st.checkbox(t, key=f"task_{t}")


def page_fmea():
    st.title(L("📋 FMEA والموثوقية", "📋 FMEA & reliability"))
    st.markdown(L("جدول FMEA بترميز ISO 14224 لأنماط الفشل. عدّل S وO وD ليُعاد حساب RPN. «D بعد» هو تقييم الاكتشاف بعد تشغيل كاشف رقيب المذكور في الصف.",
                  "FMEA with ISO 14224 failure-mode codes. Edit S, O and D to recompute RPN. 'D after' is the detection rating once the RAQIB detector on the row is live."))
    cols = ["id", "asset_ar" if AR else "asset", "fm_code", "failure_mode_ar" if AR else "failure_mode",
            "effect_ar" if AR else "effect", "S", "O", "D_before", "D_after", "raqib_detector", "action_ar" if AR else "action"]
    ed = st.data_editor(fmea[cols], width="stretch", hide_index=True, disabled=[c for c in cols if c not in ("S", "O", "D_before", "D_after")],
                        key="fmea_editor")
    f2 = fmea.copy()
    for c in ("S", "O", "D_before", "D_after"):
        f2[c] = ed[c]
    f2 = reliability.recompute_rpn(f2)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=f2["id"], y=f2["RPN_before"], name=L("RPN قبل رقيب", "RPN before RAQIB")))
    fig.add_trace(go.Bar(x=f2["id"], y=f2["RPN_after"], name=L("RPN بعد رقيب", "RPN with RAQIB")))
    fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.metric(L("خفض إجمالي RPN", "Total RPN reduction"), f"{100 * (1 - f2['RPN_after'].sum() / f2['RPN_before'].sum()):.0f}%")
    st.download_button(L("تحميل FMEA CSV", "Download FMEA CSV"), f2.to_csv(index=False).encode("utf-8-sig"), "raqib_fmea.csv")


def page_mlops():
    st.title(L("🧠 النماذج وضبط الجودة (MLOps)", "🧠 Models & MLOps"))
    info = eng.train_info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(L("نسخة النموذج", "Model version"), info["version"])
    c2.metric(L("صفوف التدريب", "Training rows"), f"{info['rows']:,}")
    c3.metric(L("أفق التنبؤ", "Prediction horizon"), f"{info['horizon_h']} h")
    c4.metric(L("زمن التدريب", "Training time"), f"{info['train_seconds']} s")
    st.markdown(L("**المحرك هجين:** التوأم يحسب الفروقات الفيزيائية، ثم نموذج Gradient Boosting مستقل لكل نمط فشل يرى فقط المؤشرات السببية لذلك النمط، مع قيود رتابة (Monotonic) حتى لا يتعلم مصادفات الطقس أو الجدول. بجانبه Isolation Forest لكشف أي شذوذ غير معروف، ونموذج تدهور أسي للزمن المتبقي.",
                  "**Hybrid engine:** the twin computes physics residuals; then one gradient-boosted model per failure mode sees only that mode's causal signals, with monotonic constraints so it cannot learn weather or schedule coincidences. An Isolation Forest flags unknown anomalies, and an exponential degradation model gives remaining life."))
    st.dataframe(pd.DataFrame([{L("نمط الفشل", "Failure mode"): cl(c), L("المؤشرات (اتجاه الرتابة)", "Signals (monotonic direction)"):
                                 ", ".join(f"{fl(f)} ({'+' if d > 0 else '−'})" for f, d in m.items())} for c, m in CLASS_FEATURES.items()]),
                 width="stretch", hide_index=True)
    st.subheader(L("اختبارات القبول على سيناريوهات لم يرها النموذج", "Acceptance tests on unseen simulation runs"))
    acc = info["evaluation"]["acceptance"].copy()
    passed = acc["passed"].mean()
    st.metric(L("نسبة النجاح", "Pass rate"), f"{passed:.0%}", help=L("بذور محاكاة منفصلة عن التدريب", "Simulation seeds held out from training"))
    acc["scenario"] = acc["scenario"].map(lambda s: L(*SCENARIO_LABEL[s]))
    st.dataframe(acc.rename(columns={"lead_h": L("الإنذار المبكر (ساعة قبل الحادث)", "Lead time (h before incident)"),
                                     "detect_after_onset_h": L("زمن الاكتشاف بعد بدء العطل (ساعة)", "Detect after onset (h)")}).round(2),
                 width="stretch", hide_index=True)
    st.caption(L("التسرب والاختراق ليس لهما إشارات مسبقة، فيُقاس زمن الاكتشاف بعد البدء. فشل المضخة يُنذر به قبل أيام عبر الزمن المتبقي (RUL)، والجدول يقيس فقط مصنف الـ 12 ساعة.",
                 "A leak or an intrusion has no precursor, so time-to-detect after onset is measured. Pump failure is flagged days ahead by RUL; the table only scores the 12 h classifier."))
    pc = info["evaluation"]["per_class"].copy()
    pc.index = [cl(c) for c in pc.index]
    st.dataframe(pc.round(3), width="stretch")
    st.caption(L("هذه مقاييس على بيانات محاكاة، وليست وعداً بأداء في موقع حقيقي. أهداف المشروع: Recall ≥ 70% وPrecision ≥ 80%.",
                 "These are metrics on simulated data, not a promise of field performance. Project targets: recall ≥ 70%, precision ≥ 80%."))
    st.subheader(L("مراقبة انجراف البيانات (PSI)", "Data drift monitoring (PSI)"))
    cur = X.iloc[-min(len(X), 24 * STEPS_PER_H):]
    d = pd.DataFrame([{"feature": fl(f), "PSI": psi(eng.feature_ref[f].to_numpy(), cur[f].to_numpy())} for f in X.columns])
    d["status"] = np.where(d["PSI"] > 0.25, L("انجراف", "drift"), np.where(d["PSI"] > 0.1, L("مراقبة", "watch"), "OK"))
    st.dataframe(d.sort_values("PSI", ascending=False).round(3), width="stretch", hide_index=True)
    st.subheader(L("درجة الشذوذ (Isolation Forest)", "Anomaly score (Isolation Forest)"))
    fig = go.Figure(go.Scatter(x=df["time"], y=P["anomaly"], fill="tozeroy"))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    st.plotly_chart(time_axis(fig, 220), width="stretch")
    st.markdown(L("**سياسة التشغيل:** 3 أشهر في وضع الظل قبل إرسال أي إنذار، إعادة تدريب شهرية مع توثيق كل نسخة، وتقييم المشغل لكل إنذار يدخل في التدريب التالي.",
                  "**Operating policy:** 3 months in shadow mode before any alert reaches operators, monthly retraining with every version recorded, and every operator verdict feeds the next training round."))


def page_security():
    st.title(L("🔐 تكامل OT والأمن السيبراني", "🔐 OT integration & cybersecurity"))
    st.markdown(L("رقيب **يقرأ فقط** من أنظمة التحكم عبر بوابة أحادية الاتجاه، والكتابة الوحيدة خارج النظام هي مسودة أمر عمل في CMMS. خط أحمر: لا يلمس نظام السلامة SIS.",
                  "RAQIB **only reads** from control systems through a one-way gateway; its only write is a work-order draft in CMMS. Red line: it never touches the SIS."))
    st.graphviz_chart(security.PURDUE_DOT, width="stretch")
    st.subheader(L("الموصلات", "Connectors"))
    st.dataframe(security.CONNECTORS, width="stretch", hide_index=True)
    st.subheader(L("جودة البيانات الآن", "Data quality now"))
    dq = security.data_quality(df_all, sh_all, i)
    st.dataframe(dq.round(3), width="stretch", hide_index=True)
    v = verdict["verdict"]
    msg = {"consistent": (L("كل القراءات متسقة مع فيزياء التوأم.", "All readings agree with the twin physics."), "success"),
           "instrument_failure": (L(f"عطل مستشعر مرجح في {verdict['tank']}: القراءة ثابتة بلا تذبذب ولا تتفق مع موازنة الكتلة.", f"Likely instrument failure on {verdict['tank']}: flat reading that disagrees with the mass balance."), "warning"),
           "suspected_cyber": (L(f"تلاعب مشتبه به في {verdict['tank']}: القراءة تبدو طبيعية (فيها تذبذب) لكنها تخالف الفيزياء، مع {verdict.get('modbus_writes_1h', 0)} أمر كتابة Modbus غير معروف في الساعة الأخيرة.", f"Suspected manipulation on {verdict['tank']}: the reading looks normal (it has noise) but breaks the physics, with {verdict.get('modbus_writes_1h', 0)} unknown Modbus writes in the last hour."), "error"),
           "physics_mismatch": (L(f"قراءة {verdict['tank']} لا تتفق مع الفيزياء.", f"{verdict['tank']} disagrees with the physics."), "warning")}[v]
    getattr(st, msg[1])(msg[0])
    st.subheader(L("اختبار البوابة أحادية الاتجاه", "One-way gateway test"))
    if st.button(L("محاولة كتابة نقطة ضبط على DCS", "Try to write a set-point to the DCS")):
        try:
            st.session_state.gateway.write("FIC-002.SP", 0.0)
        except PermissionError as e:
            st.error(f"🚫 {e}")
    st.session_state.gateway.read(df_all, "LT-101", i)
    st.dataframe(pd.DataFrame(st.session_state.gateway.audit[-8:], columns=["op", "tag", "detail"]), width="stretch", hide_index=True)
    st.subheader("IEC 62443")
    st.dataframe(security.IEC62443, width="stretch", hide_index=True)


def page_assistant():
    st.title(L("💬 المساعد المحادث", "💬 Assistant"))
    st.caption(L("يجيب من البيانات الحية ومن الإجراءات مع ذكر المصدر. لا يستخدم نموذجاً لغوياً سحابياً؛ يمكن إضافة نموذج مفتوح محلياً.",
                 "Answers from live data and procedures, citing the source. No cloud LLM; an on-prem open-weights model can be added."))
    ex = [L("لماذا ارتفع اهتزاز المضخة P-201A؟", "Why is P-201A vibration rising?"),
          L("هل مستوى الخزان TK-101 صحيح؟", "Is the TK-101 level reading right?"),
          L("ماذا أفعل عند اشتباه تسرب في الخط؟", "What do I do if a pipeline leak is suspected?"),
          L("هل هناك هجوم سيبراني على الشبكة؟", "Is there a cyber attack on the network?")]
    cols = st.columns(len(ex))
    q = None
    for c, e in zip(cols, ex):
        if c.button(e, width="stretch"):
            q = e
    q = st.chat_input(L("اكتب سؤالك...", "Ask a question...")) or q
    if q:
        with st.chat_message("user"):
            st.write(q)
        ans = get_assistant().answer(q, df, sh, p_now, rul, AR)
        with st.chat_message("assistant"):
            if ans["facts"]:
                st.markdown("\n".join(f"- {f}" for f in ans["facts"]))
            for s in ans["sources"]:
                with st.expander(f"📄 {s['doc']} › {s['section']}"):
                    st.write(s["text"])
            if not ans["facts"] and not ans["sources"]:
                st.write(L("لم أجد إجابة في البيانات أو الإجراءات.", "I found nothing in the data or procedures."))


def page_voice():
    st.title(L("🗣️ صوت المشغل وقبول النظام", "🗣️ Operator voice & acceptance"))
    st.markdown(L("أكبر مخاطر المشروع ثقة المشغلين. هنا يقيّم المشغل كل تنبيه، ونقيس عبء الإنذارات مقابل ISA-18.2.",
                  "The biggest project risk is operator trust. Here operators rate every alert, and alarm load is measured against ISA-18.2."))
    conv = pd.DataFrame({
        "H": (df[["LT-101", "LT-103"]].max(axis=1) > SWITCH_LEVEL + 0.3),
        "VIB": df["P201-VIB"] > VIB_ALARM, "LEL": df["GD-01"] > 20, "V": df["GRID-V"] < 0.9,
        "TBRG": df["P201-TBRG"] > 75, "PT": (df["PT-002"] < 3.0) & (df["FI-001"] > 50),
    })
    conv_rate = (conv.astype(int).diff().clip(lower=0).sum(axis=1)).rolling(STEPS_PER_H).sum()
    raq = pd.DataFrame({c: (P[c] >= ALERT_THR).astype(int).diff().clip(lower=0) for c in CLASSES}).sum(axis=1).rolling(STEPS_PER_H).sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=conv_rate, name=L("إنذارات تقليدية/ساعة", "Conventional alarms / h")))
    fig.add_trace(go.Scatter(x=df["time"], y=raq, name=L("تنبيهات رقيب/ساعة", "RAQIB alerts / h")))
    fig.add_hline(y=6, line_dash="dot", annotation_text="ISA-18.2 ≤ 6/h")
    st.plotly_chart(time_axis(fig, 280), width="stretch")
    st.subheader(L("تقييمات التنبيهات", "Alert verdicts"))
    fb = pd.DataFrame(st.session_state.feedback)
    if len(fb):
        st.dataframe(fb, width="stretch", hide_index=True)
        st.download_button(L("تحميل التقييمات CSV", "Download verdicts CSV"), fb.to_csv(index=False).encode("utf-8-sig"), "raqib_feedback.csv")
    else:
        st.info(L("لا توجد تقييمات بعد. قيّم التنبيهات من غرفة القيادة.", "No verdicts yet. Rate alerts in the control room."))
    st.subheader(L("استبيان القبول", "Acceptance survey"))
    a = st.slider(L("أثق في تنبيهات رقيب", "I trust RAQIB alerts"), 1, 5, 3)
    b = st.slider(L("التفسير واضح ويساعدني على القرار", "The explanation is clear and helps me decide"), 1, 5, 3)
    c = st.slider(L("الواجهة العربية مناسبة لغرفة التحكم", "The Arabic interface suits the control room"), 1, 5, 4)
    st.metric(L("مؤشر القبول", "Acceptance index"), f"{(a + b + c) / 15:.0%}")
    st.markdown(L("**معايير القبول من المشغل:** كل تنبيه يحمل سبباً وإجراءً، لا يزيد عبء الإنذارات، يعمل بالعربية، ولا يتحكم في أي شيء دون موافقة.",
                  "**Operator acceptance criteria:** every alert carries a reason and an action, alarm load does not grow, it works in Arabic, and it controls nothing without approval."))


def page_about():
    st.title(L("👥 الفريق والمعمارية", "👥 Team & architecture"))
    st.markdown(L("نموذج أولي مفتوح المصدر لمنصة **رقيب**: توأم رقمي يتنبأ بالحوادث قبل وقوعها ويقترح حلاً مختبراً على المحاكاة. بُني بأدوار فريق المرحلة الأولى:",
                  "Open-source prototype of **RAQIB**: a digital twin that predicts incidents before they happen and proposes a fix tested on the simulation. Built across the phase-1 team roles:"))
    roles = pd.DataFrame([
        (L("مهندس عمليات / محاكاة", "Process / simulation engineer"), "raqib/plant.py, raqib/twin.py", L("محاكي المحطة، التوأم في وضع الظل، المعايرة، المحاكاة الاستباقية وماذا لو", "Terminal simulator, shadow twin, calibration, look-ahead and what-if")),
        (L("مهندس موثوقية وصيانة", "Reliability & maintenance engineer"), "raqib/reliability.py, data/fmea.csv, data/cmms_history.csv", L("FMEA بترميز ISO 14224، Weibull، أوامر عمل CMMS", "ISO 14224 FMEA, Weibull, CMMS work orders")),
        (L("عالم بيانات / مهندس ML", "Data scientist / ML engineer"), "raqib/models.py", L("نماذج لكل نمط فشل، Isolation Forest، RUL، التفسير، PSI، اختبارات القبول", "Per-mode models, Isolation Forest, RUL, explanations, PSI, acceptance tests")),
        (L("مهندس بيانات وتكامل OT", "OT data & integration engineer"), "raqib/security.py", L("الموصلات، جودة البيانات، نموذج الأصول والوسوم", "Connectors, data quality, asset and tag model")),
        (L("أخصائي أمن سيبراني OT", "OT cybersecurity specialist"), "raqib/security.py", L("بوابة أحادية الاتجاه، IEC 62443، كشف التلاعب بالفيزياء", "One-way gateway, IEC 62443, physics-based tamper detection")),
        (L("مطور واجهات", "UI developer"), "app.py", L("11 شاشة ثنائية اللغة مع دعم RTL", "11 bilingual screens with RTL support")),
        (L("ممثل العمليات (مشغل أول)", "Operations representative (senior operator)"), L("شاشة صوت المشغل", "Operator voice screen"), L("تقييم التنبيهات، عبء الإنذارات، معايير القبول", "Alert verdicts, alarm load, acceptance criteria")),
    ], columns=[L("الدور", "Role"), L("الملفات", "Files"), L("ما بناه", "What it built")])
    st.dataframe(roles, width="stretch", hide_index=True)
    st.graphviz_chart(security.PURDUE_DOT, width="stretch")
    st.markdown(L("**تنبيه:** كل البيانات اصطناعية من محاكي، والإجراءات المرفقة توضيحية. هذا عرض لمفهوم، وليس نظاماً معتمداً للسلامة.",
                  "**Note:** all data is synthetic from a simulator and the bundled procedures are illustrative. This is a concept demo, not a certified safety system."))
    st.markdown("Python · Streamlit · scikit-learn · Plotly · NumPy · pandas — MIT License")


{"control": page_control, "twin": page_twin, "asset": page_asset, "advisor": page_advisor,
 "emergency": page_emergency, "fmea": page_fmea, "mlops": page_mlops, "security": page_security,
 "assistant": page_assistant, "voice": page_voice, "about": page_about}[page]()
