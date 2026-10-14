"""Operator-screen look and feel, following ISA-101 high-performance HMI practice.

Grey background, dark-grey equipment and text, values in a fixed-width face,
and colour kept for abnormal states only: red for priority 1, amber for
priority 2. Every alarm also carries a shape and a number, so it reads
without colour. Unacknowledged alarms blink until an operator rates them.

Owner role: UI developer, with the operations representative.
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio

GREY_BG = "#d4d4d4"        # display background
PANEL = "#e4e4e4"          # faceplates and panels
EDGE = "#8c8c8c"           # panel borders
EQUIP = "#4d4d4d"          # equipment outlines and normal text on graphics
FILL = "#9fb3c8"           # liquid in tanks: muted, not an alarm colour
P1 = "#e00000"             # priority 1 alarm
P2 = "#f2a900"             # priority 2 alarm
AR_FONT = 'font-family="Tajawal, Segoe UI, sans-serif"'   # labels that may be Arabic; values stay monospace
TREND = ["#1f3a68", "#5a5a5a", "#6b4c9a", "#2b7a78", "#8a6d1f", "#3f5f2f", "#9c4f2c"]

# alarm class -> the asset whose graphic carries the alarm indicator
CLASS_ASSET = {"leak": "PL-S3", "pump_trip": "P-201A", "power_trip": "EL-33KV", "vapor": "LB-01", "cyber": "OT-NET"}


def priority(alert: dict) -> int:
    """ISA-18.2 style: 1 when the operator must act now, 2 when there is time to plan (pump wear, grid stress)."""
    if alert["class"] in ("integrity", "vapor", "leak", "cyber"):
        return 1
    return 1 if alert["class"] == "overfill" and alert["p"] >= 0.8 else 2


def register_plotly_template() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor=PANEL, plot_bgcolor="#f2f2f2", colorway=TREND,
        font=dict(family="Roboto Mono, Consolas, monospace", size=12, color="#1a1a1a"),
        xaxis=dict(gridcolor="#cfcfcf", linecolor=EDGE, zeroline=False, showline=True, mirror=True),
        yaxis=dict(gridcolor="#cfcfcf", linecolor=EDGE, zeroline=False, showline=True, mirror=True),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    pio.templates["hmi"] = t
    pio.templates.default = "plotly_white+hmi"


def css(ar: bool) -> str:
    rtl = """.stMarkdown, .stAlert, [data-testid="stMetric"], [data-testid="stCaptionContainer"], h1, h2, h3, h4, label, .stRadio, .stCheckbox, [data-testid="stExpander"] summary {direction: rtl; text-align: right;}
section[data-testid="stSidebar"] {direction: rtl;}""" if ar else ""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&family=Roboto+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, label, button, input, textarea {{ font-family: 'Tajawal', 'Segoe UI', Arial, sans-serif; }}
{rtl}
.stApp {{ background: {GREY_BG}; }}
[data-testid="stHeader"] {{ background: {GREY_BG}; }}
.block-container {{ padding-top: 3.8rem; }}
section[data-testid="stSidebar"] {{ background: #bdbdbd; border-inline-end: 1px solid {EDGE}; }}
section[data-testid="stSidebar"] h2 {{ font-size: 1.05rem; letter-spacing: .04em; }}
/* display selection reads like the display buttons on a DCS console */
section[data-testid="stSidebar"] [role="radiogroup"] > label {{ background: {PANEL}; border: 1px solid {EDGE}; margin: 0 0 3px 0;
    padding: 3px 8px; border-radius: 0; width: 100%; }}
section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{ background: #4d4d4d; color: #fff; }}
section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {{ color: #fff; }}
h1, h1 span {{ color: #fff !important; }}
h1 {{ font-size: 1.25rem !important; font-weight: 700; background: #4d4d4d; padding: .35rem .7rem !important;
      border-radius: 0; letter-spacing: .03em; }}
h2, h3 {{ font-size: 1.0rem !important; font-weight: 700; color: #262626; border-bottom: 2px solid {EDGE}; padding-bottom: 2px !important;
          text-transform: uppercase; letter-spacing: .04em; }}
[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {EDGE}; padding: 6px 10px; border-radius: 0; }}
[data-testid="stMetricLabel"] p {{ font-size: .8rem; color: #333; text-transform: uppercase; }}
[data-testid="stMetricValue"] {{ font-family: 'Roboto Mono', Consolas, monospace; font-size: 1.55rem; color: #111; }}
[data-testid="stMetricValue"] div {{ direction: ltr; unicode-bidi: isolate; }}   /* keeps '3.42 mm/s' in order on Arabic screens */
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] details, .stPlotlyChart, iframe {{ border-radius: 0 !important; }}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{ background: {PANEL}; border-color: {EDGE} !important; }}
.stButton button, .stDownloadButton button {{ border-radius: 0; background: #e9e9e9; border: 1px solid #6e6e6e; color: #111; }}
.stButton button:hover {{ background: #fff; border-color: #333; color: #000; }}
[data-testid="stAlert"] > div {{ border-radius: 0; }}
[data-testid="stDataFrame"] {{ border: 1px solid {EDGE}; }}
.stTabs [data-baseweb="tab"] {{ background: {PANEL}; border: 1px solid {EDGE}; border-bottom: none; padding: 4px 14px; margin-inline-end: 3px; }}
.stTabs [aria-selected="true"] {{ background: #4d4d4d; color: #fff !important; }}
.stTabs [aria-selected="true"] p {{ color: #fff; }}
.raqib-badge {{display:inline-block;padding:1px 8px;font-weight:700;font-size:.8rem;border:1px solid #555;font-family:'Roboto Mono',monospace}}
.b-high {{background:{P1};color:#fff}} .b-medium {{background:{P2};color:#000}} .b-low {{background:{PANEL};color:#333}}
/* header strip and alarm summary */
.hmi-top {{ display:flex; flex-wrap:wrap; gap:6px 16px; align-items:center; background:#2f2f2f; color:#eee; padding:5px 10px;
            font: 14px 'Tajawal', 'Segoe UI', sans-serif; direction:ltr; }}
.m {{ font-family:'Roboto Mono', Consolas, monospace; }}
.hmi-top b {{ color:#fff; font-family:'Tajawal',sans-serif; font-size:15px; }}
.hmi-chip {{ border:1px solid #9a9a9a; padding:0 6px; }}
.hmi-chip.warn {{ background:{P2}; color:#000; border-color:{P2}; }}
.hmi-alm {{ width:100%; border-collapse:collapse; font:14px 'Tajawal', 'Segoe UI', sans-serif; background:#1e1e1e; color:#e6e6e6;
            margin-bottom:10px; direction:ltr; }}
.hmi-alm td {{ padding:3px 8px; border-bottom:1px solid #3a3a3a; white-space:nowrap; }}
.hmi-alm td.d {{ white-space:normal; width:100%; }}
.hmi-alm td.m {{ font-size:13px; }}
.hmi-alm .none {{ color:#9a9a9a; }}
.pri {{ display:inline-block; width:22px; height:18px; line-height:18px; text-align:center; font-weight:700; }}
.pri.p1 {{ background:{P1}; color:#fff; }}
.pri.p2 {{ background:{P2}; color:#000; clip-path: polygon(50% 0, 100% 100%, 0 100%); width:24px; }}
.pri.unack {{ animation: hmi-blink 1s steps(2, start) infinite; }}
@keyframes hmi-blink {{ to {{ opacity: .35; }} }}
.hmi-svg {{ background:{PANEL}; border:1px solid {EDGE}; direction:ltr; }}
</style>
"""


def header(title: str, site: str, now, scenario: str, shadow: bool, ar: bool) -> str:
    chips = [("ADVISORY · READ-ONLY" if not ar else "استشاري · قراءة فقط", ""),
             ("SIMULATED DATA" if not ar else "بيانات محاكاة", "")]
    if shadow:
        chips.append(("SHADOW MODE" if not ar else "وضع الظل", "warn"))
    c = "".join(f'<span class="hmi-chip {k}">{html.escape(t)}</span>' for t, k in chips)
    return (f'<div class="hmi-top"><b>RAQIB · رقيب</b><span>{html.escape(site)}</span>'
            f'<span>{html.escape(title)}</span><span>{html.escape(scenario)}</span>'
            f'<span class="m" style="margin-left:auto">{now:%Y-%m-%d %H:%M}</span>{c}</div>')


def alarm_summary(rows: list[dict], ar: bool, max_rows: int = 4) -> str:
    """rows: priority, start (Timestamp), tag, text, value, acked."""
    hdr = ("الأولوية", "البداية", "المعدة", "الوصف", "القيمة", "الحالة") if ar else ("PRI", "START", "TAG", "DESCRIPTION", "VALUE", "STATE")
    head = "<tr style='background:#333;color:#bbb'>" + "".join(f"<td>{h}</td>" for h in hdr) + "</tr>"
    if not rows:
        none = "لا توجد إنذارات نشطة" if ar else "NO ACTIVE ALARMS"
        return f'<table class="hmi-alm">{head}<tr><td colspan="6" class="none">{none}</td></tr></table>'
    body = []
    for r in sorted(rows, key=lambda r: (r["priority"], r["start"]))[:max_rows]:
        k = f"p{r['priority']}" + ("" if r["acked"] else " unack")
        state = ("مُقرّ" if ar else "ACK") if r["acked"] else ("غير مُقرّ" if ar else "UNACK")
        body.append(f'<tr><td><span class="pri {k}">{r["priority"]}</span></td><td class="m">{r["start"]:%m-%d %H:%M}</td>'
                    f'<td class="m">{html.escape(r["tag"])}</td><td class="d" dir="{"rtl" if ar else "ltr"}">{html.escape(r["text"])}</td>'
                    f'<td class="m">{html.escape(r["value"])}</td><td>{state}</td></tr>')
    more = len(rows) - max_rows
    if more > 0:
        body.append(f'<tr><td colspan="6" class="none">+{more} {"إنذارات أخرى" if ar else "more"}</td></tr>')
    return f'<table class="hmi-alm">{head}{"".join(body)}</table>'


# ------------------------------------------------------------------ process overview graphic
def _ind(x: float, y: float, pri: int) -> str:
    """Alarm indicator: number inside a shape, so priority reads without colour."""
    if pri == 1:
        shape = f'<rect x="{x - 11}" y="{y - 11}" width="22" height="22" fill="{P1}" stroke="#000"/>'
        col = "#fff"
    else:
        shape = f'<polygon points="{x},{y - 13} {x + 13},{y + 10} {x - 13},{y + 10}" fill="{P2}" stroke="#000"/>'
        col = "#000"
    return shape + f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-weight="700" font-size="13" fill="{col}">{pri}</text>'


def _val(x, y, tag, value, unit, alarm=False, anchor="start"):
    fill = P1 if alarm else "#111"
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="11" fill="#555" {AR_FONT}>{tag}</text>'
            f'<text x="{x}" y="{y + 15}" text-anchor="{anchor}" font-size="14" font-weight="600" fill="{fill}">{value}'
            f'<tspan font-size="11" font-weight="400" fill="#555"> {unit}</tspan></text>')


def _valve(x, y, open_: bool):
    f = EQUIP if open_ else "#ffffff"
    return (f'<polygon points="{x - 9},{y - 6} {x + 9},{y + 6} {x + 9},{y - 6} {x - 9},{y + 6}" fill="{f}" stroke="{EQUIP}" stroke-width="1.5"/>'
            f'<text x="{x + 13}" y="{y + 4}" font-size="10" fill="#555">{"OPEN" if open_ else "CLSD"}</text>')


def overview_svg(r: dict, est: dict, alarms: dict[str, int], labels: dict[str, str],
                 lshh: float = 19.0, height_m: float = 20.0) -> str:
    """Simplified P&ID of the terminal with live values.

    r:       latest row of measured tags
    est:     tank -> twin-estimated level (m)
    alarms:  asset -> alarm priority (1 or 2) for assets in alarm
    labels:  a few translated captions (refinery, loading, gasoline, diesel, est, recv)
    """
    s = []
    line = f'stroke="{EQUIP}" stroke-width="4" fill="none"'
    # pipeline from the refinery to the manifold
    pl = alarms.get("PL-S3")
    s.append(f'<path d="M20,96 H262" stroke="{P1 if pl == 1 else P2 if pl else EQUIP}" stroke-width="{6 if pl else 4}" fill="none"/>')
    s.append(f'<polygon points="262,88 276,96 262,104" fill="{EQUIP}"/>')
    s.append(f'<text x="20" y="70" font-size="13" fill="#333" {AR_FONT}>{labels["refinery"]}</text>')
    s.append(_val(24, 116, "FI-001", f'{r["FI-001"]:.0f}', "m³/h") + _val(24, 152, "PT-001", f'{r["PT-001"]:.1f}', "bar"))
    s.append(_val(176, 116, "FI-002", f'{r["FI-002"]:.0f}', "m³/h") + _val(176, 152, "PT-002", f'{r["PT-002"]:.1f}', "bar"))
    s.append('<text x="250" y="88" text-anchor="end" font-size="11" fill="#555">PL-S3 · 60 km</text>')
    if pl:
        s.append(_ind(140, 72, pl))
    # inlet header and tanks
    s.append(f'<path d="M276,96 H760" {line}/>')
    tanks = [("TK-101", 330, "LT-101"), ("TK-103", 500, "LT-103"), ("TK-102", 670, "LT-102")]
    recv = r["TAS-RECV"]
    for tk, x, lt in tanks:
        w, top, bot = 110, 150, 330
        h = bot - top
        diesel = tk == "TK-102"
        open_ = (r["FT-102IN"] > 1) if diesel else (recv == tk)
        s.append(f'<path d="M{x + w / 2},96 V{top}" {line}/>')
        s.append(_valve(x + w / 2, 124, open_))
        g, e = float(r[lt]), float(est.get(tk, r[lt]))
        lv = max(0.0, min(height_m, e))
        fy = bot - h * lv / height_m
        a = alarms.get(tk)
        stroke = P1 if a == 1 else P2 if a else EQUIP
        s.append(f'<rect x="{x}" y="{fy:.1f}" width="{w}" height="{bot - fy:.1f}" fill="{FILL}"/>')
        s.append(f'<rect x="{x}" y="{top}" width="{w}" height="{h}" fill="none" stroke="{stroke}" stroke-width="{4 if a else 2}"/>')
        ly = bot - h * lshh / height_m
        s.append(f'<line x1="{x - 6}" y1="{ly:.1f}" x2="{x + w + 6}" y2="{ly:.1f}" stroke="#333" stroke-dasharray="4 3"/>'
                 f'<text x="{x + w + 8}" y="{ly + 4:.1f}" font-size="10" fill="#333">LSHH</text>')
        gy = bot - h * max(0.0, min(height_m, g)) / height_m
        s.append(f'<polygon points="{x - 2},{gy:.1f} {x - 12},{gy - 6:.1f} {x - 12},{gy + 6:.1f}" fill="#111"/>')
        s.append(f'<text x="{x + w / 2}" y="{top + 18}" text-anchor="middle" font-size="13" font-weight="700" fill="#222">{tk}</text>')
        s.append(f'<text x="{x + w / 2}" y="{top + 34}" text-anchor="middle" font-size="12" fill="#444" {AR_FONT}>{labels["diesel" if diesel else "gasoline"]}'
                 + (f' · {labels["recv"]}' if open_ else "") + "</text>")
        diff = abs(g - e) > 0.15
        s.append(_val(x + 4, bot + 18, lt, f"{g:.2f}", "m", alarm=diff and bool(a)))
        s.append(_val(x + 4, bot + 52, labels["est"], f"{e:.2f}", "m"))
        if a:
            s.append(_ind(x + w + 16, top - 12, a))
        # outlet to the loading header (gasoline tanks only)
        if not diesel:
            s.append(f'<path d="M{x + w - 12},{bot} V{bot + 72}" {line}/>')
    s.append('<g transform="translate(0,22)">')   # loading section sits below the tank readouts
    s.append(f'<path d="M428,380 H800" {line}/>')
    # loading pumps
    run = r["PUMP-RUN"]
    for name, cy, key in (("P-201A", 380, "A"), ("P-201B", 440, "B")):
        on = run == key
        a = alarms.get(name)
        s.append(f'<circle cx="830" cy="{cy}" r="20" fill="{EQUIP if on else "#fff"}" stroke="{P1 if a == 1 else P2 if a else EQUIP}" stroke-width="{4 if a else 2}"/>'
                 f'<polygon points="822,{cy - 10} 842,{cy} 822,{cy + 10}" fill="{"#fff" if on else EQUIP}"/>'
                 f'<text x="830" y="{cy + 36}" text-anchor="middle" font-size="12" font-weight="700" fill="#222">{name}</text>'
                 f'<text x="830" y="{cy + 49}" text-anchor="middle" font-size="9" fill="#555">{"RUN" if on else "STOP"}</text>')
        if a:
            s.append(_ind(862, cy - 22, a))
    s.append(f'<path d="M800,380 V440 H810" {line}/><path d="M850,380 H900 V440 H850" {line}/>')
    s.append(_val(560, 400, "P201-VIB", f'{r["P201-VIB"]:.2f}', "mm/s", alarm=bool(alarms.get("P-201A"))))
    s.append(_val(560, 436, "P201-TBRG", f'{r["P201-TBRG"]:.0f}', "°C") + _val(660, 400, "P201-PDIS", f'{r["P201-PDIS"]:.1f}', "bar")
             + _val(660, 436, "P201-AMP", f'{r["P201-AMP"]:.0f}', "A"))
    # loading bay
    a = alarms.get("LB-01")
    s.append(f'<path d="M900,410 H930" {line}/>')
    s.append(f'<rect x="930" y="360" width="110" height="112" fill="#f2f2f2" stroke="{P1 if a == 1 else P2 if a else EQUIP}" stroke-width="{4 if a else 2}"/>'
             f'<text x="985" y="380" text-anchor="middle" font-size="13" font-weight="700" fill="#222">LB-01</text>'
             f'<text x="985" y="396" text-anchor="middle" font-size="11" fill="#444" {AR_FONT}>{labels["loading"]}</text>')
    s.append(_val(940, 412, "FT-LOAD", f'{r["FT-LOAD"]:.0f}', "m³/h") + _val(940, 444, "GD-01", f'{r["GD-01"]:.1f}', "%LEL", alarm=bool(a)))
    if a:
        s.append(_ind(1028, 372, a))
    s.append('</g>')
    # utilities and network
    for asset, x, items in (("EL-33KV", 820, [("GRID-V", f'{r["GRID-V"]:.3f}', "pu"), ("GRID-F", f'{r["GRID-F"]:.2f}', "Hz")]),
                            ("OT-NET", 940, [("MODBUS-W", f'{int(r["NET-MODBUS-W"])}', "/5min"), ("TT-900", f'{r["TT-900"]:.0f}', "°C")])):
        a = alarms.get(asset)
        s.append(f'<rect x="{x}" y="20" width="110" height="112" fill="#f2f2f2" stroke="{P1 if a == 1 else P2 if a else EQUIP}" stroke-width="{4 if a else 2}"/>'
                 f'<text x="{x + 8}" y="38" font-size="13" font-weight="700" fill="#222">{asset}</text>')
        for k, (tag, v, u) in enumerate(items):
            s.append(_val(x + 8, 58 + 36 * k, tag, v, u, alarm=bool(a)))
        if a:
            s.append(_ind(x + 96, 32, a))
    return (f'<svg class="hmi-svg" viewBox="0 0 1060 528" width="100%" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="Roboto Mono, Consolas, monospace">{"".join(s)}</svg>')
