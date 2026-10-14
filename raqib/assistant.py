"""Operator assistant: live facts + retrieval over procedures (RAG without an LLM).

Arabic and English questions are matched with character n-gram TF-IDF, which
works for Arabic morphology without a stemmer. Answers quote live values from
the twin and cite the procedure section they came from. An open-weights LLM
(served on-prem, e.g. through Ollama) can be added on top to phrase the
answer; the retrieval and the facts stay the same, so every claim stays
traceable.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB = Path(__file__).resolve().parent.parent / "data" / "knowledge"

TOPICS = {
    "P-201A": ["p-201", "p201", "مضخة", "pump", "اهتزاز", "vibration", "محمل", "bearing"],
    "TK": ["tk-10", "خزان", "tank", "مستوى", "level", "فيضان", "overfill"],
    "PL": ["خط", "pipeline", "تسرب", "leak"],
    "LB": ["أبخرة", "غاز", "vapour", "vapor", "gas", "lel", "رصيف", "loading"],
    "EL": ["كهرباء", "power", "جهد", "voltage", "انقطاع"],
    "CY": ["سيبراني", "cyber", "اختراق", "modbus", "هجوم", "attack"],
}


class Assistant:
    def __init__(self):
        self.chunks = []
        for f in sorted(KB.glob("*.md")):
            text = f.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ").strip()
            for part in re.split(r"\n(?=## )", text)[1:]:
                head = part.splitlines()[0].lstrip("# ").strip()
                body = "\n".join(part.splitlines()[1:]).strip()
                self.chunks.append({"doc": title, "section": head, "text": body, "file": f.name})
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.M = self.vec.fit_transform([c["section"] + " " + c["text"] for c in self.chunks])

    def retrieve(self, q: str, k: int = 2) -> list[dict]:
        s = cosine_similarity(self.vec.transform([q]), self.M).ravel()
        return [dict(self.chunks[j], score=float(s[j])) for j in np.argsort(-s)[:k] if s[j] > 0.05]

    @staticmethod
    def topic(q: str) -> str | None:
        ql = q.lower()
        best, hits = None, 0
        for t, kws in TOPICS.items():
            h = sum(kw in ql for kw in kws)
            if h > hits:
                best, hits = t, h
        return best

    def facts(self, topic: str | None, df: pd.DataFrame, sh: pd.DataFrame, P: pd.Series,
              rul: dict, ar: bool) -> list[str]:
        r = df.iloc[-1]
        s = sh.iloc[-1]
        L = (lambda a, e: a if ar else e)
        out = []
        if topic == "P-201A":
            vr = sh["P201-VIB_res"].iloc[-24 * 12:].dropna()
            out.append(L(f"اهتزاز P-201A الآن {r['P201-VIB']:.2f} مم/ث، والمتوقع من التوأم {s['P201-VIB_exp']:.2f} مم/ث.",
                         f"P-201A vibration is {r['P201-VIB']:.2f} mm/s; the twin expects {s['P201-VIB_exp']:.2f} mm/s."))
            if len(vr) > 12:
                out.append(L(f"الفرق (Residual) زاد من {vr.iloc[:12].mean():.2f} إلى {vr.iloc[-12:].mean():.2f} مم/ث خلال 24 ساعة، وحرارة المحمل أعلى من المتوقع بـ {np.nanmean(sh['P201-TBRG_res'].iloc[-12:]):.1f} درجة.",
                             f"The residual rose from {vr.iloc[:12].mean():.2f} to {vr.iloc[-12:].mean():.2f} mm/s over 24 h; bearing temperature is {np.nanmean(sh['P201-TBRG_res'].iloc[-12:]):.1f} degC above expected."))
            if np.isfinite(rul.get("rul_p50", np.inf)):
                out.append(L(f"الزمن المتبقي المقدر قبل حد الإيقاف: {rul['rul_p50']:.0f} ساعة (بين {rul['rul_p10']:.0f} و{rul['rul_p90']:.0f}).",
                             f"Estimated remaining life to the trip limit: {rul['rul_p50']:.0f} h (P10 {rul['rul_p10']:.0f}, P90 {rul['rul_p90']:.0f})."))
            else:
                out.append(L("لا يوجد اتجاه تدهور واضح حالياً.", "No clear degradation trend right now."))
        elif topic == "TK":
            for t in ("TK-101", "TK-103"):
                g, e = r["LT-" + t[-3:]], s[t + "_est"]
                out.append(L(f"{t}: المقياس {g:.2f} م، وتقدير التوأم {e:.2f} م (الفرق {g - e:+.2f} م).",
                             f"{t}: gauge {g:.2f} m, twin estimate {e:.2f} m (difference {g - e:+.2f} m)."))
            out.append(L(f"الخزان المستلم الآن: {r['TAS-RECV']}. احتمال الفيضان خلال 12 ساعة: {P.get('overfill', 0):.0%}.",
                         f"Receiving tank: {r['TAS-RECV']}. Overfill probability within 12 h: {P.get('overfill', 0):.0%}."))
        elif topic == "PL":
            out.append(L(f"اختلال الموازنة في الخط {s['line_imbalance_pct']:.2f}% (الطبيعي أقل من 0.5%). احتمال التسرب: {P.get('leak', 0):.0%}.",
                         f"Line imbalance {s['line_imbalance_pct']:.2f}% (normal < 0.5%). Leak probability: {P.get('leak', 0):.0%}."))
            if P.get("leak", 0) > 0.5 and np.isfinite(s["leak_pos_frac"]):
                out.append(L(f"الموقع المقدر للتسرب: {s['leak_pos_frac'] * 60:.0f} كم من محطة البداية.",
                             f"Estimated leak location: {s['leak_pos_frac'] * 60:.0f} km from the upstream station."))
        elif topic == "LB":
            out.append(L(f"كاشف GD-01 يقرأ {r['GD-01']:.1f}% LEL، والرياح {r['WIND-SPD']:.1f} م/ث. احتمال انبعاث خطر: {P.get('vapor', 0):.0%}.",
                         f"GD-01 reads {r['GD-01']:.1f}% LEL, wind {r['WIND-SPD']:.1f} m/s. Vapour hazard probability: {P.get('vapor', 0):.0%}."))
        elif topic == "EL":
            sags = int((df["GRID-V"].iloc[-72:] < 0.95).sum())
            out.append(L(f"عدد هبوطات الجهد في آخر 6 ساعات: {sags}، والحرارة {r['TT-900']:.0f} درجة. احتمال الانقطاع خلال 12 ساعة: {P.get('power_trip', 0):.0%}.",
                         f"Voltage sags in the last 6 h: {sags}; ambient {r['TT-900']:.0f} degC. Power-loss probability within 12 h: {P.get('power_trip', 0):.0%}."))
        elif topic == "CY":
            w = int(df["NET-MODBUS-W"].iloc[-12:].sum())
            out.append(L(f"أوامر كتابة Modbus غير معروفة في آخر ساعة: {w}. احتمال التلاعب: {P.get('cyber', 0):.0%}.",
                         f"Unknown Modbus writes in the last hour: {w}. Manipulation probability: {P.get('cyber', 0):.0%}."))
        return out

    def answer(self, q: str, df, sh, P, rul, ar: bool) -> dict:
        topic = self.topic(q)
        return {"topic": topic, "facts": self.facts(topic, df, sh, P, rul, ar), "sources": self.retrieve(q)}
