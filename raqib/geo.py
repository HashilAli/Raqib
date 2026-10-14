"""3D site map with CesiumJS (open source, Apache-2.0), rendered inside Streamlit.

The schematic site layout (metres) is placed at the pilot coordinates from the
project document. Equipment positions are illustrative, not surveyed.
No Cesium ion token is needed: imagery comes from Esri World Imagery tiles and
the globe uses the plain ellipsoid (the site is flat).

Owner role: UI developer.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import zlib
from urllib.parse import quote

import numpy as np

CESIUM_VERSION = "1.121.0"
ORIGIN_LAT, ORIGIN_LON = 23.4998, 58.2054      # ~300 m east of the pilot coordinates in the project document,
                                                 # on open ground so the simulated layout does not sit on real tanks
ORIGIN_XY = (180.0, 160.0)                       # site-sketch point that sits on the origin
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.radians(ORIGIN_LAT))

BAND_COLOR = {"high": "#e00000", "medium": "#f2a900", "low": "#8c8c8c"}   # ISA-101: colour only when abnormal


def to_lonlat(x: float, y: float) -> tuple[float, float]:
    return (ORIGIN_LON + (x - ORIGIN_XY[0]) / M_PER_DEG_LON,
            ORIGIN_LAT + (y - ORIGIN_XY[1]) / M_PER_DEG_LAT)


def _png(rgba: np.ndarray) -> bytes:
    """Minimal PNG encoder (RGBA8) so the plume overlay needs no imaging library."""
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[r].astype(np.uint8).tobytes() for r in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def lel_overlay(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray, floor: float = 5.0) -> dict:
    """Colour a %LEL grid (rows = ys ascending) into a transparent PNG with its lon/lat bounds."""
    stops = np.array([[5, 255, 255, 178], [25, 254, 204, 92], [50, 253, 141, 60], [100, 240, 59, 32], [150, 128, 0, 38]], float)
    z = np.clip(Z, 0, 150)
    rgba = np.zeros(z.shape + (4,))
    for k in range(3):
        rgba[..., k] = np.interp(z, stops[:, 0], stops[:, k + 1])
    rgba[..., 3] = np.where(z >= floor, np.interp(z, [floor, 100], [110, 210]), 0)
    rgba = rgba[::-1]                                   # PNG rows run north to south
    w, s = to_lonlat(xs[0], ys[0])
    e, n = to_lonlat(xs[-1], ys[-1])
    return {"png": "data:image/png;base64," + base64.b64encode(_png(rgba)).decode(), "bounds": [w, s, e, n]}


def map_payload(risk_rows: list[dict], levels: dict, ar: bool,
                overlay: dict | None = None, arrows: list[dict] | None = None,
                markers: list[dict] | None = None) -> dict:
    """Scene data for static/raqib_map.html.

    risk_rows: asset, name, class, x, y, risk, P, C, B, band
    levels:    tank -> twin-estimated liquid level (m)
    overlay:   optional plume PNG from lel_overlay()
    arrows:    optional [{x0, y0, x1, y1, color, label}]
    markers:   optional [{x, y, label, color}]
    """
    ents = []
    for r in risk_rows:
        lon, lat = to_lonlat(r["x"], r["y"])
        ents.append({"id": r["asset"], "name": r["name"], "lon": lon, "lat": lat, "band": r["band"],
                     "color": BAND_COLOR[r["band"]], "tank": r["asset"].startswith("TK"),
                     "level": float(levels.get(r["asset"], 0.0)),
                     "desc": (f"<table style='font-size:13px'>"
                              f"<tr><td>{'مؤشر الخطر' if ar else 'Risk index'}</td><td><b>{r['risk']:.0f}</b></td></tr>"
                              f"<tr><td>P</td><td>{r['P']:.0%}</td></tr><tr><td>C</td><td>{r['C']:.1f}</td></tr>"
                              f"<tr><td>B</td><td>{r['B']:.1f}</td></tr>"
                              + (f"<tr><td>{'المستوى (تقدير التوأم)' if ar else 'Level (twin estimate)'}</td><td>{levels.get(r['asset'], 0):.2f} m</td></tr>"
                                 if r["asset"] in levels else "") + "</table>")})
    line = [to_lonlat(x, 140.0) for x in (-900.0, -400.0, 30.0)]
    arr = [{"from": to_lonlat(a["x0"], a["y0"]), "to": to_lonlat(a["x1"], a["y1"]), "color": a.get("color", "#2ca02c"),
            "label": a.get("label", "")} for a in (arrows or [])]
    mk = [{"pos": to_lonlat(m["x"], m["y"]), "label": m["label"], "color": m.get("color", "#1f77b4")} for m in (markers or [])]
    return {"ents": ents, "line": line, "overlay": overlay, "arrows": arr, "markers": mk,
            "origin": [ORIGIN_LON, ORIGIN_LAT],
            "pad": [to_lonlat(x, y) for x, y in ((-20, 0), (400, 0), (400, 310), (-20, 310))],
            "lineLabel": "خط المنتجات" if ar else "Products pipeline"}


def map_url(payload: dict, page: str = "app/static/raqib_map.html") -> str:
    """URL of the static Cesium page with the scene in the hash.

    The page is served by Streamlit's static file server rather than through
    components.html: Cesium builds geometry in web workers, and workers do not
    start from the srcdoc iframe that components.html uses (its origin is null).
    RAQIB_CESIUM_BASE lets an air-gapped (on-prem) install serve Cesium from its own web server.
    """
    base = os.environ.get("RAQIB_CESIUM_BASE", "")
    q = "?base=" + quote(base, safe="") if base else ""
    h = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    return f"{page}{q}#{h}"
