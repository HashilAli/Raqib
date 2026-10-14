---
title: RAQIB Refinery Digital Twin
emoji: 🛡️
colorFrom: green
colorTo: red
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Digital twin that predicts terminal incidents before they happen
---

# 🛡️ RAQIB · رقيب
**Refinery AI for Quick Incident Prediction & Bypass**: an open-source prototype of a digital twin that watches a product terminal, predicts incidents hours to days ahead, and proposes a fix that has already been tested on the simulation.

> All data is synthetic, produced by a simulator in this repository. The bundled procedures are illustrative. This is a concept demo, not a certified safety system.

## بالعربية باختصار
«رقيب» توأم رقمي لمحطة تخزين وتوزيع منتجات نفطية. يقرأ المستشعرات (هنا من محاكي)، ويقارن كل قراءة بما **يجب** أن تكون عليه حسب الفيزياء، ويتنبأ بستة أنواع من الحوادث: فيضان الخزان، تسرب الخط، فشل المضخة، انقطاع الكهرباء، سحابة الأبخرة، والتلاعب السيبراني. لكل تنبيه يجرب 2–4 حلول على التوأم ويرتبها، ثم ينشئ مسودة أمر عمل صيانة. النظام **مستشار لا متحكم**: لا يكتب أي شيء في أنظمة التحكم أو السلامة.
الواجهة بالعربية والإنجليزية (11 شاشة) مع خريطة ثلاثية الأبعاد للموقع بـ CesiumJS، وكل المكونات مفتوحة المصدر ويمكن نشرها مجاناً.

## What you can show in two minutes
1. Open **Control room** with the scenario *Tank overfill: stuck level gauge*. The level gauge on TK-101 froze; the operator sees a normal-looking level, but the twin's mass balance says the tank is still filling. RAQIB raises "Untrusted instrument" within about 30 minutes of the gauge sticking, about 28 hours before the simulated overflow.
2. Drag **Current time** forward and watch the overfill probability rise before the red incident line.
3. Open **What-if & solutions**: four options are simulated from the twin's corrected levels; "continue" overfills, "divert to TK-103 + manual dip" does not, and it becomes the recommendation. Approve it to draft a SAP PM style work order.
4. Switch to *Cyber attack: replayed level readings*: the replayed signal looks normal (it even has noise), but breaks the physics, so RAQIB separates **manipulation** from a **broken instrument**.

Deep links work for sharing a specific screen, for example `?lang=en&scenario=pump&page=asset`.

## Screens
The screens follow ISA-101 high-performance HMI practice so they look like the control-system displays operators already use: grey background, equipment in dark grey, values in a fixed-width face, and colour only for abnormal states. A header strip and an ISA-18.2 style alarm summary (priority shape and number, start time, tag, value, acknowledged state) sit on every screen, and unacknowledged alarms blink until the operator rates them.

| Screen | What it shows | Team role |
|---|---|---|
| 🏭 Control room | Process overview graphic (simplified P&ID with live tags, gauge vs twin level, alarm indicators) and a 3D CesiumJS site map (tank fill = twin-estimated level) coloured by live risk `R = P × C × (1 − B)`, top 5 predictions, alerts with reasons, operator verdict buttons | UI developer |
| 🔁 Digital twin | Shadow mode: measured vs expected, residuals, 24 h look-ahead to the LSHH, ±2% accuracy check | Process / simulation |
| ⚙️ Asset card P-201A | Vibration trend, exponential RUL with P10–P90, deviation drivers, Weibull from CMMS history | Reliability, ML |
| 🧪 What-if & solutions | 2–4 options per alert, each simulated forward on the twin and ranked; CMMS work-order draft | Process, reliability |
| 🚨 Emergency centre | Gaussian plume from the detector reading draped on the 3D map, flammable zone, upwind evacuation, response tasks | Process safety |
| 📋 FMEA & reliability | Editable ISO 14224 FMEA, RPN before/after RAQIB | Reliability |
| 🧠 Models & MLOps | Model card, acceptance tests on unseen runs, precision/recall, PSI drift, anomaly score | Data science / ML |
| 🔐 OT integration & security | Purdue diagram, read-only connectors, data quality, tamper verdict, blocked-write demo, IEC 62443 | OT data, OT cyber |
| 💬 Assistant | Arabic/English Q&A from live data plus cited procedures (retrieval, no cloud LLM) | ML, operations |
| 🗣️ Operator voice | Alarm load vs ISA-18.2, alert verdict log, acceptance survey | Operations representative |
| 👥 Team & architecture | Role map and architecture | All |

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The first load trains the models on simulated runs (about 15 s on a laptop, longer on free cloud CPUs) and caches them.

Tests: `pytest -q` (one slow test trains the full model set).

## Deploy for free

### Option A: Streamlit Community Cloud (easiest)
1. Create a public GitHub repository, for example `raqib-prototype`, and upload every file in this folder (keep the folder structure).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app** → **Deploy a public app from GitHub**.
3. Pick the repository, branch `main`, main file `app.py`. Under **Advanced settings** choose Python 3.12.
4. Click **Deploy**. You get a URL like `https://raqib-prototype.streamlit.app` to put in your LinkedIn post.

### Option B: Hugging Face Spaces (Docker)
1. Go to <https://huggingface.co/new-space>, choose a name, **Docker** as the SDK (blank template), visibility Public.
2. Upload all files from this folder (or `git push` to the Space repository). The YAML block at the top of this README tells Spaces to build the `Dockerfile` and serve port 8501.
3. The Space builds and runs; the URL is `https://huggingface.co/spaces/<user>/<space>`.

Any container host (Render, Railway, Fly.io) can also run the `Dockerfile`.

## How it works
```
plant.py (simulated site) ──measured tags──▶ twin.py (shadow model) ──residuals──▶ models.py ──P, RUL, anomaly──▶ advisor.py ──▶ app.py
   hidden true_* columns                         mass balance, pump curve,           one monotonic GBM per        risk index, what-if,
   used only for scoring                         hydraulic gradient, look-ahead       failure mode, Isolation      work orders
                                                                                       Forest, RUL, explanations
```
* **Simulated terminal** (`raqib/plant.py`): two gasoline tanks with automatic line-up switching driven by the *measured* level, a diesel tank, a 60 km pipeline segment, loading pumps P-201A/B, a 33 kV supply, a loading bay with a gas detector, summer weather. Seven scenarios inject faults: stuck gauge with a failed LSHH (Buncefield pattern), growing pipeline leak, bearing degradation, heat-driven grid sags, a replay attack on a level gauge with Modbus writes, and a night-time vapour release in calm wind.
* **Digital twin** (`raqib/twin.py`): calibrated on the first 24 h, it computes expected values and residuals, reconciles tank levels when sensor and physics agree, and keeps a physics estimate when they do not. The look-ahead projects hours to the LSHH from the twin's level, not the gauge.
* **Prediction engine** (`raqib/models.py`): one gradient-boosted classifier per failure mode, restricted to that mode's causal signals with monotonic constraints, so it cannot learn coincidences of weather or schedule. Labels start only once a physical precursor exists. Isolation Forest scores unknown anomalies. RUL fits an exponential to the vibration residual with a moving-block bootstrap for the interval. Every alert carries its drivers.
* **Advisor** (`raqib/advisor.py`): branches the simulation from the current state (with the twin's reconciled levels) once per option and scores incident avoided, product spilled and throughput lost. Level L2: it recommends, the operator executes.
* **3D map** (`raqib/geo.py`, `static/raqib_map.html`): CesiumJS 1.121 (Apache-2.0) with Esri World Imagery, so no Cesium ion token is needed. Equipment positions are illustrative, placed on open ground about 300 m east of the pilot coordinates in the design document so the simulated layout does not overlap the real tank farm. The page is served by Streamlit's static file server (`enableStaticServing` in `.streamlit/config.toml`) because Cesium's web workers do not start inside `components.html`. Cesium loads from the jsDelivr CDN; on an air-gapped network set `RAQIB_CESIUM_BASE` to a local copy of `Build/Cesium`. The map needs WebGL in the viewer's browser.
* **Security** (`raqib/security.py`): the gateway has no write path; physics consistency plus network evidence separates instrument failure from manipulation.

## Acceptance tests (simulated data, held-out seeds)
Three unseen runs per scenario, 10 days each:

| Scenario | Passed | Mean warning before incident | Mean time to detect after onset | False alerts |
|---|---|---|---|---|
| Normal operation | 3/3 | – | – | 0 |
| Tank overfill (stuck gauge) | 3/3 | 12.1 h (classifier); twin flags the gauge ~28 h before overflow | ~0.5 h (twin residual) | 0 |
| Pipeline leak | 3/3 | – (no precursor) | 0.7 h | 0 |
| Pump bearing | 3/3 | 12.8 h (classifier); RUL warns days ahead | – | 0 |
| Grid disturbance | 3/3 | 14.9 h | – | 0 |
| Cyber replay attack | 3/3 | – | immediate (network + physics) | 0 |
| Vapour release | 3/3 | 6.1 h before 60% LEL | 0.1 h | 0 |

These numbers describe the simulator, not field performance. The project targets for a real site are recall ≥ 70%, precision ≥ 80%, ≥ 72 h warning for rotating equipment.

## From prototype to the full design
| Prototype | Production design |
|---|---|
| Python simulator | Modelica/FMI or DWSIM dynamic model (or AVEVA/Aspen), calibrated against the historian |
| In-memory frames | Kafka + OPC UA (Eclipse Milo / asyncua) behind a hardware data diode, TimescaleDB/InfluxDB |
| scikit-learn models | PyTorch LSTM / Temporal Fusion Transformer for RUL, MLflow registry, 3 months in shadow mode |
| TF-IDF retrieval | On-prem open-weights LLM with pgvector RAG over SOPs, incident reports and manuals |
| JSON work orders | SAP PM OData notifications |
| Streamlit + CesiumJS on a schematic layout | React + CesiumJS with surveyed 3D Tiles of the real site, mobile app for field crews |

## Repository layout
```
app.py                   Streamlit dashboard (11 screens, Arabic/English, RTL)
raqib/plant.py           simulated terminal and fault scenarios
raqib/twin.py            shadow-mode digital twin
raqib/models.py          prediction engine, RUL, explanations, acceptance tests
raqib/advisor.py         risk index, what-if engine, work orders
raqib/reliability.py     FMEA, CMMS history, Weibull
raqib/security.py        connectors, data quality, gateway, IEC 62443
raqib/assistant.py       retrieval assistant
raqib/hmi.py             ISA-101 look: styles, alarm summary, process overview graphic
raqib/geo.py             site coordinates and scene data for the 3D map
static/raqib_map.html    CesiumJS page embedded by the dashboard
data/                    FMEA, CMMS history, illustrative procedures
tests/                   pytest suite
docs/linkedin_post.md    LinkedIn post drafts (Arabic and English)
```

## License
MIT. Built with Python, Streamlit, scikit-learn, Plotly, CesiumJS, NumPy and pandas. Satellite imagery © Esri, Maxar, Earthstar Geographics.
