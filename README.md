# RF Sentinel v15

![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-BSD%203--Clause-orange?style=flat-square)
![HackRF](https://img.shields.io/badge/hardware-HackRF%20One-blueviolet?style=flat-square)
![Receive Only](https://img.shields.io/badge/TX--guard-receive--only-red?style=flat-square)

> **Automates RF spectrum monitoring so you don't have to stare at a waterfall in SDR# or GQRX waiting for something unusual to show up.**

RF Sentinel pulls raw IQ samples from a HackRF One, runs them through a DSP pipeline, and uses a layered set of machine learning models to flag anomalous signals -- with a live web dashboard for reviewing and labeling what it finds. It never transmits -- a hardware-level TX guard fails loud and terminates the process immediately on any emission attempt.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Web UI](#web-ui)
- [Configuration](#configuration)
- [Advanced Patch (A/B/C/D)](#advanced-patch-abcd)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Limitations & Warnings](#limitations--warnings)
- [License](#license)

---

## How It Works

**Ingestion** -- Pulls raw IQ data directly from the HackRF (receive-only; see [Limitations & Warnings](#limitations--warnings)).

**DSP pipeline** -- Uses numpy/scipy to compute FFT-based power spectral density, spectral entropy, kurtosis, sample entropy, a cyclostationary score, and a GSM FCCH detector -- per channel and per sweep. A Savitzky-Golay pre-filter smooths the PSD before scoring to suppress noise spikes.

**Anomaly detection (three layers)**

1. **Unsupervised, global** -- DBSCAN + IsolationForest trained on the whole spectrum's behaviour.
2. **Unsupervised, per signal type** -- Hidden Markov Model (hmmlearn) for frequency-hopping/spoofing patterns; Local Outlier Factor / One-Class SVM / IsolationForest for other categories. DTW pattern matching identifies FHSS hopping signatures.
3. **Supervised** -- LightGBM (-> XGBoost -> RandomForest fallback) trained on your confirmed labels, used **only to downgrade severity** -- it never escalates an alert on its own, and it refuses to train (or disables itself) if labels look imbalanced, leaky, or like it is memorising the rule table instead of the signal.

**Confidence gating** -- Alerts must survive multiple checks (persistence across scans, a power Z-score consensus between AI layers, and the RF downgrade gate) before reaching CRITICAL.

**Web dashboard** -- A local Flask server (default port 1717) shows recent events, spectrograms, and lets you confirm/reject detections to build the labeled dataset the classifier trains on.

**Storage** -- Events, spectrum history, and evidence (raw IQ + metadata) are logged to SQLite and `./rf_logs/`.

---

## Features

| Category | Detail |
|---|---|
| **AI Gate 1 -- Global** | DBSCAN + Isolation Forest trained across all channels |
| **AI Gate 2 -- Per-Type** | HMM (FHSS/Spoof) * LOF (RFID/IoT/FakeBTS) * OCSVM (Emergency) * IF (all others) |
| **AI Gate 3 -- Supervised** | LightGBM -> XGBoost -> RandomForest fallback chain, label-leakage guarded |
| **AI Gate 4 -- Downgrade-only** | Classifier can only lower severity, never raise it |
| **DSP** | Savitzky-Golay pre-filter * DTW FHSS tracker * CAF cyclostationary analysis * GSM FCCH detector |
| **Auto-labeling** | FP/TP events labeled autonomously -- no human input needed for classifier training |
| **Confidence gating** | Alerts require >= 3/5 consecutive rounds + Dual-AI consensus at CRITICAL |
| **Web dashboard** | Real-time SSE updates * Chart.js charts * CSV/JSON export * calibration wizard |
| **TX guard** | Every emission call is intercepted and terminates the process immediately |
| **Storage** | SQLite rotating log * fingerprint files * evidence snapshots |

### Monitored Channels (default watchlist)

`GPS L1/L2` * `GSM 900/1800` * `TETRA` * `ISM 433/868/915` * `PMR 446` * `LoRa` * `Drone FHSS 2.4/5.8 GHz` * `Wi-Fi 2.4 GHz` * `Bluetooth` * `ADS-B 1090 MHz` * `L-band Satcom` * `Emergency VHF/UHF`

---

## Architecture

```
HackRF One (RX only)
        |
        v
 FastSweepEngine          <- ~25 ms/channel sweep
        |
        +-> DSP pipeline
        |     +- Savitzky-Golay pre-filter   [C]
        |     +- FFT / PSD / entropy / kurtosis / GSM FCCH
        |     +- DTW FHSS tracker            [B]
        |     +- CAF cyclostationary         [D]
        |
        +-> AI [1]  HybridCognitiveAI        (DBSCAN + IF, global)
        +-> AI [2]  RFAnomalyAI              (per-type specialists)
        +-> AI [3]  RFThreatClassifier       (LightGBM/XGBoost/RF) [A]
                |
                v
         Confidence gates (Persistence * Z-score * EMA * RF downgrade)
                |
                v
         SQLite  -->  Flask Web UI  (SSE, :1717)
```

---

## Requirements

### Hardware
- [HackRF One](https://greatscottgadgets.com/hackrf/) (receive-only mode)
- `libhackrf` at the system level -- `apt install hackrf libhackrf-dev` on Debian/Ubuntu, `brew install hackrf` on macOS

### Software

| Package | Version | Notes |
|---|---|---|
| Python | >= 3.10 | [x] required |
| numpy | any | [x] required |
| scipy | any | [x] required |
| scikit-learn | any | [x] required |
| joblib | any | [x] required |
| flask | any | [x] required |
| hackrf / pyhackrf | any | [x] required (try `hackrf` first) |
| hmmlearn | any | recommended |
| lightgbm | any | optional -- Gate 3 fast path |
| xgboost | any | optional -- Gate 3 fallback |
| matplotlib | any | optional -- spectrum GUI |

---

## Installation

```bash
# 1. Clone
git clone https://github.com/jacobsdr95/Receive-only-SDR-anomaly-detection-tool.git
cd Receive-only-SDR-anomaly-detection-tool

# 2. Create virtual environment (strongly recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install core dependencies
pip install numpy scipy scikit-learn joblib flask hmmlearn matplotlib

# 4. Install HackRF driver bindings
pip install hackrf
# if that fails on your platform:
# pip install pyhackrf

# 5. Optional -- faster classifier backend
pip install lightgbm            # recommended
pip install xgboost             # fallback

# 6. Apply advanced patch (adds LightGBM, DTW, SG-filter, CAF)
python patched_sdr_blueteam.py

# 7. Verify patch was applied
python patched_sdr_blueteam.py --check
```

---

## Usage

### Normal mode -- full sweep with HackRF

```bash
python SDR-BLUE-TEAM.py
```

### Web-only / dev mode -- no hardware required

Useful for testing or modifying the Flask UI without a connected HackRF:

```bash
python SDR-BLUE-TEAM.py --web-only
```

Then open **http://localhost:1717/** in a browser.

### Force classifier retrain on startup

```bash
python SDR-BLUE-TEAM.py --train-now
```

### Apply advanced patch to a custom-named file

```bash
python patched_sdr_blueteam.py path/to/your_file.py
```

### Dry-run -- preview diff without writing

```bash
python patched_sdr_blueteam.py --dry-run
```

### Load a file with a hyphenated name (importlib)

Python cannot `import` filenames containing hyphens directly. Use `importlib.util`:

```python
import importlib.util, sys

def load_module(path, name="sdr"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

sdr = load_module("SDR-BLUE-TEAM.py")
```

---

## Web UI

Once running, open **http://127.0.0.1:1717** in a browser.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main dashboard |
| `/api/stream` | GET | Server-Sent Events -- live anomaly feed |
| `/api/events` | GET | Recent events (JSON, filterable) |
| `/api/export` | GET | Download all events as CSV or JSON |
| `/api/channels` | GET | Per-channel baseline and last Z-score |
| `/api/calibration` | GET / POST | Read or update per-band dBm offsets |
| `/api/rf_model` | GET | Classifier status, accuracy, classes |

**Filters** (query params on `/api/events`): `severity`, `threat_type`, `channel`, `since` (ISO timestamp).

---

## Configuration

All tunable constants live near the top of the main file:

```python
# Detection thresholds
POWER_Z_ALERT       = 3.0     # Z-score to trigger a power alert
ENTROPY_DROP_ALERT  = 0.35    # Spectral entropy drop threshold
KURTOSIS_ALERT      = 8.0
CYCLO_ALERT         = 0.55    # Basic cyclostationary threshold

# Confidence gating
PERSISTENCE_MIN_COUNT = 3     # Min hits within window
PERSISTENCE_WINDOW    = 5     # Rolling window (rounds)
CRITICAL_ZSCORE_GATE  = 3.5   # Z-score required for CRITICAL consensus

# Sweep
SAMPLE_RATE_HZ  = 2_000_000
FFT_SIZE        = 1024
SWEEP_DWELL_S   = 0.005

# Classifier (Gate 3)
RF_MIN_TRAINING    = 200      # Rows before training is attempted
RF_CONF_THRESHOLD  = 0.70     # Minimum confidence to act on prediction
RF_N_ESTIMATORS    = 300

# Web UI
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 1717
```

### Calibration

Power readings are **relative** until calibrated. Use the `/api/calibration` endpoint or the in-dashboard calibration wizard to enter per-band offsets measured against a known signal generator:

```
offset_dB = (true_power_dBm) - (sdr_reading_dBm)
```

Default placeholder offset is `-50.0 dB` for all bands (`CALIBRATION_VERIFIED = False`).

---

## Advanced Patch (A/B/C/D)

Run `python patched_sdr_blueteam.py` to inject four algorithm upgrades into the main file:

### [A] LightGBM / XGBoost backend
Replaces RandomForest at Gate 3. 3-5x faster inference on embedded hardware (Raspberry Pi, NUC). Fallback chain: **LightGBM -> XGBoost -> RandomForest -> disabled**. All label-leakage and degenerate-model guards are preserved. The saved model file gains a `"backend"` field.

### [B] DTW FHSS tracker
Pure-NumPy Dynamic Time Warping tracks spectral centroid history per channel and compares against four FHSS templates (DJI Lightbridge, OcuSync, ELRS 2.4 GHz, Generic FHSS). Thread-safe singleton. Adds `dtw_fhss_score` and `dtw_fhss_alert` to every anomaly result.

### [C] Savitzky-Golay pre-filter
Smooths the PSD before entropy and Z-score calculation, suppressing random noise spikes and reducing false positives. Gracefully skips if `scipy` is unavailable or the array is too short. Tunable via `SG_WINDOW_LENGTH` and `SG_POLYORDER`.

### [D] Advanced cyclostationary (CAF)
Cyclic Autocorrelation Function at lag tau=0 across 8 common baud rates (9.6 k - 2 M sym/s). Man-made signals produce a clear cyclic peak; thermal noise does not. Adds `cyclo_adv_score`, `cyclo_alpha_hz`, and `cyclo_adv_alert`.

---

## Running Tests

No hardware required. Uses Python's built-in `unittest`:

```bash
# Run all tests
python test_sdr_sentinel.py

# Verbose output
python test_sdr_sentinel.py -v

# Run a specific test group
python test_sdr_sentinel.py TestDSP
python test_sdr_sentinel.py TestRFClassifier
python test_sdr_sentinel.py TestDatabase
```

### Test groups

| Group | Covers |
|---|---|
| `TestDSP` | FFT, entropy, kurtosis, Savitzky-Golay, CAF |
| `TestChannel` | Baseline tracking, Z-score, EMA update guard |
| `TestThreatRules` | Per-threat detection logic in `analyze()` |
| `TestConfidenceGates` | `PersistenceTracker`, `apply_confidence_gate()` |
| `TestRXGuard` | TX-call interception via `ReceiveOnlySDRProxy` |
| `TestRFClassifier` | Training guards, degenerate/leaky detection, predict gating |
| `TestDatabase` | SQLite CRUD with in-memory DB, `db_log_event` return ID, `_auto_label` FP/TP logic |
| `TestSignalFixtures` | Synthetic IQ signal generators used by other tests |

---

## Project Structure

```
Receive-only-SDR-anomaly-detection-tool/
+-- SDR-BLUE-TEAM.py           # Main monitor -- entry point
+-- patched_sdr_blueteam.py    # Advanced algorithm patch script (A/B/C/D)
+-- rf_sentinel_ui.py          # Flask web UI (routes, SSE, export)
+-- test_sdr_sentinel.py       # Full test suite (no hardware needed)
+-- README.md
+-- rf_logs/                   # Created at runtime
    +-- sentinel.log
    +-- rf_sentinel.db
    +-- fingerprints/
    +-- evidence/
    +-- reports/
```

---

## Limitations & Warnings

- **Receive-only tool.** The TX guard intercepts and terminates on any emission attempt. Audit `verify_rx_guard_integrity()` and `ReceiveOnlySDRProxy` in the source if you want to verify this yourself. Do not modify or bypass it.
- **Calibration required.** Power readings are uncalibrated by default (`CALIBRATION_VERIFIED = False`). All dBm values are relative until you run a calibration pass with a signal generator.
- **~25 ms per channel sweep.** Very short bursts (< 25 ms) may be missed entirely.
- **DSSS signals** below the noise floor may evade detection even with CAF active.
- **Classifier cold start.** The supervised classifier (Gate 3) stays silent until >= 200 labeled rows across >= 2 distinct threat classes are collected. Auto-labeling accumulates this without human input -- LOW alerts are labeled FP automatically, HIGH alerts with dual-AI consensus and |Z| > 4.0 are labeled TP. Expect ~500 events before the classifier becomes active in a typical environment.
- **Legal.** Frequency scanning and any related radio use are subject to local telecommunications law -- that is on you to check for your jurisdiction and hardware. This tool does not verify compliance.

---

## License


BSD 3-Clause License

Copyright (c) 2024, jacobsdr95@gmail.com

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
































$\color{#FFFFFF}{In~memory~of~Aaron~Swartz~1986-2013~Information~is~power.~Sharing~is~a~moral~imperative}$
