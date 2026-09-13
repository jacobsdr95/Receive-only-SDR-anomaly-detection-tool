# RF Sentinel — Receive-Only SDR Anomaly Detection

Automates RF spectrum monitoring so you don't have to stare at a waterfall in
SDR# or GQRX waiting for something unusual to show up. The tool pulls raw IQ
samples from a HackRF, runs them through a DSP pipeline, and uses a layered
set of machine learning models to flag anomalous signals — with a live web
dashboard for reviewing and labeling what it finds.

## How it works

**Ingestion** — Pulls raw IQ data directly from the HackRF (receive-only; see
[Safety](#safety--scope) below).

**DSP pipeline** — Uses `numpy`/`scipy` to compute FFT-based power spectral
density, spectral entropy, kurtosis, sample entropy, a cyclostationary score,
and a GSM FCCH detector, per channel and per sweep.

**Anomaly detection (three layers)**
- *Unsupervised, global:* DBSCAN + IsolationForest trained on the whole
  spectrum's behavior.
- *Unsupervised, per signal type:* a Hidden Markov Model (`hmmlearn`) for
  frequency-hopping/spoofing patterns, plus Local Outlier Factor / One-Class
  SVM / IsolationForest for other categories.
- *Supervised:* a Random Forest trained on your confirmed labels, used only
  to **downgrade** severity — it never escalates an alert on its own, and it
  refuses to train (or disables itself) if the labels look imbalanced,
  leaky, or like it's just memorizing the rule table instead of the signal.

**Confidence gating** — Alerts have to survive multiple checks (persistence
across scans, a power z-score consensus between the AI layers, and the RF
downgrade gate above) before reaching `CRITICAL`.

**Web dashboard** — A local Flask server (default port `1717`) shows recent
events, spectrograms, and lets you confirm/reject detections to build up the
labeled dataset the Random Forest trains on.

**Storage** — Events, spectrum history, and evidence (raw IQ + metadata) are
logged to SQLite and `./rf_logs/`.

## Safety & scope

This tool is **receive-only**. Any call that looks transmit-capable is
intercepted and the process is killed immediately (fail-loud, not a warning)
— see `verify_rx_guard_integrity()` and `ReceiveOnlySDRProxy` in the source
if you want to audit that yourself. It does not transmit, jam, or replay
signals, and it isn't built to.

Two things worth knowing before you rely on it:
- **Power readings are uncalibrated by default** (`CALIBRATION_VERIFIED =
  False`). dBm values are relative until you run a calibration pass against
  a signal generator.
- Frequency scanning and any related radio use are subject to local
  telecommunications law — that's on you to check for your jurisdiction and
  hardware, not something this tool verifies for you.

## Installation

Python 3.9+ recommended. Using a virtual environment is strongly suggested:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install numpy scipy scikit-learn joblib flask hmmlearn matplotlib
pip install hackrf                 # HackRF driver bindings
# if that fails on your platform, try instead:
# pip install pyhackrf
```

You'll also need `libhackrf` installed at the system level (e.g.
`apt install hackrf libhackrf-dev` on Debian/Ubuntu, `brew install hackrf`
on macOS) for either Python binding to find the hardware.

## Running it

**Full mode** (opens the HackRF, runs the sweep, starts the dashboard):

```bash
python SDR-BLUE-TEAM.py
```

**Web-only / dev mode** — no hardware required, useful for testing or
modifying the Flask UI:

```bash
python SDR-BLUE-TEAM.py --web-only
```

Then open `http://localhost:1717/` in a browser.

## License\n\nMIT
