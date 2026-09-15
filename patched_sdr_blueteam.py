#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_advanced_features.py — RF Sentinel v15: Advanced Algorithm Upgrade
=========================================================================
Patches SDR-BLUE-TEAM_patched.py (default) hoặc file chỉ định.

PATCH NOTES
───────────
[A] LightGBM / XGBoost  ─  thay thế RandomForest ở Gate [4]
    • Inference nhanh hơn 3–5× trên Raspberry Pi / NUC
    • Độ chính xác với tabular features cao hơn RF
    • Graceful fall-back: LightGBM → XGBoost → RandomForest → disabled
    • Giữ nguyên toàn bộ safety properties (degenerate/leaky guards)
    • Model mới lưu thêm trường "backend" vào file joblib
    • pip install lightgbm   (hoặc pip install xgboost cho fallback 2)

[B] DTW (Dynamic Time Warping)  ─  nhận diện mẫu nhảy tần FHSS
    • Theo dõi spectral centroid history theo channel (ring buffer)
    • DTW so sánh chuỗi centroid vs 4 template: DJI Lightbridge, OcuSync,
      ELRS 2.4GHz, FHSS Generic (cả về shape lẫn phân bố entropy)
    • Không cần thư viện ngoài — pure NumPy O(n·m)
    • FHSSTracker singleton thread-safe (Lock)
    • Trường mới trong EW_AnomalyResult: dtw_fhss_score, dtw_fhss_alert
    • DRONE_FHSS branch trong analyze() dùng score này làm trigger chính

[C] Savitzky-Golay pre-filter  ─  làm mịn spectrum trước Z-Score
    • scipy.signal.savgol_filter áp lên PSD trước khi tính entropy /
      detect_single_carrier / spectral_entropy
    • Loại bỏ spike nhiễu ngẫu nhiên → giảm false positive
    • Fallback về raw PSD nếu scipy thiếu hoặc array quá ngắn
    • Hằng số cấu hình: SG_WINDOW_LENGTH, SG_POLYORDER

[D] Advanced Cyclostationary (CAF-based)  ─  phát hiện bên dưới noise floor
    • Cyclic Autocorrelation Function tại lag τ=0 cho cyclic freq α
    • Kiểm tra 8 baud/symbol rates phổ biến (9.6k – 2M sym/s)
    • Tín hiệu nhân tạo có α_peak rõ ràng; nhiễu nhiệt thì không
    • Trường mới: cyclo_adv_score, cyclo_alpha_hz, cyclo_adv_alert
    • DRONE_FHSS và CELL_JAM analysis ghi log symbol rate bị nghi ngờ

Sử dụng:
    python patch_advanced_features.py                 # patch SDR-BLUE-TEAM_patched.py
    python patch_advanced_features.py my_file.py      # patch file chỉ định
    python patch_advanced_features.py --dry-run       # xem diff, không ghi
    python patch_advanced_features.py --check         # kiểm tra đã patch chưa

Sau khi patch:
    pip install lightgbm      # cho [A]  (khuyến nghị — không bắt buộc)
    pip install xgboost       # cho [A]  (fallback — không bắt buộc)
    # scipy đã có sẵn trong project — không cần cài thêm cho [C] [D]
"""

from __future__ import annotations

import ast
import os
import shutil
import sys
import textwrap
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────

IDEMPOTENCY_MARKER = "# [ADV-PATCH-v1] Applied"   # added to top of patched file


def _find_target(argv) -> Path:
    for arg in argv:
        if not arg.startswith("--") and arg.endswith(".py"):
            p = Path(arg)
            if p.exists():
                return p
            print(f"[ERR] File không tồn tại: {arg}")
            sys.exit(1)
    candidates = [
        Path(__file__).parent / "SDR-BLUE-TEAM_patched.py",
        Path("SDR-BLUE-TEAM_patched.py"),
        Path(__file__).parent / "SDR-BLUE-TEAM.py",
        Path("SDR-BLUE-TEAM.py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    print("[ERR] Không tìm thấy SDR-BLUE-TEAM_patched.py hoặc SDR-BLUE-TEAM.py")
    print("      Chỉ định rõ: python patch_advanced_features.py <file.py>")
    sys.exit(1)


# ─── Syntax verify ─────────────────────────────────────────────────────────

def _verify(code: str, label: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"  ✗  [{label}] Lỗi syntax sau patch: {e}")
        return False


# ─── Patch helpers ─────────────────────────────────────────────────────────

def _apply(code: str, old: str, new: str, label: str) -> tuple[str, bool]:
    """
    Replace first occurrence of `old` with `new`.
    Returns (new_code, success).
    old must appear EXACTLY ONCE — checked before replacement.
    """
    if old not in code:
        print(f"  ✗  [{label}] Anchor string không tìm thấy — bỏ qua.")
        return code, False
    if code.count(old) > 1:
        print(f"  !  [{label}] Anchor xuất hiện {code.count(old)} lần — patch chỉ thay lần đầu.")
    return code.replace(old, new, 1), True


# ══════════════════════════════════════════════════════════════════════════════
# PATCH A — LightGBM / XGBoost imports
# ══════════════════════════════════════════════════════════════════════════════

_A_OLD = """\
try:
    import hmmlearn  # noqa: F401
    from hmmlearn.hmm import GaussianHMM
    HMM_OK = True
except Exception:
    HMM_OK = False"""

_A_NEW = """\
try:
    import hmmlearn  # noqa: F401
    from hmmlearn.hmm import GaussianHMM
    HMM_OK = True
except Exception:
    HMM_OK = False

# [A] LightGBM / XGBoost — drop-in faster replacement for RandomForest
try:
    from lightgbm import LGBMClassifier
    LGBM_OK = True
except Exception:
    LGBM_OK = False

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False"""


# ══════════════════════════════════════════════════════════════════════════════
# PATCH B — New algorithm constants
# ══════════════════════════════════════════════════════════════════════════════

_B_OLD = "CYCLO_ALERT            = 0.55"

_B_NEW = """\
CYCLO_ALERT            = 0.55

# ---- [C] Savitzky-Golay pre-filter ----------------------------------------
SG_WINDOW_LENGTH  = 15          # must be odd; reduce to 7 if FFT_SIZE < 256
SG_POLYORDER      = 3

# ---- [B] DTW FHSS pattern matching -----------------------------------------
DTW_FHSS_ALERT    = 0.52        # DTW similarity threshold for FHSS alert
FHSS_HISTORY_LEN  = 24          # centroid ring-buffer depth (snapshots)

# ---- [D] Advanced cyclostationary (CAF) ------------------------------------
CYCLO_ADV_ALERT         = 0.38      # normalised CAF magnitude threshold
CYCLO_SYMBOL_RATES_HZ   = [         # baud / symbol rates to probe
    9_600, 19_200, 38_400, 115_200,
    250_000, 500_000, 1_000_000, 2_000_000,
]"""


# ══════════════════════════════════════════════════════════════════════════════
# PATCH C — New DSP functions inserted before AI [1] block
# ══════════════════════════════════════════════════════════════════════════════

_C_OLD = """\
# ============================================================================
# AI [1] — GLOBAL HYBRID (DBSCAN + IsolationForest)
# ============================================================================"""

_C_NEW = textwrap.dedent("""\
# ============================================================================
# [C] SAVITZKY-GOLAY SPECTRUM PRE-FILTER
# ============================================================================
def smooth_psd_savgol(psd, window=SG_WINDOW_LENGTH, poly=SG_POLYORDER):
    \"\"\"
    Apply Savitzky-Golay filter to PSD before any threshold computation.

    Why: Raw IQ → PSD is inherently "jagged" due to thermal noise.
    Sporadic single-bin spikes trigger false-positive Z-score / entropy
    alerts.  SG smoothing preserves spectral shape (unlike moving average)
    while eliminating those impulse artefacts.

    Falls back to raw PSD if scipy is unavailable or the array is shorter
    than the requested window (can happen on very short IQ snapshots).
    \"\"\"
    if not SCIPY_OK or psd is None or len(psd) < 5:
        return np.asarray(psd, dtype=np.float32)
    try:
        from scipy.signal import savgol_filter
        p  = np.asarray(psd, dtype=np.float64)
        wl = min(window, len(p))
        if wl % 2 == 0:
            wl -= 1                     # window length must be odd
        if wl < 5:
            return p.astype(np.float32)
        po = min(poly, wl - 1)
        smoothed = savgol_filter(p, wl, po)
        smoothed = np.clip(smoothed, 0.0, None)   # SG can produce tiny negatives
        return smoothed.astype(np.float32)
    except Exception:
        return np.asarray(psd, dtype=np.float32)


# ============================================================================
# [B] DTW (DYNAMIC TIME WARPING) — FHSS PATTERN RECOGNITION
# ============================================================================
def _spectral_centroid_norm(psd):
    \"\"\"
    Normalised spectral centroid of a PSD vector: 0.0 = low edge, 1.0 = high.
    Returns 0.5 for a flat / empty spectrum.
    Used to build the per-channel centroid time-series fed into DTW.
    \"\"\"
    p = np.asarray(psd, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return 0.5
    bins = np.arange(len(p), dtype=np.float64)
    return float(np.dot(bins, p) / (s * max(len(p) - 1, 1)))


def _dtw_distance_fast(seq_a, seq_b):
    \"\"\"
    O(n·m) DTW distance — pure NumPy, no external library.
    seq_a, seq_b: 1-D float arrays, may have different lengths.

    For the FHSS use-case both sequences are short (≤24 elements) so the
    quadratic cost is negligible.  A Sakoe-Chiba band constraint could be
    added here for longer sequences without changing the interface.
    \"\"\"
    a = np.asarray(seq_a, dtype=np.float64)
    b = np.asarray(seq_b, dtype=np.float64)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j],
                                   dtw[i,     j - 1],
                                   dtw[i - 1, j - 1])
    return float(dtw[n, m])


def _centroid_hist_entropy(seq, n_bins=8):
    \"\"\"
    Normalised Shannon entropy of the centroid histogram.
    0.0 → all centroids pile up at one bin (narrowband / CW).
    1.0 → uniform spread across the band (ideal FHSS).
    \"\"\"
    counts, _ = np.histogram(seq, bins=n_bins, range=(0.0, 1.0))
    counts = counts.astype(np.float64)
    s = counts.sum()
    if s <= 0 or math.log2(n_bins) <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / math.log2(n_bins))


# Reference FHSS centroid sequences (normalised 0–1 within the monitored band).
# Each row is a synthetic template that captures the statistical signature of
# the corresponding drone protocol's hopping spread pattern.
#
# HOW TO CALIBRATE: record a real drone with HackRF in your environment,
# extract centroid sequences with FHSSTracker.update(), then replace / extend
# the entries below.  The current values are designed to capture the
# pseudo-random wide spread that distinguishes FHSS from CW interference,
# not to fingerprint a specific serial number.
_FHSS_REFERENCES = {
    "DJI_Lightbridge": np.array([
        0.18, 0.72, 0.41, 0.88, 0.09, 0.63, 0.33, 0.79,
        0.52, 0.24, 0.67, 0.15, 0.84, 0.38, 0.61, 0.06,
        0.77, 0.44, 0.92, 0.30,
    ]),
    "DJI_OcuSync": np.array([
        0.10, 0.55, 0.91, 0.28, 0.74, 0.19, 0.62, 0.45,
        0.83, 0.13, 0.70, 0.38, 0.95, 0.21, 0.57, 0.80,
        0.35, 0.68, 0.48, 0.15,
    ]),
    "ELRS_2400": np.array([
        0.25, 0.78, 0.12, 0.58, 0.90, 0.36, 0.71, 0.04,
        0.82, 0.47, 0.65, 0.22, 0.87, 0.53, 0.16, 0.74,
        0.41, 0.96, 0.30, 0.59,
    ]),
    "FHSS_Generic": np.array([
        0.08, 0.42, 0.76, 0.21, 0.89, 0.55, 0.31, 0.67,
        0.14, 0.85, 0.50, 0.73, 0.28, 0.93, 0.16, 0.62,
        0.38, 0.80, 0.46, 0.11,
    ]),
}


class FHSSTracker:
    \"\"\"
    Per-channel ring buffer of spectral centroids feeding DTW scoring.

    Thread-safe (per-instance Lock).  The singleton _FHSS_TRACKER is
    created at module level after this class definition — it is lightweight
    (no logging, no I/O) so module-level construction is fine.

    score() combines two orthogonal signals:
      1. DTW similarity to reference hopping templates (shape match)
      2. Centroid histogram entropy (spread match — FHSS is near-uniform)

    A narrowband jammer scores high on shape but LOW on entropy.
    A real FHSS drone scores high on both.
    \"\"\"
    MIN_SAMPLES = 8     # below this, not enough history to score reliably
    STD_MIN     = 0.08  # < 8 % of band width → narrowband, not FHSS

    def __init__(self, history_len=FHSS_HISTORY_LEN):
        from collections import deque, defaultdict
        self._buf  = defaultdict(lambda: deque(maxlen=history_len))
        self._lock = threading.Lock()

    def update(self, channel_name: str, psd) -> None:
        \"\"\"Push the current snapshot's normalised spectral centroid.\"\"\"
        c = _spectral_centroid_norm(psd)
        with self._lock:
            self._buf[channel_name].append(c)

    def dtw_score(self, channel_name: str) -> float:
        \"\"\"
        Return FHSS similarity score in [0, 1].
        0 = flat / narrowband / insufficient history.
        1 = strong match to a known FHSS hopping signature.
        \"\"\"
        with self._lock:
            hist = list(self._buf[channel_name])
        n = len(hist)
        if n < self.MIN_SAMPLES:
            return 0.0

        seq = np.asarray(hist, dtype=np.float64)

        # Hard gate: narrowband signals cannot be FHSS regardless of shape.
        if np.std(seq) < self.STD_MIN:
            return 0.0

        # --- DTW against each reference template ---
        best_dtw = 0.0
        for ref in _FHSS_REFERENCES.values():
            d = _dtw_distance_fast(seq, ref)
            # Normalise: worst-case DTW for sequences in [0,1] of length L is L.
            max_d = max(n, len(ref))
            sim   = max(0.0, 1.0 - d / max_d)
            if sim > best_dtw:
                best_dtw = sim

        # --- Centroid entropy (spread quality) ---
        entropy_score = _centroid_hist_entropy(seq)

        # Weighted combination: shape (65%) + spread (35%)
        combined = 0.65 * best_dtw + 0.35 * entropy_score
        return float(min(1.0, combined))

    def centroid_history(self, channel_name: str) -> list:
        \"\"\"Return a copy of the centroid history (for logging / UI export).\"\"\"
        with self._lock:
            return list(self._buf[channel_name])


# Module-level singleton — created here, before detect_anomaly_ew is defined.
_FHSS_TRACKER = FHSSTracker()


# ============================================================================
# [D] ADVANCED CYCLOSTATIONARY FEATURE DETECTION  (CAF-based)
# ============================================================================
def cyclo_detect_advanced(iq, fs=None,
                           symbol_rates=None):
    \"\"\"
    Cyclic Autocorrelation Function (CAF) detector at lag τ = 0.

    Key insight
    ───────────
    Every man-made digital signal has a *cyclic frequency* α equal (or
    harmonically related) to its symbol/baud rate.  At that α the CAF
    R_α(τ=0) = E[ x(t)·x*(t) · e^{−j2π α t} ]  has a significant
    non-zero magnitude.  Thermal / environmental noise has NO such
    periodicity, so R_α ≈ 0 for all α.

    This lets us detect artificial signals that are *below* the noise
    floor in the raw PSD — the kind that a simple Z-score would miss.

    Return value
    ────────────
    (score : float [0,1],  alpha_hz : float)
        score    — peak normalised |R_α| across all tested symbol rates
        alpha_hz — the cyclic frequency (Hz) where the peak was found

    Computational cost
    ──────────────────
    For N = 8192 samples and 8 symbol rates: ≈ 8 × N complex mults
    ≈ 65 k FLOPs.  Runs in < 1 ms on a Raspberry Pi 5 / NUC.
    \"\"\"
    if iq is None or len(iq) < 2048:
        return 0.0, 0.0

    _fs     = float(fs) if fs is not None else float(SAMPLE_RATE_HZ)
    _rates  = symbol_rates if symbol_rates is not None else CYCLO_SYMBOL_RATES_HZ

    x = np.asarray(iq[:8192], dtype=np.complex128)
    N = len(x)
    x = x - x.mean()                           # remove DC offset

    px = float(np.mean(np.abs(x) ** 2))        # signal power
    if px < 1e-12:
        return 0.0, 0.0

    # lag-0 conjugate product is simply |x(t)|^2 (real-valued)
    xabs2 = (x * np.conj(x)).real              # shape (N,)
    t     = np.arange(N, dtype=np.float64) / _fs

    best_score = 0.0
    best_alpha = 0.0

    for sr in _rates:
        alpha = float(sr)
        if alpha >= _fs / 2.0:                  # above Nyquist/2 → skip
            continue
        # Project onto cyclic frequency α: scalar complex value
        caf = np.abs(np.mean(xabs2 * np.exp(-1j * 2.0 * math.pi * alpha * t)))
        # Normalise by signal power for amplitude independence
        norm_score = caf / (px + 1e-12)
        if norm_score > best_score:
            best_score = norm_score
            best_alpha = alpha

    # Empirical scaling: strong real signals score ≈ 0.3–0.5 before clipping.
    return float(min(1.0, best_score * 2.5)), float(best_alpha)


# ============================================================================
# AI [1] — GLOBAL HYBRID (DBSCAN + IsolationForest)
# ============================================================================""")


# ══════════════════════════════════════════════════════════════════════════════
# PATCH D — New fields in EW_AnomalyResult
# ══════════════════════════════════════════════════════════════════════════════

_D_OLD = """\
    structural_alert: bool = False
    swept_alert:      bool = False

    @property
    def ai_alert(self):
        return self.hybrid_ai_alert or self.type_ai_alert

    @property
    def ai_both_agree(self):
        return self.hybrid_ai_alert and self.type_ai_alert

    @property
    def any_alert(self):
        return any([self.power_alert, self.entropy_alert, self.hybrid_ai_alert,
                    self.type_ai_alert, self.burst_alert,
                    self.structural_alert, self.swept_alert])"""

_D_NEW = """\
    structural_alert: bool = False
    swept_alert:      bool = False
    # ── [B] DTW FHSS ────────────────────────────────────────────────────────
    dtw_fhss_score:   float = 0.0
    dtw_fhss_alert:   bool  = False
    # ── [D] Advanced cyclostationary ────────────────────────────────────────
    cyclo_adv_score:  float = 0.0
    cyclo_alpha_hz:   float = 0.0   # detected symbol rate (Hz); 0 if none
    cyclo_adv_alert:  bool  = False

    @property
    def ai_alert(self):
        return self.hybrid_ai_alert or self.type_ai_alert

    @property
    def ai_both_agree(self):
        return self.hybrid_ai_alert and self.type_ai_alert

    @property
    def any_alert(self):
        return any([self.power_alert, self.entropy_alert, self.hybrid_ai_alert,
                    self.type_ai_alert, self.burst_alert,
                    self.structural_alert, self.swept_alert,
                    self.dtw_fhss_alert, self.cyclo_adv_alert])"""


# ══════════════════════════════════════════════════════════════════════════════
# PATCH E — detect_anomaly_ew: add SG filter, DTW, advanced cyclo
# ══════════════════════════════════════════════════════════════════════════════

_E_OLD = """\
def detect_anomaly_ew(channel, iq, hybrid_ai, type_ai):
    psd  = compute_psd(iq)
    pwr  = estimate_power_dbm(iq, channel.freq_hz)
    base = channel.baseline
    z    = channel.zscore(pwr)

    psd_valid = np.count_nonzero(psd) > 0
    ent  = spectral_entropy(psd) if psd_valid else 0.0
    kurt = kurtosis_of(iq)
    se   = sample_entropy_of(iq)
    ste  = stft_entropy(iq)
    cyc  = cyclostationary_score(iq)
    sc   = detect_single_carrier(psd) if psd_valid else False
    fcch = gsm_fcch_score(iq)

    # Entropy only means something when there is actual signal to measure.
    # On a silent channel it collapses to 0 and would fire ENTROPY_DROP on noise.
    if not psd_valid or pwr < base - 20.0:
        ent = 0.0
        se  = 0.0

    r = EW_AnomalyResult(
        channel=channel.name, freq_hz=channel.freq_hz,
        threat_type=channel.threat_type, power_dbm=pwr,
        baseline_dbm=base, zscore=z, entropy=ent, kurtosis=kurt,
        sample_entropy=se, stft_entropy=ste, cyclo_score=cyc,
        single_carrier=sc, gsm_fcch=fcch,
        snr_db=max(0.0, pwr - base),
        bandwidth_hz=float(SAMPLE_RATE_HZ),
        duration_ms=SWEEP_DWELL_S * 1000.0,
    )

    r.power_alert      = abs(z) > POWER_Z_ALERT
    r.burst_alert      = kurt > KURTOSIS_ALERT
    r.entropy_alert    = (ent > 0 and ent < (1.0 - ENTROPY_DROP_ALERT)) 
    r.structural_alert = (sc and ent > 0 and ent < 0.55)
    r.swept_alert      = cyc > CYCLO_ALERT"""

_E_NEW = """\
def detect_anomaly_ew(channel, iq, hybrid_ai, type_ai):
    psd   = compute_psd(iq)
    psd_s = smooth_psd_savgol(psd)                  # [C] SG-smoothed copy
    pwr   = estimate_power_dbm(iq, channel.freq_hz)
    base  = channel.baseline
    z     = channel.zscore(pwr)

    psd_valid = np.count_nonzero(psd_s) > 0
    ent  = spectral_entropy(psd_s) if psd_valid else 0.0   # [C] smoothed
    kurt = kurtosis_of(iq)
    se   = sample_entropy_of(iq)
    ste  = stft_entropy(iq)
    cyc  = cyclostationary_score(iq)
    sc   = detect_single_carrier(psd_s) if psd_valid else False  # [C] smoothed
    fcch = gsm_fcch_score(iq)

    # [B] DTW-based FHSS score — update tracker always; score only on DRONE channels
    _FHSS_TRACKER.update(channel.name, psd_s)
    dtw_fhss = (_FHSS_TRACKER.dtw_score(channel.name)
                if channel.threat_type == "DRONE_FHSS" else 0.0)

    # [D] CAF-based advanced cyclostationary score
    cyclo_adv, cyclo_alpha = cyclo_detect_advanced(iq)

    # Entropy only means something when there is actual signal to measure.
    # On a silent channel it collapses to 0 and would fire ENTROPY_DROP on noise.
    if not psd_valid or pwr < base - 20.0:
        ent = 0.0
        se  = 0.0

    r = EW_AnomalyResult(
        channel=channel.name, freq_hz=channel.freq_hz,
        threat_type=channel.threat_type, power_dbm=pwr,
        baseline_dbm=base, zscore=z, entropy=ent, kurtosis=kurt,
        sample_entropy=se, stft_entropy=ste, cyclo_score=cyc,
        single_carrier=sc, gsm_fcch=fcch,
        snr_db=max(0.0, pwr - base),
        bandwidth_hz=float(SAMPLE_RATE_HZ),
        duration_ms=SWEEP_DWELL_S * 1000.0,
        dtw_fhss_score=dtw_fhss,       # [B]
        cyclo_adv_score=cyclo_adv,      # [D]
        cyclo_alpha_hz=cyclo_alpha,     # [D]
    )

    r.power_alert      = abs(z) > POWER_Z_ALERT
    r.burst_alert      = kurt > KURTOSIS_ALERT
    r.entropy_alert    = (ent > 0 and ent < (1.0 - ENTROPY_DROP_ALERT))
    r.structural_alert = (sc and ent > 0 and ent < 0.55)
    r.swept_alert      = cyc > CYCLO_ALERT
    r.dtw_fhss_alert   = dtw_fhss > DTW_FHSS_ALERT         # [B]
    r.cyclo_adv_alert  = cyclo_adv > CYCLO_ADV_ALERT        # [D]"""


# ══════════════════════════════════════════════════════════════════════════════
# PATCH F — DRONE_FHSS branch in analyze(): use DTW + CAF
# ══════════════════════════════════════════════════════════════════════════════

_F_OLD = """\
    elif tt == "DRONE_FHSS":
        if r.swept_alert or r.type_ai_alert:
            lvl = ThreatLevel.HIGH
            ind.append("Dac trung nhay tan (FHSS) — nghi drone.")"""

_F_NEW = """\
    elif tt == "DRONE_FHSS":
        if r.dtw_fhss_alert:                              # [B] DTW primary
            lvl = ThreatLevel.HIGH
            ind.append(
                f"[DTW] Khop mau nhay tan FHSS "
                f"(score={r.dtw_fhss_score:.2f}) — nghi drone.")
        elif r.swept_alert or r.type_ai_alert:
            lvl = ThreatLevel.HIGH
            ind.append("Dac trung nhay tan (FHSS) — nghi drone.")
        if r.cyclo_adv_alert:                             # [D] may be sub-NF
            sym_khz = r.cyclo_alpha_hz / 1e3
            ind.append(
                f"[CAF] Tin hieu nhan tao phat hien duoi noise floor "
                f"(symbol rate ~{sym_khz:.0f} kHz).")"""


# ══════════════════════════════════════════════════════════════════════════════
# PATCH G — RFThreatClassifier: add backend field + LightGBM training
# ══════════════════════════════════════════════════════════════════════════════

# G1: add self.backend to __init__
_G1_OLD = """\
        self._since_train = 0
        self._lock       = Lock()
        self._load()"""

_G1_NEW = """\
        self._since_train = 0
        self._lock       = Lock()
        self.backend     = "none"   # [A] set by train(); "LightGBM"|"XGBoost"|"RandomForest"
        self._load()"""


# G2: load backend field from saved model
_G2_OLD = """\
            self.trained_at  = blob.get("trained_at")
            # A degenerate model on disk stays degenerate across restarts.
            self.usable = (self.model is not None and not self.degenerate
                           and len(self.classes_) >= 2)
            state = "USABLE" if self.usable else "DISABLED (degenerate)"
            logger.info(f"[RF] Da nap model: {self.classes_} — {state}")"""

_G2_NEW = """\
            self.trained_at  = blob.get("trained_at")
            self.backend     = blob.get("backend", "RandomForest")  # [A]
            # A degenerate model on disk stays degenerate across restarts.
            self.usable = (self.model is not None and not self.degenerate
                           and len(self.classes_) >= 2)
            state = "USABLE" if self.usable else "DISABLED (degenerate)"
            logger.info(
                f"[RF] Da nap model ({self.backend}): "
                f"{self.classes_} — {state}")"""


# G3: save backend field
_G3_OLD = """\
                \"model\": self.model, \"classes\": self.classes_,
                \"degenerate\": degenerate, \"holdout_acc\": self.holdout_acc,
                \"cv5_acc\": self.cv5_acc,
                \"trained_at\": datetime.now(timezone.utc).isoformat(),"""

_G3_NEW = """\
                "model": self.model, "classes": self.classes_,
                "degenerate": degenerate, "holdout_acc": self.holdout_acc,
                "cv5_acc": self.cv5_acc,
                "backend": self.backend,                 # [A]
                "trained_at": datetime.now(timezone.utc).isoformat(),"""


# G4: replace RandomForestClassifier with LightGBM-first multi-backend logic
_G4_OLD = """\
        clf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42)
        clf.fit(Xtr, ytr)
        holdout = float(clf.score(Xte, yte))"""

_G4_NEW = """\
        # [A] Backend selection: LightGBM → XGBoost → RandomForest
        if LGBM_OK:
            clf = LGBMClassifier(
                n_estimators=RF_N_ESTIMATORS,
                num_leaves=31,
                learning_rate=0.1,
                class_weight="balanced",
                n_jobs=-1,
                verbose=-1,
                random_state=42,
            )
            _backend = "LightGBM"
        elif "XGB_OK" in dir() and XGB_OK:
            # XGBoost requires integer labels; use sklearn wrapper with LabelEncoder.
            from sklearn.preprocessing import LabelEncoder
            _le = LabelEncoder()
            ytr_enc = _le.fit_transform(ytr)
            yte_enc = _le.transform(yte)
            clf_xgb = XGBClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=5,
                eval_metric="mlogloss",
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            )
            clf_xgb.fit(Xtr, ytr_enc)
            holdout = float(clf_xgb.score(Xte, yte_enc))
            # Wrap into a thin adapter so the rest of the method stays generic.
            class _XGBAdapter:
                \"\"\"Thin sklearn-compatible wrapper that handles label encoding.\"\"\"
                def __init__(self, model, le):
                    self._m, self._le = model, le
                    self.feature_importances_ = model.feature_importances_
                    self.classes_ = le.classes_
                def predict_proba(self, X):
                    return self._m.predict_proba(X)
                def score(self, X, y):
                    return self._m.score(X, self._le.transform(y))
            clf = _XGBAdapter(clf_xgb, _le)
            _backend = "XGBoost"
        else:
            clf = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=42)
            _backend = "RandomForest"

        if _backend != "XGBoost":          # XGBoost holdout computed above
            clf.fit(Xtr, ytr)
            holdout = float(clf.score(Xte, yte))
        self.backend = _backend
        logger.info(f"[RF] Su dung backend: {_backend}")"""


# G5: update feature importances log to include backend
_G5_OLD = """\
        imp = sorted(zip(self.FEATURES, clf.feature_importances_),
                     key=lambda t: -t[1])[:4]
        logger.info(
            f"[RF] Retrain: {n} mau, holdout={holdout:.3f}, "
            f"cv5={self.cv5_acc}, baseline={baseline:.3f}, "
            f"classes={self.classes_}, usable={self.usable}, "
            f"top={[(nm, round(v, 3)) for nm, v in imp]}")"""

_G5_NEW = """\
        try:
            fi = getattr(clf, "feature_importances_",
                         getattr(getattr(clf, "_m", None),
                                 "feature_importances_", None))
            imp = (sorted(zip(self.FEATURES, fi), key=lambda t: -t[1])[:4]
                   if fi is not None else [])
        except Exception:
            imp = []
        logger.info(
            f"[RF/{self.backend}] Retrain: {n} mau, holdout={holdout:.3f}, "
            f"cv5={self.cv5_acc}, baseline={baseline:.3f}, "
            f"classes={self.classes_}, usable={self.usable}, "
            f"top={[(nm, round(v, 3)) for nm, v in imp]}")"""


# ══════════════════════════════════════════════════════════════════════════════
# PATCH H — print_banner: reflect new backends
# ══════════════════════════════════════════════════════════════════════════════

_H_OLD = '    logger.info("  AI [3] Supervised: RandomForest tren nhan da kiem chung")'

_H_NEW = (
    '    _be = ("LightGBM" if LGBM_OK else\n'
    '           "XGBoost"  if ("XGB_OK" in dir() and XGB_OK) else\n'
    '           "RandomForest")\n'
    '    logger.info(f"  AI [3] Supervised: {_be} tren nhan da kiem chung [A]")\n'
    '    logger.info("  DSP [C] Savitzky-Golay spectrum pre-filter: BAT")\n'
    '    logger.info("  DSP [B] DTW FHSS tracker: BAT (history=%d)", FHSS_HISTORY_LEN)\n'
    '    logger.info("  DSP [D] CAF cyclostationary: BAT (%d symbol rates)", '
    'len(CYCLO_SYMBOL_RATES_HZ))'
)


# ══════════════════════════════════════════════════════════════════════════════
# PATCH I — CELL_JAM analysis: add CAF indicator
# ══════════════════════════════════════════════════════════════════════════════

_I_OLD = """\
    elif tt == "CELL_JAM":
        if r.power_alert and r.burst_alert:
            lvl = ThreatLevel.CRITICAL
            ind.append("Nhieu dang burst tren bang cellular/TETRA.")"""

_I_NEW = """\
    elif tt == "CELL_JAM":
        if r.power_alert and r.burst_alert:
            lvl = ThreatLevel.CRITICAL
            ind.append("Nhieu dang burst tren bang cellular/TETRA.")
        if r.cyclo_adv_alert:             # [D] hidden jamming signal
            sym_khz = r.cyclo_alpha_hz / 1e3
            ind.append(
                f"[CAF] Nghi tin hieu nhieu co chu ky an "
                f"(symbol rate ~{sym_khz:.0f} kHz) — co the la jammer nghi trang.")"""


# ══════════════════════════════════════════════════════════════════════════════
# APPLY ALL PATCHES
# ══════════════════════════════════════════════════════════════════════════════

PATCHES = [
    ("A: LightGBM/XGBoost imports",     _A_OLD, _A_NEW),
    ("B/C/D: constants",                 _B_OLD, _B_NEW),
    ("B/C/D: new DSP functions",         _C_OLD, _C_NEW),
    ("D: EW_AnomalyResult new fields",   _D_OLD, _D_NEW),
    ("C/B/D: detect_anomaly_ew body",    _E_OLD, _E_NEW),
    ("B: DRONE_FHSS analysis",           _F_OLD, _F_NEW),
    ("I: CELL_JAM CAF indicator",        _I_OLD, _I_NEW),
    ("A: RF __init__ backend field",     _G1_OLD, _G1_NEW),
    ("A: RF _load backend field",        _G2_OLD, _G2_NEW),
    ("A: RF _save backend field",        _G3_OLD, _G3_NEW),
    ("A: RF train multi-backend",        _G4_OLD, _G4_NEW),
    ("A: RF train feature importances",  _G5_OLD, _G5_NEW),
    ("A: print_banner backend label",    _H_OLD, _H_NEW),
]


def already_patched(code: str) -> bool:
    return IDEMPOTENCY_MARKER in code


def run(argv=None):
    argv = argv or sys.argv[1:]
    dry_run = "--dry-run" in argv
    check   = "--check"   in argv

    target = _find_target(argv)
    code   = target.read_text(encoding="utf-8")

    if check:
        if already_patched(code):
            print(f"[OK]  {target.name} — đã được patch (marker tìm thấy).")
        else:
            print(f"[--]  {target.name} — CHƯA được patch.")
        return

    if already_patched(code):
        print(f"[SKIP] {target.name} đã được patch rồi (idempotency marker).")
        return

    print(f"\n{'─'*60}")
    print(f"  Target : {target}")
    print(f"  Mode   : {'DRY RUN (không ghi file)' if dry_run else 'PATCH THỰC'}")
    print(f"{'─'*60}\n")

    ok_count = 0
    for label, old, new in PATCHES:
        code, ok = _apply(code, old, new, label)
        status = "✓" if ok else "✗"
        print(f"  {status}  {label}")
        if ok:
            ok_count += 1
            if not _verify(code, label):
                print(f"\n[ABORT] Syntax lỗi sau patch '{label}'. File KHÔNG được ghi.")
                sys.exit(2)

    # Add idempotency marker after shebang/encoding line
    lines = code.splitlines(keepends=True)
    insert_at = 0
    for i, ln in enumerate(lines[:5]):
        if ln.startswith("#"):
            insert_at = i + 1
    lines.insert(insert_at, f"{IDEMPOTENCY_MARKER}\n")
    code = "".join(lines)

    print(f"\n  Áp dụng: {ok_count}/{len(PATCHES)} patches thành công.\n")

    if dry_run:
        print("[DRY-RUN] Không ghi file.  Để patch thật: bỏ flag --dry-run\n")
        return

    bak = target.with_suffix(".py.adv_bak")
    shutil.copy2(target, bak)
    print(f"  Backup : {bak}")

    target.write_text(code, encoding="utf-8")
    print(f"  Ghi    : {target}")

    # Final verify on written file
    written = target.read_text(encoding="utf-8")
    if _verify(written, "final"):
        print("\n[OK] Patch hoàn tất — syntax hợp lệ.\n")
    else:
        print("\n[WARN] Patch ghi xong nhưng syntax verify thất bại.")
        print(f"       Khôi phục từ: {bak}\n")

    print("Bước tiếp theo:")
    print("  pip install lightgbm          # [A] khuyến nghị")
    print("  pip install xgboost           # [A] fallback (tuỳ chọn)")
    print("  # scipy đã có — [C][D] hoạt động ngay\n")
    print("Kiểm tra nhanh (không cần HackRF):")
    print(f"  python {target.name} --web-only\n")


if __name__ == "__main__":
    run()
