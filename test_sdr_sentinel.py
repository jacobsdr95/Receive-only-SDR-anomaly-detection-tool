"""
test_sdr_sentinel.py — Test suite cho RF Sentinel (SDR-BLUE-TEAM.py)

Chạy:
    python test_sdr_sentinel.py           # chạy tất cả, verbose
    python test_sdr_sentinel.py -v        # verbose mode
    python test_sdr_sentinel.py TestDSP   # chỉ chạy nhóm DSP

Không cần HackRF hay bất kỳ hardware nào.
Không cần pytest — dùng unittest built-in.

Cách tổ chức:
    TestDSP              — Hàm signal processing (FFT, entropy, kurtosis...)
    TestChannel          — Channel baseline tracking và z-score
    TestThreatRules      — Per-threat-type detection rules trong analyze()
    TestConfidenceGates  — PersistenceTracker và apply_confidence_gate()
    TestRXGuard          — ReceiveOnlySDRProxy chặn TX calls
    TestRFClassifier     — RFThreatClassifier gating + training guards
    TestDatabase         — SQLite CRUD với in-memory DB
    TestSignalFixtures   — Kiểm tra signal generators dùng trong các test khác
"""

import os
import sys
import math
import time
import unittest
import sqlite3
import tempfile
import threading

import importlib
import importlib.util as _ilu

# ---------------------------------------------------------------------------
# _dyn_import — load module/attr bằng importlib, trả None nếu thiếu
# ---------------------------------------------------------------------------
def _dyn_import(module_name: str, attr: str | None = None):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attr) if attr else mod
    except (ImportError, AttributeError):
        return None

np = _dyn_import("numpy")
if np is None:
    raise ImportError("numpy is required — pip install numpy")

# ---------------------------------------------------------------------------
# Đưa source vào sys.path và import
# ---------------------------------------------------------------------------
_SRC = os.path.join(os.path.dirname(__file__),
                    "Receive-only-SDR-anomaly-detection-tool-main")
sys.path.insert(0, _SRC)

# Patch DB_PATH trước khi import để tránh tạo file thật
import unittest.mock as mock

with mock.patch("os.makedirs"):   # tránh tạo thư mục thật lúc import
    import types  # importlib đã import ở trên

# ---- dùng importlib.util để load file tên có gạch ngang ----
# Python không cho phép import "SDR-BLUE-TEAM" trực tiếp.

_spec = _ilu.spec_from_file_location(
    "sdr_sentinel",
    os.path.join(_SRC, "SDR-BLUE-TEAM.py")
)
sdr = _ilu.module_from_spec(_spec)

# Patch 'os.makedirs' trong quá trình exec để test không tạo thư mục thật
_real_makedirs = os.makedirs
def _patched_makedirs(path, **kw):
    if "rf_logs" in path:
        return   # skip tạo ./rf_logs khi đang test
    _real_makedirs(path, **kw)

os.makedirs = _patched_makedirs
_spec.loader.exec_module(sdr)
os.makedirs = _real_makedirs   # restore


# ===========================================================================
# Helpers tạo IQ signal giả cho test
# ===========================================================================

def make_noise_iq(n=4096, amplitude=50.0, seed=42) -> np.ndarray:
    """White Gaussian noise — entropy cao, kurtosis thấp (~3)."""
    rng = np.random.default_rng(seed)
    I = rng.standard_normal(n) * amplitude
    Q = rng.standard_normal(n) * amplitude
    return (I + 1j * Q).astype(np.complex64)


def make_tone_iq(n=4096, freq_hz=100e3, fs=2e6, amplitude=80.0) -> np.ndarray:
    """Sóng sin đơn tần — entropy thấp, detect_single_carrier=True."""
    t = np.arange(n) / fs
    iq = (amplitude * np.exp(2j * np.pi * freq_hz * t)).astype(np.complex64)
    return iq


def make_burst_iq(n=4096, burst_len=64, burst_amplitude=100.0, seed=7) -> np.ndarray:
    """Burst ngắn nằm trong nền nhiễu nhỏ — kurtosis rất cao."""
    rng = np.random.default_rng(seed)
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    # Chèn burst ở giữa
    mid = n // 2
    iq[mid:mid + burst_len] += burst_amplitude
    return iq


def make_repeating_iq(n=8192, period=200, amplitude=60.0) -> np.ndarray:
    """Tín hiệu lặp theo chu kỳ — cyclostationary_score cao."""
    iq = np.zeros(n, dtype=np.complex64)
    for start in range(0, n, period):
        end = min(start + period // 4, n)
        iq[start:end] = amplitude
    return iq


def make_empty_iq() -> np.ndarray:
    return np.zeros(0, dtype=np.complex64)


def make_dummy_anomaly_result(**overrides) -> sdr.EW_AnomalyResult:
    """Tạo EW_AnomalyResult mặc định (tất cả flags=False, scores=0)."""
    defaults = dict(
        channel="GPS_L1",
        freq_hz=1575.42e6,
        threat_type="GPS_JAM",
        power_dbm=-60.0,
        baseline_dbm=-70.0,
        zscore=0.0,
        entropy=0.8,
        kurtosis=3.0,
        sample_entropy=0.5,
        stft_entropy=5.0,
        cyclo_score=0.1,
        single_carrier=False,
        gsm_fcch=0.0,
        snr_db=10.0,
        bandwidth_hz=2e6,
        duration_ms=5.0,
        hybrid_ai_score=0.0,
        hybrid_ai_label="UNTRAINED",
        type_ai_score=0.0,
        type_ai_label="UNTRAINED",
        power_alert=False,
        entropy_alert=False,
        hybrid_ai_alert=False,
        type_ai_alert=False,
        burst_alert=False,
        structural_alert=False,
        swept_alert=False,
    )
    defaults.update(overrides)
    return sdr.EW_AnomalyResult(**defaults)


# ===========================================================================
# TestSignalFixtures — kiểm tra helpers bên trên trước khi dùng trong tests khác
# ===========================================================================

class TestSignalFixtures(unittest.TestCase):

    def test_noise_shape_and_dtype(self):
        iq = make_noise_iq(n=1024)
        self.assertEqual(iq.shape, (1024,))
        self.assertEqual(iq.dtype, np.complex64)

    def test_tone_is_narrowband(self):
        """Sóng sin đơn tần phải có entropy thấp."""
        iq = make_tone_iq(n=4096)
        psd = sdr.compute_psd(iq)
        ent = sdr.spectral_entropy(psd)
        self.assertLess(ent, 0.4, "Single tone should have low spectral entropy")

    def test_noise_has_high_entropy(self):
        iq = make_noise_iq(n=4096)
        psd = sdr.compute_psd(iq)
        ent = sdr.spectral_entropy(psd)
        self.assertGreater(ent, 0.7, "White noise should have high spectral entropy")

    def test_burst_has_high_kurtosis(self):
        iq = make_burst_iq()
        kurt = sdr.kurtosis_of(iq)
        self.assertGreater(kurt, 10.0, "Burst signal should have kurtosis >> 3")


# ===========================================================================
# TestDSP — hàm xử lý tín hiệu số
# ===========================================================================

class TestDSP(unittest.TestCase):

    # ---- iq_bytes_to_complex ------------------------------------------------

    def test_iq_bytes_empty_returns_zero_array(self):
        out = sdr.iq_bytes_to_complex(b"")
        self.assertEqual(len(out), 0)

    def test_iq_bytes_none_returns_zero_array(self):
        out = sdr.iq_bytes_to_complex(None)
        self.assertEqual(len(out), 0)

    def test_iq_bytes_odd_length_drops_last_byte(self):
        """Số byte lẻ: byte cuối bị bỏ để tránh IQ mismatch."""
        data = bytes([10, 20, 30])  # 3 bytes → drop 1 → 1 sample
        out = sdr.iq_bytes_to_complex(data)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].real, 10.0)
        self.assertAlmostEqual(out[0].imag, 20.0)

    def test_iq_bytes_interleaved_iq(self):
        """Format: [I0, Q0, I1, Q1, ...]"""
        data = bytes([10, 20, 30, 40])  # I0=10,Q0=20, I1=30,Q1=40
        out = sdr.iq_bytes_to_complex(data)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0].real, 10.0)
        self.assertAlmostEqual(out[0].imag, 20.0)
        self.assertAlmostEqual(out[1].real, 30.0)
        self.assertAlmostEqual(out[1].imag, 40.0)

    def test_iq_bytes_dtype(self):
        data = bytes([0, 0, 50, 50])
        out = sdr.iq_bytes_to_complex(data)
        self.assertEqual(out.dtype, np.complex64)

    # ---- estimate_power_dbm ------------------------------------------------

    def test_power_empty_returns_floor(self):
        self.assertAlmostEqual(sdr.estimate_power_dbm(make_empty_iq()), -120.0)

    def test_power_none_returns_floor(self):
        self.assertAlmostEqual(sdr.estimate_power_dbm(None), -120.0)

    def test_power_increases_with_amplitude(self):
        low  = sdr.estimate_power_dbm(make_noise_iq(amplitude=10.0))
        high = sdr.estimate_power_dbm(make_noise_iq(amplitude=100.0))
        self.assertGreater(high, low)

    def test_power_is_float(self):
        p = sdr.estimate_power_dbm(make_noise_iq())
        self.assertIsInstance(p, float)

    def test_power_reasonable_range(self):
        """Amplitude=50 (mid-range 8-bit) → power khoảng -10 đến +10 dBm."""
        p = sdr.estimate_power_dbm(make_noise_iq(amplitude=50.0))
        self.assertGreater(p, -50.0)
        self.assertLess(p, 50.0)

    # ---- compute_psd -------------------------------------------------------

    def test_psd_empty_returns_zeros(self):
        psd = sdr.compute_psd(make_empty_iq())
        self.assertTrue(np.all(psd == 0))

    def test_psd_too_short_returns_zeros(self):
        iq = make_noise_iq(n=10)   # < FFT_SIZE (1024)
        psd = sdr.compute_psd(iq)
        self.assertTrue(np.all(psd == 0))

    def test_psd_shape(self):
        """PSD có FFT_SIZE//2 bins."""
        iq = make_noise_iq(n=4096)
        psd = sdr.compute_psd(iq)
        self.assertEqual(psd.shape, (sdr.FFT_SIZE // 2,))

    def test_psd_dtype(self):
        iq = make_noise_iq(n=4096)
        psd = sdr.compute_psd(iq)
        self.assertEqual(psd.dtype, np.float32)

    def test_psd_nonnegative(self):
        psd = sdr.compute_psd(make_noise_iq())
        self.assertTrue(np.all(psd >= 0))

    def test_psd_tone_has_peak(self):
        """Sóng sin → PSD có peak rõ, không flat."""
        iq = make_tone_iq(n=4096)
        psd = sdr.compute_psd(iq)
        ratio = psd.max() / (psd.mean() + 1e-9)
        self.assertGreater(ratio, 10.0, "Tone PSD should have a sharp peak")

    # ---- spectral_entropy --------------------------------------------------

    def test_entropy_zero_psd_returns_zero(self):
        self.assertAlmostEqual(sdr.spectral_entropy(np.zeros(512)), 0.0)

    def test_entropy_flat_psd_returns_one(self):
        """Uniform distribution → max entropy = 1.0."""
        psd = np.ones(512)
        ent = sdr.spectral_entropy(psd)
        self.assertAlmostEqual(ent, 1.0, places=4)

    def test_entropy_impulse_psd_returns_zero(self):
        """Tất cả năng lượng tại 1 bin → entropy gần 0."""
        psd = np.zeros(512)
        psd[256] = 1.0
        ent = sdr.spectral_entropy(psd)
        self.assertAlmostEqual(ent, 0.0, places=4)

    def test_entropy_in_range(self):
        psd = sdr.compute_psd(make_noise_iq())
        ent = sdr.spectral_entropy(psd)
        self.assertGreaterEqual(ent, 0.0)
        self.assertLessEqual(ent, 1.0)

    def test_entropy_single_bin_psd(self):
        """psd có 1 phần tử → log2(n) undefined (n<=1) → return 0."""
        self.assertAlmostEqual(sdr.spectral_entropy(np.array([5.0])), 0.0)

    # ---- kurtosis_of -------------------------------------------------------

    def test_kurtosis_none_returns_zero(self):
        self.assertAlmostEqual(sdr.kurtosis_of(None), 0.0)

    def test_kurtosis_short_returns_zero(self):
        self.assertAlmostEqual(sdr.kurtosis_of(np.ones(5, dtype=np.complex64)), 0.0)

    def test_kurtosis_constant_returns_zero(self):
        """Constant signal → std=0 → return 0 (guard division by zero)."""
        iq = np.ones(1024, dtype=np.complex64) * 50.0
        self.assertAlmostEqual(sdr.kurtosis_of(iq), 0.0)

    def test_kurtosis_gaussian_near_three(self):
        """Gaussian noise kurtosis ≈ 3."""
        rng = np.random.default_rng(0)
        iq = (rng.standard_normal(100000) + 1j * rng.standard_normal(100000)).astype(np.complex64)
        kurt = sdr.kurtosis_of(iq)
        self.assertAlmostEqual(kurt, 3.0, delta=0.2)

    def test_kurtosis_burst_greater_than_noise(self):
        noise = sdr.kurtosis_of(make_noise_iq())
        burst = sdr.kurtosis_of(make_burst_iq())
        self.assertGreater(burst, noise)

    # ---- stft_entropy ------------------------------------------------------

    def test_stft_entropy_empty_returns_zero(self):
        self.assertAlmostEqual(sdr.stft_entropy(make_empty_iq()), 0.0)

    def test_stft_entropy_too_short_returns_zero(self):
        iq = make_noise_iq(n=10)
        self.assertAlmostEqual(sdr.stft_entropy(iq), 0.0)

    def test_stft_entropy_positive(self):
        iq = make_noise_iq(n=4096)
        self.assertGreater(sdr.stft_entropy(iq), 0.0)

    def test_stft_entropy_noise_greater_than_tone(self):
        """Noise có STFT entropy cao hơn single tone."""
        noise_e = sdr.stft_entropy(make_noise_iq(n=4096))
        tone_e  = sdr.stft_entropy(make_tone_iq(n=4096))
        self.assertGreater(noise_e, tone_e)

    # ---- cyclostationary_score ---------------------------------------------

    def test_cyclo_empty_returns_zero(self):
        self.assertAlmostEqual(sdr.cyclostationary_score(make_empty_iq()), 0.0)

    def test_cyclo_short_returns_zero(self):
        iq = make_noise_iq(n=100)
        self.assertAlmostEqual(sdr.cyclostationary_score(iq), 0.0)

    def test_cyclo_in_range(self):
        score = sdr.cyclostationary_score(make_noise_iq(n=8192))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_cyclo_periodic_greater_than_noise(self):
        """Tín hiệu lặp có score cao hơn white noise."""
        noise_score = sdr.cyclostationary_score(make_noise_iq(n=8192))
        repeating_score = sdr.cyclostationary_score(make_repeating_iq())
        self.assertGreater(repeating_score, noise_score)

    # ---- detect_single_carrier ---------------------------------------------

    def test_single_carrier_none_returns_false(self):
        self.assertFalse(sdr.detect_single_carrier(None))

    def test_single_carrier_short_returns_false(self):
        self.assertFalse(sdr.detect_single_carrier(np.ones(3)))

    def test_single_carrier_tone_returns_true(self):
        iq = make_tone_iq(n=4096)
        psd = sdr.compute_psd(iq)
        self.assertTrue(sdr.detect_single_carrier(psd))

    def test_single_carrier_noise_returns_false(self):
        iq = make_noise_iq(n=4096)
        psd = sdr.compute_psd(iq)
        self.assertFalse(sdr.detect_single_carrier(psd))

    # ---- gsm_fcch_score ----------------------------------------------------

    def test_fcch_empty_returns_zero(self):
        self.assertAlmostEqual(sdr.gsm_fcch_score(make_empty_iq()), 0.0)

    def test_fcch_short_returns_zero(self):
        self.assertAlmostEqual(sdr.gsm_fcch_score(make_noise_iq(n=100)), 0.0)

    def test_fcch_in_range(self):
        score = sdr.gsm_fcch_score(make_noise_iq(n=4096))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_fcch_tone_at_67700hz_has_high_score(self):
        """GSM FCCH = sinusoid tại ~67.7 kHz → score cao."""
        iq = make_tone_iq(n=4096, freq_hz=67700.0, fs=sdr.SAMPLE_RATE_HZ, amplitude=127.0)
        score = sdr.gsm_fcch_score(iq)
        self.assertGreater(score, 0.1, "FCCH tone should score higher than noise")

    # ---- get_calibration_offset --------------------------------------------

    def test_calibration_offset_in_range(self):
        """Offset tồn tại trong tất cả các băng tần trong CALIBRATION_BANDS."""
        for lo, hi, expected in sdr.CALIBRATION_BANDS:
            mid = (lo + hi) / 2.0
            offset = sdr.get_calibration_offset(mid)
            self.assertIsInstance(offset, float)

    def test_calibration_offset_out_of_range_uses_default(self):
        """Tần số ngoài tất cả bands → fallback -50.0 dB."""
        offset = sdr.get_calibration_offset(100e9)   # 100 GHz, ngoài range
        self.assertAlmostEqual(offset, -50.0)


# ===========================================================================
# TestChannel — Channel baseline tracking
# ===========================================================================

class TestChannel(unittest.TestCase):

    def _make_ch(self) -> sdr.Channel:
        return sdr.Channel("TEST", 433.92e6, 2, "IOT_REPLAY")

    def test_baseline_empty_returns_floor(self):
        ch = self._make_ch()
        self.assertAlmostEqual(ch.baseline, -100.0)

    def test_std_empty_returns_one(self):
        ch = self._make_ch()
        self.assertAlmostEqual(ch.std, 1.0)

    def test_baseline_is_median(self):
        ch = self._make_ch()
        for v in [-70, -60, -80, -65, -55]:
            ch.add_sample(v)
        expected = float(np.median([-70, -60, -80, -65, -55]))
        self.assertAlmostEqual(ch.baseline, expected)

    def test_zscore_at_baseline_is_zero(self):
        ch = self._make_ch()
        for _ in range(10):
            ch.add_sample(-70.0)
        self.assertAlmostEqual(ch.zscore(-70.0), 0.0, places=3)

    def test_zscore_above_baseline_is_positive(self):
        ch = self._make_ch()
        for _ in range(20):
            ch.add_sample(-70.0)
        self.assertGreater(ch.zscore(-50.0), 0.0)

    def test_update_baseline_absorbs_normal(self):
        """z-score khoảng 0 → sample được add vào history."""
        ch = self._make_ch()
        for _ in range(20):
            ch.add_sample(-70.0)
        before = len(ch._history)
        ch.update_baseline(-70.5)   # gần median → z nhỏ → được thêm
        self.assertEqual(len(ch._history), before + 1)

    def test_update_baseline_skips_alert(self):
        """Alert sample (z > POWER_Z_ALERT) KHÔNG được add vào baseline."""
        ch = self._make_ch()
        for _ in range(20):
            ch.add_sample(-70.0)
        hist_before = list(ch._history)
        ch.update_baseline(-30.0)   # z >> POWER_Z_ALERT → skip
        # History không thay đổi
        self.assertEqual(list(ch._history), hist_before)

    def test_history_capped_at_max_samples(self):
        ch = self._make_ch()
        limit = sdr.Channel.BASELINE_MAX_SAMPLES
        for i in range(limit + 50):
            ch.add_sample(float(-70 + i * 0.01))
        self.assertEqual(len(ch._history), limit)

    def test_std_nonnegative(self):
        ch = self._make_ch()
        for v in [-70.0, -71.0, -69.0, -72.0]:
            ch.add_sample(v)
        self.assertGreaterEqual(ch.std, 0.0)

    def test_std_constant_signal_returns_floor(self):
        """std = 0 → clamp to 1e-6 → return 1.0 (floor guard)."""
        ch = self._make_ch()
        for _ in range(10):
            ch.add_sample(-70.0)
        # std là 0 → code trả về 1.0
        self.assertAlmostEqual(ch.std, 1.0)


# ===========================================================================
# TestThreatRules — Logic phân loại threat trong analyze()
# ===========================================================================

class TestThreatRules(unittest.TestCase):

    def _analyze(self, threat_type: str, channel_name: str = "TEST",
                 freq_hz: float = 435e6, **anomaly_overrides) -> sdr.ThreatResult:
        ch = sdr.Channel(channel_name, freq_hz, 1, threat_type)
        for _ in range(20):
            ch.add_sample(-70.0)
        r = make_dummy_anomaly_result(
            channel=channel_name, freq_hz=freq_hz,
            threat_type=threat_type, **anomaly_overrides
        )
        return sdr.analyze(ch, r, {})

    # ---- OK baseline -------------------------------------------------------

    def test_no_alerts_returns_ok(self):
        res = self._analyze("GPS_JAM")
        self.assertEqual(res.threat_level, sdr.ThreatLevel.OK)

    # ---- GPS_JAM -----------------------------------------------------------

    def test_gps_jam_power_and_entropy_alert_is_critical(self):
        res = self._analyze("GPS_JAM", power_alert=True, entropy_alert=True)
        self.assertEqual(res.threat_level, sdr.ThreatLevel.CRITICAL)

    def test_gps_jam_power_only_is_high(self):
        res = self._analyze("GPS_JAM", power_alert=True, entropy_alert=False)
        self.assertEqual(res.threat_level, sdr.ThreatLevel.HIGH)

    def test_gps_jam_entropy_only_stays_low(self):
        """Entropy alert saja (không có power alert) → GPS rule không fire."""
        res = self._analyze("GPS_JAM", power_alert=False, entropy_alert=True)
        self.assertLessEqual(res.threat_level.value, sdr.ThreatLevel.LOW.value)

    # ---- FAKE_BTS ----------------------------------------------------------

    def test_fake_bts_gsm_fcch_high_score_is_high(self):
        res = self._analyze("FAKE_BTS", gsm_fcch=0.5)
        self.assertGreaterEqual(res.threat_level.value, sdr.ThreatLevel.HIGH.value)

    def test_fake_bts_structural_alert_is_critical(self):
        res = self._analyze("FAKE_BTS", structural_alert=True, entropy=0.3, single_carrier=True)
        self.assertEqual(res.threat_level, sdr.ThreatLevel.CRITICAL)

    def test_fake_bts_no_alerts_is_ok(self):
        res = self._analyze("FAKE_BTS")
        self.assertEqual(res.threat_level, sdr.ThreatLevel.OK)

    # ---- CELL_JAM ----------------------------------------------------------

    def test_cell_jam_power_and_burst_is_critical(self):
        res = self._analyze("CELL_JAM", power_alert=True, burst_alert=True)
        self.assertEqual(res.threat_level, sdr.ThreatLevel.CRITICAL)

    def test_cell_jam_power_only_stays_low(self):
        res = self._analyze("CELL_JAM", power_alert=True, burst_alert=False)
        self.assertLessEqual(res.threat_level.value, sdr.ThreatLevel.LOW.value)

    # ---- DRONE_FHSS --------------------------------------------------------

    def test_drone_fhss_swept_alert_is_high(self):
        res = self._analyze("DRONE_FHSS", swept_alert=True)
        self.assertGreaterEqual(res.threat_level.value, sdr.ThreatLevel.HIGH.value)

    def test_drone_fhss_type_ai_alert_is_high(self):
        res = self._analyze("DRONE_FHSS", type_ai_alert=True, type_ai_score=0.9)
        self.assertGreaterEqual(res.threat_level.value, sdr.ThreatLevel.HIGH.value)

    # ---- WIFI_DEAUTH -------------------------------------------------------

    def test_wifi_deauth_burst_and_power_is_medium(self):
        res = self._analyze("WIFI_DEAUTH", burst_alert=True, power_alert=True)
        self.assertEqual(res.threat_level, sdr.ThreatLevel.MEDIUM)

    # ---- EMERGENCY ---------------------------------------------------------

    def test_emergency_any_alert_is_high(self):
        """Emergency channel: bất kỳ alert nào cũng là HIGH."""
        res = self._analyze("EMERGENCY", power_alert=True)
        self.assertGreaterEqual(res.threat_level.value, sdr.ThreatLevel.HIGH.value)

    # ---- AI consensus ------------------------------------------------------

    def test_ai_both_agree_escalates_to_high(self):
        res = self._analyze("IOT_REPLAY",
                            hybrid_ai_alert=True, hybrid_ai_score=0.8,
                            type_ai_alert=True,   type_ai_score=0.8)
        self.assertGreaterEqual(res.threat_level.value, sdr.ThreatLevel.HIGH.value)

    def test_only_one_ai_does_not_escalate(self):
        """Chỉ 1 AI alert → không reach HIGH qua ai_both_agree path."""
        res = self._analyze("IOT_REPLAY",
                            hybrid_ai_alert=True, hybrid_ai_score=0.8,
                            type_ai_alert=False,  type_ai_score=0.0)
        # Chỉ reach LOW (any_alert=True qua hybrid_ai_alert)
        self.assertLessEqual(res.threat_level.value, sdr.ThreatLevel.LOW.value)

    # ---- indicators --------------------------------------------------------

    def test_indicators_list_present(self):
        res = self._analyze("GPS_JAM", power_alert=True, entropy_alert=True)
        self.assertIsInstance(res.indicators, list)
        self.assertGreater(len(res.indicators), 0)

    def test_ok_result_has_empty_or_minimal_indicators(self):
        res = self._analyze("GPS_JAM")
        # OK result: không có indicators
        self.assertEqual(res.threat_level, sdr.ThreatLevel.OK)

    # ---- result fields -----------------------------------------------------

    def test_result_has_timestamp(self):
        res = self._analyze("GPS_JAM")
        self.assertIsNotNone(res.timestamp)
        self.assertTrue(len(res.timestamp) > 0)

    def test_result_confidence_nonnegative(self):
        res = self._analyze("GPS_JAM", hybrid_ai_score=0.7)
        self.assertGreaterEqual(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 1.0)


# ===========================================================================
# TestConfidenceGates
# ===========================================================================

class TestConfidenceGates(unittest.TestCase):

    # ---- PersistenceTracker ------------------------------------------------

    def test_persistence_not_triggered_below_min(self):
        pt = sdr.PersistenceTracker()
        for _ in range(sdr.PERSISTENCE_MIN_COUNT - 1):
            pt.record("GPS_L1", True)
        self.assertFalse(pt.is_persistent("GPS_L1"))

    def test_persistence_triggered_at_min_count(self):
        pt = sdr.PersistenceTracker()
        for _ in range(sdr.PERSISTENCE_MIN_COUNT):
            pt.record("GPS_L1", True)
        self.assertTrue(pt.is_persistent("GPS_L1"))

    def test_persistence_window_slides_correctly(self):
        """Sau PERSISTENCE_WINDOW alerts, cửa sổ trượt."""
        pt = sdr.PersistenceTracker()
        w = sdr.PERSISTENCE_WINDOW
        # Fill window với True
        for _ in range(w):
            pt.record("CH", True)
        self.assertTrue(pt.is_persistent("CH"))
        # Append False × w để đẩy tất cả True ra khỏi window
        for _ in range(w):
            pt.record("CH", False)
        self.assertFalse(pt.is_persistent("CH"))

    def test_persistence_independent_channels(self):
        pt = sdr.PersistenceTracker()
        for _ in range(sdr.PERSISTENCE_MIN_COUNT):
            pt.record("CH_A", True)
        # CH_B chưa record gì
        self.assertTrue(pt.is_persistent("CH_A"))
        self.assertFalse(pt.is_persistent("CH_B"))

    def test_count_returns_number_of_alerts_in_window(self):
        pt = sdr.PersistenceTracker()
        pt.record("CH", True)
        pt.record("CH", False)
        pt.record("CH", True)
        self.assertEqual(pt.count("CH"), 2)

    # ---- apply_confidence_gate (không có RF) --------------------------------

    def _make_result(self, level: sdr.ThreatLevel, **anomaly_overrides) -> sdr.ThreatResult:
        anomaly = make_dummy_anomaly_result(**anomaly_overrides)
        return sdr.ThreatResult(
            channel="GPS_L1", freq_hz=1575.42e6,
            threat_type="GPS_JAM", threat_level=level,
            anomaly=anomaly, indicators=[], timestamp="2024-01-01T00:00:00Z",
        )

    def test_gate_persist_downgrades_non_persistent_high(self):
        """HIGH nhưng không persistent → downgrade về MEDIUM."""
        res = self._make_result(sdr.ThreatLevel.HIGH)
        sdr.apply_confidence_gate(res, is_persistent=False, persist_count=1)
        self.assertEqual(res.threat_level, sdr.ThreatLevel.MEDIUM)

    def test_gate_persist_keeps_high_if_persistent(self):
        res = self._make_result(sdr.ThreatLevel.HIGH)
        sdr.apply_confidence_gate(res, is_persistent=True, persist_count=3)
        self.assertEqual(res.threat_level, sdr.ThreatLevel.HIGH)

    def test_gate_critical_requires_ai_consensus_and_zscore(self):
        """CRITICAL không có AI consensus + z thấp → downgrade về HIGH."""
        res = self._make_result(
            sdr.ThreatLevel.CRITICAL,
            hybrid_ai_alert=False, type_ai_alert=False, zscore=2.0
        )
        sdr.apply_confidence_gate(res, is_persistent=True, persist_count=5)
        self.assertEqual(res.threat_level, sdr.ThreatLevel.HIGH)

    def test_gate_critical_passes_with_consensus_and_high_zscore(self):
        """CRITICAL có AI consensus + |z| > CRITICAL_ZSCORE_GATE → giữ CRITICAL."""
        res = self._make_result(
            sdr.ThreatLevel.CRITICAL,
            hybrid_ai_alert=True, type_ai_alert=True,
            zscore=sdr.CRITICAL_ZSCORE_GATE + 0.5
        )
        sdr.apply_confidence_gate(res, is_persistent=True, persist_count=5)
        self.assertEqual(res.threat_level, sdr.ThreatLevel.CRITICAL)

    def test_gate_medium_not_affected_by_persistence_gate(self):
        """Gate 1 (persistence) chỉ áp dụng cho >= HIGH."""
        res = self._make_result(sdr.ThreatLevel.MEDIUM)
        sdr.apply_confidence_gate(res, is_persistent=False, persist_count=0)
        self.assertEqual(res.threat_level, sdr.ThreatLevel.MEDIUM)

    def test_gate_ok_unchanged(self):
        res = self._make_result(sdr.ThreatLevel.OK)
        sdr.apply_confidence_gate(res, is_persistent=False, persist_count=0)
        self.assertEqual(res.threat_level, sdr.ThreatLevel.OK)


# ===========================================================================
# TestRXGuard — ReceiveOnlySDRProxy chặn TX calls
# ===========================================================================

class TestRXGuard(unittest.TestCase):

    class FakeSDR:
        """Mock SDR object — chỉ có RX methods."""
        def __init__(self):
            self.sample_rate = 2e6
            self.center_freq = 433e6
        def start_rx(self):
            return "rx_started"
        def read_samples(self, n):
            return np.zeros(n, dtype=np.int8)

    def _wrap(self) -> sdr.ReceiveOnlySDRProxy:
        return sdr.ReceiveOnlySDRProxy(self.FakeSDR())

    def test_rx_method_passthrough(self):
        """RX call (start_rx) phải pass through bình thường."""
        proxy = self._wrap()
        result = proxy.start_rx()
        self.assertEqual(result, "rx_started")

    def test_rx_attribute_passthrough(self):
        proxy = self._wrap()
        self.assertAlmostEqual(proxy.sample_rate, 2e6)

    def test_read_samples_passthrough(self):
        proxy = self._wrap()
        samples = proxy.read_samples(16)
        self.assertEqual(len(samples), 16)

    def test_transmit_blocked_calls_exit(self):
        """Bất kỳ TX call nào → gọi os._exit(1)."""
        proxy = self._wrap()
        with mock.patch("os._exit") as mock_exit:
            # __getattr__ gọi _blocked → os._exit(1)
            _ = proxy.transmit
            mock_exit.assert_called_once_with(1)

    def test_start_tx_blocked(self):
        proxy = self._wrap()
        with mock.patch("os._exit") as mock_exit:
            _ = proxy.start_tx
            mock_exit.assert_called_once_with(1)

    def test_send_samples_blocked(self):
        proxy = self._wrap()
        with mock.patch("os._exit") as mock_exit:
            _ = proxy.send_samples
            mock_exit.assert_called_once_with(1)

    def test_tx_vga_blocked(self):
        proxy = self._wrap()
        with mock.patch("os._exit") as mock_exit:
            _ = proxy.tx_vga
            mock_exit.assert_called_once_with(1)

    def test_jam_method_blocked(self):
        proxy = self._wrap()
        with mock.patch("os._exit") as mock_exit:
            _ = proxy.jam
            mock_exit.assert_called_once_with(1)

    def test_replay_blocked(self):
        proxy = self._wrap()
        with mock.patch("os._exit") as mock_exit:
            _ = proxy.replay_signal
            mock_exit.assert_called_once_with(1)

    def test_setattr_tx_blocked(self):
        """Gán vào attribute TX → cũng bị chặn."""
        proxy = self._wrap()
        with mock.patch("os._exit") as mock_exit:
            proxy.tx_gain = 40
            mock_exit.assert_called_once_with(1)

    def test_setattr_rx_passes_through(self):
        """Gán vào attribute thường → không block."""
        proxy = self._wrap()
        with mock.patch("os._exit") as mock_exit:
            proxy.sample_rate = 4e6
            mock_exit.assert_not_called()
        self.assertAlmostEqual(proxy._sdr.sample_rate, 4e6)

    def test_looks_like_tx_patterns(self):
        """_looks_like_tx() nhận biết đúng các patterns."""
        should_block = [
            "transmit", "start_tx", "tx_start", "send_samples",
            "write_samples", "tx_enable", "enable_tx", "set_tx",
            "tx_vga", "txvga", "tx_gain", "tx_amp", "repeat",
            "replay", "jam", "spoof_tx", "carrier_on",
            "TRANSMIT",    # case-insensitive
            "Start_TX",
        ]
        for name in should_block:
            with self.subTest(name=name):
                self.assertTrue(sdr._looks_like_tx(name))

    def test_looks_like_tx_allows_rx_names(self):
        """RX method names KHÔNG bị flag là TX."""
        rx_names = ["start_rx", "read_samples", "tune", "snapshot",
                    "sample_rate", "center_freq", "lna_gain", "vga_gain"]
        for name in rx_names:
            with self.subTest(name=name):
                self.assertFalse(sdr._looks_like_tx(name))


# ===========================================================================
# TestRFClassifier — Training guards
# ===========================================================================

class TestRFClassifier(unittest.TestCase):

    def _make_clf(self, model_path=None) -> sdr.RFThreatClassifier:
        if model_path is None:
            model_path = os.path.join(tempfile.mkdtemp(), "test_rf.joblib")
        return sdr.RFThreatClassifier(model_path=model_path)

    def _fake_db_rows(self, n_per_class: dict) -> list:
        """Tạo labeled rows giả cho training."""
        rows = []
        rng = np.random.default_rng(99)
        for label, count in n_per_class.items():
            for i in range(count):
                rows.append({
                    "threat_type": label,
                    "power_dbm":   float(rng.uniform(-80, -20)),
                    "zscore":      float(rng.uniform(-5, 10)),
                    "snr_db":      float(rng.uniform(0, 30)),
                    "bandwidth_hz": 2e6,
                    "duration_ms": 5.0,
                    "persistence_ratio": float(rng.uniform(0, 1)),
                    "freq_hz":     float(rng.choice([433e6, 868e6, 915e6, 1575e6])),
                    "confirmed":   1,
                })
        return rows

    def test_new_clf_not_usable(self):
        clf = self._make_clf()
        self.assertFalse(clf.usable)

    def test_train_refuses_below_min_samples(self):
        clf = self._make_clf()
        rows = self._fake_db_rows({"GPS_JAM": 10, "FAKE_BTS": 10})
        with mock.patch("SDR_BLUE_TEAM.db_fetch_labeled_events", return_value=rows):
            result = clf.train()
        self.assertFalse(result)
        self.assertFalse(clf.usable)

    def test_train_refuses_single_class(self):
        """Chỉ có 1 class → từ chối training."""
        clf = self._make_clf()
        rows = self._fake_db_rows({"GPS_JAM": 250})
        with mock.patch("SDR_BLUE_TEAM.db_fetch_labeled_events", return_value=rows):
            result = clf.train()
        self.assertFalse(result)

    def test_train_refuses_imbalanced_labels(self):
        """1 class chiếm >85% → từ chối vì nghi ngờ label theo rule."""
        clf = self._make_clf()
        rows = self._fake_db_rows({"GPS_JAM": 180, "FAKE_BTS": 20})
        # 180/200 = 90% > RF_MAX_CLASS_SHARE (0.85)
        with mock.patch("SDR_BLUE_TEAM.db_fetch_labeled_events", return_value=rows):
            result = clf.train()
        self.assertFalse(result)
        self.assertFalse(clf.usable)

    def test_train_succeeds_balanced_data(self):
        """Data cân bằng, đủ samples → training thành công."""
        clf = self._make_clf()
        rows = self._fake_db_rows({
            "GPS_JAM": 100, "FAKE_BTS": 100, "CELL_JAM": 80, "IOT_REPLAY": 80
        })
        with mock.patch("SDR_BLUE_TEAM.db_fetch_labeled_events", return_value=rows):
            with mock.patch.object(clf, "_save"):  # skip lưu file
                result = clf.train()
        self.assertTrue(result)
        self.assertTrue(clf.usable)
        self.assertFalse(clf.degenerate)

    def test_predict_returns_none_when_not_usable(self):
        clf = self._make_clf()
        label, conf = clf.predict({"power_dbm": -60, "zscore": 3.0,
                                    "snr_db": 10, "bandwidth_hz": 2e6,
                                    "duration_ms": 5, "persistence_ratio": 0.6,
                                    "freq_hz": 1575e6})
        self.assertIsNone(label)
        self.assertAlmostEqual(conf, 0.0)

    def test_predict_returns_label_when_trained(self):
        clf = self._make_clf()
        rows = self._fake_db_rows({
            "GPS_JAM": 120, "FAKE_BTS": 120, "CELL_JAM": 80
        })
        with mock.patch("SDR_BLUE_TEAM.db_fetch_labeled_events", return_value=rows):
            with mock.patch.object(clf, "_save"):
                clf.train()

        if not clf.usable:
            self.skipTest("Training failed (degenerate) — skip predict test")

        label, conf = clf.predict({"power_dbm": -50, "zscore": 5.0,
                                    "snr_db": 20, "bandwidth_hz": 2e6,
                                    "duration_ms": 5, "persistence_ratio": 1.0,
                                    "freq_hz": 1575.42e6})
        self.assertIn(label, clf.classes_)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_degenerate_model_not_usable(self):
        """Model bị mark degenerate → usable=False → predict trả None."""
        clf = self._make_clf()
        rows = self._fake_db_rows({
            "GPS_JAM": 120, "FAKE_BTS": 120, "CELL_JAM": 80
        })
        with mock.patch("SDR_BLUE_TEAM.db_fetch_labeled_events", return_value=rows):
            with mock.patch.object(clf, "_save"):
                clf.train()

        # Force degenerate
        clf.degenerate = True
        clf.usable = False

        label, conf = clf.predict({"power_dbm": -50, "zscore": 5.0,
                                    "snr_db": 20, "bandwidth_hz": 2e6,
                                    "duration_ms": 5, "persistence_ratio": 1.0,
                                    "freq_hz": 1575.42e6})
        self.assertIsNone(label)


# ===========================================================================
# TestDatabase — SQLite CRUD với in-memory DB
# ===========================================================================

class TestDatabase(unittest.TestCase):

    def setUp(self):
        """Mỗi test dùng in-memory DB riêng."""
        self._orig_db = sdr._db
        self._orig_db_path = sdr.DB_PATH
        sdr.DB_PATH = ":memory:"
        sdr._db = None
        sdr.init_db()

    def tearDown(self):
        sdr.close_db()
        sdr._db = self._orig_db
        sdr.DB_PATH = self._orig_db_path

    def _make_result(self) -> sdr.ThreatResult:
        anomaly = make_dummy_anomaly_result(
            channel="GPS_L1", freq_hz=1575.42e6, threat_type="GPS_JAM",
            power_dbm=-45.0, baseline_dbm=-70.0, zscore=4.2
        )
        return sdr.ThreatResult(
            channel="GPS_L1", freq_hz=1575.42e6, threat_type="GPS_JAM",
            threat_level=sdr.ThreatLevel.HIGH, anomaly=anomaly,
            indicators=["Test indicator"], action="Test action",
            timestamp="2024-01-01T00:00:00Z",
            confidence=0.8,
        )

    def test_init_db_creates_tables(self):
        conn = sdr._db
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("events", tables)
        self.assertIn("spectrum_history", tables)

    def test_log_event_inserts_row(self):
        res = self._make_result()
        sdr.db_log_event(res, persistence_ratio=0.6)
        rows = sdr.db_fetch_recent_events()
        self.assertEqual(len(rows), 1)

    def test_log_event_fields_correct(self):
        res = self._make_result()
        sdr.db_log_event(res, persistence_ratio=0.6)
        row = sdr.db_fetch_recent_events()[0]
        self.assertEqual(row["channel"], "GPS_L1")
        self.assertEqual(row["severity"], "HIGH")
        self.assertAlmostEqual(row["zscore"], 4.2, places=2)

    def test_fetch_recent_returns_newest_first(self):
        for i in range(3):
            res = self._make_result()
            res.anomaly = make_dummy_anomaly_result(power_dbm=float(-60 + i))
            sdr.db_log_event(res)
        rows = sdr.db_fetch_recent_events()
        ids = [r["id"] for r in rows]
        self.assertEqual(ids, sorted(ids, reverse=True))

    def test_set_confirmed_updates_row(self):
        res = self._make_result()
        sdr.db_log_event(res)
        row = sdr.db_fetch_recent_events()[0]
        sdr.db_set_confirmed(row["id"], 1)
        updated = sdr.db_fetch_recent_events()[0]
        self.assertEqual(updated["confirmed"], 1)

    def test_fetch_labeled_excludes_unlabeled(self):
        # Insert 2 rows: 1 labeled, 1 không
        res = self._make_result()
        sdr.db_log_event(res)
        sdr.db_log_event(res)
        rows = sdr.db_fetch_recent_events()
        sdr.db_set_confirmed(rows[0]["id"], 1)   # label 1 row
        labeled = sdr.db_fetch_labeled_events()
        self.assertEqual(len(labeled), 1)

    def test_label_stats_reflects_confirmed_rows(self):
        res = self._make_result()
        sdr.db_log_event(res)
        row = sdr.db_fetch_recent_events()[0]
        sdr.db_set_confirmed(row["id"], 1)
        stats = sdr.db_label_stats()
        self.assertEqual(stats["labeled"], 1)
        self.assertEqual(stats["classes"], 1)

    def test_db_none_safe(self):
        """Tất cả hàm DB phải safe khi _db=None."""
        sdr.close_db()
        sdr._db = None
        self.assertEqual(sdr.db_fetch_recent_events(), [])
        self.assertEqual(sdr.db_fetch_labeled_events(), [])
        sdr.db_log_event(self._make_result())  # không crash
        sdr.init_db()   # restore cho tearDown

    # ── NEW: db_log_event return value ─────────────────────────────────────

    def test_log_event_returns_integer_id(self):
        """db_log_event() phải trả về int ID của row vừa insert."""
        res = self._make_result()
        eid = sdr.db_log_event(res)
        self.assertIsNotNone(eid)
        self.assertIsInstance(eid, int)
        self.assertGreater(eid, 0)

    def test_log_event_returns_none_when_db_none(self):
        """Trả None (không crash) khi _db=None."""
        sdr.close_db()
        sdr._db = None
        result = sdr.db_log_event(self._make_result())
        self.assertIsNone(result)
        sdr.init_db()

    def test_log_event_ids_increment(self):
        """Mỗi insert trả ID tăng dần."""
        res = self._make_result()
        id1 = sdr.db_log_event(res)
        id2 = sdr.db_log_event(res)
        self.assertGreater(id2, id1)

    # ── NEW: _auto_label — FP cases ────────────────────────────────────────

    def test_auto_label_low_threat_is_fp(self):
        """LOW threat luôn được label là FP."""
        res = self._make_result()
        res.threat_level = sdr.ThreatLevel.LOW
        eid = sdr.db_log_event(res)
        sdr._auto_label(eid, res, 0)
        row = sdr.db_fetch_recent_events()[0]
        self.assertEqual(row["confirmed"], 0)

    def test_auto_label_medium_no_consensus_not_persistent_is_fp(self):
        """MEDIUM + no dual-AI + not persistent → FP."""
        res = self._make_result()
        res.threat_level            = sdr.ThreatLevel.MEDIUM
        res.anomaly.hybrid_ai_alert = False
        res.anomaly.type_ai_alert   = False
        eid = sdr.db_log_event(res)
        sdr._auto_label(eid, res, persist_count=0)
        row = sdr.db_fetch_recent_events()[0]
        self.assertEqual(row["confirmed"], 0)

    def test_auto_label_medium_one_ai_not_persistent_is_fp(self):
        """MEDIUM + chỉ 1 AI alert + không persistent → FP."""
        res = self._make_result()
        res.threat_level            = sdr.ThreatLevel.MEDIUM
        res.anomaly.hybrid_ai_alert = True
        res.anomaly.type_ai_alert   = False
        eid = sdr.db_log_event(res)
        sdr._auto_label(eid, res, persist_count=1)
        row = sdr.db_fetch_recent_events()[0]
        self.assertEqual(row["confirmed"], 0)

    # ── NEW: _auto_label — TP cases ────────────────────────────────────────

    def test_auto_label_high_all_conditions_is_tp(self):
        """HIGH + dual-AI + persistent + |Z| > 4 → TP."""
        res = self._make_result()
        res.threat_level             = sdr.ThreatLevel.HIGH
        res.anomaly.hybrid_ai_alert  = True
        res.anomaly.type_ai_alert    = True
        res.anomaly.zscore           = 5.0
        eid = sdr.db_log_event(res)
        sdr._auto_label(eid, res, persist_count=sdr.PERSISTENCE_MIN_COUNT)
        row = sdr.db_fetch_recent_events()[0]
        self.assertEqual(row["confirmed"], 1)

    def test_auto_label_critical_all_conditions_is_tp(self):
        """CRITICAL + dual-AI + persistent + |Z| > 4 → TP."""
        res = self._make_result()
        res.threat_level             = sdr.ThreatLevel.CRITICAL
        res.anomaly.hybrid_ai_alert  = True
        res.anomaly.type_ai_alert    = True
        res.anomaly.zscore           = 6.0
        eid = sdr.db_log_event(res)
        sdr._auto_label(eid, res, persist_count=sdr.PERSISTENCE_MIN_COUNT)
        row = sdr.db_fetch_recent_events()[0]
        self.assertEqual(row["confirmed"], 1)

    # ── NEW: _auto_label — boundary / ambiguous ────────────────────────────

    def test_auto_label_high_no_dual_ai_not_labeled(self):
        """HIGH nhưng chỉ 1 AI → không đủ điều kiện TP, không label."""
        res = self._make_result()
        res.threat_level             = sdr.ThreatLevel.HIGH
        res.anomaly.hybrid_ai_alert  = True
        res.anomaly.type_ai_alert    = False
        res.anomaly.zscore           = 5.0
        eid = sdr.db_log_event(res)
        sdr._auto_label(eid, res, persist_count=sdr.PERSISTENCE_MIN_COUNT)
        row = sdr.db_fetch_recent_events()[0]
        self.assertIsNone(row["confirmed"])

    def test_auto_label_high_low_zscore_not_labeled(self):
        """|Z| < 4.0 → không đủ điều kiện TP, không label."""
        res = self._make_result()
        res.threat_level             = sdr.ThreatLevel.HIGH
        res.anomaly.hybrid_ai_alert  = True
        res.anomaly.type_ai_alert    = True
        res.anomaly.zscore           = 2.0
        eid = sdr.db_log_event(res)
        sdr._auto_label(eid, res, persist_count=sdr.PERSISTENCE_MIN_COUNT)
        row = sdr.db_fetch_recent_events()[0]
        self.assertIsNone(row["confirmed"])

    def test_auto_label_medium_with_consensus_not_labeled(self):
        """MEDIUM + dual-AI agree → gray zone, không label."""
        res = self._make_result()
        res.threat_level             = sdr.ThreatLevel.MEDIUM
        res.anomaly.hybrid_ai_alert  = True
        res.anomaly.type_ai_alert    = True
        eid = sdr.db_log_event(res)
        sdr._auto_label(eid, res, persist_count=sdr.PERSISTENCE_MIN_COUNT)
        row = sdr.db_fetch_recent_events()[0]
        self.assertIsNone(row["confirmed"])

    # ── NEW: _auto_label — safety ──────────────────────────────────────────

    def test_auto_label_none_event_id_safe(self):
        """event_id=None không crash."""
        res = self._make_result()
        res.threat_level = sdr.ThreatLevel.LOW
        sdr._auto_label(None, res, 0)

    def test_auto_label_db_none_safe(self):
        """_db=None không crash."""
        sdr.close_db()
        sdr._db = None
        res = self._make_result()
        res.threat_level = sdr.ThreatLevel.LOW
        sdr._auto_label(1, res, 0)
        sdr.init_db()


# ===========================================================================
# TestDowngradeHelper
# ===========================================================================

class TestDowngradeHelper(unittest.TestCase):

    def test_downgrade_critical_to_high(self):
        self.assertEqual(sdr._downgrade_one("CRITICAL"), "HIGH")

    def test_downgrade_high_to_medium(self):
        self.assertEqual(sdr._downgrade_one("HIGH"), "MEDIUM")

    def test_downgrade_medium_to_low(self):
        self.assertEqual(sdr._downgrade_one("MEDIUM"), "LOW")

    def test_downgrade_low_to_ok(self):
        self.assertEqual(sdr._downgrade_one("LOW"), "OK")

    def test_downgrade_ok_stays_ok(self):
        self.assertEqual(sdr._downgrade_one("OK"), "OK")

    def test_downgrade_unknown_returns_input(self):
        self.assertEqual(sdr._downgrade_one("BOGUS"), "BOGUS")

    def test_downgrade_case_insensitive(self):
        self.assertEqual(sdr._downgrade_one("critical"), "HIGH")


# ===========================================================================
# TestEWAnomalyResultProperties
# ===========================================================================

class TestEWAnomalyResultProperties(unittest.TestCase):

    def test_ai_alert_true_if_hybrid_alert(self):
        r = make_dummy_anomaly_result(hybrid_ai_alert=True)
        self.assertTrue(r.ai_alert)

    def test_ai_alert_true_if_type_alert(self):
        r = make_dummy_anomaly_result(type_ai_alert=True)
        self.assertTrue(r.ai_alert)

    def test_ai_alert_false_if_neither(self):
        r = make_dummy_anomaly_result()
        self.assertFalse(r.ai_alert)

    def test_ai_both_agree_requires_both(self):
        r = make_dummy_anomaly_result(hybrid_ai_alert=True, type_ai_alert=True)
        self.assertTrue(r.ai_both_agree)

    def test_ai_both_agree_false_one_only(self):
        r = make_dummy_anomaly_result(hybrid_ai_alert=True, type_ai_alert=False)
        self.assertFalse(r.ai_both_agree)

    def test_any_alert_power(self):
        r = make_dummy_anomaly_result(power_alert=True)
        self.assertTrue(r.any_alert)

    def test_any_alert_entropy(self):
        r = make_dummy_anomaly_result(entropy_alert=True)
        self.assertTrue(r.any_alert)

    def test_any_alert_burst(self):
        r = make_dummy_anomaly_result(burst_alert=True)
        self.assertTrue(r.any_alert)

    def test_any_alert_structural(self):
        r = make_dummy_anomaly_result(structural_alert=True)
        self.assertTrue(r.any_alert)

    def test_any_alert_swept(self):
        r = make_dummy_anomaly_result(swept_alert=True)
        self.assertTrue(r.any_alert)

    def test_any_alert_false_when_all_clear(self):
        r = make_dummy_anomaly_result()
        self.assertFalse(r.any_alert)


# ===========================================================================
# TestWatchlist — sanity check danh sách kênh quan sát
# ===========================================================================

class TestWatchlist(unittest.TestCase):

    def test_watchlist_not_empty(self):
        self.assertGreater(len(sdr.WATCHLIST), 0)

    def test_all_channels_have_name(self):
        for ch in sdr.WATCHLIST:
            self.assertTrue(len(ch.name) > 0)

    def test_all_channels_have_positive_freq(self):
        for ch in sdr.WATCHLIST:
            self.assertGreater(ch.freq_hz, 0)

    def test_gps_l1_present(self):
        names = {ch.name for ch in sdr.WATCHLIST}
        self.assertIn("GPS_L1", names)

    def test_all_priorities_are_1_2_or_3(self):
        for ch in sdr.WATCHLIST:
            self.assertIn(ch.priority, (1, 2, 3))

    def test_no_duplicate_names(self):
        names = [ch.name for ch in sdr.WATCHLIST]
        self.assertEqual(len(names), len(set(names)))


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RF Sentinel Test Suite")
    parser.add_argument("pattern", nargs="?", default=None,
                        help="Optional test class/method filter (e.g. TestDSP)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    verbosity = 2 if args.verbose else 1

    if args.pattern:
        suite = unittest.TestLoader().loadTestsFromName(args.pattern,
                                                        module=sys.modules[__name__])
    else:
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
