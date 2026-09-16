# Contributing to RF Sentinel

First off, thanks for taking the time to contribute! 🎉

RF Sentinel is a **receive-only** passive RF monitoring tool. All contributions must respect this core principle — no transmission features will ever be accepted.

---

## 🚀 How to Contribute

### 1. Reporting Bugs

Before submitting a bug report:
- Check the [existing issues](../../issues) to avoid duplicates
- Make sure you're running the latest version

When submitting a bug report, include:
- Your OS and Python version
- HackRF firmware version (if applicable)
- Steps to reproduce
- Expected vs actual behavior
- Logs from `rf_logs/` directory

### 2. Suggesting Features

Open an issue with the `[Feature Request]` prefix. Include:
- What problem does this solve?
- How would it work?
- Any relevant RF/DSP theory

### 3. Submitting Code

```bash
# 1. Fork the repo
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/Receive-only-SDR-anomaly-detection-tool.git

# 3. Create a branch
git checkout -b feature/your-feature-name

# 4. Make your changes
# 5. Run tests
python3 test_sdr_sentinel.py -v

# 6. Commit
git commit -m "feat: your feature description"

# 7. Push and open PR
git push origin feature/your-feature-name
```

---

## 📋 Contribution Guidelines

### Code Style
- Follow PEP 8
- Use type hints (`def foo(x: int) -> str:`)
- Add docstrings to all functions/classes
- Keep functions focused and small

### Testing
- Add tests for new features in `test_sdr_sentinel.py`
- All existing tests must still pass
- Test with and without optional dependencies (lightgbm, hmmlearn)

### Commit Messages
Use conventional commits:
```
feat: add new anomaly detection algorithm
fix: correct frequency calibration offset
docs: update installation guide
test: add unit tests for DTW module
refactor: simplify ML pipeline
```

---

## ⚠️ What We Won't Accept

- ❌ Any active transmission (TX) features
- ❌ Features that require illegal spectrum use
- ❌ Code without tests
- ❌ Breaking changes without discussion first
- ❌ Dependencies that don't have graceful fallbacks

---

## 🧪 Running Tests

```bash
# Full test suite
python3 test_sdr_sentinel.py -v

# Specific test class
python3 test_sdr_sentinel.py -v TestRFAnalysis

# With optional packages
pip install lightgbm hmmlearn scipy
python3 test_sdr_sentinel.py -v
```

---

## 💙 Thank You!

Every contribution, no matter how small, helps make RF Sentinel better for the whole community. 

If you found this project useful, please consider giving it a ⭐ star on GitHub!
