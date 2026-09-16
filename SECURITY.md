# Security Policy

## ⚠️ Important: Receive-Only Tool

RF Sentinel is a **passive, receive-only** RF monitoring tool. It is designed exclusively for **defensive spectrum monitoring** and anomaly detection.

- ✅ It **receives** RF signals for analysis
- ❌ It **never transmits** — a hardware-level TX guard terminates the process immediately on any emission attempt
- ✅ It is intended for **lawful spectrum monitoring** only

---

## Supported Versions

| Version | Supported |
|---------|-----------|
| v15.x (latest) | ✅ Active support |
| < v15 | ❌ No support |

---

## Reporting a Vulnerability

If you discover a security vulnerability in RF Sentinel, please **do not open a public issue**.

### How to Report

1. **Open a private security advisory** via GitHub:
   - Go to the [Security tab](../../security/advisories/new)
   - Click "Report a vulnerability"

2. **Or contact the maintainer directly** via GitHub profile

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

| Stage | Timeline |
|-------|----------|
| Acknowledgement | Within 48 hours |
| Initial assessment | Within 7 days |
| Fix/patch | Within 30 days (critical: ASAP) |
| Public disclosure | After fix is released |

---

## Scope

### In Scope 🔴
- Code execution vulnerabilities in the DSP pipeline
- SQL injection in the SQLite logging module
- Web UI (Flask) security issues (XSS, CSRF, etc.)
- ML model poisoning attacks
- Unauthorized access to the web dashboard

### Out of Scope 🟢
- Theoretical RF interference (tool is receive-only)
- Issues requiring physical access to hardware
- Social engineering attacks
- Third-party library vulnerabilities (report to those projects directly)

---

## Legal Notice

RF Sentinel is built for **lawful, authorized use only**. Users are responsible for complying with all applicable laws and regulations regarding radio frequency monitoring in their jurisdiction.

**Misuse of this tool may violate local telecommunications laws.**
