# RF SENTINEL v15 - REGULATORY COMPLIANCE & LEGAL FRAMEWORK

===========================================================================
LEGAL COMPLIANCE, TERMS OF SERVICE (ToS), AND ANTI-ABUSE CONSTITUTION
===========================================================================
Last Updated: 2026-09-13
Author: Software Owner / Repository Maintainer

This document governs all usage, modification, distribution, compilation,
and deployment of the RF SENTINEL v15 software core. By downloading or 
using this source code, you unconditionally agree to this framework.

---------------------------------------------------------------------------
1. LICENSE & ANTI-ABUSE POLICY (DUAL-LICENSING)
---------------------------------------------------------------------------
This project is licensed under the strict GNU Affero General Public License 
v3.0 (AGPL-3.0). This ensures that the core signal processing technology 
remains open and cannot be turned into a closed-source surveillance service
or software-as-a-service (SaaS) without contributing back to the public.

### How to avoid "Source Code Contamination" (For Lawful Users):
If you want to integrate this SDR core into your private system or any 
commercial project without infecting your own stack with AGPL copyleft, 
you MUST contact me directly for a custom proprietary license exception.

To get approved for an exception, open a private channel and provide:
1. Your full identity, company name, or accredited organization.
2. Your exact technical use-case, operating frequencies, and architecture.

I will securely store our chat history as official, legally binding 
proof of authorization and customized software clearance.

### The Absolute Red Line:
Commercial or private clearance will NEVER be granted under any circumstance 
if your deployment architecture involves mounting this software onto:
- Portable devices, hand-held scanners, or tactical field laptops.
- Covert hardware designed for active or stealthy signal eavesdropping.
- IMSI catchers, rogue cellular base stations, or tracking cells.
- Electronic warfare nodes designed for broad spectrum disruption.

Keep the software static, keep it academic, and keep it strictly lawful.

---------------------------------------------------------------------------
2. TERMS OF SERVICE (ToS) & CODE OF CONDUCT
---------------------------------------------------------------------------
### Article 2.1: Intended Educational Purpose
RF SENTINEL v15 is built exclusively as a passive Software Defined Radio 
(SDR) analyzer for academic research, digital signal processing (DSP) 
education, and licensed amateur radio spectral observation.

### Article 2.2: Prohibition of Transmission Modding
The codebase is structurally passive (Receive-Only). Any attempt to inject 
active transmit (TX) routines, modulation loops, or signal replay loops 
using this codebase as a baseline is a direct breach of this ToS.

### Article 2.3: Data Privacy and Interception Limits
Users are strictly prohibited from utilizing the IQ processing pipelines 
to extract payload data, demodulate unencrypted voice streams of local 
emergency services, or systematically log uncoordinated commercial traffic.

### Article 2.4: Immediate Revocation of Rights
Any violation of these terms triggers an immediate, non-negotiable 
termination of your right to use, modify, or host this software. Your 
repository forks will be targeted for copyright enforcement action.

---------------------------------------------------------------------------
3. SPECIFIC STATUTORY WARNINGS & APPLICABLE LAWS
---------------------------------------------------------------------------
Deploying or modifying SDR equipment outside of a strict Faraday cage may 
violate local, national, and international telecommunications laws. 

### Section 3.1: United States Federal Laws
- Federal Communications Act (47 U.S.C. § 301): Prohibits unauthorized 
  radio operations and transmissions without valid FCC licenses.
- Electronic Communications Privacy Act (18 U.S.C. § 2511): Wiretap Act 
  violations apply to any willful interception of radio communications 
  not readily accessible to the general public (e.g., encrypted/scrambled).

### Section 3.2: International Radio Regulations (ITU)
- ITU Radio Regulations (Article 18): Licenses for stations must be issued 
  by the respective sovereign government. No pirate operations allowed.
- ITU Article 45: All stations are forbidden from causing harmful 
  interference to radio services or communications of emergency services.

### Section 3.3: Vietnam Telecommunications & Radio Laws
- Luật Tần số Vô tuyến điện (Law on Radio Frequencies No. 42/2009/QH12): 
  Strictly regulates the use of any device interacting with the national 
  radio frequency spectrum. Unauthorized spectrum scanning or tracking 
  of critical infrastructure frequencies is strictly illegal.
- Điều 12 (Luật Tần số VTĐ): Nghiêm cấm hành vi sử dụng thiết bị vô tuyến 
  gây nhiễu có hại cho mạng thông tin vô tuyến điện quốc gia, an ninh, 
  quốc phòng, hoặc hệ thống thông tin cứu nạn, cứu hộ.
- Bộ luật Hình sự (Criminal Code No. 100/2015/QH13 - Điều 287): Tội cản trở 
  hoặc gây rối luận hoạt động của mạng máy tính, mạng viễn thông, phương 
  tiện điện tử công cộng có thể bị xử lý hình sự phạt tù đến 12 năm.

---------------------------------------------------------------------------
4. RISK ACKNOWLEDGMENT & END-USER WARRANTY VOID
---------------------------------------------------------------------------
### Clause 4.1: Pure As-Is Distribution
The author provides this codebase with no warranties of any kind. You 
acknowledge that radio wave analysis involves physical hardware that can 
damage host machine USB controllers if uncalibrated.

### Clause 4.2: Zero Indemnification
The copyright holder shall not be held liable for any legal summons, 
regulatory fines, asset confiscations, or criminal investigations 
resulting from your operational negligence or illegal deployment.

---------------------------------------------------------------------------
5. INTELLECTUAL PROPERTY & FORK ENFORCEMENT
---------------------------------------------------------------------------
### Clause 5.1: Attribution Integrity
All repository forks, derivative projects, or packaged containers must 
prominently display the original author copyright notice and maintain 
a direct hyperlink to the original upstream repository on GitHub.

### Clause 5.2: Automated Compliance Scanning
The upstream project utilizes automated scrapers to monitor public code 
forks. If a fork is discovered stripping these legal notices, removing 
the AGPL-3.0 headers, or packaging the core into an unapproved closed 
commercial application, a DMCA takedown notice will be filed with GitHub 
legal operations automatically without any prior friendly warning.

---------------------------------------------------------------------------
6. HARDWARE COMPATIBILITY & ISOLATION MANDATE
---------------------------------------------------------------------------
### Directive 6.1: Passive SDR Dongle Scope
This engine is structurally tuned for passive devices like RTL-SDR, 
Airspy, and the receive pipeline of HackRF One. It is designed to evaluate 
environmental signals, not to generate active carrier waves.

### Directive 6.2: Lab Testing Isolation
If you are testing the boundaries of the Dual-AI algorithm or using custom 
high-gain amplifiers, you are legally obligated to execute all processing 
inside a shielded RF enclosure (Faraday box) to guarantee zero leakage 
into public airwaves.

---------------------------------------------------------------------------
7. EMERGENCY PROTOCOLS & GOVERNMENT COOPERATION
---------------------------------------------------------------------------
### Protocol 7.1: Zero Tolerance for Critical Infrastructure Sniffing
The software logs spectral anomalies. If you use this software to map, 
benchmark, or evaluate the active defense systems of airports, military 
outposts, or government communication relays, you act entirely on your 
own legal peril.

### Protocol 7.2: Public Safety Fallback
If any government law enforcement agency requests verification of the open 
nature of this tool, the repository maintainer will fully cooperate by 
demontaining that the codebase is completely transparent, auditable, and 
contains no hidden active signals generation mechanisms or cyber weapons.

---------------------------------------------------------------------------
8. SEVERABILITY AND ENTIRE AGREEMENT
---------------------------------------------------------------------------
If any section, clause, or provision of these Terms of Service is found 
to be invalid or unenforceable under local telecommunications laws by a 
competest court, that specific provision shall be severed, and the 
remaining architecture of the ToS shall remain in full legal force.

By proceeding to compile, execute, or deploy the code blocks contained within 
this repository, you execute an irrevocable digital signature affirming 
that you have read, understood, and agreed to be bound by every boundary, 
regulation, and strict penal warning defined in this charter.

===========================================================================
                      [ END OF LEGAL REGULATORY FILE ]
===========================================================================

---

## 🚀 Quick Start (For Authorized & Lawful Use)

1. **Clone the repository legally:**
   ```bash
   git clone https://github.com
   ```
2. **Review the AGPL-3.0 License file** included in the root directory.
3. **Verify your hardware configuration** is set to Receive-Only (RX).
4. **Run the baseline training** in a controlled static environment.

Have questions regarding custom corporate licensing clearance? 
Shoot a message directly to the maintainer via the designated channel.
### 🏛️ Special Sovereign & Government Exemption (Confidential Pipeline):
If you represent a sovereign state entity, national security agency, or defense research laboratory and wish to deploy this core within classified or closed networks without triggering the AGPL-3.0 source contamination:

1. **Secure Contact:** You must establish a private, verified communication channel with the maintainer.
2. **Sovereign Proof:** Official, encrypted credentials or institutional proof verification will be required to validate your agency's authority.
3. **Strict NDA Commitment:** The maintainer guarantees **100% confidentiality**. No government identity, network architecture details, or chat history under this specific pipeline will ever be leaked, disclosed, or pushed to public channels. 

A dedicated, isolated commercial/sovereign license waiver will be issued privately upon successful verification.

