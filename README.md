# Q-SOC “Starlight” – Quantum-Enhanced Threat-Hunting Co-Pilot

> **Author:** Md Tanvir Rana
> **Repository:** [q-soc-starlight](https://github.com/tanviiiiir-r/q-soc-starlight)
> **Last Updated:** 2025-05-28

---

## 🧠 Project Overview

**Q-SOC “Starlight”** is a research-grade, post-quantum-ready Security Operations Center (SOC) simulation platform designed to enhance cyber threat detection through quantum machine learning and automated incident response. The project blends **Wazuh SIEM logs**, **quantum kernel PCA**, **LLM-based incident narration**, and **Dilithium-signed SOAR playbooks**, all deployed in a **hub-and-spoke Zero Trust VNet** on Microsoft Azure.

This is part of a larger Erasmus/DAAD portfolio demonstrating future-proof cybersecurity innovation for NIS-2 compliance.

---

## ❓ Problem Statement

Traditional SOC systems struggle with:

* Detecting **sparse kill-chains** that don't fit existing signatures
* Ensuring **tamper-resistant SOAR workflows**
* Describing complex security events in **natural language**
* Preparing for the era of **quantum computing threats**

---

## ✅ Proposed Solution

This project provides:

1. **Quantum Kernel PCA** on log data to identify anomalies missed by classical systems
2. **LLM-based Incident Narrator** using LLaMA-2 to auto-generate ATT\&CK-aligned incident stories
3. **SOAR Playbooks** signed with post-quantum Dilithium keys to ensure tamper-proof automation
4. **Zero Trust Hub-Spoke Architecture** to ensure strict East-West and North-South isolation
5. **Cost-effective simulation** using Azure free and consumption-tier resources with real QPU jobs on Quantinuum/IonQ

---

## 🧱 Project Architecture

* **Log Pipeline**: Wazuh logs → Azure Storage (CMK encrypted)
* **Quantum Pipeline**: Angle-encoded 64-dim vectors → Qiskit → Quantum Kernel PCA → Elbow analysis
* **Narration Layer**: LLaMA-2 ONNX fine-tuned with MITRE data → `mitre_prompts.jsonl`
* **Automation Layer**: Logic App triggered via HTTP, signed by Dilithium script
* **Security Enforcement**: Zero Trust VNet + NSGs + CMK + Private Endpoints

---

## 🧪 Phased Implementation

### 📦 Phase 1: Log Extraction

* Exported 10k Wazuh logs
* Labeled manually using regex + alert levels
* Stored in CMK-encrypted private Blob

### 🔍 Phase 2: Feature Encoding

* Angle-encoded logs into 64-dim quantum feature vectors
* Validated with Qiskit simulator
* Saved as `quantum_encoded_features.csv`

### 🧬 Phase 3: Quantum Kernel PCA

* Ran Kernel PCA with quantum kernel simulator
* Elbow-curve determined optimal 3D space
* Output `quantum_pca_projection.png`

### ⚛️ Phase 4: Hardware Validation

* Submitted ≤10 jobs to Quantinuum and IonQ via Azure Quantum
* Measured cost, latency, variance delta

### 🧠 Phase 5: LLM Incident Narrator

* Fine-tuned LLaMA-2 (ONNX) using `enterprise-attack.json`
* Generated `mitre_prompts.jsonl` with instruction-format examples
* Output incident summaries like: “Explain T1059.003 in plain language”

### 🚨 Phase 6: SOAR + Dilithium Signatures

* Logic App with trigger + webhook JSON forwarding
* Signatures added via `dilithium_signing_script.py`
* Dashboard includes MTTR + integrity hash logs

---

## 📁 Project Structure

```
q-soc-starlight/
├── datasets/
│   ├── raw_wazuh_logs.csv
│   ├── quantum_encoded_features.csv
│   └── quantum_labels.csv
├── quantum_pipeline/
│   ├── 2_feature_encoder.py
│   ├── 3_kernel_pca_simulator.ipynb
│   └── 4_hardware_jobs_submission.py
├── llm_narrator/
│   ├── 5_prompt_builder.py
│   ├── finetune_llama2.py
│   ├── mitre_prompts.jsonl
│   └── inference_sample.py
├── soar_pipeline/
│   ├── logic_app_playbook.json
│   ├── dilithium_signing_script.py
│   └── secure_blob_upload.py
├── architecture/
│   ├── phase-flow.mermaid
│   └── zero-trust-vnet-diagram.png
├── utils/
│   └── secure_blob_upload.py
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🔐 Security Model

| Control               | Implementation                                      |
| --------------------- | --------------------------------------------------- |
| Zero Trust Networking | Hub-and-spoke VNet, NSG isolation, no public IPs    |
| Data Protection       | CMK-encrypted blob, private endpoint access only    |
| Identity Protection   | Token-authenticated Logic App, Azure AD + Role RBAC |
| Quantum-Safe Signing  | Dilithium signing of SOAR payloads                  |

---

## 💸 Cost Optimization

* VMs deallocated after test runs
* Logic Apps only consumption triggered
* Azure Quantum jobs capped at 10 hardware runs
* No App Services or Premium tiers used

---

## 📊 Results & Takeaways

* Detected anomalies that were **missed by signature-based systems**
* Generated human-readable summaries aligned with MITRE ATT\&CK
* Delivered quantum-secure playbooks with <5s MTTR
* All logs auditable and integrity-enforced

---

## 📂 Related Research Topics

* NIS-2 Resilience Metrics
* Quantum Kernel Methods
* Cryptographic Agility (CRA)
* SOC Automation Integrity

---

## ✅ How to Reproduce

```bash
git clone https://github.com/tanviiiiir-r/q-soc-starlight.git
cd q-soc-starlight
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run phase 2 → 3 → 5 → 6 sequentially
python quantum_pipeline/2_feature_encoder.py
python quantum_pipeline/3_kernel_pca_simulator.ipynb
python llm_narrator/5_prompt_builder.py
python soar_pipeline/dilithium_signing_script.py
```

---

## 📜 License

MIT License © 2025 Md Tanvir Rana

---
