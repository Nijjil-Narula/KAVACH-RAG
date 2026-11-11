# KAVACH-RAG Phase 1.5 Calibration Report

- **Generated at**: `2026-11-11T18:02:29.704995+00:00`
- **Selected logit threshold `T`**: `7.0` (`floor_to_0.25(min_in_scope_dev_logit - 0.5)`)
- **Dev ROC AUC (`inscope_dev_100` vs `oos_dev_60`)**: `0.9997`
- **Reranker model / precision / max_len**: `cross-encoder/ettin-reranker-150m-v1` / `fp32` / `640` (Contingency L: `False`)
- **Dev OOS abstain rate @ T**: `0.9667` (system-level: `0.9667`)
- **Dev In-Scope false-abstain rate @ T**: `0.0000`

## 1. Per-Category Dev Results

### OOS Dev (`oos_dev_60`)
| Category | N | Router HARD_REFUSAL | Reranker < T | System Blocked | System False Accepts | Mean Logit | Max Logit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `general_off_topic` | 30 | 80.00% | 100.00% | 100.00% | 0 | 0.061 | 3.556 |
| `lexical_decoy` | 10 | 60.00% | 100.00% | 100.00% | 0 | 0.127 | 2.704 |
| `org_proprietary` | 10 | 0.00% | 100.00% | 100.00% | 0 | 2.430 | 6.124 |
| `unindexed_public_security` | 10 | 0.00% | 80.00% | 80.00% | 2 | 5.827 | 7.995 |

### In-Scope Dev (`inscope_dev_100`)
| Category | N | False Abstain @ T | Doc Recall@3 | Doc Recall@8 | Min Logit | Mean Logit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cluttered_context` | 10 | 0.00% | 1.0000 | 1.0000 | 7.980 | 9.823 |
| `factual` | 40 | 0.00% | 0.9750 | 0.9750 | 9.543 | 11.227 |
| `malformed` | 10 | 0.00% | 0.9500 | 0.9500 | 9.122 | 10.564 |
| `multi_hop` | 20 | 0.00% | 0.8500 | 0.9500 | 8.743 | 10.138 |
| `owasp_version_agnostic` | 10 | 0.00% | 0.8500 | 0.8500 | 7.526 | 9.823 |
| `short_keyword` | 10 | 0.00% | 1.0000 | 1.0000 | 10.483 | 11.397 |

## 2. Diagnostic Tail Cases

### Lowest 10 In-Scope Dev Logits
| ID | Category | Top-1 Logit | Route | Expected Docs | Top-1 Doc | Query |
| --- | --- | ---: | --- | --- | --- | --- |
| `DEV_IN_069` | `owasp_version_agnostic` | 7.526 | `SEMANTIC_RAG` | `OWASP_v1.1_LLM06,OWASP_v2.0_LLM02` | `OWASP_v2.0_LLM08` | What access-boundary mistakes cause multi-user RAG retrieval pipelines to leak confidential user records under OWASP? |
| `DEV_IN_097` | `cluttered_context` | 7.980 | `SEMANTIC_RAG` | `MITRE_T1566` | `MITRE_T1566` | timestamp/sensor/src_ip/dst_port/signature 2025-04-11T11:03:51Z/segw-mail-01/203.0.113.77/443/SPEARPHISH_HTML_LINK_CLICK |
| `DEV_IN_067` | `owasp_version_agnostic` | 8.729 | `SEMANTIC_RAG` | `OWASP_v1.1_LLM08,OWASP_v2.0_LLM06` | `OWASP_v2.0_LLM06` | Why must autonomous tool-calling agents enforce strict human approval under OWASP LLM agency controls? |
| `DEV_IN_058` | `multi_hop` | 8.743 | `SEMANTIC_RAG` | `CVE-2022-22965,MITRE_T1190` | `CVE-2022-22965` | Link Spring Framework class-loader manipulation in CVE-2022-22965 to initial access technique T1190. |
| `DEV_IN_096` | `cluttered_context` | 8.917 | `SEMANTIC_RAG` | `MITRE_T1210` | `MITRE_T1210` | timestamp/sensor/src_ip/dst_port/signature 2025-04-11T09:44:22Z/ids-core-west/10.14.88.19/445/LATERAL_SMB_PROBE_MULTIHOS |
| `DEV_IN_041` | `multi_hop` | 8.960 | `SEMANTIC_RAG` | `CVE-2023-23397,MITRE_T1187` | `MITRE_T1187` | How does NTLM hash theft in CVE-2023-23397 operationalize ATT&CK Forced Authentication T1187? |
| `DEV_IN_083` | `malformed` | 9.122 | `SEMANTIC_RAG` | `CVE-2023-23397` | `CVE-2023-23397` | summrize kredenshal thft fllowng CVE-2023-23397 ntlm relai |
| `DEV_IN_098` | `cluttered_context` | 9.200 | `SEMANTIC_RAG` | `MITRE_T1195.002` | `MITRE_T1195.002` | From: secops-triage@enterprise.example To: platform-security@enterprise.example Subject: FW: Build pipeline anomaly chec |
| `DEV_IN_048` | `multi_hop` | 9.438 | `SEMANTIC_RAG` | `CVE-2020-1472,MITRE_T1210` | `CVE-2020-1472` | How do ZeroLogon Netlogon abuses in CVE-2020-1472 map onto remote service exploitation under T1210? |
| `DEV_IN_093` | `cluttered_context` | 9.439 | `SEMANTIC_RAG` | `CVE-2022-41040` | `CVE-2022-41040` | incident_ticket:   alert_id: INC-90501   host: mail-gw-03   request: /autodiscover/autodiscover.json?@evil.local/ews/exc |

### Highest 10 OOS Dev Logits
| ID | Category | Top-1 Logit | Route | Top-1 Doc | Query |
| --- | --- | ---: | --- | --- | --- |
| `DEV_OOS_056` | `unindexed_public_security` | 7.995 | `SEMANTIC_RAG` | `MITRE_T1078` | Describe adversary persistence via SSH authorized keys covered by ATT&CK T1098.004. |
| `DEV_OOS_051` | `unindexed_public_security` | 7.260 | `SEMANTIC_RAG` | `CVE-2023-4863` | Summarize the session token memory leakage vulnerability tracked as CVE-2023-4966. |
| `DEV_OOS_048` | `org_proprietary` | 6.124 | `SEMANTIC_RAG` | `CVE-2021-44228` | What internal pager alias escalates CrescentAir crew scheduling portal sign-in interruptions? |
| `DEV_OOS_060` | `unindexed_public_security` | 6.018 | `SEMANTIC_RAG` | `MITRE_T1566` | Outline workforce phishing awareness review milestones recommended by CIS Control 14. |
| `DEV_OOS_052` | `unindexed_public_security` | 5.974 | `SEMANTIC_RAG` | `CVE-2019-0708` | How severe is the Windows Remote Desktop Licensing buffer flaw CVE-2024-38077? |
| `DEV_OOS_057` | `unindexed_public_security` | 5.833 | `SEMANTIC_RAG` | `MITRE_T1659` | What telemetry reveals adversary cloud administration command execution under T1651? |
| `DEV_OOS_053` | `unindexed_public_security` | 5.775 | `SEMANTIC_RAG` | `CVE-2019-11510` | What authentication bypass mechanism is documented for CVE-2023-46805 in Ivanti appliances? |
| `DEV_OOS_054` | `unindexed_public_security` | 5.344 | `SEMANTIC_RAG` | `CVE-2019-2725` | Explain the HTTP parameter authentication bypass associated with CVE-2024-27198. |
| `DEV_OOS_055` | `unindexed_public_security` | 4.969 | `SEMANTIC_RAG` | `MITRE_T1210` | How do defenders detect Kerberoasting ticket requests under ATT&CK T1558.003? |
| `DEV_OOS_058` | `unindexed_public_security` | 4.671 | `SEMANTIC_RAG` | `MITRE_T1199` | List the baseline account provisioning requirements in NIST SP 800-53 control AC-2. |

## 3. Held-Out `golden_150.jsonl` Verification (Fixed `T`)
- **System overall block rate (router + abstention)**: `6.67%`
- **Golden OOS (`out_of_scope`) block rate**: `100.00%`
- **Golden In-Scope false-abstain rate**: `0.00%`

| Question Type | N | Router HARD_REFUSAL | Reranker < T (Non-Refused) | Combined Blocked | Mean Logit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CONFLICT_TRAP` | 15 | 0.00% | 0.00% | 0.00% | 10.203 |
| `CONTEXT_NOISE` | 10 | 0.00% | 0.00% | 0.00% | 8.229 |
| `FACTUAL` | 60 | 0.00% | 0.00% | 0.00% | 10.910 |
| `MALFORMED` | 10 | 0.00% | 0.00% | 0.00% | 10.075 |
| `MULTI_HOP` | 45 | 0.00% | 0.00% | 0.00% | 9.052 |
| `OUT_OF_SCOPE` | 10 | 90.00% | 10.00% | 100.00% | 0.892 |

### Golden `OUT_OF_SCOPE` Per-Item Outcomes (`TEST_121..TEST_130`)
| Query ID | Router Route | Top-1 Logit | Gate Abstained (`< T`) | System Blocked | Query |
| --- | --- | ---: | --- | --- | --- |
| `TEST_121` | `HARD_REFUSAL` | -0.948 | `True` | `True` | How do I bake a sourdough loaf? |
| `TEST_122` | `HARD_REFUSAL` | 1.551 | `True` | `True` | Who won the 2022 FIFA World Cup final? |
| `TEST_123` | `HARD_REFUSAL` | 0.381 | `True` | `True` | What is the best way to train for a marathon in six weeks? |
| `TEST_124` | `HARD_REFUSAL` | 0.657 | `True` | `True` | Recommend a good sci-fi novel for a long flight. |
| `TEST_125` | `HARD_REFUSAL` | 0.924 | `True` | `True` | What's the weather forecast for Delhi this weekend? |
| `TEST_126` | `HARD_REFUSAL` | -2.277 | `True` | `True` | How do I fix a leaking kitchen faucet? |
| `TEST_127` | `HARD_REFUSAL` | 1.150 | `True` | `True` | Explain how compound interest works on a savings account. |
| `TEST_128` | `HARD_REFUSAL` | -1.799 | `True` | `True` | Translate 'good morning' into Japanese and French. |
| `TEST_129` | `SEMANTIC_RAG` | 3.666 | `True` | `True` | What are the internal firewall rules of Acme Corp's production network? |
| `TEST_130` | `HARD_REFUSAL` | 5.621 | `True` | `True` | What is the password policy of my company's HR portal? |

## 4. Full Threshold Sweep Grid (`[-2.0, 12.0]`, step `0.25`)
| T | InScope False Abstain | InScope TPR | OOS False Accept | OOS TNR | Balanced Acc | System OOS False Accept |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -2.00 | 0.0000 | 1.0000 | 0.9667 | 0.0333 | 0.5167 | 0.4833 |
| -1.75 | 0.0000 | 1.0000 | 0.9500 | 0.0500 | 0.5250 | 0.4833 |
| -1.50 | 0.0000 | 1.0000 | 0.9333 | 0.0667 | 0.5333 | 0.4833 |
| -1.25 | 0.0000 | 1.0000 | 0.8833 | 0.1167 | 0.5583 | 0.4833 |
| -1.00 | 0.0000 | 1.0000 | 0.8500 | 0.1500 | 0.5750 | 0.4667 |
| -0.75 | 0.0000 | 1.0000 | 0.8000 | 0.2000 | 0.6000 | 0.4667 |
| -0.50 | 0.0000 | 1.0000 | 0.7167 | 0.2833 | 0.6417 | 0.4333 |
| -0.25 | 0.0000 | 1.0000 | 0.6833 | 0.3167 | 0.6583 | 0.4167 |
| 0.00 | 0.0000 | 1.0000 | 0.6500 | 0.3500 | 0.6750 | 0.4000 |
| 0.25 | 0.0000 | 1.0000 | 0.6000 | 0.4000 | 0.7000 | 0.4000 |
| 0.50 | 0.0000 | 1.0000 | 0.5667 | 0.4333 | 0.7167 | 0.4000 |
| 0.75 | 0.0000 | 1.0000 | 0.5167 | 0.4833 | 0.7417 | 0.4000 |
| 1.00 | 0.0000 | 1.0000 | 0.4667 | 0.5333 | 0.7667 | 0.3667 |
| 1.25 | 0.0000 | 1.0000 | 0.4167 | 0.5833 | 0.7917 | 0.3500 |
| 1.50 | 0.0000 | 1.0000 | 0.4000 | 0.6000 | 0.8000 | 0.3333 |
| 1.75 | 0.0000 | 1.0000 | 0.3667 | 0.6333 | 0.8167 | 0.3333 |
| 2.00 | 0.0000 | 1.0000 | 0.3167 | 0.6833 | 0.8417 | 0.2833 |
| 2.25 | 0.0000 | 1.0000 | 0.3000 | 0.7000 | 0.8500 | 0.2667 |
| 2.50 | 0.0000 | 1.0000 | 0.2333 | 0.7667 | 0.8833 | 0.2000 |
| 2.75 | 0.0000 | 1.0000 | 0.2167 | 0.7833 | 0.8917 | 0.2000 |
| 3.00 | 0.0000 | 1.0000 | 0.2167 | 0.7833 | 0.8917 | 0.2000 |
| 3.25 | 0.0000 | 1.0000 | 0.2167 | 0.7833 | 0.8917 | 0.2000 |
| 3.50 | 0.0000 | 1.0000 | 0.2167 | 0.7833 | 0.8917 | 0.2000 |
| 3.75 | 0.0000 | 1.0000 | 0.2000 | 0.8000 | 0.9000 | 0.2000 |
| 4.00 | 0.0000 | 1.0000 | 0.2000 | 0.8000 | 0.9000 | 0.2000 |
| 4.25 | 0.0000 | 1.0000 | 0.2000 | 0.8000 | 0.9000 | 0.2000 |
| 4.50 | 0.0000 | 1.0000 | 0.1833 | 0.8167 | 0.9083 | 0.1833 |
| 4.75 | 0.0000 | 1.0000 | 0.1500 | 0.8500 | 0.9250 | 0.1500 |
| 5.00 | 0.0000 | 1.0000 | 0.1333 | 0.8667 | 0.9333 | 0.1333 |
| 5.25 | 0.0000 | 1.0000 | 0.1333 | 0.8667 | 0.9333 | 0.1333 |
| 5.50 | 0.0000 | 1.0000 | 0.1167 | 0.8833 | 0.9417 | 0.1167 |
| 5.75 | 0.0000 | 1.0000 | 0.1167 | 0.8833 | 0.9417 | 0.1167 |
| 6.00 | 0.0000 | 1.0000 | 0.0667 | 0.9333 | 0.9667 | 0.0667 |
| 6.25 | 0.0000 | 1.0000 | 0.0333 | 0.9667 | 0.9833 | 0.0333 |
| 6.50 | 0.0000 | 1.0000 | 0.0333 | 0.9667 | 0.9833 | 0.0333 |
| 6.75 | 0.0000 | 1.0000 | 0.0333 | 0.9667 | 0.9833 | 0.0333 |
| 7.00 **`<-- T`** | 0.0000 | 1.0000 | 0.0333 | 0.9667 | 0.9833 | 0.0333 |
| 7.25 | 0.0000 | 1.0000 | 0.0333 | 0.9667 | 0.9833 | 0.0333 |
| 7.50 | 0.0000 | 1.0000 | 0.0167 | 0.9833 | 0.9917 | 0.0167 |
| 7.75 | 0.0100 | 0.9900 | 0.0167 | 0.9833 | 0.9867 | 0.0167 |
| 8.00 | 0.0200 | 0.9800 | 0.0000 | 1.0000 | 0.9900 | 0.0000 |
| 8.25 | 0.0200 | 0.9800 | 0.0000 | 1.0000 | 0.9900 | 0.0000 |
| 8.50 | 0.0200 | 0.9800 | 0.0000 | 1.0000 | 0.9900 | 0.0000 |
| 8.75 | 0.0400 | 0.9600 | 0.0000 | 1.0000 | 0.9800 | 0.0000 |
| 9.00 | 0.0600 | 0.9400 | 0.0000 | 1.0000 | 0.9700 | 0.0000 |
| 9.25 | 0.0800 | 0.9200 | 0.0000 | 1.0000 | 0.9600 | 0.0000 |
| 9.50 | 0.1100 | 0.8900 | 0.0000 | 1.0000 | 0.9450 | 0.0000 |
| 9.75 | 0.1700 | 0.8300 | 0.0000 | 1.0000 | 0.9150 | 0.0000 |
| 10.00 | 0.2200 | 0.7800 | 0.0000 | 1.0000 | 0.8900 | 0.0000 |
| 10.25 | 0.2800 | 0.7200 | 0.0000 | 1.0000 | 0.8600 | 0.0000 |
| 10.50 | 0.3800 | 0.6200 | 0.0000 | 1.0000 | 0.8100 | 0.0000 |
| 10.75 | 0.5000 | 0.5000 | 0.0000 | 1.0000 | 0.7500 | 0.0000 |
| 11.00 | 0.5800 | 0.4200 | 0.0000 | 1.0000 | 0.7100 | 0.0000 |
| 11.25 | 0.7000 | 0.3000 | 0.0000 | 1.0000 | 0.6500 | 0.0000 |
| 11.50 | 0.8100 | 0.1900 | 0.0000 | 1.0000 | 0.5950 | 0.0000 |
| 11.75 | 0.8900 | 0.1100 | 0.0000 | 1.0000 | 0.5550 | 0.0000 |
| 12.00 | 0.9500 | 0.0500 | 0.0000 | 1.0000 | 0.5250 | 0.0000 |

## 5. Phase 2 Recommendations
- Monitor `unindexed_public_security` and `lexical_decoy` OOS queries where `router_route != HARD_REFUSAL`; expand Phase 2 corpus / identifier existence check if non-indexed CVEs share strong semantic priors with indexed CVEs.
- Track `malformed` and `cluttered_context` false abstentions vs reranker `rerank_max_length` truncation; use Contingency L (`rerank_long_query_max_length`) if clutter tokens push core questions beyond the cross-encoder window.
- Add router hard-negative examples specifically contrasting `unindexed_public_security` and `lexical_decoy` phrasing in Phase 2 fine-tuning while preserving strict `golden_150` holdout separation.
