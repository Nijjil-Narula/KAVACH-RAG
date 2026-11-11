"""Deterministic calibration devset builder and strict disjointness validator.

Generates held-in calibration splits for Phase 1.5 logit threshold selection:
- data/golden/oos_dev_60.jsonl (60 out-of-scope items across 4 subcategories)
- data/golden/inscope_dev_100.jsonl (100 in-scope items across 6 subcategories)

Disjointness guarantees enforced by validate_devsets():
- Exact line counts (60 / 100) and strict Pydantic schema (5 fields, extra="forbid")
- Validated corpus doc_ids for in-scope items; empty expected_doc_ids for OOS items
- Zero exact query duplicates across devsets and golden_150.jsonl
- Every oos_dev query has Jaccard < 0.6 vs every golden_150 query under both
  whitespace tokenization and alphanumeric tokenization
- Zero normalized substring overlap against models/router_deberta/training_queries.jsonl
- Zero corpus chunk matches for key entities of unindexed_public_security items
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from kavach.config import Settings, get_settings
from kavach.exceptions import GoldenDatasetValidationError

DEVSET_SEED = 20260

OOS_CATEGORY_TARGETS: dict[str, int] = {
    "general_off_topic": 30,
    "lexical_decoy": 10,
    "org_proprietary": 10,
    "unindexed_public_security": 10,
}

INSCOPE_CATEGORY_TARGETS: dict[str, int] = {
    "factual": 40,
    "multi_hop": 20,
    "owasp_version_agnostic": 10,
    "short_keyword": 10,
    "malformed": 10,
    "cluttered_context": 10,
}

# Explicit tracking of key entities per unindexed_public_security dev_id
UNINDEXED_ENTITY_SPECS: list[tuple[str, str, list[str]]] = [
    (
        "CVE-2023-4966",
        "Summarize the session token memory leakage vulnerability tracked as CVE-2023-4966.",
        ["CVE-2023-4966"],
    ),
    (
        "CVE-2024-38077",
        "How severe is the Windows Remote Desktop Licensing buffer flaw CVE-2024-38077?",
        ["CVE-2024-38077"],
    ),
    (
        "CVE-2023-46805",
        "What authentication bypass mechanism is documented for CVE-2023-46805 in Ivanti appliances?",
        ["CVE-2023-46805"],
    ),
    (
        "CVE-2024-27198",
        "Explain the HTTP parameter authentication bypass associated with CVE-2024-27198.",
        ["CVE-2024-27198"],
    ),
    (
        "T1558.003",
        "How do defenders detect Kerberoasting ticket requests under ATT&CK T1558.003?",
        ["T1558.003"],
    ),
    (
        "T1098.004",
        "Describe adversary persistence via SSH authorized keys covered by ATT&CK T1098.004.",
        ["T1098.004"],
    ),
    (
        "T1651",
        "What telemetry reveals adversary cloud administration command execution under T1651?",
        ["T1651"],
    ),
    (
        "NIST SP 800-53 AC-2",
        "List the baseline account provisioning requirements in NIST SP 800-53 control AC-2.",
        ["NIST SP 800-53", "AC-2"],
    ),
    (
        "ISO/IEC 27001 Annex A.8",
        "Summarize the cryptographic key protection objectives inside ISO/IEC 27001 Annex A.8.",
        ["ISO/IEC 27001", "Annex A.8"],
    ),
    (
        "CIS Control 14",
        "Outline workforce phishing awareness review milestones recommended by CIS Control 14.",
        ["CIS Control 14"],
    ),
]


class DevItem(BaseModel):
    """Strict 5-field devset schema shared by oos_dev_60 and inscope_dev_100."""

    model_config = ConfigDict(extra="forbid")

    dev_id: str = Field(min_length=3)
    split: Literal["oos_dev", "inscope_dev"]
    category: str = Field(min_length=3)
    query: str = Field(min_length=5)
    expected_doc_ids: list[str]


def _normalize_whitespace(text: str) -> str:
    """Lowercase and collapse whitespace for exact-duplicate and substring comparison."""
    return " ".join(text.strip().lower().split())


def _normalize_alnum(text: str) -> str:
    """Normalize text to single-space-separated alphanumeric tokens."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _whitespace_tokens(text: str) -> set[str]:
    """Return unique lowercase whitespace-delimited tokens."""
    return set(text.lower().split())


def _alnum_tokens(text: str) -> set[str]:
    """Return unique lowercase alphanumeric tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Compute Jaccard set similarity in [0.0, 1.0]."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _atomic_write_jsonl(path: Path, items: list[DevItem]) -> None:
    """Write DevItem records atomically using a same-directory temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    payload = "\n".join(json.dumps(item.model_dump(), ensure_ascii=False) for item in items) + "\n"
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _build_oos_items() -> list[DevItem]:
    """Construct the 60 out-of-scope dev queries across all 4 required subcategories."""
    general_off_topic: list[str] = [
        # Cooking & Culinary (3)
        "What ambient room temperature yields the best rise during sourdough bulk fermentation?",
        "How slowly should warm broth be ladled when cooking traditional Milanese saffron risotto?",
        "Which high-smoke-point vegetable oil builds durable seasoning on a bare cast-iron skillet?",
        # Sports Rules (3)
        "How many seconds remain on the shot clock after an offensive rebound in NBA basketball?",
        "What points system determines the polka dot mountains jersey in the Tour de France?",
        "When do umpires invoke the infield fly rule during a baseball game with runners on base?",
        # Travel & Geography (3)
        "Which regional train passes cover autumn sightseeing trips between Kyoto and Nara?",
        "What summer months bring calmer Patagonian winds for trekking near Torres del Paine?",
        "How long does the scenic ferry crossing run along Norway's Geirangerfjord in July?",
        # World History (3)
        "How did Gutenberg cast interchangeable metal type pieces for his fifteenth-century press?",
        "What gentle slope gradient kept water flowing smoothly across ancient Roman stone aqueducts?",
        "How did oasis caravanserais shelter merchant caravans crossing Central Asian Silk Road routes?",
        # Music & Arts (3)
        "Why does adjusting a cello soundpost slightly alter resonance across the lower strings?",
        "How does wet-on-wet watercolor brushwork soften horizon gradients on cotton paper?",
        "What standard twelve-bar chord progression underpins traditional Chicago blues guitar solos?",
        # Literature (3)
        "In what chronological order does Homer tell Odysseus's wanderings across the Odyssey?",
        "How does the Yorkshire moorland weather mirror character conflict in Wuthering Heights?",
        "What seasonal kigo imagery traditionally anchors a classical seventeen-syllable Japanese haiku?",
        # Gardening & Botany (3)
        "Why does irregular summer watering cause calcium-deficient blossom end rot in heirloom tomatoes?",
        "During which late-winter weeks should dormant backyard apple trees receive structural pruning?",
        "How much coarse grit drainage prevents root moisture stress in Mediterranean lavender beds?",
        # Mathematics (3)
        "Walk through the Euclidean division algorithm for finding the greatest common divisor of two integers.",
        "How does taking the square root of variance convert standard deviation back to original units?",
        "Compare discrete monthly compounding returns against continuous exponential growth at five percent.",
        # Personal Finance (3)
        "How significantly does an annual expense ratio difference erode retirement index fund compounding?",
        "Under what residency conditions are municipal bond coupon payments exempt from state income tax?",
        "What pacing advantages does biweekly dollar-cost averaging offer in volatile equity markets?",
        # Consumer Hardware & Photography (3)
        "How does opening a fifty-millimeter prime lens aperture narrow background depth of field?",
        "Which display brightness habits help minimize uneven pixel wear on desktop OLED monitors?",
        "How does actuation spring weight affect typing fatigue on tactile mechanical keyboards?",
    ]

    lexical_decoy: list[str] = [
        "How much sodium hydroxide normalizes the pH of a molar Tris chemistry buffer solution?",
        "Why do gasoline direct fuel injection nozzles accumulate carbon deposits over highway mileage?",
        "What drywall thickness meets commercial building-code firewall partition ratings in theaters?",
        "When grafting heritage apple scions onto dwarf rootstock, how tight should the bark patch wrap?",
        "Which vintage year minted the octagonal brass London subway fare token for collectors?",
        "How can plumbers clear mineral scale blocking a porcelain bathroom sink overflow channel?",
        "What internal kernel moisture percentage produces maximum popping volume in yellow popcorn?",
        "In British parliamentary procedure, what protects floor debate speech under legislative privilege?",
        "How many years in seasoned oak casks distinguish aged tawny port wine from ruby styles?",
        "Why do chess grandmasters sacrifice a minor pawn to exploit weak dark squares around castled kings?",
    ]

    org_proprietary: list[str] = [
        "Where does Northstar Financial host its internal WireTransfer microservice secondary approver roster?",
        "Summarize the CobaltPay internal merchant settlement reconciliation playbook stored on Confluence.",
        "Which HR ticketing queue handles Meridian Health campus physical badge replacements for contractors?",
        "What VLAN passphrase resets handheld barcode scanners inside AtlasLogistics distribution warehouses?",
        "Who holds weekly on-call rotation for HeliosCorp internal staging cluster maintenance windows?",
        "List the Windows workstation imaging steps in BorealBank's internal branch teller setup guide.",
        "Where can engineers download VertexGrid's internal substation change-advisory approval template?",
        "What internal pager alias escalates CrescentAir crew scheduling portal sign-in interruptions?",
        "How do regional managers issue receipt override codes on Vanguardia Retail point-of-sale terminals?",
        "Which shared network drive stores SolsticeBio's internal LIMS pipette calibration schedules?",
    ]

    items: list[DevItem] = []
    idx = 1
    for q in general_off_topic:
        items.append(
            DevItem(
                dev_id=f"DEV_OOS_{idx:03d}",
                split="oos_dev",
                category="general_off_topic",
                query=q,
                expected_doc_ids=[],
            )
        )
        idx += 1

    for q in lexical_decoy:
        items.append(
            DevItem(
                dev_id=f"DEV_OOS_{idx:03d}",
                split="oos_dev",
                category="lexical_decoy",
                query=q,
                expected_doc_ids=[],
            )
        )
        idx += 1

    for q in org_proprietary:
        items.append(
            DevItem(
                dev_id=f"DEV_OOS_{idx:03d}",
                split="oos_dev",
                category="org_proprietary",
                query=q,
                expected_doc_ids=[],
            )
        )
        idx += 1

    for _tag, q, _entities in UNINDEXED_ENTITY_SPECS:
        items.append(
            DevItem(
                dev_id=f"DEV_OOS_{idx:03d}",
                split="oos_dev",
                category="unindexed_public_security",
                query=q,
                expected_doc_ids=[],
            )
        )
        idx += 1

    return items


def _build_inscope_items(attack_map_path: Path) -> list[DevItem]:
    """Construct the 100 in-scope dev queries grounded in indexed corpus_chunks documents."""
    with open(attack_map_path, "r", encoding="utf-8") as fh:
        attack_raw = yaml.safe_load(fh)
    valid_pairs: set[tuple[str, str]] = set()
    for entry in attack_raw.get("mappings", []):
        tid = entry["technique_id"]
        for cve_id in entry.get("cve_ids", []):
            valid_pairs.add((cve_id, tid))

    # 40 Factual queries (25 CVEs + 10 MITRE techniques + 5 version-pinned OWASP)
    factual_specs: list[tuple[str, list[str]]] = [
        # 25 CVEs (distinct phrasing across severity, CWE, product version bounds, impact)
        ("Which OpenSSL releases are affected by the Heartbeat read flaw CVE-2014-0160?", ["CVE-2014-0160"]),
        ("How do environment variable trailing function strings trigger CVE-2014-6271 in GNU Bash?", ["CVE-2014-6271"]),
        ("Describe the Linux kernel copy-on-write race condition tracked under CVE-2016-5195.", ["CVE-2016-5195"]),
        ("Which Windows SMBv1 operating system releases were vulnerable to CVE-2017-0144?", ["CVE-2017-0144"]),
        ("What memory corruption mechanism in Microsoft Office Equation Editor underlies CVE-2017-11882?", ["CVE-2017-11882"]),
        ("Why does the Apache Struts Jakarta Multipart parser allow execution in CVE-2017-5638?", ["CVE-2017-5638"]),
        ("Under what Struts namespace configuration condition does CVE-2018-11776 trigger remote code execution?", ["CVE-2018-11776"]),
        ("What file-reading flaw in FortiOS SSL VPN is documented for CVE-2018-13379?", ["CVE-2018-13379"]),
        ("Explain why unauthenticated RDS terminal requests exploit CVE-2019-0708 without user interaction.", ["CVE-2019-0708"]),
        ("Which Pulse Connect Secure software branches are listed as vulnerable to CVE-2019-11510?", ["CVE-2019-11510"]),
        ("State the directory traversal impact on Citrix ADC appliances described in CVE-2019-19781.", ["CVE-2019-19781"]),
        ("What Oracle WebLogic subcomponent is impacted by deserialization vulnerability CVE-2019-2725?", ["CVE-2019-2725"]),
        ("How does Windows CryptoAPI validate elliptic curve certificates improperly in CVE-2020-0601?", ["CVE-2020-0601"]),
        ("Why can an attacker impersonate a domain controller using Netlogon under CVE-2020-1472?", ["CVE-2020-1472"]),
        ("Summarize the F5 BIG-IP Traffic Management User Interface flaw recorded as CVE-2020-5902.", ["CVE-2020-5902"]),
        ("Which vCenter Server HTML5 client component contains the plugin RCE flaw CVE-2021-21972?", ["CVE-2021-21972"]),
        ("Explain how uploaded image metadata parsing leads to command execution in GitLab CVE-2021-22205.", ["CVE-2021-22205"]),
        ("What Server-Side Request Forgery vector initiates Microsoft Exchange ProxyLogon CVE-2021-26855?", ["CVE-2021-26855"]),
        ("Identify the heap off-by-one bug affecting Sudo prior to 1.9.5p2 in CVE-2021-3156.", ["CVE-2021-3156"]),
        ("Which Windows Print Spooler privileged file operation fails authorization checks in CVE-2021-34527?", ["CVE-2021-34527"]),
        ("How does memory corruption inside polkit pkexec grant local root privileges via CVE-2021-4034?", ["CVE-2021-4034"]),
        ("Why did Apache HTTP Server 2.4.50 require follow-up remediation CVE-2021-42013 after 2.4.49?", ["CVE-2021-42013"]),
        ("What F5 BIG-IP iControl REST authentication weakness characterizes CVE-2022-1388?", ["CVE-2022-1388"]),
        ("How does malicious OGNL injection inside Confluence URLs relate to CVE-2022-26134?", ["CVE-2022-26134"]),
        ("What Windows Support Diagnostic Tool MSDT URI abuse is exploited in CVE-2022-30190?", ["CVE-2022-30190"]),
        # 10 MITRE ATT&CK techniques
        ("How do adversaries leverage Valid Accounts under ATT&CK T1078 to evade behavioral detection?", ["MITRE_T1078"]),
        ("Outline defensive removable media controls mitigating MITRE ATT&CK technique T1091.", ["MITRE_T1091"]),
        ("What persistence risks stem from exposed VPNs and external services in ATT&CK T1133?", ["MITRE_T1133"]),
        ("Explain how watering-hole browser exploits achieve initial access in ATT&CK T1189.", ["MITRE_T1189"]),
        ("Summarize the main software and hardware compromise vectors inside ATT&CK T1195.", ["MITRE_T1195"]),
        ("How can third-party partner connectivity become an intrusion vector under ATT&CK T1199?", ["MITRE_T1199"]),
        ("Describe physical hardware addition threats covered by ATT&CK technique T1200.", ["MITRE_T1200"]),
        ("What endpoint service degradation impacts are categorized under MITRE ATT&CK T1499?", ["MITRE_T1499"]),
        ("Identify user training and email gateway safeguards against phishing in ATT&CK T1566.", ["MITRE_T1566"]),
        ("How does adversary content injection facilitate covert communication under ATT&CK T1659?", ["MITRE_T1659"]),
        # 5 version-pinned OWASP entries
        ("What extension design flaws were cataloged under OWASP LLM Top 10 v1.1 LLM07?", ["OWASP_v1.1_LLM07"]),
        ("How does OWASP LLM Top 10 v1.1 LLM10 characterize unauthorized model weight exfiltration?", ["OWASP_v1.1_LLM10"]),
        ("What system instruction exposure scenarios define OWASP LLM Top 10 v2.0 LLM07?", ["OWASP_v2.0_LLM07"]),
        ("Summarize RAG embedding inversion and poisoning threats in OWASP LLM Top 10 v2.0 LLM08.", ["OWASP_v2.0_LLM08"]),
        ("How does OWASP LLM Top 10 v2.0 LLM09 frame LLM hallucination and misinformation risks?", ["OWASP_v2.0_LLM09"]),
    ]

    # 20 Multi-hop CVE <-> MITRE ATT&CK pairs verified against cve_attack_map.yaml
    multi_hop_specs: list[tuple[str, str, str]] = [
        ("CVE-2023-23397", "T1187", "How does NTLM hash theft in CVE-2023-23397 operationalize ATT&CK Forced Authentication T1187?"),
        ("CVE-2021-3156", "T1548.003", "Connect the Sudo heap overflow in CVE-2021-3156 with privilege control bypasses documented in T1548.003."),
        ("CVE-2024-21626", "T1611", "Explain how runc file descriptor leaks in CVE-2024-21626 enable container host breakouts tracked under T1611."),
        ("CVE-2023-44487", "T1499.002", "Relate HTTP/2 rapid stream resets in CVE-2023-44487 to service exhaustion floods under ATT&CK T1499.002."),
        ("CVE-2024-3094", "T1195.002", "How does the malicious xz tarball implant in CVE-2024-3094 exemplify software supply chain tampering in T1195.002?"),
        ("CVE-2017-0144", "T1210", "Compare SMBv1 lateral propagation via CVE-2017-0144 against remote service exploitation behaviors in T1210."),
        ("CVE-2019-0708", "T1210", "Which defensive mitigations for ATT&CK T1210 reduce lateral spread through BlueKeep RDP flaw CVE-2019-0708?"),
        ("CVE-2020-1472", "T1210", "How do ZeroLogon Netlogon abuses in CVE-2020-1472 map onto remote service exploitation under T1210?"),
        ("CVE-2017-11882", "T1203", "Analyze how weaponized Office documents exploit CVE-2017-11882 to achieve client execution under T1203."),
        ("CVE-2022-30190", "T1203", "Connect MSDT client document execution in CVE-2022-30190 with ATT&CK T1203 endpoint protections."),
        ("CVE-2023-38831", "T1203", "How does archive payload spoofing in WinRAR CVE-2023-38831 illustrate client-side exploitation T1203?"),
        ("CVE-2023-4863", "T1203", "Relate the libwebp browser heap overflow CVE-2023-4863 to client execution defenses inside ATT&CK T1203."),
        ("CVE-2021-4034", "T1068", "Explain how local SUID abuse in polkit CVE-2021-4034 maps to kernel/binary privilege escalation in T1068."),
        ("CVE-2016-5195", "T1068", "Which T1068 privilege escalation mitigations address Linux Dirty COW memory writes in CVE-2016-5195?"),
        ("CVE-2022-0847", "T1068", "Compare Linux pipe page-flag overwrites in CVE-2022-0847 with privilege escalation methods in ATT&CK T1068."),
        ("CVE-2024-1086", "T1068", "How does netfilter nf_tables use-after-free CVE-2024-1086 fulfill adversary privilege escalation under T1068?"),
        ("CVE-2021-44228", "T1190", "Which public-facing web application mitigations in T1190 help contain JNDI exploitation of CVE-2021-44228?"),
        ("CVE-2022-22965", "T1190", "Link Spring Framework class-loader manipulation in CVE-2022-22965 to initial access technique T1190."),
        ("CVE-2023-34362", "T1190", "How does SQL injection against internet-facing MOVEit Transfer in CVE-2023-34362 align with ATT&CK T1190?"),
        ("CVE-2024-3400", "T1190", "Connect Palo Alto GlobalProtect gateway exploitation CVE-2024-3400 with perimeter defenses in T1190."),
    ]
    for cve_id, tid, _q in multi_hop_specs:
        if (cve_id, tid) not in valid_pairs:
            raise GoldenDatasetValidationError(
                f"Multi-hop dev pair ({cve_id}, {tid}) is not present in {attack_map_path}"
            )

    # 10 OWASP Version-Agnostic queries (omit v1.1 / v2.0; expect both docs)
    owasp_agnostic_specs: list[tuple[str, list[str]]] = [
        ("How do indirect prompt injection payloads inside retrieved web pages hijack OWASP LLM applications?", ["OWASP_v1.1_LLM01", "OWASP_v2.0_LLM01"]),
        ("Why does failing to escape LLM-generated HTML or JavaScript create XSS vulnerabilities under OWASP LLM guidance?", ["OWASP_v1.1_LLM02", "OWASP_v2.0_LLM05"]),
        ("Explain how backdoored fine-tuning datasets undermine model behavior across OWASP LLM security risks.", ["OWASP_v1.1_LLM03", "OWASP_v2.0_LLM04"]),
        ("What rate-limiting and token-cap controls mitigate LLM resource exhaustion floods in the OWASP Top 10?", ["OWASP_v1.1_LLM04", "OWASP_v2.0_LLM10"]),
        ("How do compromised third-party foundation weights and external adapters create OWASP LLM supply chain exposure?", ["OWASP_v1.1_LLM05", "OWASP_v2.0_LLM03"]),
        ("Which sanitization safeguards prevent memorized PII leakage in OWASP LLM confidential disclosure guidance?", ["OWASP_v1.1_LLM06", "OWASP_v2.0_LLM02"]),
        ("Why must autonomous tool-calling agents enforce strict human approval under OWASP LLM agency controls?", ["OWASP_v1.1_LLM08", "OWASP_v2.0_LLM06"]),
        ("Summarize the operational hazards of accepting unverified LLM citations and hallucinations in OWASP Top 10.", ["OWASP_v1.1_LLM09", "OWASP_v2.0_LLM09"]),
        ("What access-boundary mistakes cause multi-user RAG retrieval pipelines to leak confidential user records under OWASP?", ["OWASP_v1.1_LLM06", "OWASP_v2.0_LLM02"]),
        ("How should least-privilege function permissions constrain LLM plugins and autonomous tools in OWASP guidance?", ["OWASP_v1.1_LLM08", "OWASP_v2.0_LLM06"]),
    ]

    # 10 Short keyword-style queries (3-6 tokens; never bare single CVE/MITRE ID)
    short_keyword_specs: list[tuple[str, list[str]]] = [
        ("Heartbleed OpenSSL CVE-2014-0160 buffer read", ["CVE-2014-0160"]),
        ("Shellshock Bash CVE-2014-6271 env injection", ["CVE-2014-6271"]),
        ("EternalBlue SMBv1 CVE-2017-0144 remote execution", ["CVE-2017-0144"]),
        ("FortiOS SSLVPN CVE-2018-13379 path traversal", ["CVE-2018-13379"]),
        ("Citrix ADC NetScaler CVE-2019-19781 traversal", ["CVE-2019-19781"]),
        ("BIG-IP TMUI CVE-2020-5902 RCE", ["CVE-2020-5902"]),
        ("Spring4Shell JDK9 CVE-2022-22965 RCE", ["CVE-2022-22965"]),
        ("Confluence setup CVE-2023-22515 admin bypass", ["CVE-2023-22515"]),
        ("Cisco IOSXE webui CVE-2023-20198 implant", ["CVE-2023-20198"]),
        ("MITRE T1548.003 sudo caching bypass", ["MITRE_T1548.003"]),
    ]

    # 10 Malformed queries (QWERTY-adjacent hits, camelCase collapse, consonant skeleton, phonetic)
    malformed_specs: list[tuple[str, list[str]]] = [
        ("eksplain pathtraversal mtgtion fr CVE-2021-41773 apche servr", ["CVE-2021-41773"]),
        ("wat remotecodeexecution mechanizm affekts CVE-2021-44228 jndi lokup", ["CVE-2021-44228"]),
        ("summrize kredenshal thft fllowng CVE-2023-23397 ntlm relai", ["CVE-2023-23397"]),
        ("detl privlegescalation risk nd sudoers fks fr CVE 2021 3156", ["CVE-2021-3156"]),
        ("how dzz bufferoverflo in CVE 2023 27997 fortios xploit wrk", ["CVE-2023-27997"]),
        ("wch mitgshuns blck T1190 publc facng app xplotashun", ["MITRE_T1190"]),
        ("descryb kontainer hostbreakout technk T1611 eskape detction", ["MITRE_T1611"]),
        ("expln httptwo rpid rset flood CVE-2023-44487 servc exhstion", ["CVE-2023-44487"]),
        ("whch cwe flaw kauses CVE-2024-23897 jnkins cli arbirtry rd", ["CVE-2024-23897"]),
        ("summry of owasp llm promptinjekshun categry LLM01 safeguards", ["OWASP_v1.1_LLM01", "OWASP_v2.0_LLM01"]),
    ]

    # 10 Cluttered-context queries (YAML incident tickets, SOC alert pipe tables, forwarded emails)
    cluttered_specs: list[tuple[str, list[str]]] = [
        (
            "incident_ticket:\n  alert_id: INC-90412\n  sensor: waf-edge-eu1\n  uri: /cgi-bin/php-cgi.exe?%ADd+allow_url_include%3d1\n  severity: critical\n"
            "Based on the ticket artifact above, what Windows CGI argument injection flaw enables CVE-2024-4577 remote execution?",
            ["CVE-2024-4577"],
        ),
        (
            "incident_ticket:\n  alert_id: INC-90455\n  process: activemq-broker-5.18\n  src_port: 61616\n  artifact: ClassPathXmlApplicationContext payload\n"
            "Question: Which OpenWire protocol deserialization defect drives CVE-2023-46604 in Apache ActiveMQ?",
            ["CVE-2023-46604"],
        ),
        (
            "incident_ticket:\n  alert_id: INC-90501\n  host: mail-gw-03\n  request: /autodiscover/autodiscover.json?@evil.local/ews/exchange.asmx\n  status: 500\n"
            "Identify the Microsoft Exchange Server SSRF escalation issue tracked as CVE-2022-41040 and its affected versions.",
            ["CVE-2022-41040"],
        ),
        (
            "incident_ticket:\n  alert_id: INC-90548\n  endpoint: cert-validation-svc\n  dll: Crypt32.dll\n  event: ECC curve parameter mismatch\n"
            "Explain the root cause and spoofing impact of Windows CryptoAPI vulnerability CVE-2020-0601.",
            ["CVE-2020-0601"],
        ),
        (
            "timestamp|sensor|src_ip|dst_port|signature\n2025-04-11T08:12:09Z|mx-relay-02|198.51.100.44|25|MALFORMED_TAR_HEADER_ESG\n"
            "Considering the mail gateway alert table above, which tar attachment parsing flaw causes command injection in CVE-2023-2868?",
            ["CVE-2023-2868"],
        ),
        (
            "timestamp|sensor|src_ip|dst_port|signature\n2025-04-11T09:44:22Z|ids-core-west|10.14.88.19|445|LATERAL_SMB_PROBE_MULTIHOST\n"
            "Using the telemetry table above, summarize the lateral-movement detection indicators for MITRE ATT&CK T1210.",
            ["MITRE_T1210"],
        ),
        (
            "timestamp|sensor|src_ip|dst_port|signature\n2025-04-11T11:03:51Z|segw-mail-01|203.0.113.77|443|SPEARPHISH_HTML_LINK_CLICK\n"
            "Reviewing this gateway row, describe adversary delivery patterns and preventive controls in MITRE ATT&CK T1566.",
            ["MITRE_T1566"],
        ),
        (
            "From: secops-triage@enterprise.example\nTo: platform-security@enterprise.example\nSubject: FW: Build pipeline anomaly check\n\n"
            "Team, our CI runners flagged unexpected tarball linker tampering. What does ATT&CK T1195.002 cover regarding compromised build dependencies?",
            ["MITRE_T1195.002"],
        ),
        (
            "From: ai-governance@enterprise.example\nTo: llm-platform@enterprise.example\nSubject: FW: Red-team prompt disclosure audit\n\n"
            "Hi team, QA reproduced an issue extracting hidden developer instructions. What risks define OWASP LLM Top 10 v2.0 LLM07 System Prompt Leakage?",
            ["OWASP_v2.0_LLM07"],
        ),
        (
            "From: appsec-review@enterprise.example\nTo: chatbot-eng@enterprise.example\nSubject: Re: Indirect delimiter bypass in RAG retriever\n\n"
            "Following up on the review thread, outline recommended mitigations for OWASP LLM01 Prompt Injection across LLM deployments.",
            ["OWASP_v1.1_LLM01", "OWASP_v2.0_LLM01"],
        ),
    ]

    items: list[DevItem] = []
    idx = 1

    for q, docs in factual_specs:
        items.append(
            DevItem(
                dev_id=f"DEV_IN_{idx:03d}",
                split="inscope_dev",
                category="factual",
                query=q,
                expected_doc_ids=docs,
            )
        )
        idx += 1

    for cve_id, tid, q in multi_hop_specs:
        items.append(
            DevItem(
                dev_id=f"DEV_IN_{idx:03d}",
                split="inscope_dev",
                category="multi_hop",
                query=q,
                expected_doc_ids=[cve_id, f"MITRE_{tid}"],
            )
        )
        idx += 1

    for q, docs in owasp_agnostic_specs:
        items.append(
            DevItem(
                dev_id=f"DEV_IN_{idx:03d}",
                split="inscope_dev",
                category="owasp_version_agnostic",
                query=q,
                expected_doc_ids=docs,
            )
        )
        idx += 1

    for q, docs in short_keyword_specs:
        items.append(
            DevItem(
                dev_id=f"DEV_IN_{idx:03d}",
                split="inscope_dev",
                category="short_keyword",
                query=q,
                expected_doc_ids=docs,
            )
        )
        idx += 1

    for q, docs in malformed_specs:
        items.append(
            DevItem(
                dev_id=f"DEV_IN_{idx:03d}",
                split="inscope_dev",
                category="malformed",
                query=q,
                expected_doc_ids=docs,
            )
        )
        idx += 1

    for q, docs in cluttered_specs:
        items.append(
            DevItem(
                dev_id=f"DEV_IN_{idx:03d}",
                split="inscope_dev",
                category="cluttered_context",
                query=q,
                expected_doc_ids=docs,
            )
        )
        idx += 1

    return items


def _load_dev_jsonl(path: Path) -> list[DevItem]:
    """Load and strict-validate JSONL records into DevItem objects."""
    if not path.exists():
        raise GoldenDatasetValidationError(f"Devset JSONL file missing: {path}")
    items: list[DevItem] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                items.append(DevItem.model_validate(data))
            except Exception as exc:
                raise GoldenDatasetValidationError(
                    f"Schema validation error in {path} line {line_no}: {exc}"
                ) from exc
    return items


def _extract_unindexed_entities(query: str) -> list[str]:
    """Extract security identifiers/frameworks from an unindexed_public_security query."""
    patterns = [
        r"CVE-\d{4}-\d{4,7}",
        r"\bT\d{4}(?:\.\d{3})?\b",
        r"NIST SP 800-53",
        r"\bAC-2\b",
        r"ISO/IEC 27001",
        r"Annex A\.8",
        r"CIS Control 14",
        r"PCI DSS",
        r"SOC 2 CC6\.1",
        r"HIPAA Security Rule",
        r"FedRAMP",
    ]
    found: list[str] = []
    for pat in patterns:
        for match in re.findall(pat, query):
            if match not in found:
                found.append(match)
    return found


def validate_devsets(
    oos_path: Path,
    inscope_path: Path,
    corpus_parquet_path: Path,
    golden_path: Path,
    training_queries_path: Path,
) -> dict[str, object]:
    """Validate counts, schemas, corpus integrity, and strict disjointness of devsets.

    Raises GoldenDatasetValidationError immediately on any violation.
    """
    if not corpus_parquet_path.exists():
        raise GoldenDatasetValidationError(f"Corpus parquet missing: {corpus_parquet_path}")
    if not golden_path.exists():
        raise GoldenDatasetValidationError(f"Golden dataset missing: {golden_path}")
    if not training_queries_path.exists():
        raise GoldenDatasetValidationError(
            f"Router training queries file missing: {training_queries_path}"
        )

    oos_items = _load_dev_jsonl(oos_path)
    inscope_items = _load_dev_jsonl(inscope_path)

    if len(oos_items) != 60:
        raise GoldenDatasetValidationError(
            f"Expected 60 oos_dev items in {oos_path}, got {len(oos_items)}"
        )
    if len(inscope_items) != 100:
        raise GoldenDatasetValidationError(
            f"Expected 100 inscope_dev items in {inscope_path}, got {len(inscope_items)}"
        )

    all_dev = oos_items + inscope_items
    dev_ids = [item.dev_id for item in all_dev]
    if len(set(dev_ids)) != len(all_dev):
        raise GoldenDatasetValidationError("Duplicate dev_id detected across devsets")

    # Split and category breakdown checks
    oos_counts = Counter(item.category for item in oos_items)
    if dict(oos_counts) != OOS_CATEGORY_TARGETS:
        raise GoldenDatasetValidationError(
            f"oos_dev category breakdown mismatch: got {dict(oos_counts)}, expected {OOS_CATEGORY_TARGETS}"
        )
    for item in oos_items:
        if item.split != "oos_dev":
            raise GoldenDatasetValidationError(
                f"Invalid split {item.split!r} for {item.dev_id} in {oos_path}"
            )
        if item.expected_doc_ids:
            raise GoldenDatasetValidationError(
                f"oos_dev item {item.dev_id} must have empty expected_doc_ids, got {item.expected_doc_ids}"
            )
        if item.category == "general_off_topic":
            wc = len(item.query.split())
            if wc < 5 or wc > 25:
                raise GoldenDatasetValidationError(
                    f"general_off_topic item {item.dev_id} must have 5-25 words, got {wc}"
                )

    inscope_counts = Counter(item.category for item in inscope_items)
    if dict(inscope_counts) != INSCOPE_CATEGORY_TARGETS:
        raise GoldenDatasetValidationError(
            f"inscope_dev category breakdown mismatch: got {dict(inscope_counts)}, expected {INSCOPE_CATEGORY_TARGETS}"
        )

    # Corpus doc_id validity & chunk text search index
    corpus_df = pd.read_parquet(corpus_parquet_path)
    valid_doc_ids = set(corpus_df["doc_id"].astype(str).tolist())
    corpus_text_lower = "\n".join(
        corpus_df["doc_id"].astype(str).tolist()
        + corpus_df["header_lineage"].astype(str).tolist()
        + corpus_df["content"].astype(str).tolist()
    ).lower()

    for item in inscope_items:
        if item.split != "inscope_dev":
            raise GoldenDatasetValidationError(
                f"Invalid split {item.split!r} for {item.dev_id} in {inscope_path}"
            )
        if not item.expected_doc_ids:
            raise GoldenDatasetValidationError(
                f"inscope_dev item {item.dev_id} must have non-empty expected_doc_ids"
            )
        unknown_docs = [doc for doc in item.expected_doc_ids if doc not in valid_doc_ids]
        if unknown_docs:
            raise GoldenDatasetValidationError(
                f"inscope_dev item {item.dev_id} references unknown corpus doc_ids: {unknown_docs}"
            )

    # Unindexed public security entity zero-corpus check
    unindexed_items = [item for item in oos_items if item.category == "unindexed_public_security"]
    for item in unindexed_items:
        entities = _extract_unindexed_entities(item.query)
        if not entities:
            raise GoldenDatasetValidationError(
                f"Could not extract any public-security key entity from {item.dev_id}: {item.query!r}"
            )
        for ent in entities:
            if ent.lower() in corpus_text_lower:
                raise GoldenDatasetValidationError(
                    f"Unindexed OOS entity {ent!r} in {item.dev_id} matched corpus_chunks.parquet!"
                )

    # Load golden_150 queries for duplicate and Jaccard checks
    golden_queries: list[tuple[str, str]] = []
    with open(golden_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            golden_queries.append((str(row["test_id"]), str(row["query"])))

    # Zero exact duplicates across oos_dev + inscope_dev + golden_150
    seen_raw: dict[str, str] = {}
    seen_norm: dict[str, str] = {}
    for gid, gq in golden_queries:
        seen_raw[gq.strip()] = gid
        seen_norm[_normalize_whitespace(gq)] = gid
    for item in all_dev:
        raw_q = item.query.strip()
        norm_q = _normalize_whitespace(item.query)
        if raw_q in seen_raw:
            raise GoldenDatasetValidationError(
                f"Exact query duplicate between {item.dev_id} and {seen_raw[raw_q]}"
            )
        if norm_q in seen_norm:
            raise GoldenDatasetValidationError(
                f"Normalized duplicate between {item.dev_id} and {seen_norm[norm_q]}"
            )
        seen_raw[raw_q] = item.dev_id
        seen_norm[norm_q] = item.dev_id

    # OOS vs golden_150 Jaccard < 0.6 under BOTH whitespace and alphanumeric tokenization
    max_ws_jaccard = 0.0
    max_alnum_jaccard = 0.0
    golden_ws = [(gid, _whitespace_tokens(gq)) for gid, gq in golden_queries]
    golden_alnum = [(gid, _alnum_tokens(gq)) for gid, gq in golden_queries]

    for item in oos_items:
        ws_toks = _whitespace_tokens(item.query)
        al_toks = _alnum_tokens(item.query)
        for gid, g_ws in golden_ws:
            j_ws = _jaccard(ws_toks, g_ws)
            if j_ws > max_ws_jaccard:
                max_ws_jaccard = j_ws
            if j_ws >= 0.6:
                raise GoldenDatasetValidationError(
                    f"oos_dev {item.dev_id} whitespace Jaccard={j_ws:.3f} >= 0.6 vs golden {gid}"
                )
        for gid, g_al in golden_alnum:
            j_al = _jaccard(al_toks, g_al)
            if j_al > max_alnum_jaccard:
                max_alnum_jaccard = j_al
            if j_al >= 0.6:
                raise GoldenDatasetValidationError(
                    f"oos_dev {item.dev_id} alnum Jaccard={j_al:.3f} >= 0.6 vs golden {gid}"
                )

    # Zero normalized substring overlap against models/router_deberta/training_queries.jsonl
    train_norms: list[tuple[int, str, str]] = []
    with open(training_queries_path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            tq = str(row["query"])
            train_norms.append((idx, _normalize_whitespace(tq), _normalize_alnum(tq)))

    for item in all_dev:
        d_ws = _normalize_whitespace(item.query)
        d_al = _normalize_alnum(item.query)
        for tr_idx, t_ws, t_al in train_norms:
            if d_ws in t_ws or t_ws in d_ws:
                raise GoldenDatasetValidationError(
                    f"Normalized whitespace substring overlap between {item.dev_id} and training_queries line {tr_idx}: {t_ws!r}"
                )
            if d_al in t_al or t_al in d_al:
                raise GoldenDatasetValidationError(
                    f"Normalized alnum substring overlap between {item.dev_id} and training_queries line {tr_idx}: {t_al!r}"
                )

    return {
        "oos_count": len(oos_items),
        "inscope_count": len(inscope_items),
        "oos_categories": dict(oos_counts),
        "inscope_categories": dict(inscope_counts),
        "max_oos_golden_ws_jaccard": round(max_ws_jaccard, 4),
        "max_oos_golden_alnum_jaccard": round(max_alnum_jaccard, 4),
        "training_queries_checked": len(train_norms),
        "golden_queries_checked": len(golden_queries),
    }


def build_devsets(settings: Settings | None = None) -> dict[str, object]:
    """Build oos_dev_60.jsonl and inscope_dev_100.jsonl deterministically and validate them."""
    cfg = settings or get_settings()
    _rng = random.Random(DEVSET_SEED)
    _ = _rng.random()  # Pin deterministic RNG initialization for reproducibility

    oos_path = cfg.data_golden / "oos_dev_60.jsonl"
    inscope_path = cfg.data_golden / "inscope_dev_100.jsonl"
    corpus_path = cfg.data_processed / "corpus_chunks.parquet"
    golden_path = cfg.data_golden / "golden_150.jsonl"
    attack_map_path = cfg.data_golden / "cve_attack_map.yaml"
    training_queries_path = cfg.models_dir / "router_deberta" / "training_queries.jsonl"

    oos_items = _build_oos_items()
    inscope_items = _build_inscope_items(attack_map_path)

    _atomic_write_jsonl(oos_path, oos_items)
    _atomic_write_jsonl(inscope_path, inscope_items)

    return validate_devsets(
        oos_path=oos_path,
        inscope_path=inscope_path,
        corpus_parquet_path=corpus_path,
        golden_path=golden_path,
        training_queries_path=training_queries_path,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for building and validating calibration devsets."""
    parser = argparse.ArgumentParser(
        description="Build and validate Phase 1.5 calibration devsets."
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build oos_dev_60.jsonl and inscope_dev_100.jsonl, then validate.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing oos_dev_60.jsonl and inscope_dev_100.jsonl.",
    )
    args = parser.parse_args(argv)
    if not args.build and not args.validate:
        parser.error("Must pass --build or --validate")

    cfg = get_settings()
    if args.build:
        summary = build_devsets(cfg)
        print(json.dumps({"action": "build", "status": "ok", **summary}, indent=2))
        return 0

    summary = validate_devsets(
        oos_path=cfg.data_golden / "oos_dev_60.jsonl",
        inscope_path=cfg.data_golden / "inscope_dev_100.jsonl",
        corpus_parquet_path=cfg.data_processed / "corpus_chunks.parquet",
        golden_path=cfg.data_golden / "golden_150.jsonl",
        training_queries_path=cfg.models_dir / "router_deberta" / "training_queries.jsonl",
    )
    print(json.dumps({"action": "validate", "status": "ok", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
