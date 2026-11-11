"""
kavach.retrieval.train_router — Train and evaluate the Tier 1 DeBERTa router.

Usage:
  python -m kavach.retrieval.train_router
  python -m kavach.retrieval.train_router --verify-disjoint

The router is a 3-class fine-tuned DeBERTa-v3-small:
  Class 0 — SEMANTIC_RAG     (~500 training examples)
  Class 1 — HARD_REFUSAL     (~200 training examples)
  Class 2 — MALFORMED_INPUT  (~100 training examples)

Training is pure PyTorch (no Trainer, no accelerate), CPU, 2 threads.
Model and tokenizer are saved to models/router_deberta/.
Evaluation results are written to reports/router_eval.json.
"""
from __future__ import annotations

import os

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_ROOT, "models", "hf_cache"))

from kavach.runtime_guard import apply_hardware_guard, rss_mb, vram_probe  # noqa: E402

import argparse
import gc
import json
import logging
import random
import sys
import unicodedata
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from kavach.config import Settings, get_settings

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

LABEL_SEMANTIC = 0
LABEL_REFUSAL = 1
LABEL_MALFORMED = 2
CLASS_NAMES = ["SEMANTIC_RAG", "HARD_REFUSAL", "MALFORMED_INPUT"]

ROOT = Path(_PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ─────────────────────────────────────────────────────────────────────────────

_SEED_CVE_IDS: list[str] = [
    "CVE-2024-3094", "CVE-2021-44228", "CVE-2022-26134", "CVE-2023-23397",
    "CVE-2024-21762", "CVE-2021-26855", "CVE-2020-1472", "CVE-2019-0708",
    "CVE-2023-44487", "CVE-2022-30190", "CVE-2021-21985", "CVE-2022-22965",
    "CVE-2023-23376", "CVE-2022-41040", "CVE-2023-36025", "CVE-2024-1709",
    "CVE-2023-27350", "CVE-2022-3786", "CVE-2021-40444", "CVE-2023-20198",
    "CVE-2024-20353", "CVE-2024-6387", "CVE-2024-38112", "CVE-2024-21338",
    "CVE-2023-34362", "CVE-2023-42115", "CVE-2024-30088", "CVE-2024-26169",
    "CVE-2023-28252", "CVE-2024-1086",
]

_SEED_MITRE_TECHNIQUES: list[str] = [
    "T1190 Exploit Public-Facing Application",
    "T1566 Phishing",
    "T1059 Command and Scripting Interpreter",
    "T1078 Valid Accounts",
    "T1486 Data Encrypted for Impact",
    "T1071 Application Layer Protocol",
    "T1021 Remote Services",
    "T1055 Process Injection",
    "T1082 System Information Discovery",
    "T1562 Impair Defenses",
    "T1105 Ingress Tool Transfer",
    "T1003 OS Credential Dumping",
    "T1046 Network Service Discovery",
    "T1548 Abuse Elevation Control Mechanism",
    "T1140 Deobfuscate/Decode Files or Information",
    "TA0001 Initial Access",
    "TA0002 Execution",
    "TA0003 Persistence",
    "TA0005 Defense Evasion",
    "TA0006 Credential Access",
]

_SEED_OWASP_TITLES: list[str] = [
    "LLM01 Prompt Injection",
    "LLM02 Insecure Output Handling",
    "LLM03 Training Data Poisoning",
    "LLM04 Model Denial of Service",
    "LLM05 Supply Chain Vulnerabilities",
    "LLM06 Sensitive Information Disclosure",
    "LLM07 Insecure Plugin Design",
    "LLM08 Excessive Agency",
    "LLM09 Overreliance",
    "LLM10 Model Theft",
]

_SEED_CWE_IDS: list[str] = [
    "CWE-79", "CWE-89", "CWE-22", "CWE-78", "CWE-20", "CWE-787",
    "CWE-125", "CWE-416", "CWE-476", "CWE-119",
]

_SEMANTIC_TEMPLATES: list[str] = [
    "What is the CVSS score for {cve}?",
    "Describe the vulnerability in {cve} and its attack vector.",
    "What mitigations are recommended for {cve}?",
    "Which systems are affected by {cve}?",
    "Has {cve} been exploited in the wild?",
    "What is the patch or workaround for {cve}?",
    "What is the CVSS v3 severity rating for {cve}?",
    "Explain the technical details of the {cve} vulnerability.",
    "What authentication bypass does {cve} enable?",
    "How does {cve} impact network-facing services?",
    "What MITRE ATT&CK technique does the {cve} attack chain use?",
    "Is {cve} listed in the CISA Known Exploited Vulnerabilities catalog?",
    "What is the attack complexity of {cve}?",
    "How was {cve} first publicly disclosed?",
    "What software components does {cve} affect?",
    "Describe the MITRE technique {technique} and its sub-techniques.",
    "What defences counter {technique}?",
    "Which threat actors use {technique}?",
    "How is {technique} detected in network logs?",
    "Provide detection rules for {technique} in SIEM.",
    "Explain the OWASP risk {owasp} and provide mitigation guidance.",
    "How does {owasp} differ from traditional injection vulnerabilities?",
    "What testing methods identify {owasp} in LLM deployments?",
    "List real-world examples of {owasp} exploitation.",
    "What is the weakness described in {cwe}?",
    "How can {cwe} be exploited by an attacker?",
    "What coding practices prevent {cwe}?",
    "Which languages are most susceptible to {cwe}?",
    "What security tools detect {cwe} in source code?",
    "What is the relationship between {cwe} and memory safety?",
    "Compare the CVSS scores of {cve} and other critical CVEs from 2024.",
    "What is the difference between a supply chain attack and {owasp}?",
    "How does network segmentation reduce exposure to {technique}?",
    "What endpoint detection capabilities catch {technique} in real time?",
    "Summarise the incident response steps following exploitation of {cve}.",
    "What logging and monitoring controls detect exploitation of {cve}?",
    "What vendor advisories exist for {cve}?",
    "How do cloud provider security groups mitigate {cve} exposure?",
    "List indicators of compromise associated with {cve} exploitation.",
    "How does {owasp} affect agentic LLM systems with tool access?",
    "What is the difference between direct and indirect {owasp}?",
    "How does MITRE {technique} relate to ransomware kill chains?",
    "What open-source tools detect exploitation of {cve}?",
    "How does patch management reduce exposure to critical CVEs like {cve}?",
    "What is the EPSS score for {cve}?",
    "Which MITRE mitigation technique addresses {technique}?",
    "What NVD enrichment data is available for {cve}?",
    "How is {cwe} categorised in the CWE Top 25?",
    "Provide YARA rules that detect artefacts associated with {cve}.",
    "What container security controls mitigate {owasp}?",
    "How do adversaries chain {technique} with lateral movement?",
    "What red team tooling simulates {technique}?",
    "What is the relationship between {cve} and supply chain security?",
    "Describe the memory corruption caused by {cwe} in C/C++ code.",
    "How do automated scanners identify instances of {cwe}?",
    "What is the CVSS environmental score adjustment for {cve} in cloud deployments?",
    "How did threat actors exploit {cve} in documented campaigns?",
    "List the CPE identifiers affected by {cve}.",
    "What is the temporal CVSS score change for {cve} after patch release?",
    "How does {technique} evade traditional antivirus detection?",
    "What privilege escalation does {cve} enable?",
    "How is {cve} remediated in containerised environments?",
    "What firewalling rules block exploitation of {cve}?",
    "Explain how {owasp} can be used to exfiltrate training data.",
    "What governance controls address {owasp} risk in enterprise LLM use?",
    "How does {cwe} manifest differently in interpreted versus compiled languages?",
    "What static analysis rules flag potential {cwe} in Java code?",
    "How does {technique} use legitimate system binaries to avoid detection?",
    "What is the relationship between {cve} and zero-day exploitation?",
    "How does {technique} facilitate credential theft?",
    "What network traffic signatures indicate {technique} activity?",
    "Provide a timeline of {cve} from disclosure to patch availability.",
    "How does adversarial robustness testing relate to {owasp}?",
]

# >=120 off-topic prompts for HARD_REFUSAL — kept disjoint from golden OOS queries
_REFUSAL_TEMPLATES: list[str] = [
    "What ingredients go into a classic tiramisu?",
    "Give me a recipe for homemade pasta dough.",
    "How do I score a perfect cricket century?",
    "Who won the most recent FIFA World Cup?",
    "What is the weather forecast for Mumbai this weekend?",
    "Best hiking trails in the Swiss Alps?",
    "How do I invest in index funds for retirement?",
    "What films won the Palme d'Or in 2023?",
    "Tell me how to bake chocolate chip cookies.",
    "What is the best football team in the Premier League?",
    "How do I make authentic ramen broth at home?",
    "What is the plot of the novel 1984 by George Orwell?",
    "How do I train for a 5K run as a beginner?",
    "Give me tips for growing tomatoes in a small garden.",
    "What are the best beaches in Thailand?",
    "How do I refinance my home mortgage?",
    "What yoga poses help with lower back pain?",
    "Who is the highest paid actor in Hollywood?",
    "How do I make a classic French omelette?",
    "What is the best smartphone to buy in 2024?",
    "Give me a vegetarian lasagne recipe.",
    "How do I learn to play the guitar from scratch?",
    "What are the best universities for studying economics?",
    "How do I write a cover letter for a marketing job?",
    "What is the capital of Brazil?",
    "How do I learn to meditate for stress relief?",
    "What are the top tourist attractions in Japan?",
    "How do I make espresso without a machine?",
    "Tell me about the history of the Roman Empire.",
    "What is the best diet for weight loss?",
    "How do I start a podcast?",
    "What is the difference between whisky and whiskey?",
    "Give me tips for taking better travel photos.",
    "How do I invest in cryptocurrency responsibly?",
    "What are the signs of vitamin D deficiency?",
    "How do I grow my social media following?",
    "What is the best way to learn Spanish quickly?",
    "Tell me about the life of Leonardo da Vinci.",
    "How do I make a perfect margarita?",
    "What are the benefits of cold water swimming?",
    "How do I write a short story with a twist ending?",
    "What national parks are worth visiting in the USA?",
    "How do I fix a dripping tap at home?",
    "What are the best board games for a family game night?",
    "How do I make sourdough starter from scratch?",
    "What is the proper way to taste wine?",
    "Tell me about the life cycle of a butterfly.",
    "How do I negotiate a higher salary?",
    "What is the best way to learn chess strategy?",
    "How do I produce a music album on a low budget?",
    "What are the top programming languages for data science?",
    "Tell me how photosynthesis works.",
    "What are the healthiest superfoods to add to my diet?",
    "How do I build muscle mass at home without equipment?",
    "What is the best app for learning a new language?",
    "Give me a simple curry recipe for a weeknight dinner.",
    "How does the stock market work?",
    "What is the best way to plan a road trip across Europe?",
    "Tell me interesting facts about black holes.",
    "How do I start journaling for mental health?",
    "What is the best way to clean a cast iron pan?",
    "How do I audition for a theatre production?",
    "What are the most popular sports in Australia?",
    "Tell me about the history of jazz music.",
    "How do I make kombucha at home?",
    "What is the difference between AI and machine learning?",
    "How do I create a successful budget for my household?",
    "What is the plot of Romeo and Juliet?",
    "How do I get a visa for visiting the United States?",
    "Tell me how to care for indoor succulents.",
    "What are the rules of cricket?",
    "How do I start a small business from scratch?",
    "What is the best way to study for exams?",
    "How do I make a classic beef burger?",
    "Tell me about the history of the Olympics.",
    "What is the best skincare routine for oily skin?",
    "How do I write a business plan?",
    "What are the best streaming services right now?",
    "Tell me about the life of Marie Curie.",
    "How do I make a traditional paella?",
    "What exercises help reduce belly fat?",
    "What are the benefits of intermittent fasting?",
    "Tell me about the best novels published in 2024.",
    "How do I learn to draw portraits?",
    "What is the best way to teach my child to read?",
    "How do I make my own candles at home?",
    "What are the top travel destinations in South America?",
    "How do I fall asleep faster?",
    "Tell me about the discovery of penicillin.",
    "How do I invest in real estate with little money?",
    "What is the difference between vegan and vegetarian diets?",
    "Tell me how to care for a rescue dog.",
    "What are the best ways to reduce household energy bills?",
    "How do I make my garden more wildlife-friendly?",
    "What is the history of the Beatles?",
    "How do I make my own bread at home?",
    "What are the best podcasts about history?",
    "How do I get started with oil painting?",
    "Tell me about effective parenting strategies for teenagers.",
    "What is the best way to preserve seasonal vegetables?",
    "How do I write a compelling novel opening chapter?",
    "What are the safest countries for solo female travel?",
    "Tell me how volcanoes form.",
    "How do I create a capsule wardrobe?",
    "What are the best productivity techniques for remote work?",
    "How do I cook the perfect steak?",
    "What are the signs of burnout and how do I recover?",
    "Tell me about ancient Egyptian civilization.",
    "How do I learn to surf?",
    "What is the best way to plan a wedding on a budget?",
    "How do I make my own cheese at home?",
    "What are the classic novels everyone should read?",
    "How do I become a freelance graphic designer?",
    "Tell me about the life of Winston Churchill.",
    "How do I develop a morning routine for productivity?",
    "What is the best hiking gear for beginners?",
    "Tell me about the life of Albert Einstein.",
    "How do I make my own sushi at home?",
    "What are the best board games for two players?",
    "How do I write a good poem?",
    "Tell me about the history of coffee.",
    "How do I learn to knit from scratch?",
    "What are the top European cities to visit in winter?",
    "How do I grow herbs on my windowsill?",
    "What is the best way to volunteer abroad?",
    "Tell me about the history of democracy.",
    "How do I make a traditional Mexican guacamole?",
]

_MALFORMED_TEMPLATES: list[str] = [
    "cve2024-3094 xz util liblzma bckdoor",
    "cvss 9.8 vuln xz util mitgation",
    "log4shll rce exploi patc",
    "cwe 787 out of bound write mtigtion",
    "mitre t1190 exploi pub facin app defen",
    "owasp llm0 injctn mitigte",
    "cve 2021 44228 log4j log4shll rmve fix",
    "deberta smll routr 3clas classifr",
    "vulnrability disclsr patc 0day crtical",
    "xz utils backdor sshd rsauthentication disble",
    "sql injecton cww 89 prevnt sanitiz",
    "phishin spearphish malicous email atak",
    "reansomware encryptd data impct mitigat",
    "ldap injctn log4shll jndi lookp",
    "cve2022-26134 confluenc ognl injctn",
    "mtre attck chain inital accs persistnce",
    "owasp llm03 poisnin traning data mniplte",
    "cve 2023 23397 outlook ntlm hash credntl",
    "zerodayy exploit publicfacng appication rce",
    "crtical vuln cvss 10.0 patch immeditly",
    "vuln xz util liblzma backdor",
    "mitgation cve rce exploi",
    "cwe22 path travsal fil inlcusion",
    "owasp llm prompt injctn directt indirct",
    "mtre t1566 phisin spearphsh link attchmnt",
    "log4j log4shll jndi ldap rce critcal",
    "cve 2024 21762 fortios sslvpn rce",
    "mitre ta001 inital acces techniqs",
    "owasp llm overrelnc overrely hallucnat",
    "cve 2020 1472 zerologon priv escalt",
    "cwe 416 use aftr fre memry corpt",
    "mtre m1051 updte softwr patchg",
    "cve 2023 44487 http2 rapid rst dos",
    "sqli xss csrf owasp top10 preventn",
    "revrs shell webshll persistnce backdor",
    "privlge escaltn tok impersonatn windws",
    "bufr ovflw stck smsh rce explot",
    "xss dom base reflctd store persistnt",
    "csrf tokn validtn prevnt mitigat",
    "cve 2019 0708 bluekep rdp worm",
    "smb eternalblue ms17-010 exploiton",
    "cve 2021 26855 proxylogon exchng",
    "heapspry jit spray exploitatin",
    "aslr dep nx bit mitigatin byps",
    "vulnerblity scannr nessus openvas sast",
    "pentest recon enum exploi post-exploi",
    "cve 2024 6387 openssh regressn rce",
    "ldap injctn activdir autheticatn bypass",
    "cve 2022 30190 follina msdt msoffic",
    "mtre t1055 procs injctn code exec",
    "owasp llm supplychan vul dpndncy",
    "xz util backdor systemd liblzm brkdoor",
    "cve 2023 20198 ios xe unauth accss",
    "cve 2023 34362 moveit sql inj",
    "mtre t1486 dat encryptd rnsmwr impct",
    "nvd cvss v3 basescre cvssvctr",
    "cisa kev knwn exploitd vulnrblts catlog",
    "epss exploitabity prdictn scre cve",
    "cpe identfr affctd softwr nvd databas",
    "owasp llm08 excsive agncy toool use",
    "cve 2024 1086 linux krnl use aftr fre",
    "reachbl code anaysis sast dast iast",
    "devsecops shift lft secrity pipline ci",
    "cve 2023 28252 windws clfs priv escalt",
    "mtre t1021 remte srvcs rdp ssh smb",
    "cwe 125 out of bound red buffer",
    "owasp llm05 supplychn vulnrablt modl",
    "cve 2022 3786 openssl pufferfish ovfl",
    "cve 2024 20353 ciscio asa ftd dos crtcl",
    "mtre t1562 impair defns log del",
    "cve 2024 26169 winfws certsvc priv esc",
    "heap spraying rce brwsr exploitaton",
    "cve 2023 36025 windws smartscren byps",
    "owasp llm04 mdl denial svc dos resrc",
    "mtre t1046 ntwrk svc discvry nmap",
    "cwe 476 null pointr derrefernc crash",
    "cve 2021 40444 activex mshtml rce offc",
    "mtre t1003 oscrednl dmpng lsass",
    "owasp llm02 insecur ouput handlg xss",
    "cve 2024 38112 windws mshtml rce 0day",
    "mtre t1548 abse elev ctrl mec uac byps",
    "cve 2024 1709 connectwis screenconn auth byps",
    "cve 2024 30088 windws krnl priv escalt",
    "cve 2022 41040 proxynotshell exchng",
    "owasp llm07 insecr plugn desig api",
    "mtre t1140 deobfsc decode fil info",
    "cve 2023 42115 exim smtp rce authentitn",
    "cwe 20 improppr inpt validtn sanitizn",
    "cve 2021 21985 vcntr servr rce plugin",
    "cve 2022 22965 spring4shll spring frmwk",
    "mtre t1105 ingrss tool trnsfr malwr downld",
    "owasp llm10 mdl theft model extrctn",
    "cve 2024 21338 windws applocker byps priv",
    "cwe 78 improppr os cmd injectn shell",
    "cve 2022 3786 openssl buffr overun puny",
]


def _load_corpus_entities() -> tuple[list[str], list[str], list[str]]:
    """Load CVE IDs, MITRE technique names, and OWASP titles from raw corpus dirs."""
    cve_ids: list[str] = list(_SEED_CVE_IDS)
    mitre_names: list[str] = list(_SEED_MITRE_TECHNIQUES)
    owasp_titles: list[str] = list(_SEED_OWASP_TITLES)

    cve_dir = ROOT / "data" / "raw" / "cve_json"
    if cve_dir.exists():
        for f in sorted(cve_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                cve_id = data.get("id") or data.get("cveId") or f.stem
                if cve_id and cve_id not in cve_ids:
                    cve_ids.append(cve_id)
            except Exception as exc:
                log.warning("Skipping unreadable CVE file %s: %s", f.name, exc)

    mitre_dir = ROOT / "data" / "raw" / "mitre_markdown"
    if mitre_dir.exists():
        for f in sorted(mitre_dir.glob("*.md")):
            stem = f.stem
            if stem not in mitre_names:
                mitre_names.append(stem)

    owasp_dir = ROOT / "data" / "raw" / "owasp_markdown"
    if owasp_dir.exists():
        for f in sorted(owasp_dir.rglob("*.md")):
            stem = f.stem
            if stem not in owasp_titles:
                owasp_titles.append(stem)

    return cve_ids, mitre_names, owasp_titles


def _fill_template(
    template: str,
    cve_ids: list[str],
    mitre_names: list[str],
    owasp_titles: list[str],
    cwe_ids: list[str],
    rng: random.Random,
) -> str:
    """Fill placeholders in a template string."""
    result = template
    if "{cve}" in result:
        result = result.replace("{cve}", rng.choice(cve_ids))
    if "{technique}" in result:
        result = result.replace("{technique}", rng.choice(mitre_names))
    if "{owasp}" in result:
        result = result.replace("{owasp}", rng.choice(owasp_titles))
    if "{cwe}" in result:
        result = result.replace("{cwe}", rng.choice(cwe_ids))
    return result


def _normalize(text: str) -> str:
    """Simple normalisation for disjoint checks."""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    return " ".join(text.split())


def generate_training_data(rng: random.Random) -> list[tuple[str, int]]:
    """Generate ~800 labelled examples across the 3 Track B intent classes."""
    cve_ids, mitre_names, owasp_titles = _load_corpus_entities()
    cwe_ids = _SEED_CWE_IDS

    examples: list[tuple[str, int]] = []

    # SEMANTIC_RAG: 500 examples
    filled_semantic: list[str] = []
    seen_semantic: set[str] = set()
    attempts = 0
    while len(filled_semantic) < 500 and attempts < 3000:
        tmpl = rng.choice(_SEMANTIC_TEMPLATES)
        text = _fill_template(tmpl, cve_ids, mitre_names, owasp_titles, cwe_ids, rng)
        if text not in seen_semantic:
            seen_semantic.add(text)
            filled_semantic.append(text)
        attempts += 1
    for text in filled_semantic:
        examples.append((text, LABEL_SEMANTIC))

    # HARD_REFUSAL: 200 examples from pool of >=120 unique templates
    all_refusal = list(_REFUSAL_TEMPLATES)
    extras = [f"{t} Please give detailed advice." for t in all_refusal[:60]]
    all_refusal = all_refusal + extras
    rng.shuffle(all_refusal)
    for text in all_refusal[:200]:
        examples.append((text, LABEL_REFUSAL))

    # MALFORMED_INPUT: 100 examples
    all_malformed = list(_MALFORMED_TEMPLATES)
    while len(all_malformed) < 100:
        base = rng.choice(_MALFORMED_TEMPLATES)
        suffix = rng.choice(["pls help", "explain", "???", "urgent", "need info"])
        all_malformed.append(f"{base} {suffix}")
    rng.shuffle(all_malformed)
    for text in all_malformed[:100]:
        examples.append((text, LABEL_MALFORMED))

    rng.shuffle(examples)
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class RouterDataset(Dataset):
    """PyTorch Dataset storing raw text and integer class labels."""

    def __init__(self, examples: list[tuple[str, int]]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self._examples[idx]


def _make_collate_fn(tokenizer: AutoTokenizer, max_length: int):
    """Build a dynamic-padding collate function for fast CPU batching."""

    def _collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": torch.tensor(labels, dtype=torch.long),
        }

    return _collate


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def train(
    examples: list[tuple[str, int]],
    base_model_id: str,
    save_dir: Path,
    epochs: int = 3,
    lr: float = 3e-5,
    batch_size: int = 16,
    max_length: int = 128,
) -> dict[str, Any]:
    """Fine-tune DeBERTa-v3-small on *examples* and return eval metrics.

    Trains encoder + classification head in float32 (keeping word_embeddings
    frozen so RSS stays well below Settings.rss_ceiling_mb (4096.0 MB) and AdamW
    steps do not underflow bfloat16 precision), then converts to bfloat16 for
    evaluation and saving.
    """
    apply_hardware_guard()
    torch.manual_seed(42)
    np.random.seed(42)

    split_idx = int(len(examples) * 0.8)
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]

    log.info(
        "Training set: %d  |  Eval set: %d", len(train_examples), len(eval_examples)
    )

    log.info("Loading tokenizer: %s", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    log.info("Loading model: %s (3-class head)", base_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=3,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    model.config.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

    # Freeze large word_embeddings table (128k vocab) to avoid overfitting & RSS spikes
    word_emb = getattr(
        getattr(getattr(model, "deberta", None), "embeddings", None),
        "word_embeddings",
        None,
    )
    if word_emb is not None:
        word_emb.weight.requires_grad_(False)
        log.info("Froze word_embeddings during fine-tuning (RSS=%.1f MB).", rss_mb())

    collate_fn = _make_collate_fn(tokenizer, max_length=max_length)
    train_ds = RouterDataset(train_examples)
    eval_ds = RouterDataset(eval_examples)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits.float()
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(len(train_loader), 1)
        log.info(
            "Epoch %d/%d  avg_loss=%.4f  RSS=%.1f MB",
            epoch,
            epochs,
            avg_loss,
            rss_mb(),
        )

    # Free optimizer state and cast to bfloat16 so eval matches runtime inference dtype
    del optimizer
    gc.collect()
    model = model.to(torch.bfloat16)
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.inference_mode():
        for batch in eval_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            preds = outputs.logits.float().argmax(dim=-1).tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].tolist())

    correct = sum(p == t for p, t in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0.0

    conf_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for pred, true in zip(all_preds, all_labels):
        conf_matrix[true][pred] += 1

    log.info("Eval accuracy (bf16): %.4f  (%d/%d)", accuracy, correct, len(all_labels))
    log.info("Confusion matrix: %s", conf_matrix)

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    log.info("Model saved to %s", save_dir)

    return {
        "held_out_accuracy": round(accuracy, 4),
        "confusion_matrix": conf_matrix,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "peak_rss_mb": round(rss_mb(), 1),
        "vram_probe": dict(vram_probe()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Disjoint verification (AMENDMENT A1)
# ─────────────────────────────────────────────────────────────────────────────


def verify_disjoint(
    examples: list[tuple[str, int]],
    golden_path: Path,
) -> tuple[list[tuple[str, int]], bool, list[dict[str, str]]]:
    """Check for overlap between training examples and golden_150.jsonl.

    Raises FileNotFoundError if golden_path does not exist (AMENDMENT A1).
    """
    if not golden_path.exists():
        raise FileNotFoundError(
            f"golden_150.jsonl not found at {golden_path}. Run golden.py --build first."
        )

    golden_queries: list[str] = []
    oos_golden_norm: list[str] = []
    with golden_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            q = item.get("query", "")
            if q:
                golden_queries.append(q)
                if item.get("question_type") == "OUT_OF_SCOPE":
                    oos_golden_norm.append(_normalize(q))

    golden_norm = [_normalize(q) for q in golden_queries]
    log.info(
        "Loaded %d golden queries (%d OOS) for disjoint check.",
        len(golden_queries),
        len(oos_golden_norm),
    )

    overlap_pairs: list[dict[str, str]] = []
    keep: list[tuple[str, int]] = []

    for text, label in examples:
        norm_train = _normalize(text)
        conflict = False
        for golden_q, golden_n in zip(golden_queries, golden_norm):
            if not norm_train or not golden_n:
                continue
            if norm_train == golden_n:
                overlap_pairs.append(
                    {
                        "train": text,
                        "golden": golden_q,
                        "overlap_type": "exact_normalised",
                    }
                )
                conflict = True
                break
            if norm_train in golden_n or golden_n in norm_train:
                overlap_pairs.append(
                    {
                        "train": text,
                        "golden": golden_q,
                        "overlap_type": "substring",
                    }
                )
                conflict = True
                break
        if not conflict:
            keep.append((text, label))

    overlap_found = len(overlap_pairs) > 0
    if overlap_found:
        log.warning(
            "Overlap found: %d training examples conflict with golden dataset. "
            "Removing from training data.",
            len(overlap_pairs),
        )
        for pair in overlap_pairs:
            log.warning(
                "  OVERLAP (%s): '%s' <-> '%s'",
                pair["overlap_type"],
                pair["train"][:60],
                pair["golden"][:60],
            )
    else:
        log.info("No overlap found between training data and golden dataset.")

    # Invariant: zero overlap remains with golden OOS queries
    for text, _ in keep:
        norm_train = _normalize(text)
        for golden_n in oos_golden_norm:
            assert norm_train != golden_n, (
                f"Training example matches OOS golden query!\n"
                f"  Train: {text!r}\n  Golden: {golden_n!r}"
            )

    return keep, overlap_found, overlap_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def _write_eval_json(
    path: Path,
    metrics: dict[str, Any],
    verify_disjoint_run: bool,
    overlap_found: bool,
    overlap_pairs: list[dict[str, str]],
) -> None:
    """Write reports/router_eval.json atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "held_out_accuracy": metrics["held_out_accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_names": CLASS_NAMES,
        "train_examples": metrics["train_examples"],
        "eval_examples": metrics["eval_examples"],
        "verify_disjoint_run": verify_disjoint_run,
        "overlap_found": overlap_found,
        "overlap_pairs": overlap_pairs,
        "peak_rss_mb": metrics["peak_rss_mb"],
        "vram_probe": metrics["vram_probe"],
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    log.info("Written: %s", path)


def get_deterministic_splits(
    golden_path: Path | None = None,
    apply_disjoint: bool = True,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Deterministically generate synthetic router examples and 80/20 splits (seed=42)."""
    rng = random.Random(42)
    examples = generate_training_data(rng)
    g_path = (
        Path(golden_path)
        if golden_path is not None
        else get_settings().data_golden / "golden_150.jsonl"
    )
    if apply_disjoint and g_path.exists():
        examples, _, _ = verify_disjoint(examples, g_path)
    split_idx = int(len(examples) * 0.8)
    return examples[:split_idx], examples[split_idx:]


def dump_training_queries(output_path: Path | None = None) -> Path:
    """Write deterministic synthetic train/eval queries atomically to JSONL without loading models."""
    settings = get_settings()
    target = (
        Path(output_path)
        if output_path is not None
        else settings.models_dir / "router_deberta" / "training_queries.jsonl"
    )
    train_examples, eval_examples = get_deterministic_splits(
        golden_path=settings.data_golden / "golden_150.jsonl",
        apply_disjoint=True,
    )

    lines: list[str] = []
    for split_name, subset in (("train", train_examples), ("eval", eval_examples)):
        for text, label_id in subset:
            record = {
                "query": text,
                "label": CLASS_NAMES[label_id],
                "label_id": int(label_id),
                "split": split_name,
            }
            lines.append(json.dumps(record, ensure_ascii=False))

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    log.info(
        "Dumped %d training queries (%d train, %d eval) to %s",
        len(lines),
        len(train_examples),
        len(eval_examples),
        target,
    )
    return target


def evaluate_router_precision(
    precision: Literal["bf16", "fp32", "int8"],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Evaluate trained Tier 1 router on golden_150.jsonl and held-out synthetic split."""
    from kavach.retrieval.router import AgentIntentRouter
    from kavach.retrieval.schema import GoldenItem

    apply_hardware_guard()
    base_cfg = settings or get_settings()
    cfg = base_cfg.model_copy(update={"router_precision": precision})

    router = AgentIntentRouter(cfg)
    router.load()

    # 1. Evaluate golden_150.jsonl (full Track A + Track B routing)
    golden_path = cfg.data_golden / "golden_150.jsonl"
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {golden_path}")

    golden_items = [
        GoldenItem.model_validate_json(line)
        for line in golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    golden_correct = 0
    golden_misclassified: list[dict[str, Any]] = []
    for item in golden_items:
        dec = router.route(item.query)
        if dec.route == item.expected_route:
            golden_correct += 1
        else:
            golden_misclassified.append(
                {
                    "test_id": item.test_id,
                    "query": item.query,
                    "expected_route": item.expected_route,
                    "predicted_route": dec.route,
                    "confidence": round(float(dec.confidence), 6),
                }
            )
    golden_acc = golden_correct / len(golden_items) if golden_items else 0.0

    # 2. Evaluate deterministic held-out synthetic split (Track B classifier directly)
    _, eval_examples = get_deterministic_splits(
        golden_path=golden_path,
        apply_disjoint=True,
    )
    assert router._tokenizer is not None and router._model is not None
    conf_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    held_out_correct = 0
    for text, true_label in eval_examples:
        enc = router._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False,
        )
        with torch.inference_mode():
            logits = router._model(**enc).logits
        pred_label = int(logits[0].float().argmax(dim=-1).item())
        conf_matrix[int(true_label)][pred_label] += 1
        if pred_label == int(true_label):
            held_out_correct += 1

    held_out_acc = held_out_correct / len(eval_examples) if eval_examples else 0.0

    return {
        "precision": precision,
        "golden_total": len(golden_items),
        "golden_correct": golden_correct,
        "golden_accuracy": round(golden_acc, 6),
        "golden_misclassified": golden_misclassified,
        "held_out_total": len(eval_examples),
        "held_out_correct": held_out_correct,
        "held_out_accuracy": round(held_out_acc, 6),
        "held_out_confusion_matrix": conf_matrix,
        "rss_mb": round(rss_mb(), 2),
        "vram_probe": dict(vram_probe()),
    }


def main(argv: list[str] | None = None) -> None:
    """Train the router; optionally verify disjointness or dump synthetic queries."""
    apply_hardware_guard()

    parser = argparse.ArgumentParser(description="Train KAVACH-RAG intent router.")
    parser.add_argument(
        "--verify-disjoint",
        action="store_true",
        help="Verify no overlap with golden_150.jsonl before training.",
    )
    parser.add_argument(
        "--dump-train-set",
        action="store_true",
        help="Deterministically write synthetic train/eval queries JSONL without loading DeBERTa.",
    )
    args = parser.parse_args(argv)

    if args.dump_train_set:
        out_path = dump_training_queries()
        print(f"Wrote deterministic training queries to {out_path}", flush=True)
        return

    settings = get_settings()
    rng = random.Random(42)

    save_dir = settings.models_dir / "router_deberta"
    eval_path = settings.reports_dir / "router_eval.json"
    golden_path = settings.data_golden / "golden_150.jsonl"

    log.info("Generating synthetic training data...")
    examples = generate_training_data(rng)
    log.info("Total examples: %d", len(examples))

    overlap_found = False
    overlap_pairs: list[dict[str, str]] = []
    verify_disjoint_run = bool(args.verify_disjoint)

    if args.verify_disjoint:
        log.info("Running disjoint verification against %s ...", golden_path)
        try:
            examples, overlap_found, overlap_pairs = verify_disjoint(
                examples, golden_path
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr, flush=True)
            sys.exit(1)
        log.info(
            "After disjoint check: %d training examples remain.", len(examples)
        )

    base_model_id = settings.router_base_model_id
    log.info("Starting training from base model: %s", base_model_id)
    metrics = train(
        examples=examples,
        base_model_id=base_model_id,
        save_dir=save_dir,
        epochs=3,
        lr=3e-5,
        batch_size=16,
        max_length=128,
    )

    _write_eval_json(
        eval_path,
        metrics=metrics,
        verify_disjoint_run=verify_disjoint_run,
        overlap_found=overlap_found,
        overlap_pairs=overlap_pairs,
    )
    log.info("Training complete. Accuracy: %.4f", metrics["held_out_accuracy"])


if __name__ == "__main__":
    main()
