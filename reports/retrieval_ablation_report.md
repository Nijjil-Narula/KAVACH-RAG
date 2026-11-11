# KAVACH-RAG Phase 1.5 CPU-Only Retrieval Ablation Report

- **Generated (UTC):** 2026-11-11T19:22:25.804050+00:00
- **Hardware Provenance:** CPU-only `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel` (10P / 16L cores), OS `Windows-11-10.0.26200-SP0`, torch `2.15.0+cpu` (quantized engine `onednn`), affinity `p_core_distinct`
- **Model Precisions & Truncation:** embed=`bf16`, rerank=`fp32`, router=`fp32`, rerank_max_length=`640`, contingency_l=`False`, abstain_threshold_T=`7.0`
- **Memory & GPU Guard:** Peak RSS=`2948.9 MB` (ceiling `<= 4096.0 MB`), CUDA available=`False`, PID in nvidia-smi=`False`
- **Corpus Chunks:** w512=505, w256=1028
- **Golden Queries:** 150 (140 in-scope, 10 OUT_OF_SCOPE)
- **Embedder:** `codefuse-ai/F2LLM-v2-0.6B` | **Reranker:** `cross-encoder/ettin-reranker-150m-v1` (fallback=`False`)

## 1. Headline Document-Level Ablation Results (140 In-Scope Queries)

| Run | Configuration | k | Recall@1 | Recall@3 | Recall@k | MRR | nDCG@3 | nDCG@k | Mean ms | P50 ms | P95 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 1 | w256 | FlatIndex(1024) full1024 | 3 | 0.6536 | 0.7071 | 0.7071 | 0.8714 | 0.7316 | 0.7316 | 3240.66 | 2021.17 | 18277.77 |
| Run 2 | w512 | FlatIndex(1024) full1024 | 3 | 0.6393 | 0.7071 | 0.7071 | 0.8524 | 0.7267 | 0.7267 | 3240.57 | 2021.12 | 18277.72 |
| Run 3 | w512 | SQ8Index mrl512 | 3 | 0.6107 | 0.6929 | 0.6929 | 0.8107 | 0.7033 | 0.7033 | 3240.75 | 2021.26 | 18277.88 |
| Run 4 | w512 | SQ8Index mrl512 (expanded) | 8 | 0.6107 | 0.7071 | 0.7750 | 0.8301 | 0.7119 | 0.7437 | 3240.75 | 2021.26 | 18277.88 |
| Run 5 | w512 | SQ8-MRL512 + Reranker | 3 | 0.6786 | 0.7286 | 0.7286 | 0.8726 | 0.7576 | 0.7576 | 8714.65 | 8108.91 | 25495.44 |
| Run 6 | w512 | SQ8-MRL512 + Reranker (k=5) | 5 | 0.6786 | 0.7321 | 0.7571 | 0.8821 | 0.7598 | 0.7726 | 8714.65 | 8108.91 | 25495.44 |

## 2. Chunk-Level Ablation Results (Runs 2–6, w512)

| Run | Configuration | k | Recall@1 | Recall@3 | Recall@k | Prec@3 | Prec@k | MRR | nDCG@3 | nDCG@k |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 2 | w512 | FlatIndex(1024) full1024 | 3 | 0.2439 | 0.4089 | 0.4089 | 0.2024 | 0.2024 | 0.4607 | 0.3906 | 0.3906 |
| Run 3 | w512 | SQ8Index mrl512 | 3 | 0.2296 | 0.3989 | 0.3989 | 0.1929 | 0.1929 | 0.4369 | 0.3756 | 0.3756 |
| Run 4 | w512 | SQ8Index mrl512 (expanded) | 8 | 0.2296 | 0.3989 | 0.5361 | 0.1929 | 0.1045 | 0.4751 | 0.3756 | 0.4140 |
| Run 5 | w512 | SQ8-MRL512 + Reranker | 3 | 0.4725 | 0.5096 | 0.5096 | 0.2476 | 0.2476 | 0.6762 | 0.5609 | 0.5609 |
| Run 6 | w512 | SQ8-MRL512 + Reranker (k=5) | 5 | 0.4725 | 0.5096 | 0.5275 | 0.2476 | 0.1586 | 0.6876 | 0.5609 | 0.5421 |

## 3. Per-Question-Type Latency Breakdown (Runs 3 & 5)

| Run | Question Type | Count | Mean ms | P50 ms | P95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Run 3 | `FACTUAL` | 60 | 1871.78 | 1943.35 | 2366.91 |
| Run 3 | `MULTI_HOP` | 45 | 2287.82 | 2435.26 | 2618.19 |
| Run 3 | `CONFLICT_TRAP` | 15 | 1759.66 | 1796.56 | 1982.99 |
| Run 3 | `MALFORMED` | 10 | 2072.99 | 2120.01 | 2454.19 |
| Run 3 | `CONTEXT_NOISE` | 10 | 21044.26 | 22305.10 | 23270.20 |
| Run 3 | `OUT_OF_SCOPE` | 10 | 1328.61 | 1321.61 | 1508.06 |
| Run 5 | `FACTUAL` | 60 | 8373.60 | 9553.09 | 12055.75 |
| Run 5 | `MULTI_HOP` | 45 | 7154.83 | 6947.62 | 9062.56 |
| Run 5 | `CONFLICT_TRAP` | 15 | 5790.15 | 5964.42 | 6622.28 |
| Run 5 | `MALFORMED` | 10 | 7974.21 | 8307.27 | 8930.33 |
| Run 5 | `CONTEXT_NOISE` | 10 | 27770.19 | 28398.30 | 30621.63 |
| Run 5 | `OUT_OF_SCOPE` | 10 | 3851.84 | 3586.72 | 6432.36 |

## 4. Per-Question-Type Retrieval Breakdown (Ungated)

| Run | Question Type | Count | Doc Recall@k | Doc MRR | Doc nDCG@k | Chunk Recall@k | Chunk MRR | Both Versions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Run 1 | FACTUAL | 60 | 0.9667 | 0.9500 | 0.9544 | N/A | N/A | — |
| Run 1 | MULTI_HOP | 45 | 0.4889 | 0.9667 | 0.5945 | N/A | N/A | — |
| Run 1 | CONFLICT_TRAP | 15 | 0.4333 | 0.5556 | 0.4248 | N/A | N/A | 0.2000 (3/15) |
| Run 1 | MALFORMED | 10 | 1.0000 | 0.9333 | 0.9500 | N/A | N/A | — |
| Run 1 | CONTEXT_NOISE | 10 | 0.2500 | 0.3833 | 0.2533 | N/A | N/A | — |
| Run 2 | FACTUAL | 60 | 0.9500 | 0.9333 | 0.9377 | 0.6500 | 0.5000 | — |
| Run 2 | MULTI_HOP | 45 | 0.4778 | 0.9111 | 0.5708 | 0.1344 | 0.5185 | — |
| Run 2 | CONFLICT_TRAP | 15 | 0.5667 | 0.6444 | 0.5431 | 0.2000 | 0.2444 | 0.3333 (5/15) |
| Run 2 | MALFORMED | 10 | 1.0000 | 0.9333 | 0.9500 | 0.9000 | 0.7167 | — |
| Run 2 | CONTEXT_NOISE | 10 | 0.2000 | 0.3333 | 0.2146 | 0.0200 | 0.0333 | — |
| Run 3 | FACTUAL | 60 | 0.9500 | 0.9250 | 0.9315 | 0.6500 | 0.4917 | — |
| Run 3 | MULTI_HOP | 45 | 0.4778 | 0.8593 | 0.5522 | 0.1300 | 0.5037 | — |
| Run 3 | CONFLICT_TRAP | 15 | 0.4667 | 0.5000 | 0.4409 | 0.1333 | 0.1222 | 0.3333 (5/15) |
| Run 3 | MALFORMED | 10 | 1.0000 | 0.9333 | 0.9500 | 0.9000 | 0.7167 | — |
| Run 3 | CONTEXT_NOISE | 10 | 0.1500 | 0.2500 | 0.1613 | 0.0000 | 0.0000 | — |
| Run 4 | FACTUAL | 60 | 0.9833 | 0.9347 | 0.9471 | 0.8333 | 0.5296 | — |
| Run 4 | MULTI_HOP | 45 | 0.5667 | 0.8681 | 0.5974 | 0.1878 | 0.5265 | — |
| Run 4 | CONFLICT_TRAP | 15 | 0.7333 | 0.5778 | 0.5736 | 0.4000 | 0.2161 | 0.5333 (8/15) |
| Run 4 | MALFORMED | 10 | 1.0000 | 0.9333 | 0.9500 | 1.0000 | 0.7367 | — |
| Run 4 | CONTEXT_NOISE | 10 | 0.3000 | 0.3060 | 0.2300 | 0.0600 | 0.0435 | — |
| Run 5 | FACTUAL | 60 | 0.9833 | 0.9833 | 0.9833 | 0.8167 | 0.8083 | — |
| Run 5 | MULTI_HOP | 45 | 0.4667 | 0.8370 | 0.5418 | 0.1478 | 0.6444 | — |
| Run 5 | CONFLICT_TRAP | 15 | 0.7333 | 0.9333 | 0.7786 | 0.3667 | 0.4556 | 0.5333 (8/15) |
| Run 5 | MALFORMED | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — |
| Run 5 | CONTEXT_NOISE | 10 | 0.1000 | 0.1500 | 0.1000 | 0.0200 | 0.0333 | — |
| Run 6 | FACTUAL | 60 | 0.9833 | 0.9833 | 0.9833 | 0.8333 | 0.8125 | — |
| Run 6 | MULTI_HOP | 45 | 0.5444 | 0.8611 | 0.5826 | 0.1700 | 0.6689 | — |
| Run 6 | CONFLICT_TRAP | 15 | 0.7333 | 0.9333 | 0.7786 | 0.4000 | 0.4722 | 0.5333 (8/15) |
| Run 6 | MALFORMED | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — |
| Run 6 | CONTEXT_NOISE | 10 | 0.1500 | 0.1750 | 0.1264 | 0.0200 | 0.0333 | — |

## 5. Calibrated Logit Abstention & Gated Effective Retrieval (Runs 5 & 6)

| Run | k | Threshold T | OOS Abstention | In-Scope False Abstain | Ungated Doc Recall@k | Gated Effective Doc Recall@k | Gated Doc MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 5 | 3 | 7.0 | 1.0000 (10/10) | 0.0214 (3/140) | 0.7286 | 0.7250 | 0.8690 |
| Run 6 | 5 | 7.0 | 1.0000 (10/10) | 0.0214 (3/140) | 0.7571 | 0.7536 | 0.8786 |

### Reranker Score & Raw-Logit Distributions

| Subset | Metric | Count | Min | Median | Mean | P95 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-1 In-Scope | `rerank_raw_logit` | 140 | 6.5969 | 9.9352 | 9.9592 | 11.8809 | 12.0703 |
| Top-1 In-Scope | `rerank_score` | 140 | 0.9986 | 1.0000 | 0.9999 | 1.0000 | 1.0000 |
| Top-1 OUT_OF_SCOPE | `rerank_raw_logit` | 10 | -2.2773 | 0.7901 | 0.8925 | 4.7414 | 5.6210 |
| Top-1 OUT_OF_SCOPE | `rerank_score` | 10 | 0.0930 | 0.6871 | 0.6039 | 0.9868 | 0.9964 |
| Run 5 Gated In-Scope | `rerank_raw_logit` | 411 | 3.3827 | 9.3209 | 9.0850 | 11.3580 | 12.0703 |
| Run 5 Gated In-Scope | `rerank_score` | 411 | 0.9672 | 0.9999 | 0.9994 | 1.0000 | 1.0000 |
| Run 5 Gated OUT_OF_SCOPE | `rerank_raw_logit` | 0 | N/A | N/A | N/A | N/A | N/A |
| Run 5 Gated OUT_OF_SCOPE | `rerank_score` | 0 | N/A | N/A | N/A | N/A | N/A |

## 6. Tier 1 Router & End-to-End Live Pipeline Evaluation

- **Router Accuracy:** 0.9933 (149/150), Latency Mean=65.007 ms / P50=63.753 ms / P95=116.755 ms
- **End-to-End Pipeline OOS Block Rate:** 1.0000 (10/10), **In-Scope False Block Rate:** 0.0000 (0/140), **Latency:** Mean=8060.21 ms / P50=7852.97 ms / P95=25308.91 ms

## 7. Comparison vs Superseded GPU-Hybrid Reference

| Run | CPU Doc MRR | Superseded Ref MRR | Delta MRR | CPU Recall@k | Ref Recall@k | Delta Recall@k | Drift > 0.01 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Run 1 | 0.8714 | 0.8714 | +0.0000 | 0.7071 | 0.7071 | +0.0000 | ok |
| Run 2 | 0.8524 | 0.8524 | +0.0000 | 0.7071 | 0.7071 | +0.0000 | ok |
| Run 3 | 0.8107 | 0.8107 | +0.0000 | 0.6929 | 0.6929 | +0.0000 | ok |
| Run 4 | 0.8301 | 0.8301 | +0.0000 | 0.7750 | 0.7750 | +0.0000 | ok |
| Run 5 | 0.8726 | 0.8762 | -0.0036 | 0.7286 | 0.7286 | +0.0000 | ok |
| Run 6 | 0.8821 | 0.8875 | -0.0054 | 0.7571 | 0.7607 | -0.0036 | ok |
