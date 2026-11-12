# KAVACH-RAG Phase 4 Audit Report: Two-Stage Cross-Family Judge Cascade

- Generated at (UTC): 2026-11-12T09:18:41Z
- Items graded: 150 (checkpoint file `data/processed/tier6_eval_checkpoints.jsonl`)
- Judges: `gemini-3.5-flash-lite` (thinking_applied=True), `gemini-3.8-flash` (thinking_applied=True), `openai/gpt-oss-120b` (thinking_applied=None). No Qwen model judges Qwen output.
- Stage mix: {'STAGE_2_JURY_CONSENSUS': 20, 'STAGE_1_BASELINE': 120, 'STAGE_0_DETERMINISTIC': 10}; jury panels with a failed member: 18

## 1. Verdict distribution

| Question type | PASS | PARTIAL | FAIL | Faithfulness mean |
|---|---:|---:|---:|---:|
| CONFLICT_TRAP | 8 | 6 | 1 | 0.7567 |
| CONTEXT_NOISE | 8 | 0 | 2 | 0.8000 |
| FACTUAL | 53 | 6 | 1 | 0.9217 |
| MALFORMED | 6 | 4 | 0 | 0.9340 |
| MULTI_HOP | 30 | 3 | 12 | 0.7000 |
| OUT_OF_SCOPE | 10 | 0 | 0 | 1.0000 |
| **Total** | **115** | **19** | **16** | **0.8366** |

## 2. CI gate metrics

| Metric | Observed | Threshold | Gated |
|---|---:|---:|---|
| Average faithfulness, all 150 items | 0.8366 | >= 0.85 | yes |
| Out-of-scope refusal rate | 10/10 | 100% | yes |
| FAIL verdicts | 16 | <= 12 | yes |
| Faithfulness, answered items only (n=92) | 0.7336 | reported | no |
| Context relevance, in-scope items | 0.5893 | reported (retrieval-bound) | no |
| Context relevance, context-sufficient subset (n=72) | 0.9792 | target >= 0.90 | no |
| Conflict-trap items with versions kept separate | 9/15 | reported | no |

Faithfulness over all items counts a correct refusal as faithful. Answered-only faithfulness is the stricter view of hallucination risk. Overall relevance is dominated by retrieval misses on multi-hop and context-noise items, not by generation quality.

## 3. Failures

FAIL test_ids: TEST_045, TEST_062, TEST_072, TEST_073, TEST_084, TEST_086, TEST_092, TEST_093, TEST_098, TEST_099, TEST_101, TEST_102, TEST_104, TEST_117, TEST_145, TEST_146

## 4. Reference: Phase 2 single-stage judge

- Faithfulness mean 0.8628, relevance mean 0.5679, relevance on context-sufficient items 0.9236. The Phase 2 judge scores claim-level support on served items only; the Phase 4 cascade scores all items on a PASS/PARTIAL/FAIL scale, so the numbers are related but not identical.

## 5. Human calibration

- Items: 30 | observed agreement 0.9333 | chance agreement 0.3778
- Cohen's kappa: **0.8929** (bootstrap 95% CI [0.7207, 1.0]) against the 0.70 floor: **PASS**

| Human \ Judge | PASS | PARTIAL | FAIL |
|---|---:|---:|---:|
| PASS | 14 | 0 | 0 |
| PARTIAL | 1 | 9 | 0 |
| FAIL | 0 | 1 | 5 |

Limits: n=30 gives a wide confidence interval; the review set over-samples non-PASS judge verdicts to avoid a PASS-dominated chance baseline.
