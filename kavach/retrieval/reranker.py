"""
kavach.retrieval.reranker — Tier 4 Cross-Encoder Reranker.

Supports both:
  1. Sentence-Transformers v5 modular CrossEncoders (such as
     ``cross-encoder/ettin-reranker-150m-v1``, composed of ``ModernBertModel`` +
     ``1_Pooling`` CLS + ``2_Dense`` GELU + ``3_LayerNorm`` + ``4_Dense``
     score projection).
  2. Standard HuggingFace ``AutoModelForSequenceClassification`` cross-encoders
     (such as fallback ``cross-encoder/ms-marco-MiniLM-L-6-v2``).

Eagerly loads on CPU under the configured ``rerank_precision`` (``fp32``,
``bf16``, or backbone-only dynamic ``int8`` with ``fp32`` head), constructs
pairs passage-first (raising ``RetrievalError`` if any passage exceeds the
token budget; truncating query tokens only), sorts descending by
``rerank_raw_logit``, and attaches informational ``rerank_score`` (sigmoid).
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

from kavach.runtime_guard import apply_hardware_guard  # noqa: E402

import dataclasses
import gc
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from kavach.config import Settings, get_settings
from kavach.exceptions import ModelLoadError, RetrievalError
from kavach.retrieval.quantization import quantize_linear_layers

log = logging.getLogger(__name__)

_FALLBACK_RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANK_BATCH_SIZE = 4
_PASSAGE_MAX_TOKENS = 512

_ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "torch.nn.modules.activation.GELU": nn.GELU,
    "torch.nn.modules.activation.Tanh": nn.Tanh,
    "torch.nn.modules.activation.ReLU": nn.ReLU,
    "torch.nn.modules.activation.SiLU": nn.SiLU,
    "torch.nn.modules.linear.Identity": nn.Identity,
}

_BACKBONE_DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "int8": torch.float32,
}


@dataclasses.dataclass
class _ClassifierOutput:
    """Minimal container matching SequenceClassifierOutput.logits."""

    logits: torch.Tensor


def _resolve_repo_file(
    model_id: str, filename: str, local_files_only: bool = False
) -> Optional[Path]:
    """Return local path to *filename* from local directory or HuggingFace Hub."""
    local_dir = Path(model_id)
    if local_dir.is_dir():
        candidate = local_dir / filename
        return candidate if candidate.is_file() else None
    try:
        resolved = hf_hub_download(
            repo_id=model_id, filename=filename, local_files_only=True
        )
        return Path(resolved)
    except Exception:
        if local_files_only:
            return None
    try:
        resolved = hf_hub_download(repo_id=model_id, filename=filename)
        return Path(resolved)
    except Exception:
        return None


def _strip_prefix(
    state_dict: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """Strip expected module prefix from loaded safetensors keys."""
    return {
        (k[len(prefix) :] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def _build_activation(name: Optional[str]) -> nn.Module:
    """Instantiate Sentence-Transformers Dense activation module."""
    if not name:
        return nn.Identity()
    cls = _ACTIVATION_MAP.get(name)
    if cls is None:
        raise ModelLoadError(
            f"Unsupported modular cross-encoder activation function: {name!r}",
            details={"activation_function": name},
        )
    return cls()


class _ModularCrossEncoder(nn.Module):
    """Sentence-Transformers modular CrossEncoder (Backbone -> CLS -> Dense -> LN -> Dense)."""

    def __init__(
        self,
        model_id: str,
        backbone_dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        pooling_cfg_path = _resolve_repo_file(
            model_id, "1_Pooling/config.json", local_files_only=local_files_only
        )
        dense2_cfg_path = _resolve_repo_file(
            model_id, "2_Dense/config.json", local_files_only=local_files_only
        )
        dense2_wt_path = _resolve_repo_file(
            model_id, "2_Dense/model.safetensors", local_files_only=local_files_only
        )
        ln_cfg_path = _resolve_repo_file(
            model_id, "3_LayerNorm/config.json", local_files_only=local_files_only
        )
        ln_wt_path = _resolve_repo_file(
            model_id, "3_LayerNorm/model.safetensors", local_files_only=local_files_only
        )
        dense4_cfg_path = _resolve_repo_file(
            model_id, "4_Dense/config.json", local_files_only=local_files_only
        )
        dense4_wt_path = _resolve_repo_file(
            model_id, "4_Dense/model.safetensors", local_files_only=local_files_only
        )

        required = [
            pooling_cfg_path,
            dense2_cfg_path,
            dense2_wt_path,
            ln_cfg_path,
            ln_wt_path,
            dense4_cfg_path,
            dense4_wt_path,
        ]
        if any(p is None for p in required):
            raise ModelLoadError(
                f"Incomplete modular CrossEncoder head files for '{model_id}'",
                details={"model_id": model_id},
            )

        assert pooling_cfg_path is not None
        assert dense2_cfg_path is not None and dense2_wt_path is not None
        assert ln_cfg_path is not None and ln_wt_path is not None
        assert dense4_cfg_path is not None and dense4_wt_path is not None

        pooling_cfg = json.loads(pooling_cfg_path.read_text(encoding="utf-8"))
        pooling_mode = str(pooling_cfg.get("pooling_mode", "")).lower()
        if pooling_mode != "cls" and not pooling_cfg.get("pooling_mode_cls_token"):
            raise ModelLoadError(
                f"Unsupported pooling config for '{model_id}': {pooling_cfg}",
                details={"model_id": model_id, "pooling_cfg": pooling_cfg},
            )

        d2_cfg = json.loads(dense2_cfg_path.read_text(encoding="utf-8"))
        ln_cfg = json.loads(ln_cfg_path.read_text(encoding="utf-8"))
        d4_cfg = json.loads(dense4_cfg_path.read_text(encoding="utf-8"))

        self.backbone = AutoModel.from_pretrained(
            model_id, dtype=backbone_dtype, local_files_only=local_files_only
        )

        # Head layers always stay in float32 regardless of backbone dtype/quantization
        self.dense1 = nn.Linear(
            int(d2_cfg["in_features"]),
            int(d2_cfg["out_features"]),
            bias=bool(d2_cfg.get("bias", False)),
            dtype=torch.float32,
        )
        self.dense1.load_state_dict(
            {
                k: v.to(torch.float32)
                for k, v in _strip_prefix(load_file(str(dense2_wt_path)), "linear.").items()
            },
            strict=True,
        )
        self.act1 = _build_activation(d2_cfg.get("activation_function"))

        self.norm = nn.LayerNorm(
            int(ln_cfg["dimension"]),
            dtype=torch.float32,
        )
        self.norm.load_state_dict(
            {
                k: v.to(torch.float32)
                for k, v in _strip_prefix(load_file(str(ln_wt_path)), "norm.").items()
            },
            strict=True,
        )

        self.dense2 = nn.Linear(
            int(d4_cfg["in_features"]),
            int(d4_cfg["out_features"]),
            bias=bool(d4_cfg.get("bias", True)),
            dtype=torch.float32,
        )
        self.dense2.load_state_dict(
            {
                k: v.to(torch.float32)
                for k, v in _strip_prefix(load_file(str(dense4_wt_path)), "linear.").items()
            },
            strict=True,
        )
        self.act2 = _build_activation(d4_cfg.get("activation_function"))

    def forward(self, **inputs: torch.Tensor) -> _ClassifierOutput:
        """Compute cross-encoder logits using CLS token + float32 Sentence-Transformers head."""
        backbone_out = self.backbone(**inputs)
        cls_emb = backbone_out.last_hidden_state[:, 0, :].to(torch.float32)
        hidden = self.act1(self.dense1(cls_emb))
        hidden = self.norm(hidden)
        scores = self.act2(self.dense2(hidden))
        return _ClassifierOutput(logits=scores)


def logits_to_probs_and_raw(
    logits: torch.Tensor,
) -> tuple[list[float], list[float]]:
    """Convert raw cross-encoder logits to ``(probs, raw_logits)``.

    - 1-logit head ``(B,)`` or ``(B, 1)`` -> ``sigmoid``, scalar raw logit
    - 2-logit head ``(B, 2)`` -> ``softmax[:, 1]``, log-odds ``logits[:, 1] - logits[:, 0]``
    - >2-logit head ``(B, C)`` -> ``softmax[:, 1]``, ``logits[:, 1]``
    """
    logits_f32 = logits.float()
    if logits_f32.ndim == 1:
        raw = logits_f32
        probs = torch.sigmoid(raw)
    elif logits_f32.shape[-1] == 1:
        raw = logits_f32.squeeze(-1)
        probs = torch.sigmoid(raw)
    elif logits_f32.shape[-1] == 2:
        raw = logits_f32[:, 1] - logits_f32[:, 0]
        probs = torch.softmax(logits_f32, dim=-1)[:, 1]
    else:
        raw = logits_f32[:, 1]
        probs = torch.softmax(logits_f32, dim=-1)[:, 1]
    prob_list = [float(p) for p in probs.clamp(0.0, 1.0).reshape(-1).cpu().tolist()]
    raw_list = [float(r) for r in raw.reshape(-1).cpu().tolist()]
    return prob_list, raw_list


def logits_to_probs(logits: torch.Tensor) -> list[float]:
    """Convert raw cross-encoder logits to relevance probabilities in [0, 1]."""
    probs, _ = logits_to_probs_and_raw(logits)
    return probs


def _has_modular_cross_encoder(
    model_id: str, local_files_only: bool = False
) -> bool:
    """Detect whether *model_id* exposes a Sentence-Transformers modular CrossEncoder."""
    cfg = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
    architectures = list(getattr(cfg, "architectures", None) or [])
    if any(arch.endswith("ForSequenceClassification") for arch in architectures):
        return False
    modules_path = _resolve_repo_file(
        model_id, "modules.json", local_files_only=local_files_only
    )
    if modules_path is None:
        return False
    try:
        modules = json.loads(modules_path.read_text(encoding="utf-8"))
        paths = {str(m.get("path", "")) for m in modules if isinstance(m, dict)}
        return {"1_Pooling", "2_Dense", "3_LayerNorm", "4_Dense"}.issubset(paths)
    except Exception:
        return False


def _load_model_variant(
    mid: str,
    precision: str = "fp32",
) -> tuple[Any, nn.Module, str]:
    """Load tokenizer + model for *mid* in *precision*, preferring local cache first."""
    norm_prec = precision.lower()
    if norm_prec not in _BACKBONE_DTYPE_MAP:
        raise ModelLoadError(
            f"Unsupported rerank_precision: {precision!r}",
            details={"rerank_precision": precision},
        )
    backbone_dtype = _BACKBONE_DTYPE_MAP[norm_prec]

    last_err: Exception | None = None
    for local_only in (True, False):
        try:
            tok = AutoTokenizer.from_pretrained(mid, local_files_only=local_only)
            if tok.pad_token is None and (tok.eos_token or tok.unk_token):
                tok.pad_token = tok.eos_token or tok.unk_token
            if _has_modular_cross_encoder(mid, local_files_only=local_only):
                mdl: nn.Module = _ModularCrossEncoder(
                    mid,
                    backbone_dtype=backbone_dtype,
                    local_files_only=local_only,
                )
                arch_label = f"modular CrossEncoder {norm_prec} backbone / fp32 head"
            else:
                mdl = AutoModelForSequenceClassification.from_pretrained(
                    mid, dtype=backbone_dtype, local_files_only=local_only
                )
                arch_label = f"SequenceClassification {norm_prec}"

            mdl.eval()
            if norm_prec == "int8":
                q_mdl = quantize_linear_layers(
                    mdl,
                    exclude_names=(
                        "dense1",
                        "dense2",
                        "classifier",
                        "score",
                        "head",
                        "pooler",
                        "embeddings",
                        "tok_embeddings",
                        "word_embeddings",
                    ),
                )
                del mdl
                mdl = q_mdl
            return tok, mdl, arch_label
        except OSError as exc:
            last_err = exc
            if not local_only:
                raise
    assert last_err is not None
    raise last_err


@dataclasses.dataclass
class _PairTemplate:
    """Extracted special-token layout ``prefix + q_ids + mid + p_ids + suffix``."""

    prefix: list[int]
    mid: list[int]
    suffix: list[int]
    special_token_count: int
    uses_token_type_ids: bool


def _find_sublist(seq: list[int], sub: list[int], start: int = 0) -> int:
    """Return first index of *sub* inside *seq[start:]*, or -1 if absent."""
    if not sub:
        return start
    limit = len(seq) - len(sub)
    for idx in range(start, limit + 1):
        if seq[idx : idx + len(sub)] == sub:
            return idx
    return -1


def _detect_pair_template(tokenizer: Any) -> _PairTemplate:
    """Inspect *tokenizer* pair special-token layout across v4/v5 tokenizers."""
    uses_ttids = "token_type_ids" in list(getattr(tokenizer, "model_input_names", []) or [])

    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        empty = list(tokenizer.build_inputs_with_special_tokens([], []))
        probe_q = [10101]
        probe_p = [20202]
        built = list(tokenizer.build_inputs_with_special_tokens(probe_q, probe_p))
        q_pos = _find_sublist(built, probe_q, 0)
        p_pos = _find_sublist(built, probe_p, q_pos + 1)
        if q_pos >= 0 and p_pos > q_pos:
            prefix = built[:q_pos]
            mid = built[q_pos + 1 : p_pos]
            suffix = built[p_pos + 1 :]
            return _PairTemplate(
                prefix=prefix,
                mid=mid,
                suffix=suffix,
                special_token_count=len(empty),
                uses_token_type_ids=uses_ttids,
            )

    # Transformers v5 TokenizersBackend path: infer exact template from real pair encoding
    q_probe_ids = list(tokenizer.encode("alpha", add_special_tokens=False))
    p_probe_ids = list(tokenizer.encode("beta", add_special_tokens=False))
    pair_enc = tokenizer("alpha", "beta", add_special_tokens=True)
    pair_ids = list(pair_enc["input_ids"])
    uses_ttids = uses_ttids or ("token_type_ids" in pair_enc)

    q_pos = _find_sublist(pair_ids, q_probe_ids, 0)
    p_pos = (
        _find_sublist(pair_ids, p_probe_ids, q_pos + len(q_probe_ids))
        if q_pos >= 0
        else -1
    )
    if q_pos >= 0 and p_pos >= q_pos + len(q_probe_ids):
        prefix = pair_ids[:q_pos]
        mid = pair_ids[q_pos + len(q_probe_ids) : p_pos]
        suffix = pair_ids[p_pos + len(p_probe_ids) :]
        special_count = len(prefix) + len(mid) + len(suffix)
        return _PairTemplate(
            prefix=prefix,
            mid=mid,
            suffix=suffix,
            special_token_count=special_count,
            uses_token_type_ids=uses_ttids,
        )

    # Safe CLS/SEP fallback if tokenizer behaves unexpectedly
    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    if cls_id is not None and sep_id is not None:
        return _PairTemplate(
            prefix=[int(cls_id)],
            mid=[int(sep_id)],
            suffix=[int(sep_id)],
            special_token_count=3,
            uses_token_type_ids=uses_ttids,
        )

    raise ModelLoadError(
        "Unable to determine cross-encoder pair special-token layout for tokenizer.",
        details={"tokenizer_type": type(tokenizer).__name__},
    )


def _assemble_pair_tokens(
    tokenizer: Any,
    template: _PairTemplate,
    q_ids: list[int],
    p_ids: list[int],
) -> tuple[list[int], list[int], list[int] | None]:
    """Assemble ``(input_ids, attention_mask, token_type_ids)`` for ``(q_ids, p_ids)``."""
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        input_ids = list(tokenizer.build_inputs_with_special_tokens(q_ids, p_ids))
        ttids: list[int] | None = None
        if template.uses_token_type_ids and hasattr(
            tokenizer, "create_token_type_ids_from_sequences"
        ):
            ttids = list(tokenizer.create_token_type_ids_from_sequences(q_ids, p_ids))
    else:
        input_ids = template.prefix + q_ids + template.mid + p_ids + template.suffix
        ttids = (
            ([0] * (len(template.prefix) + len(q_ids) + len(template.mid)))
            + ([1] * (len(p_ids) + len(template.suffix)))
            if template.uses_token_type_ids
            else None
        )
    attn = [1] * len(input_ids)
    return input_ids, attn, ttids


class Reranker:
    """CPU-resident cross-encoder reranker with D2 passage-first pair construction."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        apply_hardware_guard()
        self._settings = settings or get_settings()
        self.model_id: str = self._resolve_model_id()
        self.precision: str = self._settings.rerank_precision.lower()
        self._tokenizer: Any = None
        self._model: Optional[nn.Module] = None
        self._pair_template: Optional[_PairTemplate] = None
        self.is_fallback: bool = False
        self._truncation_stats: dict[str, int] = {
            "pairs_scored": 0,
            "queries_truncated": 0,
            "passages_truncated": 0,
            "contingency_l_pairs": 0,
        }
        self._load_model()

    @property
    def is_loaded(self) -> bool:
        """Return True if the cross-encoder model is currently resident in memory."""
        return self._model is not None and self._tokenizer is not None

    @property
    def truncation_stats(self) -> dict[str, int]:
        """Return cumulative D2 truncation and Contingency L statistics."""
        return dict(self._truncation_stats)

    def reset_truncation_stats(self) -> None:
        """Reset cumulative truncation counters to zero."""
        self._truncation_stats = {
            "pairs_scored": 0,
            "queries_truncated": 0,
            "passages_truncated": 0,
            "contingency_l_pairs": 0,
        }

    def _load_model(self) -> None:
        """Eagerly load the cross-encoder (or fallback) onto CPU."""
        candidates = [self.model_id]
        if self.model_id != _FALLBACK_RERANKER_ID:
            candidates.append(_FALLBACK_RERANKER_ID)

        last_exc: Exception | None = None
        for idx, mid in enumerate(candidates):
            try:
                tok, mdl, arch_label = _load_model_variant(
                    mid, precision=self.precision
                )
                self._tokenizer = tok
                self._model = mdl
                self._pair_template = _detect_pair_template(tok)
                self.model_id = mid
                self.is_fallback = idx > 0 or (
                    mid == _FALLBACK_RERANKER_ID
                    and self._settings.rerank_model_id != _FALLBACK_RERANKER_ID
                )
                gc.collect()
                log.info("Reranker ready: %s (%s, CPU)", self.model_id, arch_label)
                return
            except Exception as exc:
                last_exc = exc
                log.warning("Could not load reranker '%s': %s", mid, exc)

        raise ModelLoadError(
            f"Failed to load reranker '{self.model_id}' or fallback "
            f"'{_FALLBACK_RERANKER_ID}': {last_exc}",
            details={
                "model_id": self.model_id,
                "precision": self.precision,
                "error": str(last_exc),
            },
        ) from last_exc

    def _resolve_model_id(self) -> str:
        """Read reports/preflight.json if present to honor fallback resolution."""
        preflight_path = self._settings.reports_dir / "preflight.json"
        if preflight_path.exists():
            try:
                data = json.loads(preflight_path.read_text(encoding="utf-8"))
                if data.get("reranker_is_fallback") and isinstance(
                    data.get("rerank_model_id"), str
                ):
                    return data["rerank_model_id"]
                if isinstance(data.get("rerank_model_id"), str) and data["rerank_model_id"]:
                    return data["rerank_model_id"]
            except Exception as exc:
                log.warning("Could not inspect %s: %s", preflight_path, exc)
        return self._settings.rerank_model_id

    def free_tokenizer_caches(self) -> None:
        """Trigger garbage collection to release transient tokenizer buffers."""
        gc.collect()

    def build_pair_inputs(self, query: str, passage: str) -> dict[str, Any]:
        """Construct D2 passage-first pair token IDs and attention mask.

        Enforces ``len(p_ids) <= 512`` and never truncates the passage. Only
        query tokens exceeding ``max_pair_len - len(p_ids) - special_tokens``
        are truncated, with every truncation logged and tracked.
        """
        if not self.is_loaded or self._tokenizer is None:
            raise ModelLoadError(
                "Reranker must be loaded before constructing pairs.",
                details={"model_id": self.model_id},
            )
        if self._pair_template is None:
            self._pair_template = _detect_pair_template(self._tokenizer)

        tok = self._tokenizer
        template = self._pair_template
        special_tokens = template.special_token_count

        p_ids = list(tok.encode(passage, add_special_tokens=False))
        q_ids_raw = list(tok.encode(query, add_special_tokens=False))

        max_pair_len = int(self._settings.rerank_max_length)
        used_contingency_l = False
        if (
            self._settings.contingency_l_applied
            and self._settings.rerank_long_query_max_length is not None
            and len(q_ids_raw) > int(self._settings.rerank_long_query_token_threshold)
        ):
            max_pair_len = int(self._settings.rerank_long_query_max_length)
            used_contingency_l = True

        q_budget = max_pair_len - len(p_ids) - special_tokens
        if len(p_ids) > _PASSAGE_MAX_TOKENS or q_budget < 0:
            raise RetrievalError(
                f"Passage token count {len(p_ids)} would truncate at "
                f"max_pair_len={max_pair_len} (special_tokens={special_tokens}, q_budget={q_budget}).",
                details={
                    "query": query[:120],
                    "passage_tokens": len(p_ids),
                    "max_pair_len": max_pair_len,
                    "special_tokens": special_tokens,
                    "q_budget": q_budget,
                },
            )

        self._truncation_stats["pairs_scored"] += 1
        if used_contingency_l:
            self._truncation_stats["contingency_l_pairs"] += 1

        if len(q_ids_raw) > q_budget:
            q_ids = q_ids_raw[:q_budget]
            self._truncation_stats["queries_truncated"] += 1
            log.warning(
                "Reranker D2 query truncation: %d -> %d tokens (passage_tokens=%d, max_pair_len=%d, special_tokens=%d)",
                len(q_ids_raw),
                len(q_ids),
                len(p_ids),
                max_pair_len,
                special_tokens,
            )
        else:
            q_ids = q_ids_raw

        input_ids, attn_mask, ttids = _assemble_pair_tokens(
            tok, template, q_ids, p_ids
        )
        # Sanity assertion: untruncated non-empty pairs match tokenizer(query, passage)
        if len(q_ids_raw) == len(q_ids) and q_ids and p_ids:
            expected = list(tok(query, passage, add_special_tokens=True)["input_ids"])
            if input_ids != expected:
                raise RetrievalError(
                    "Manual D2 pair construction diverged from tokenizer(query, passage).",
                    details={
                        "built_len": len(input_ids),
                        "expected_len": len(expected),
                    },
                )

        out: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        }
        if ttids is not None:
            out["token_type_ids"] = ttids
        return out

    def _pad_pair_batch(
        self, built_pairs: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        """Pad a batch of pre-built pair token lists to rectangular PyTorch tensors."""
        tok = self._tokenizer
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        max_len = max(len(item["input_ids"]) for item in built_pairs)
        padding_side = getattr(tok, "padding_side", "right")

        padded_ids: list[list[int]] = []
        padded_mask: list[list[int]] = []
        has_ttids = "token_type_ids" in built_pairs[0]
        padded_ttids: list[list[int]] = []

        for item in built_pairs:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            pad_len = max_len - len(ids)
            if padding_side == "left":
                padded_ids.append(([int(pad_id)] * pad_len) + ids)
                padded_mask.append(([0] * pad_len) + mask)
                if has_ttids:
                    padded_ttids.append(([0] * pad_len) + item["token_type_ids"])
            else:
                padded_ids.append(ids + ([int(pad_id)] * pad_len))
                padded_mask.append(mask + ([0] * pad_len))
                if has_ttids:
                    padded_ttids.append(item["token_type_ids"] + ([0] * pad_len))

        tensors: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
        }
        if has_ttids:
            tensors["token_type_ids"] = torch.tensor(padded_ttids, dtype=torch.long)
        return tensors

    def warmup(self, passes: int = 3) -> None:
        """Run *passes* warmup passes with candidate_k=8 full-length synthetic pairs."""
        if not self.is_loaded or self._tokenizer is None:
            raise ModelLoadError(
                "Reranker must be loaded before warmup.",
                details={"model_id": self.model_id},
            )
        tok = self._tokenizer
        if self._pair_template is None:
            self._pair_template = _detect_pair_template(tok)
        special_tokens = self._pair_template.special_token_count
        target_total = int(self._settings.rerank_max_length)
        target_p_tokens = min(_PASSAGE_MAX_TOKENS - 8, max(64, target_total - 96))
        target_q_tokens = max(8, target_total - target_p_tokens - special_tokens)

        def _exact_token_text(seed_word: str, count: int) -> str:
            base_ids = list(tok.encode((seed_word + " ") * (count + 16), add_special_tokens=False))
            return str(tok.decode(base_ids[:count], skip_special_tokens=True))

        warm_query = _exact_token_text("vulnerability", target_q_tokens)
        warm_passage = _exact_token_text("mitigation", target_p_tokens)
        # Trim if BPE re-encode grew by a token
        while len(tok.encode(warm_passage, add_special_tokens=False)) > _PASSAGE_MAX_TOKENS:
            p_ids = list(tok.encode(warm_passage, add_special_tokens=False))[:-4]
            warm_passage = str(tok.decode(p_ids, skip_special_tokens=True))

        batch_k = max(1, int(self._settings.candidate_k))
        texts = [warm_passage] * batch_k
        for _ in range(max(1, int(passes))):
            self._score_pairs(warm_query, texts)
        self.reset_truncation_stats()

    def _score_pairs(self, query: str, texts: list[str]) -> list[tuple[float, float]]:
        """Run cross-encoder forward passes in length-bucketed batches of up to 4.

        Returns a list of ``(calibrated_prob, raw_logit)`` pairs aligned to *texts*.
        """
        if not texts:
            return []
        if not self.is_loaded or self._model is None:
            raise ModelLoadError(
                "Reranker weights are not resident.",
                details={"model_id": self.model_id},
            )

        built_pairs = [self.build_pair_inputs(query, text) for text in texts]
        order = sorted(range(len(built_pairs)), key=lambda idx: len(built_pairs[idx]["input_ids"]))
        ordered_results: list[tuple[int, float, float]] = []

        for start in range(0, len(order), _RERANK_BATCH_SIZE):
            batch_indices = order[start : start + _RERANK_BATCH_SIZE]
            batch_pairs = [built_pairs[idx] for idx in batch_indices]
            enc = self._pad_pair_batch(batch_pairs)
            with torch.inference_mode():
                logits = self._model(**enc).logits
            probs, raw_logits = logits_to_probs_and_raw(logits)
            for idx, prob, raw_val in zip(batch_indices, probs, raw_logits):
                ordered_results.append((idx, prob, raw_val))

        out: list[tuple[float, float]] = [(0.0, 0.0)] * len(texts)
        for idx, prob, raw_val in ordered_results:
            out[idx] = (prob, raw_val)
        return out

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 3,
        return_all: bool = False,
    ) -> list[dict[str, Any]]:
        """Score all candidates once and sort descending by ``rerank_raw_logit``.

        Parameters
        ----------
        query:
            Normalized or raw user query string.
        candidates:
            Candidate chunk dictionaries (must contain ``content`` or ``text``).
        top_k:
            Maximum results to return when ``return_all=False`` (capped at
            ``settings.rerank_top_k_max``).
        return_all:
            When True, return all scored candidates without slicing to *top_k*.
        """
        if not candidates:
            return []

        texts = [str(c.get("content") or c.get("text") or "") for c in candidates]
        raw_scores = self._score_pairs(query, texts)

        scored: list[dict[str, Any]] = []
        for cand, score_item in zip(candidates, raw_scores):
            copy = dict(cand)
            if isinstance(score_item, (tuple, list)):
                prob = float(score_item[0])
                raw_logit = float(score_item[1])
            else:
                prob = float(score_item)
                clamped = min(max(prob, 1e-7), 1.0 - 1e-7)
                raw_logit = float(math.log(clamped / (1.0 - clamped)))
            copy["rerank_score"] = prob
            copy["rerank_raw_logit"] = raw_logit
            scored.append(copy)

        scored.sort(
            key=lambda item: (item["rerank_raw_logit"], item["rerank_score"]),
            reverse=True,
        )

        if return_all:
            return scored

        cap = max(0, min(int(top_k), self._settings.rerank_top_k_max))
        return scored[:cap]
