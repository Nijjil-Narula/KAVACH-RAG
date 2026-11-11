"""
kavach.retrieval.quantization — Dynamic INT8 quantization helpers for CPU inference.

Provides:
  - ``check_quantize_dynamic_safety()``: verifies ``quantize_dynamic`` import,
    selects a supported quantized backend engine (preferring ``fbgemm`` if
    available, otherwise ``onednn`` / ``x86``), captures deprecation warnings,
    and checks FP32-vs-INT8 output cosine similarity (> 0.99) on a small
    ``nn.Linear`` stack.
  - ``quantize_linear_layers(model, exclude_names=())``: deep-copies *model* in
    float32 eval mode and dynamically quantizes non-excluded ``nn.Linear``
    layers to INT8 while leaving excluded layers (e.g., classification heads
    and embeddings) in float32.
"""
from __future__ import annotations

import copy
import logging
import warnings
from typing import Any, Callable

import torch
import torch.nn as nn

log = logging.getLogger(__name__)
_LOGGED_ENGINE_SELECTION: bool = False


def _resolve_quantize_dynamic() -> tuple[Callable[..., nn.Module], str]:
    """Import quantize_dynamic preferring torch.ao.quantization with fallback."""
    try:
        from torch.ao.quantization import quantize_dynamic

        return quantize_dynamic, "torch.ao.quantization.quantize_dynamic"
    except ImportError:
        from torch.quantization import quantize_dynamic

        return quantize_dynamic, "torch.quantization.quantize_dynamic"


def _select_quantized_engine() -> tuple[str, list[str]]:
    """Select and set a supported PyTorch quantized CPU backend engine."""
    global _LOGGED_ENGINE_SELECTION
    supported = list(getattr(torch.backends.quantized, "supported_engines", []))
    chosen = ""
    for candidate in ("fbgemm", "onednn", "x86"):
        if candidate in supported:
            chosen = candidate
            break
    if not chosen and supported:
        chosen = supported[0]
    if chosen:
        torch.backends.quantized.engine = chosen
    active = str(torch.backends.quantized.engine)
    if not _LOGGED_ENGINE_SELECTION:
        _LOGGED_ENGINE_SELECTION = True
        if "fbgemm" not in supported:
            log.info(
                "Quantized backend 'fbgemm' absent from supported_engines=%s; "
                "using fallback engine=%r",
                supported,
                active,
            )
        else:
            log.info(
                "Selected PyTorch quantized backend engine=%r (supported_engines=%s)",
                active,
                supported,
            )
    return active, supported


def get_quantized_engine_info() -> dict[str, Any]:
    """Select and return the active quantized CPU engine and supported engines list."""
    engine, supported = _select_quantized_engine()
    return {
        "engine": engine,
        "supported_engines": supported,
    }


def _is_excluded(module_name: str, exclude_names: set[str] | tuple[str, ...]) -> bool:
    """Return True if *module_name* matches or contains any exclusion token/prefix."""
    if not exclude_names:
        return False
    parts = module_name.split(".")
    for ex in exclude_names:
        if not ex:
            continue
        if (
            module_name == ex
            or module_name.startswith(ex + ".")
            or ex in parts
            or ex in module_name
        ):
            return True
    return False


def check_quantize_dynamic_safety() -> dict[str, Any]:
    """Verify dynamic INT8 quantization safety and FP32-vs-INT8 cosine fidelity."""
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quantize_fn, import_path = _resolve_quantize_dynamic()
        engine, supported = _select_quantized_engine()

        torch.manual_seed(42)
        stack = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        ).to(torch.float32).eval()
        sample_x = torch.randn(4, 64, dtype=torch.float32)

        with torch.inference_mode():
            fp32_out = stack(sample_x)
            q_stack = quantize_fn(stack, qconfig_spec={nn.Linear}, dtype=torch.qint8)
            int8_out = q_stack(sample_x)

        cosine = float(
            torch.nn.functional.cosine_similarity(
                fp32_out.reshape(-1), int8_out.reshape(-1), dim=0
            ).item()
        )
        quantized_count = sum(
            1
            for mod in q_stack.modules()
            if "quantized" in type(mod).__module__ and type(mod).__name__ == "Linear"
        )

    for w in caught:
        msg = f"{w.category.__name__}: {w.message}"
        if msg not in captured_warnings:
            captured_warnings.append(msg)

    smoke_ok = bool(engine in supported and quantized_count == 2 and cosine > 0.99)
    return {
        "import_path": import_path,
        "engine": engine,
        "supported_engines": supported,
        "warnings": captured_warnings,
        "cosine_similarity": round(cosine, 6),
        "quantized_linear_count": quantized_count,
        "smoke_ok": smoke_ok,
    }


def quantize_linear_layers(
    model: nn.Module,
    exclude_names: set[str] | tuple[str, ...] = (),
) -> nn.Module:
    """Deep-copy *model* in float32 eval mode and dynamically quantize Linear layers.

    Any ``nn.Linear`` submodule whose qualified name matches or contains an
    entry in *exclude_names* stays as a ``torch.nn.Linear`` in float32.
    """
    _select_quantized_engine()
    quantize_fn, _ = _resolve_quantize_dynamic()

    q_model = copy.deepcopy(model).to(torch.float32).eval()
    for mod in q_model.modules():
        if hasattr(mod, "qconfig"):
            delattr(mod, "qconfig")

    if isinstance(q_model, nn.Linear):
        if exclude_names:
            return q_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", UserWarning)
            return quantize_fn(q_model, qconfig_spec={nn.Linear}, dtype=torch.qint8)

    target_names: set[str] = {
        name
        for name, mod in q_model.named_modules()
        if isinstance(mod, nn.Linear) and name and not _is_excluded(name, exclude_names)
    }
    if not target_names:
        return q_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        quantized = quantize_fn(
            q_model,
            qconfig_spec=target_names,
            dtype=torch.qint8,
            inplace=True,
        )

    # Verify excluded Linear layers stayed torch.nn.Linear (float32)
    for name, mod in quantized.named_modules():
        if name and _is_excluded(name, exclude_names) and isinstance(mod, nn.Linear):
            if "quantized" in type(mod).__module__:
                raise RuntimeError(
                    f"Excluded Linear module {name!r} was unexpectedly quantized: {type(mod)}"
                )

    return quantized
