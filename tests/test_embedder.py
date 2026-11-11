"""
tests.test_embedder — Unit and model tests for Tier 2 Embedder.

Run with:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_embedder.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from kavach.config import get_settings
from kavach.retrieval.embedder import Embedder, EmbeddingBatch, last_token_pool


def _embedder_weights_available() -> bool:
    """Check whether preflight has populated embedder_notes.md and hf_cache."""
    settings = get_settings()
    notes_ok = (settings.models_dir / "embedder_notes.md").exists()
    hf_cache = settings.models_dir / "hf_cache"
    hub_dir = hf_cache / "hub"
    has_cached = hub_dir.exists() and any(hub_dir.iterdir())
    return notes_ok and has_cached


def test_last_token_pool_left_and_right_padding() -> None:
    """last_token_pool must select the final non-pad token for both padding sides."""
    # Batch size 2, seq_len 4, hidden_dim 3
    # Sequence 0 has 2 tokens [1, 2]; Sequence 1 has 4 tokens [10, 20, 30, 40]
    hidden_right = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [99.0, 99.0, 99.0], [99.0, 99.0, 99.0]],
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0], [40.0, 0.0, 0.0]],
        ]
    )
    mask_right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    out_right = last_token_pool(hidden_right, mask_right, padding_side="right")
    assert torch.allclose(out_right[0], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(out_right[1], torch.tensor([40.0, 0.0, 0.0]))

    hidden_left = torch.tensor(
        [
            [[99.0, 99.0, 99.0], [99.0, 99.0, 99.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0], [40.0, 0.0, 0.0]],
        ]
    )
    mask_left = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    out_left = last_token_pool(hidden_left, mask_left, padding_side="left")
    assert torch.allclose(out_left[0], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(out_left[1], torch.tensor([40.0, 0.0, 0.0]))


def test_embedder_shapes_norms_mrl_and_asymmetry() -> None:
    """Verify full1024/mrl512 shapes, L2 norms, MRL re-normalization, and prefixing."""
    if not _embedder_weights_available():
        pytest.skip("Embedder weights not yet cached by preflight.py")

    settings = get_settings()
    embedder = Embedder(settings)

    passages = [
        "CVE-2024-3094 backdoor in XZ Utils liblzma 5.6.0 and 5.6.1.",
        "MITRE ATT&CK T1190 Exploit Public-Facing Application.",
        "OWASP LLM03 Training Data Poisoning in v1.1 vs Supply Chain in v2.0.",
    ]
    p_batch = embedder.embed(passages, kind="passage", batch_size=2)
    assert isinstance(p_batch, EmbeddingBatch)
    assert p_batch.full1024.shape == (3, settings.embed_dim_full)
    assert p_batch.mrl512.shape == (3, settings.embed_dim_mrl)
    assert p_batch.full1024.dtype == np.float32
    assert p_batch.mrl512.dtype == np.float32

    # L2 norm == 1 for both full1024 and mrl512
    full_norms = np.linalg.norm(p_batch.full1024, axis=1)
    mrl_norms = np.linalg.norm(p_batch.mrl512, axis=1)
    assert np.allclose(full_norms, 1.0, atol=1e-4)
    assert np.allclose(mrl_norms, 1.0, atol=1e-4)

    # Verify mrl512 equals L2-renormalized first 512 dims of full1024
    sliced = p_batch.full1024[:, : settings.embed_dim_mrl]
    expected_mrl = sliced / np.linalg.norm(sliced, axis=1, keepdims=True)
    assert np.allclose(p_batch.mrl512, expected_mrl, atol=1e-5)

    # Query vs passage prefix asymmetry
    text = "CVE-2024-3094 XZ Utils backdoor"
    q_vec = embedder.embed([text], kind="query").mrl512[0]
    p_vec = embedder.embed([text], kind="passage").mrl512[0]
    assert q_vec.shape == (settings.embed_dim_mrl,)
    assert not np.allclose(q_vec, p_vec, atol=1e-4)

    # Memo returns exact identical vector on repeat call
    q_vec_2 = embedder.embed([text], kind="query").mrl512[0]
    assert np.array_equal(q_vec, q_vec_2)
