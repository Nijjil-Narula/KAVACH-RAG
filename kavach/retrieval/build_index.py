"""
kavach.retrieval.build_index — Build corpus embeddings and FAISS SQ8 index.

Usage:
  python -m kavach.retrieval.build_index --build-threads 2

Steps:
  1. Read corpus_chunks.parquet (w512) and corpus_chunks_w256.parquet (w256).
  2. Deduplicate passage texts across both corpora via SHA-256 content hash.
  3. Resume from data/processed/embeddings/checkpoint/batch_*.npz if present,
     then embed pending passages in batches of 32 with atomic checkpointing.
  4. Log batch progress, throughput, ETA, and RSS to logs/build_index.log.
  5. Atomically write:
       - data/processed/embeddings/w512.npz (chunk_ids, full1024, mrl512)
       - data/processed/embeddings/w256.npz (chunk_ids, full1024, mrl512)
       - data/processed/vectors_mrl512.parquet (chunk_id, doc_id, vector)
       - models/index_sq8_mrl512.faiss + models/index_sq8_mrl512.idmap.json
"""
from __future__ import annotations

import argparse
import os
import sys


def _preconfigure_env_from_argv() -> int:
    """Parse --build-threads before importing torch/faiss/transformers."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-threads", type=int, default=2)
    known, _ = parser.parse_known_args(sys.argv[1:])
    threads = max(1, int(known.build_threads))
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("HF_HOME", os.path.join(project_root, "models", "hf_cache"))
    return threads


_REQUESTED_THREADS = _preconfigure_env_from_argv()

from kavach.runtime_guard import apply_hardware_guard, rss_mb, vram_probe  # noqa: E402

import logging  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
import faiss  # noqa: E402, F401 — imported strictly after torch

from kavach.config import Settings, get_settings  # noqa: E402
from kavach.exceptions import CorpusIntegrityError  # noqa: E402
from kavach.retrieval.embedder import Embedder, content_hash  # noqa: E402
from kavach.retrieval.vector_store import SQ8Index  # noqa: E402

log = logging.getLogger("kavach.build_index")


def _configure_logging(log_path: Path) -> None:
    """Attach UTF-8 file handler and stream handler cleanly."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Remove pre-existing handlers to avoid duplicate lines if re-entered
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    root_logger.addHandler(sh)
    root_logger.addHandler(fh)


def _save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    """Write an .npz archive atomically via a temporary .tmp.npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _load_checkpoints(
    ckpt_dir: Path,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
    """Load completed batch_*.npz checkpoints and return next batch index."""
    embedded_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    max_batch_idx = -1

    if not ckpt_dir.exists():
        return embedded_map, 0

    for ckpt_file in sorted(ckpt_dir.glob("batch_*.npz")):
        m = re.match(r"batch_(\d+)\.npz$", ckpt_file.name)
        if m:
            max_batch_idx = max(max_batch_idx, int(m.group(1)))
        try:
            with np.load(ckpt_file, allow_pickle=False) as data:
                hashes = data["hashes"]
                full1024 = data["full1024"]
                mrl512 = data["mrl512"]
                for i, h in enumerate(hashes):
                    embedded_map[str(h)] = (
                        np.asarray(full1024[i], dtype=np.float32),
                        np.asarray(mrl512[i], dtype=np.float32),
                    )
        except Exception as exc:
            log.warning("Skipping corrupt checkpoint %s: %s", ckpt_file.name, exc)

    return embedded_map, max_batch_idx + 1


def _assemble_corpus_embeddings(
    df: pd.DataFrame,
    embedded_map: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble (chunk_ids, full1024, mrl512) arrays matching *df* row order."""
    chunk_ids = np.asarray(df["chunk_id"].astype(str).tolist(), dtype=np.str_)
    full_rows: list[np.ndarray] = []
    mrl_rows: list[np.ndarray] = []
    for text in df["content"].astype(str).tolist():
        h = content_hash("passage", text)
        vec_full, vec_mrl = embedded_map[h]
        full_rows.append(vec_full)
        mrl_rows.append(vec_mrl)
    full1024 = np.vstack(full_rows).astype(np.float32)
    mrl512 = np.vstack(mrl_rows).astype(np.float32)
    return chunk_ids, full1024, mrl512


def _write_vectors_parquet_atomic(
    path: Path,
    chunk_ids: np.ndarray,
    doc_ids: list[str],
    mrl512: np.ndarray,
    dim: int,
) -> None:
    """Write data/processed/vectors_mrl512.parquet atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "chunk_id": pa.array(chunk_ids.tolist(), type=pa.string()),
            "doc_id": pa.array(doc_ids, type=pa.string()),
            "vector": pa.array(mrl512.tolist(), type=pa.list_(pa.float32(), dim)),
        }
    )
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def build_all(settings: Settings, build_threads: int = 2, batch_size: int = 32) -> None:
    """Execute the full embedding + SQ8 index build pipeline."""
    apply_hardware_guard()
    if build_threads != 2:
        torch.set_num_threads(build_threads)
        faiss.omp_set_num_threads(build_threads)

    p512 = settings.data_processed / "corpus_chunks.parquet"
    p256 = settings.data_processed / "corpus_chunks_w256.parquet"
    for p in (p512, p256):
        if not p.exists():
            raise CorpusIntegrityError(
                f"Required corpus parquet missing: {p}",
                details={"missing_path": str(p)},
            )

    df512 = pd.read_parquet(p512)
    df256 = pd.read_parquet(p256)
    log.info(
        "Loaded corpora: w512=%d chunks, w256=%d chunks (total=%d rows)",
        len(df512),
        len(df256),
        len(df512) + len(df256),
    )

    # Content-hash deduplication across both corpora
    unique_items: list[tuple[str, str]] = []
    seen_hashes: set[str] = set()
    for df in (df512, df256):
        for text in df["content"].astype(str).tolist():
            h = content_hash("passage", text)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_items.append((h, text))

    log.info(
        "Unique passages after content-hash dedup: %d (saved %d duplicate embeds)",
        len(unique_items),
        (len(df512) + len(df256)) - len(unique_items),
    )

    emb_dir = settings.data_processed / "embeddings"
    ckpt_dir = emb_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    embedded_map, next_batch_idx = _load_checkpoints(ckpt_dir)
    if embedded_map:
        log.info(
            "Resumed %d passages from existing checkpoints (next batch idx=%d).",
            len(embedded_map),
            next_batch_idx,
        )

    pending = [(h, t) for h, t in unique_items if h not in embedded_map]
    log.info("Pending unique passages to embed: %d", len(pending))

    if pending:
        embedder = Embedder(settings)
        total_batches = (len(pending) + batch_size - 1) // batch_size
        t_start = time.perf_counter()
        done_count = 0

        for b_num, start in enumerate(range(0, len(pending), batch_size), start=1):
            batch = pending[start : start + batch_size]
            batch_hashes = [h for h, _ in batch]
            batch_texts = [t for _, t in batch]

            result = embedder.embed(batch_texts, kind="passage", batch_size=batch_size)
            embedder.clear_cache()

            ckpt_path = ckpt_dir / f"batch_{next_batch_idx:05d}.npz"
            _save_npz_atomic(
                ckpt_path,
                hashes=np.asarray(batch_hashes, dtype=np.str_),
                full1024=result.full1024,
                mrl512=result.mrl512,
            )
            next_batch_idx += 1

            for idx, h in enumerate(batch_hashes):
                embedded_map[h] = (result.full1024[idx], result.mrl512[idx])

            done_count += len(batch)
            elapsed = time.perf_counter() - t_start
            rate = done_count / elapsed if elapsed > 0 else 0.0
            remaining = len(pending) - done_count
            eta_sec = remaining / rate if rate > 0 else 0.0

            log.info(
                "Batch %d/%d (%d/%d passages) | %.2f chunks/s | ETA %.1fs | RSS %.1f MB",
                b_num,
                total_batches,
                done_count,
                len(pending),
                rate,
                eta_sec,
                rss_mb(),
            )

    # Assemble & write w512.npz and w256.npz atomically
    cids_512, full_512, mrl_512 = _assemble_corpus_embeddings(df512, embedded_map)
    w512_path = emb_dir / "w512.npz"
    _save_npz_atomic(w512_path, chunk_ids=cids_512, full1024=full_512, mrl512=mrl_512)
    log.info("Written %s: shape=%s", w512_path, full_512.shape)

    cids_256, full_256, mrl_256 = _assemble_corpus_embeddings(df256, embedded_map)
    w256_path = emb_dir / "w256.npz"
    _save_npz_atomic(w256_path, chunk_ids=cids_256, full1024=full_256, mrl512=mrl_256)
    log.info("Written %s: shape=%s", w256_path, full_256.shape)

    # Write data/processed/vectors_mrl512.parquet
    vectors_pq_path = settings.data_processed / "vectors_mrl512.parquet"
    _write_vectors_parquet_atomic(
        vectors_pq_path,
        chunk_ids=cids_512,
        doc_ids=df512["doc_id"].astype(str).tolist(),
        mrl512=mrl_512,
        dim=settings.embed_dim_mrl,
    )
    log.info("Written %s (%d rows)", vectors_pq_path, len(cids_512))

    # Build and save FAISS SQ8 index on w512 MRL-512 vectors
    sq8_path = settings.models_dir / "index_sq8_mrl512.faiss"
    sq8 = SQ8Index(dim=settings.embed_dim_mrl)
    sq8.build(mrl_512, cids_512.tolist())
    sq8.save(sq8_path)

    log.info(
        "Build complete. Peak/Current RSS=%.1f MB | VRAM probe=%s",
        rss_mb(),
        dict(vram_probe()),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for corpus embedding & index building."""
    parser = argparse.ArgumentParser(
        description="Build KAVACH-RAG passage embeddings and FAISS SQ8 index."
    )
    parser.add_argument(
        "--build-threads",
        type=int,
        default=2,
        help="CPU threads for torch & faiss during index build (default: 2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Passage embedding batch size (default: 32).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    _configure_logging(settings.logs_dir / "build_index.log")
    log.info(
        "Starting build_index (build_threads=%d, batch_size=%d, HF_HOME=%s)",
        args.build_threads,
        args.batch_size,
        os.environ.get("HF_HOME"),
    )
    build_all(settings, build_threads=args.build_threads, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
