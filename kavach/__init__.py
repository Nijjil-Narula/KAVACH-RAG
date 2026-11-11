"""
kavach — KAVACH-RAG cybersecurity retrieval pipeline.

Importing any kavach submodule automatically applies the hardware isolation
guard (CPU-only PyTorch, thread limits, zero VRAM). This call is idempotent.
"""
from kavach.runtime_guard import apply_hardware_guard

apply_hardware_guard()
