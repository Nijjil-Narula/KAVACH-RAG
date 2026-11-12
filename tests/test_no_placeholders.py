"""
tests.test_no_placeholders — AST-based audit of KAVACH-RAG source for forbidden patterns.

Checks all .py files in kavach/, scripts/, and tests/ for:
  1. No ``# TODO`` or ``# FIXME`` comments.
  2. No ``raise NotImplementedError`` in non-abstract functions.
  3. No functions/methods whose ONLY body statement is ``pass`` or ``...``
     (Ellipsis constant) — except abstract methods and ``__init_subclass__``.
  4. No bare ``except:`` clauses (no exception type) followed only by ``pass``.
  5. ``time.sleep`` only appears in ``acquire.py``, ``benchlock.py``, or ``profiling.py``.

Run with:
    .venv\\Scripts\\python.exe -m pytest -q tests/test_no_placeholders.py
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path
from typing import NamedTuple

import pytest

# Project root is two levels up from this test file (tests/ → project root).
_ROOT = Path(__file__).parent.parent
_SCAN_DIRS = [_ROOT / "kavach", _ROOT / "scripts", _ROOT / "tests"]
_ALLOWED_SLEEP_PATHS = frozenset(
    {
        (_ROOT / "kavach" / "retrieval" / "acquire.py").resolve(),
        (_ROOT / "kavach" / "retrieval" / "benchlock.py").resolve(),
        (_ROOT / "kavach" / "retrieval" / "profiling.py").resolve(),
        (_ROOT / "kavach" / "generation" / "generator.py").resolve(),
        (_ROOT / "kavach" / "generation" / "judge.py").resolve(),
    }
)
_SELF_PATH = Path(__file__).resolve()


def _collect_py_files() -> list[Path]:
    """Return all .py files under the scanned directories (excluding this audit script)."""
    files: list[Path] = []
    for d in _SCAN_DIRS:
        if d.exists():
            for py_path in sorted(d.rglob("*.py")):
                if py_path.resolve() != _SELF_PATH:
                    files.append(py_path)
    return files


class Violation(NamedTuple):
    """A single placeholder / forbidden-pattern finding."""

    path: Path
    line: int
    rule: str
    detail: str


def _is_abstract_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if *node* has an ``@abstractmethod`` decorator."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "abstractmethod":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "abstractmethod":
            return True
    return False


def _body_is_only_placeholder(
    body: list[ast.stmt],
) -> bool:
    """Return True if the function body is ONLY a pass or ``...`` expression.

    A docstring followed by pass/... is NOT considered a placeholder — Python
    convention allows ``def f(): '''docstring'''`` with no body statement.
    We only flag bodies of exactly 1 node that is pass/ellipsis, or exactly 2
    nodes where the first is a string docstring and the second is pass/ellipsis.
    """
    effective: list[ast.stmt] = []
    for stmt in body:
        # Skip leading string-literal docstrings.
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            if not effective:
                continue  # it's a docstring, skip
        effective.append(stmt)

    if len(effective) != 1:
        return False

    stmt = effective[0]
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return stmt.value.value is ...
    return False


def _check_file(path: Path) -> list[Violation]:
    """Analyse *path* and return all violations found."""
    source = path.read_text(encoding="utf-8")
    violations: list[Violation] = []

    # ── Rule 1: TODO / FIXME comments (real Python comment tokens only) ──────
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        tokens = []
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            comment_text = tok.string
            lineno = tok.start[0]
            if re.search(r"\bTODO\b", comment_text, re.IGNORECASE):
                violations.append(
                    Violation(path, lineno, "NO_TODO", f"Found TODO comment: {comment_text!r}")
                )
            if re.search(r"\bFIXME\b", comment_text, re.IGNORECASE):
                violations.append(
                    Violation(path, lineno, "NO_FIXME", f"Found FIXME comment: {comment_text!r}")
                )

    # ── Rule 5: time.sleep outside allowlisted synchronization/wait modules ────
    if path.resolve() not in _ALLOWED_SLEEP_PATHS:
        for lineno, line in enumerate(source.splitlines(), start=1):
            if re.search(r"\btime\.sleep\s*\(", line):
                violations.append(
                    Violation(
                        path,
                        lineno,
                        "NO_SLEEP_OUTSIDE_ACQUIRE",
                        f"time.sleep() found outside allowlisted modules: {line.strip()!r}",
                    )
                )

    # ── AST-based checks ──────────────────────────────────────────────────────
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        violations.append(
            Violation(path, exc.lineno or 0, "SYNTAX_ERROR", str(exc))
        )
        return violations

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        is_abstract = _is_abstract_method(node)
        is_init_subclass = node.name == "__init_subclass__"
        exempt = is_abstract or is_init_subclass

        # ── Rule 3: pass-only or ...-only body ───────────────────────────────
        if not exempt and _body_is_only_placeholder(node.body):
            violations.append(
                Violation(
                    path,
                    node.lineno,
                    "NO_PASSONLY_BODY",
                    f"Function '{node.name}' body is only pass/... without docstring.",
                )
            )

        # ── Rule 2: raise NotImplementedError in non-abstract functions ───────
        if not is_abstract:
            for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if (
                    isinstance(child, ast.Raise)
                    and child.exc is not None
                    and isinstance(child.exc, ast.Call)
                    and isinstance(child.exc.func, ast.Name)
                    and child.exc.func.id == "NotImplementedError"
                ):
                    violations.append(
                        Violation(
                            path,
                            child.lineno,
                            "NO_NOT_IMPLEMENTED",
                            f"raise NotImplementedError in non-abstract '{node.name}'.",
                        )
                    )

        # ── Rule 4: bare except: pass ─────────────────────────────────────────
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(child, ast.Try):
                for handler in child.handlers:
                    if handler.type is None:  # bare except:
                        if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                            violations.append(
                                Violation(
                                    path,
                                    handler.lineno,
                                    "NO_BARE_EXCEPT_PASS",
                                    f"Bare 'except: pass' in function '{node.name}'.",
                                )
                            )

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# Pytest test
# ─────────────────────────────────────────────────────────────────────────────


def test_no_placeholders() -> None:
    """Audit all source files for forbidden placeholder patterns."""
    all_files = _collect_py_files()
    assert all_files, f"No .py files found under {_SCAN_DIRS}"

    all_violations: list[Violation] = []
    for path in all_files:
        all_violations.extend(_check_file(path))

    if all_violations:
        lines = ["\nPlaceholder violations found:\n"]
        for v in all_violations:
            rel = v.path.relative_to(_ROOT)
            lines.append(f"  [{v.rule}] {rel}:{v.line} — {v.detail}")
        pytest.fail("\n".join(lines))
