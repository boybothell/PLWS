"""Answer grading guarded against silently degraded backends.

Every Acc number in this repo is a mean over grader flags, so a missing LaTeX
backend, an absent gold answer, or a crashed comparison must never reach an
artifact looking like a wrong answer. Callers get an explicit error instead.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


GRADER_VERIFICATION_VERSION = "plws-unified-grader-v1"


class GraderUnavailable(RuntimeError):
    """The grader cannot settle LaTeX equivalence, so every flag would be wrong."""


class MissingGold(ValueError):
    """Grading was attempted without a gold answer."""


# Pairs only the sympy/LaTeX backend can settle. When antlr4 or latex2sympy2 is
# missing, `parse_latex` fails and each of these compares unequal, which turns
# every LaTeX answer into a silent miss.
_SELFTEST_EQUAL: tuple[tuple[str, str], ...] = (
    (r"\dfrac{19}{2}", "9.5"),
    ("0.09", r"\frac{9}{100}"),
    (r"\dfrac{-p^2 + 1}{3}", r"$\frac{1-p^{2}}{3}$"),
    (
        r"\begin{pmatrix} \dfrac{1}{5} \\ -\dfrac{18}{5} \end{pmatrix}",
        r"\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}",
    ),
)
_SELFTEST_UNEQUAL: tuple[tuple[str, str], ...] = (("7", "8"),)

_CHECK: Callable[..., Any] | None = None
_VERIFIED = False


def puma_root() -> Path:
    """Locate the frozen PUMA checkout that owns the grader."""

    env = os.environ.get("PUMA_ROOT")
    if env:
        return Path(env).resolve()
    return (Path(__file__).resolve().parents[2].parent / "PUMA").resolve()


def _load_grader() -> Callable[..., Any]:
    global _CHECK
    if _CHECK is None:
        import sys

        puma = puma_root()
        for entry in (str(puma), str(puma / "puma")):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        from math_grader import check_is_correct  # noqa: PLC0415

        _CHECK = check_is_correct
    return _CHECK


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def has_gold(value: Any) -> bool:
    """Report whether a gold answer is usable for grading."""

    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return any(has_gold(item) for item in value)
    return bool(_text(value).strip())


def require_grader() -> None:
    """Fail loudly when the grader would understate every LaTeX answer.

    Cheap enough (about a second) to run at the start of any scoring or
    reporting entry point.
    """

    global _VERIFIED
    if _VERIFIED:
        return
    check = _load_grader()
    broken: list[str] = []
    for pred, gold in _SELFTEST_EQUAL:
        try:
            ok = bool(check(pred, gold))
        except Exception as exc:  # noqa: BLE001 - reported below
            broken.append(f"{pred!r} vs {gold!r} raised {type(exc).__name__}: {exc}")
            continue
        if not ok:
            broken.append(f"{pred!r} vs {gold!r} graded unequal")
    for pred, gold in _SELFTEST_UNEQUAL:
        try:
            if bool(check(pred, gold)):
                broken.append(f"{pred!r} vs {gold!r} graded equal")
        except Exception as exc:  # noqa: BLE001 - reported below
            broken.append(f"{pred!r} vs {gold!r} raised {type(exc).__name__}: {exc}")
    if broken:
        raise GraderUnavailable(
            "math grader self-test failed; Acc would be understated for every "
            "LaTeX answer. Install the pinned backend "
            "(antlr4-python3-runtime==4.11.1, latex2sympy2==1.9.1, word2number). "
            "Failures: " + "; ".join(broken)
        )
    _VERIFIED = True


def grade(pred: Any, gold: Any) -> tuple[bool, str]:
    """Grade one answer as ``(correct, error)``.

    A blank gold raises instead of scoring the answer wrong. A crashed
    comparison returns its message so callers can persist it; treat any
    non-empty error as unscored rather than incorrect.
    """

    if not has_gold(gold):
        raise MissingGold(
            "gold answer is empty; re-export jobs once PUMA statistics exist"
        )
    check = _load_grader()
    try:
        return bool(check(_text(pred), _text(gold))), ""
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller
        return False, f"{type(exc).__name__}: {exc}"[:200]


def must_grade(pred: Any, gold: Any) -> bool:
    """Grade one answer or raise. Baseline runners use this for Acc flags."""

    require_grader()
    ok, error = grade(pred, gold)
    if error:
        raise RuntimeError(f"grader failed: {error}")
    return ok


def verify_baseline_records(
    records: Mapping[int, dict[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    *,
    workers: int = 16,
) -> set[int]:
    """Re-grade resumable baseline records before they can be reused.

    Safe False-to-True corrections are applied in memory. A stored True that
    now grades False, a missing gold answer, or a grader error blocks reuse.
    The returned indices are the records that callers must atomically rewrite.
    """

    require_grader()
    if not records:
        return set()

    indexed: list[tuple[int, str]] = []
    pairs: list[tuple[Any, Any]] = []
    for index, record in sorted(records.items()):
        if index < 0 or index >= len(samples):
            raise RuntimeError(f"baseline record question_idx out of range: {index}")
        sample = samples[index]
        gold = sample.get("ground_truth_answer")
        if not has_gold(gold):
            raise MissingGold(f"baseline sample {index} has no ground_truth_answer")
        indexed.extend(((index, "correct"), (index, "original_correct")))
        pairs.extend(
            (
                (record.get("answer"), gold),
                (sample.get("model_answer", ""), gold),
            )
        )

    fresh: dict[int, dict[str, bool]] = {}
    problems: list[str] = []
    for (index, flag), (ok, error) in zip(
        indexed,
        grade_many(
            pairs,
            workers=min(workers, max(1, len(pairs))),
            chunksize=8,
        ),
    ):
        if error:
            problems.append(f"q{index} {flag}: {error}")
            continue
        if bool(records[index].get(flag)) and not ok:
            problems.append(f"q{index} {flag}: stored True regraded False")
            continue
        fresh.setdefault(index, {})[flag] = ok
    if problems:
        raise RuntimeError(
            "baseline grade verification blocked: " + "; ".join(problems[:8])
        )

    changed: set[int] = set()
    for index, flags in fresh.items():
        record = records[index]
        before = (
            bool(record.get("correct")),
            bool(record.get("original_correct")),
            record.get("grader_verification"),
            record.get("grade_error"),
        )
        record["correct"] = flags["correct"]
        record["original_correct"] = flags["original_correct"]
        record["grader_verification"] = GRADER_VERIFICATION_VERSION
        record["grade_error"] = ""
        after = (
            record["correct"],
            record["original_correct"],
            record["grader_verification"],
            record["grade_error"],
        )
        if before != after:
            changed.add(index)
    return changed


def _grade_pair(pair: tuple[Any, Any]) -> tuple[bool, str]:
    return grade(pair[0], pair[1])


def _worker_init() -> None:
    _load_grader()


def grade_many(
    pairs: Sequence[tuple[Any, Any]] | Iterable[tuple[Any, Any]],
    *,
    workers: int = 8,
    chunksize: int = 8,
) -> list[tuple[bool, str]]:
    """Grade many ``(pred, gold)`` pairs in parallel.

    Uses ``ProcessPoolExecutor`` on purpose: ``multiprocessing.Pool`` workers are
    daemonic, and the grader's own timeout helper spawns a child process, so a
    Pool turns every hard comparison into ``AssertionError`` and then a miss.
    """

    items = list(pairs)
    if not items:
        return []
    if workers <= 1:
        return [_grade_pair(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        return list(pool.map(_grade_pair, items, chunksize=chunksize))
