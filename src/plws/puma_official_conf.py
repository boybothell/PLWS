"""Resolve the PUMA official knobs file for a first-time cell.

Historical runs copied an existing ``_local.conf`` from ``results/``. A clean
rental checkout has no QwQ/32B seed-42 template and also lacks the local-only
``puma_offline_r1_7b/.../_DS-7B.local.conf`` fallback, so ``run_puma_official.sh``
exited before any trial generation.

Order:
1. same-model seed-42 / AMC23 / GPQA templates already in ``results/``
2. official ``$PUMA_ROOT/configs/$ALIGN_CONF`` (the file the wrapper already
   selected but previously ignored)
3. local-machine r1_7b leftovers, if they exist
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from plws.runtime import models_root

CONF_NAMES = (
    "_DS-7B.local.conf",
    "_DS-14B.local.conf",
    "_Nemotron.local.conf",
    "_Q30B-T.local.conf",
    "_local.conf",
)
EMBEDDING_DIRNAME = "qwen3-embedding-redundancy-detector-0.6B"
_SIMILARITY_RE = re.compile(r"^SIMILARITY_THRESHOLD=", re.M)
_EMBED_RE = re.compile(r"^EMBEDDING_MODEL=.*$", re.M)


def conf_usable(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return bool(_SIMILARITY_RE.search(text))


def embedding_model_path(root: Path | None = None) -> Path:
    return (root or models_root()) / EMBEDDING_DIRNAME


def candidate_result_roots(ae: Path, model_tag: str, dataset: str) -> list[Path]:
    roots: list[Path] = []
    if model_tag == "r1_7b" and dataset == "math-500":
        roots.append(ae / "results/baselines/official/math500_official/puma_ds7b")
    puma = ae / "results/baselines/puma"
    roots.append(puma / f"puma_offline_{model_tag}" / dataset)
    roots.append(puma / f"puma_offline_{model_tag}" / "amc23")
    roots.append(puma / f"puma_offline_{model_tag}" / "gpqa-diamond")
    return roots


def _first_named_conf(root: Path) -> Path | None:
    for name in CONF_NAMES:
        path = root / name
        if conf_usable(path):
            return path
    extras = sorted(root.glob("_*.conf"))
    for path in extras:
        if conf_usable(path):
            return path
    return None


def pick_official_conf(
    ae: Path,
    puma_root: Path,
    model_tag: str,
    dataset: str,
    align_conf: str,
) -> Path:
    for root in candidate_result_roots(ae, model_tag, dataset):
        hit = _first_named_conf(root)
        if hit is not None:
            return hit
    official = puma_root / "configs" / align_conf
    if conf_usable(official):
        return official
    local_amc = (
        ae / "results/baselines/puma/puma_offline_r1_7b/amc23/_local.conf"
    )
    if conf_usable(local_amc):
        return local_amc
    local_gpqa = (
        ae
        / "results/baselines/puma/puma_offline_r1_7b/gpqa-diamond/_DS-7B.local.conf"
    )
    if conf_usable(local_gpqa):
        return local_gpqa
    raise FileNotFoundError(
        "no usable PUMA conf for "
        f"{model_tag} {dataset} (ALIGN_CONF={align_conf}); "
        f"looked at prior results and {official}"
    )


def render_local_conf(
    source: Path,
    seed: int,
    embed_path: Path | None = None,
) -> str:
    text = source.read_text(encoding="utf-8")
    embed = embed_path if embed_path is not None else embedding_model_path()
    if (embed / "config.json").is_file():
        replacement = f'EMBEDDING_MODEL="{embed}"'
        if _EMBED_RE.search(text):
            text = _EMBED_RE.sub(replacement, text)
        else:
            text = text.rstrip() + "\n" + replacement + "\n"
    text = text.rstrip() + f"\nSEED={seed}\n"
    return text


def require_embedding_model(root: Path | None = None) -> Path:
    path = embedding_model_path(root)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"missing PUMA embedding model: {path} "
            "(set PLWS_MODELS_ROOT or place "
            f"{EMBEDDING_DIRNAME} under the models root)"
        )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    pick = sub.add_parser("pick")
    pick.add_argument("--ae", type=Path, required=True)
    pick.add_argument("--puma-root", type=Path, required=True)
    pick.add_argument("--model-tag", required=True)
    pick.add_argument("--dataset", required=True)
    pick.add_argument("--align-conf", required=True)
    write = sub.add_parser("write")
    write.add_argument("--src", type=Path, required=True)
    write.add_argument("--dst", type=Path, required=True)
    write.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "pick":
        print(
            pick_official_conf(
                args.ae,
                args.puma_root,
                args.model_tag,
                args.dataset,
                args.align_conf,
            )
        )
        return 0
    require_embedding_model()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.dst.with_name(f"{args.dst.name}.tmp.{os.getpid()}")
    tmp.write_text(render_local_conf(args.src, args.seed), encoding="utf-8")
    tmp.replace(args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
