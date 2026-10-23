"""Design-level train / validation split of a graph directory (symlinks, no copies).

    python -m trifield.data.split data/graphs --train data/train --val data/val \
        --val_frac 0.15 --seed 0 --max_fins 8

All snapshots of one design land on the same side, stratified by fin count, so
validation measures generalisation to unseen geometries, not unseen bias points.
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

from trifield.data.dataset import group_by_design, load_graph


def main(argv=None):
    ap = argparse.ArgumentParser(description="design-level split of a graph directory")
    ap.add_argument("src")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_fins", type=int, default=None, help="drop designs with more fins")
    ap.add_argument("--min_snapshots", type=int, default=1)
    args = ap.parse_args(argv)

    groups = group_by_design(Path(args.src).glob("*.pt"))
    by_fins: dict[int, list[str]] = defaultdict(list)
    for name, files in groups.items():
        if len(files) < args.min_snapshots:
            continue
        nf = int(getattr(load_graph(files[0]), "nfins", 1))
        if args.max_fins is None or nf <= args.max_fins:
            by_fins[nf].append(name)
    rng = random.Random(args.seed)
    val = set()
    for nf, names in sorted(by_fins.items()):
        names = sorted(names)
        rng.shuffle(names)
        k = max(1, round(args.val_frac * len(names))) if len(names) > 1 else 0
        val.update(names[:k])
    for d in (args.train, args.val):
        Path(d).mkdir(parents=True, exist_ok=True)
        for old in Path(d).glob("*.pt"):
            old.unlink()
    n = {"train": 0, "val": 0}
    for nf, names in by_fins.items():
        for name in names:
            side = "val" if name in val else "train"
            for f in groups[name]:
                os.symlink(f.resolve(), Path(getattr(args, side)) / f.name)
                n[side] += 1
    print(f"designs per fin count: { {k: len(v) for k, v in sorted(by_fins.items())} }")
    print(f"train: {sum(len(v) for v in by_fins.values()) - len(val)} designs / {n['train']} graphs; "
          f"val: {len(val)} designs / {n['val']} graphs")


if __name__ == "__main__":
    main()
