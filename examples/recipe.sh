#!/usr/bin/env bash
# End-to-end recipe: Sentaurus data -> surrogate ensemble -> warm-start digital twin.
# Needs a licensed Sentaurus (sde, sdevice, tdx on PATH) and parameter.par
# (see trifield/tcad/README.md). Adjust WORKERS to the CPU cores you can use;
# every Sentaurus job is single-threaded.
set -euo pipefail
cd "$(dirname "$0")/.."
PAR=${PAR:-parameter.par}
WORKERS=${WORKERS:-16}

# 1. Training data: full production solves of random designs (1-8 fins), graphs at the
#    nine output-curve biases plus the Vd = 12 V transfer curve. ~100 designs suffice.
python -m trifield.tcad.campaign --par "$PAR" --work_dir runs/train --out_dir runs/train_graphs \
    --fins 1 2 3 4 5 6 8 --n_per_fins 15 --keep --workers "$WORKERS" --seed 42
python -m trifield.tcad.extract runs/train/sim_* --out_dir data/graphs --workers "$WORKERS"
python -m trifield.data.split data/graphs --train data/train --val data/val --val_frac 0.15

# 2. Surrogate ensemble: one member per GPU / seed. Data-only loss with per-(material, field)
#    homoscedastic balancing, standardised inputs, fin-local y and no LayerNorm: this recipe gave the
#    lowest field error in our tests (the continuity term trades field accuracy for current
#    conservation; enable it with --physics_max_weight 0.1).
for seed in 0 1; do
  CUDA_VISIBLE_DEVICES=$seed python -m trifield.train --data_dir data/train --val_data_dir data/val \
      --epochs 35 --max_nodes_per_batch 40000 --grad_checkpoint --cache_data --weight_decay 1e-4 \
      --input_norm --local_y --drop_vor --norm none --balance --gate_weight 8 --physics_max_weight 0 \
      --run_name ens --seed $seed &
done
wait

# 3. Digital twin on fresh designs, benchmarked against the production pipeline.
python -m trifield.tcad.campaign --par "$PAR" --work_dir runs/bench --out_dir runs/bench_graphs \
    --fins 1 2 4 6 8 --n_per_fins 3 --keep --workers "$WORKERS" --seed 2026
python -m trifield.tcad.benchmark --runs runs/bench --log runs/bench_graphs/campaign_log.csv \
    --ckpts checkpoints/ens/last_seed0.pt checkpoints/ens/last_seed1.pt \
    --modes warm cold --device cuda --refine_device auto --refine_cap 30 --escalate \
    --out runs/bench_twin.csv --workers $((WORKERS / 3))

# 4. Next acquisition: mesh-only candidate pool, four-criterion score, simulate the top-K.
python -m trifield.tcad.campaign --par "$PAR" --work_dir runs/cand --out_dir runs/cand_graphs \
    --fins 1 2 3 4 5 6 8 --n_per_fins 40 --mesh_only --keep --workers "$WORKERS" --seed 7
python -m trifield.active_learning --ckpts checkpoints/ens/last_seed*.pt \
    --candidates runs/cand_graphs --train_designs data/train --top_k 16 --out_csv runs/selection.csv
