#!/usr/bin/env bash
# Train the toy models (Section 5.1): build the shared tokenizer, then train one
# model per seed. Every setting below can be overridden from the environment:
#
#   bash toy/train.sh                              # tokenizer + 5 seeds
#   SEEDS="0" EPOCHS=50 bash toy/train.sh          # quick single run
#   DRY_RUN=1 bash toy/train.sh                    # build everything, one forward pass, no training
#   EXCLUDE_DOC_IDS=12,57 OUT_DIR=runs/toy_cf bash toy/train.sh   # counterfactual retrain
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python}

DATA_DIR=${DATA_DIR:-$ROOT/data}
OUT_DIR=${OUT_DIR:-$ROOT/runs/toy}
TOKENIZER_DIR=${TOKENIZER_DIR:-$OUT_DIR/tokenizer}
SEEDS=${SEEDS:-"0 1 2 3 4"}  # independently trained models for TRAK to average over

# Model: Pythia (GPT-NeoX) architecture, scaled down. Head dim = HIDDEN / N_HEADS = 64, as in Pythia.
N_LAYERS=${N_LAYERS:-4}
HIDDEN=${HIDDEN:-256}
N_HEADS=${N_HEADS:-4}
CTX=${CTX:-64}

# Optimization: overfit on purpose, the model should memorize every fact.
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_FRAC=${WARMUP_FRAC:-0.05}
EVAL_EVERY=${EVAL_EVERY:-10}
SAVE_EVERY=${SAVE_EVERY:-50}
EXCLUDE_DOC_IDS=${EXCLUDE_DOC_IDS:-}
DRY_RUN=${DRY_RUN:-0}

for f in docs.jsonl queries.jsonl; do
  [[ -f "$DATA_DIR/$f" ]] || { echo "missing $DATA_DIR/$f (generate the dataset first)" >&2; exit 1; }
done

# 1. Tokenizer: built once and shared, so token ids mean the same thing in every model.
if [[ -f "$TOKENIZER_DIR/tokenizer.json" ]]; then
  echo "reusing tokenizer at $TOKENIZER_DIR"
else
  "$PYTHON" "$ROOT/toy/tokenizer.py" \
    --docs "$DATA_DIR/docs.jsonl" --queries "$DATA_DIR/queries.jsonl" --out "$TOKENIZER_DIR"
fi

# 2. One model per seed. The seed sets both the init and the data order.
for seed in $SEEDS; do
  run_dir="$OUT_DIR/seed$seed"
  log="$run_dir/train.log"
  extra=()
  [[ "$DRY_RUN" == 1 ]] && { extra+=(--dry-run); log="$run_dir/dry_run.log"; }
  [[ -n "$EXCLUDE_DOC_IDS" ]] && extra+=(--exclude-doc-ids "$EXCLUDE_DOC_IDS")

  echo "=== seed $seed -> $run_dir"
  mkdir -p "$run_dir"
  "$PYTHON" "$ROOT/toy/train.py" \
    --docs "$DATA_DIR/docs.jsonl" --queries "$DATA_DIR/queries.jsonl" \
    --tokenizer "$TOKENIZER_DIR" --out "$run_dir" --seed "$seed" \
    --n-layers "$N_LAYERS" --hidden-size "$HIDDEN" --n-heads "$N_HEADS" --context-length "$CTX" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" --weight-decay "$WEIGHT_DECAY" \
    --warmup-frac "$WARMUP_FRAC" --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" \
    "${extra[@]}" 2>&1 | tee "$log"
done
