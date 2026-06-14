#!/usr/bin/env bash
# Repeatability driver: run each thinking_budget config REPS times.
# Resumable: counts existing summary.json per config and only runs what's missing,
# so it survives being killed/relaunched.
cd "$(dirname "$0")/.."
source grad2_env/bin/activate

Q=vera_lite/guiding_questions_v3.json
REPS=${REPS:-3}

configs=(baseline tb2000)
args_baseline="--max-tokens 16000"
args_tb2000="--thinking-budget 2000 --max-tokens 3000"

for cfg in "${configs[@]}"; do
  dir="vera_lite/runs/repeat/$cfg"
  mkdir -p "$dir"
  eval "extra=\$args_$cfg"
  done=$(ls -d "$dir"/*/summary.json 2>/dev/null | wc -l)
  while [ "$done" -lt "$REPS" ]; do
    echo "=== $cfg rep $((done + 1))/$REPS ==="
    python -m vera_lite.run_iteration --model qwen3.6-plus --skip-optimizer \
      --questions "$Q" $extra --out-dir "$dir"
    done=$(ls -d "$dir"/*/summary.json 2>/dev/null | wc -l)
  done
done
echo "ALL DONE"
