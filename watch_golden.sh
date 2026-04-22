#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/config.json}"

WATCH_DIR="${WATCH_DIR:-/workspace/best-pest-detector}"
GOLDEN="${GOLDEN_DIR:-/workspace/_golden}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

if [ -f "$CONFIG_PATH" ]; then
  CFG_PATHS=$(
    python3 - "$CONFIG_PATH" << 'PY'
import json, re, sys
path = sys.argv[1]
text = open(path, encoding='utf-8').read()
text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
text = re.sub(r'//.*', '', text)
cfg = json.loads(text)
p = cfg.get('paths', {})
print(p.get('final_output_dir', ''))
print(p.get('golden_dir', ''))
PY
  )
  CFG_WATCH_DIR=$(echo "$CFG_PATHS" | sed -n '1p')
  CFG_GOLDEN_DIR=$(echo "$CFG_PATHS" | sed -n '2p')
  if [ -n "$CFG_WATCH_DIR" ]; then WATCH_DIR="$CFG_WATCH_DIR"; fi
  if [ -n "$CFG_GOLDEN_DIR" ]; then GOLDEN="$CFG_GOLDEN_DIR"; fi
fi

mkdir -p "$GOLDEN"
LOG="$GOLDEN/watcher.log"
SRC_MARKER="$GOLDEN/best_source.txt"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] 감시 시작: $WATCH_DIR 모니터링" | tee -a "$LOG"

while true; do
  latest_ckpt=$(
    find "$WATCH_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2> /dev/null \
      | sort -nr \
      | head -1 \
      | cut -d' ' -f2-
  )
  if [ -z "$latest_ckpt" ] || [ ! -f "$latest_ckpt/trainer_state.json" ]; then
    sleep "$POLL_INTERVAL"
    continue
  fi

  best_path=$(python3 -c "
import json, sys
try:
    d = json.load(open('$latest_ckpt/trainer_state.json'))
    print(d.get('best_model_checkpoint') or '')
except Exception:
    sys.exit(0)
" 2> /dev/null)

  if [ -z "$best_path" ] || [ ! -d "$best_path" ]; then
    sleep "$POLL_INTERVAL"
    continue
  fi

  last_copied=$(cat "$SRC_MARKER" 2> /dev/null || echo "")
  if [ "$best_path" != "$last_copied" ]; then
    ts=$(date -u +%H:%M:%S)
    loss=$(python3 -c "
import json
try:
    d = json.load(open('$latest_ckpt/trainer_state.json'))
    print(f\"{d.get('best_metric', float('nan')):.6f}\")
except Exception:
    print('?')
" 2> /dev/null)
    echo "[$ts] 새 최고 성능: $(basename "$best_path") (eval_loss=$loss)" | tee -a "$LOG"
    rm -rf "$GOLDEN/best_ckpt.tmp"
    cp -r "$best_path" "$GOLDEN/best_ckpt.tmp"
    rm -rf "$GOLDEN/best_ckpt"
    mv "$GOLDEN/best_ckpt.tmp" "$GOLDEN/best_ckpt"
    echo "$best_path" > "$SRC_MARKER"
  fi

  sleep "$POLL_INTERVAL"
done
