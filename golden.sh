#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/config.json}"

WATCH_DIR="${WATCH_DIR:-/workspace/best-pest-detector}"
GOLDEN="${GOLDEN_DIR:-/workspace/_golden}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MIN_IMPROVEMENT="${MIN_IMPROVEMENT:-0}"
MIN_COPY_INTERVAL_SEC="${MIN_COPY_INTERVAL_SEC:-0}"

assert_safe_dir() {
  local path="$1"
  local name="$2"
  case "$path" in
  "" | "/" | ".")
    echo "오류: 위험한 $name 경로입니다: '$path'"
    exit 1
    ;;
  esac
}

if [ -f "$CONFIG_PATH" ]; then
  CFG_PATHS=$(
    python3 - "$CONFIG_PATH" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    cfg = json.load(f)
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

assert_safe_dir "$WATCH_DIR" "WATCH_DIR"
assert_safe_dir "$GOLDEN" "GOLDEN_DIR"

mkdir -p "$GOLDEN"
LOG="$GOLDEN/watcher.log"
SRC_MARKER="$GOLDEN/best_source.txt"
METRIC_MARKER="$GOLDEN/best_metric.txt"
TS_MARKER="$GOLDEN/last_copy_ts.txt"
BEST_TMP="$GOLDEN/best_ckpt.tmp"
BEST_DST="$GOLDEN/best_ckpt"

assert_safe_dir "$BEST_TMP" "best_ckpt.tmp"
assert_safe_dir "$BEST_DST" "best_ckpt"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] 감시 시작: $WATCH_DIR 모니터링" | tee -a "$LOG"

while true; do
  latest_ckpt=$(
    find "$WATCH_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null |
      sort -nr |
      head -1 |
      cut -d' ' -f2-
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
" 2>/dev/null)

  if [ -z "$best_path" ] || [ ! -d "$best_path" ]; then
    sleep "$POLL_INTERVAL"
    continue
  fi

  last_copied=$(cat "$SRC_MARKER" 2>/dev/null || echo "")
  if [ "$best_path" != "$last_copied" ]; then
    ts=$(date -u +%H:%M:%S)
    loss=$(python3 -c "
import json
try:
    d = json.load(open('$latest_ckpt/trainer_state.json'))
    print(f\"{d.get('best_metric', float('nan')):.6f}\")
except Exception:
    print('?')
" 2>/dev/null)
    should_copy=$(
      python3 - "$loss" "$MIN_IMPROVEMENT" "$MIN_COPY_INTERVAL_SEC" "$METRIC_MARKER" "$TS_MARKER" <<'PY'
import math, os, sys, time

loss_s, min_imp_s, min_int_s, metric_path, ts_path = sys.argv[1:]
try:
    loss = float(loss_s)
except ValueError:
    loss = math.nan
min_imp = float(min_imp_s)
min_int = int(float(min_int_s))

if not os.path.exists(metric_path) or not os.path.exists(ts_path):
    print("1")
    sys.exit(0)

try:
    prev_metric = float(open(metric_path, encoding="utf-8").read().strip())
except Exception:
    prev_metric = math.nan

try:
    prev_ts = int(float(open(ts_path, encoding="utf-8").read().strip()))
except Exception:
    prev_ts = 0

now = int(time.time())
if min_int > 0 and now - prev_ts >= min_int:
    print("1")
    sys.exit(0)

if (
    min_imp <= 0
    or math.isnan(loss)
    or math.isnan(prev_metric)
    or (prev_metric - loss) >= min_imp
):
    print("1")
else:
    print("0")
PY
    )
    if [ "$should_copy" = "1" ]; then
      echo "[$ts] 새 최고 성능: $(basename "$best_path") (eval_loss=$loss)" | tee -a "$LOG"
      rm -rf -- "$BEST_TMP"
      cp -r -- "$best_path" "$BEST_TMP"
      rm -rf -- "$BEST_DST"
      mv -- "$BEST_TMP" "$BEST_DST"
      echo "$best_path" >"$SRC_MARKER"
      echo "$loss" >"$METRIC_MARKER"
      date +%s >"$TS_MARKER"
    fi
  fi

  sleep "$POLL_INTERVAL"
done
