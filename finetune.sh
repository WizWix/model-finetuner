#!/usr/bin/env bash
set -eo pipefail

CONFIG_PATH="${CONFIG_PATH:-config.json}"
SESSION="${SESSION_NAME:-finetune}"
LOG="${LOG_FILE:-/workspace/finetune.log}"
EPOCHS="${EPOCHS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SAVE_STEPS="${SAVE_STEPS:-}"
EVAL_STEPS="${EVAL_STEPS:-}"
LOGGING_STEPS="${LOGGING_STEPS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="$SCRIPT_DIR/script/train.py"
WATCH_SH="$SCRIPT_DIR/golden.sh"
DISABLE_GOLDEN_WATCH="${DISABLE_GOLDEN_WATCH:-1}"
PY_RUN="python3"
if command -v uv >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
  PY_RUN="uv run --no-sync python"
fi

if [ ! -f "$TRAIN_PY" ]; then
  echo "오류: train.py를 찾을 수 없습니다: $TRAIN_PY"
  exit 1
fi

read_cfg() {
  python3 - "$CONFIG_PATH" "$1" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
with open(path, encoding='utf-8') as f:
    cfg = json.load(f)
obj = cfg
for part in key.split('.'):
    if isinstance(obj, dict):
        obj = obj.get(part)
    else:
        obj = None
        break
if obj is None:
    print('')
else:
    print(obj)
PY
}

if [ -f "$CONFIG_PATH" ]; then
  CFG_SAVE_STEPS="$(read_cfg runtime.predefined_save_steps)"
  CFG_EVAL_STEPS="$(read_cfg runtime.predefined_eval_steps)"
  CFG_LOGGING_STEPS="$(read_cfg runtime.predefined_logging_steps)"
else
  CFG_SAVE_STEPS=""
  CFG_EVAL_STEPS=""
  CFG_LOGGING_STEPS=""
fi

if [ -z "$SAVE_STEPS" ]; then SAVE_STEPS="${CFG_SAVE_STEPS:-50}"; fi
if [ -z "$EVAL_STEPS" ]; then EVAL_STEPS="${CFG_EVAL_STEPS:-50}"; fi
if [ -z "$LOGGING_STEPS" ]; then LOGGING_STEPS="${CFG_LOGGING_STEPS:-10}"; fi

case "$SAVE_STEPS" in
'' | *[!0-9]* | 0)
  echo "오류: SAVE_STEPS는 1 이상의 정수여야 합니다. 현재값='$SAVE_STEPS'"
  exit 1
  ;;
esac
case "$EVAL_STEPS" in
'' | *[!0-9]*)
  echo "오류: EVAL_STEPS는 0 이상의 정수여야 합니다. 현재값='$EVAL_STEPS'"
  exit 1
  ;;
esac
case "$LOGGING_STEPS" in
'' | *[!0-9]* | 0)
  echo "오류: LOGGING_STEPS는 1 이상의 정수여야 합니다. 현재값='$LOGGING_STEPS'"
  exit 1
  ;;
esac

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux 세션 '$SESSION'이 이미 존재합니다."
  exit 1
fi

tmux new-session -d -s "$SESSION" bash -c "
  set -o pipefail
  cd '$SCRIPT_DIR'

  $PY_RUN -u '$TRAIN_PY' --config '$CONFIG_PATH' --epochs $EPOCHS --save-steps $SAVE_STEPS --eval-steps $EVAL_STEPS --logging-steps $LOGGING_STEPS $EXTRA_ARGS 2>&1 | tee '$LOG'
  EXIT=\${PIPESTATUS[0]}
  if [ \$EXIT -ne 0 ]; then
    $PY_RUN '$SCRIPT_DIR/script/common/notify_cli.py' --config '$CONFIG_PATH' --title '고정 HP 파인튜닝 실패' --exit-code \$EXIT --message 'train.py 실행이 비정상 종료되었습니다.' || true
  fi
  exec bash
"

if [ "$DISABLE_GOLDEN_WATCH" != "1" ] && [ -f "$WATCH_SH" ] && ! tmux has-session -t golden 2>/dev/null; then
  CONFIG_PATH="$CONFIG_PATH" tmux new-session -d -s golden "bash '$WATCH_SH'"
fi

echo "tmux 세션 '$SESSION'에서 고정 HP 파인튜닝을 시작했습니다."
echo "  접속: tmux -u attach -t $SESSION"
echo "  로그: $LOG"
echo "  step 설정(save/eval/logging): $SAVE_STEPS/$EVAL_STEPS/$LOGGING_STEPS"
if [ "$DISABLE_GOLDEN_WATCH" = "1" ]; then
  echo "  golden watcher: 비활성화(기본값, DISABLE_GOLDEN_WATCH=1)"
else
  echo "  golden watcher: 활성화(DISABLE_GOLDEN_WATCH=0)"
fi
