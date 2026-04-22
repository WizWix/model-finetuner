#!/usr/bin/env bash
set -eo pipefail

CONFIG_PATH="${CONFIG_PATH:-config.json}"
SESSION="${SESSION_NAME:-predefined-finetune}"
LOG="${LOG_FILE:-/workspace/predefined_finetune.log}"
EPOCHS="${EPOCHS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="$SCRIPT_DIR/train.py"
WATCH_SH="$SCRIPT_DIR/watch_golden.sh"
PY_RUN="python3"
if command -v uv > /dev/null 2>&1 && [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
  PY_RUN="uv run --no-sync python"
fi

if [ ! -f "$TRAIN_PY" ]; then
  echo "오류: train.py를 찾을 수 없습니다: $TRAIN_PY"
  exit 1
fi

if tmux has-session -t "$SESSION" 2> /dev/null; then
  echo "tmux 세션 '$SESSION'이 이미 존재합니다."
  exit 1
fi

tmux new-session -d -s "$SESSION" bash -c "
  set -o pipefail
  cd '$SCRIPT_DIR'

  $PY_RUN -u '$TRAIN_PY' --config '$CONFIG_PATH' --epochs $EPOCHS $EXTRA_ARGS 2>&1 | tee '$LOG'
  EXIT=\${PIPESTATUS[0]}
  if [ \$EXIT -ne 0 ]; then
    $PY_RUN '$SCRIPT_DIR/common/notify_cli.py' --config '$CONFIG_PATH' --title '고정 HP 파인튜닝 실패' --exit-code \$EXIT --message 'train.py 실행이 비정상 종료되었습니다.' || true
  fi
  exec bash
"

if [ -f "$WATCH_SH" ] && ! tmux has-session -t golden 2> /dev/null; then
  CONFIG_PATH="$CONFIG_PATH" tmux new-session -d -s golden "bash '$WATCH_SH'"
fi

echo "tmux 세션 '$SESSION'에서 고정 HP 파인튜닝을 시작했습니다."
echo "  접속: tmux -u attach -t $SESSION"
echo "  로그: $LOG"
