#!/usr/bin/env bash
set -eo pipefail

CONFIG_PATH="${CONFIG_PATH:-config.json}"
SESSION="${SESSION_NAME:-hp-search}"
LOG="${LOG_FILE:-/workspace/hp_search_runner.log}"
N_TRIALS="${N_TRIALS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HP_PY="$SCRIPT_DIR/script/hp_search.py"
PY_RUN="python3"
if command -v uv >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
  PY_RUN="uv run --no-sync python"
fi

if [ ! -f "$HP_PY" ]; then
  echo "오류: hp_search.py를 찾을 수 없습니다: $HP_PY"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux 세션 '$SESSION'이 이미 존재합니다."
  exit 1
fi

TRIAL_ARG=""
if [ -n "$N_TRIALS" ]; then
  TRIAL_ARG="--n-trials $N_TRIALS"
fi

tmux new-session -d -s "$SESSION" bash -c "
  set -o pipefail
  cd '$SCRIPT_DIR'

  $PY_RUN -u '$HP_PY' --config '$CONFIG_PATH' $TRIAL_ARG $EXTRA_ARGS 2>&1 | tee '$LOG'
  EXIT=\${PIPESTATUS[0]}
  if [ \$EXIT -ne 0 ]; then
    $PY_RUN '$SCRIPT_DIR/script/common/notify_cli.py' --config '$CONFIG_PATH' --title 'HP 탐색 실패' --exit-code \$EXIT --message 'hp_search.py 실행이 비정상 종료되었습니다.' || true
  fi
  exec bash
"

echo "tmux 세션 '$SESSION'에서 HP 탐색을 시작했습니다."
echo "  접속: tmux -u attach -t $SESSION"
echo "  로그: $LOG"
