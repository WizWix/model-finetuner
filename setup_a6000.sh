#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/config.json}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "오류: config 파일을 찾을 수 없습니다: $CONFIG_PATH"
  exit 1
fi

read_cfg() {
  python3 - "$CONFIG_PATH" "$1" << 'PY'
import json, re, sys
path, key = sys.argv[1], sys.argv[2]
text = open(path, encoding='utf-8').read()
text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
text = re.sub(r'//.*', '', text)
cfg = json.loads(text)
obj = cfg
for part in key.split('.'):
    if isinstance(obj, dict):
        obj = obj.get(part)
    else:
        obj = None
        break
if obj is None:
    print('')
elif isinstance(obj, list):
    print(' '.join(str(x) for x in obj))
else:
    print(obj)
PY
}

FINAL_OUTPUT_DIR="$(read_cfg paths.final_output_dir)"
HF_TOKEN_CFG="$(read_cfg auth.hf_token)"
WANDB_API_KEY_CFG="$(read_cfg auth.wandb_api_key)"
HF_REPO_ID_CFG="$(read_cfg huggingface.hf_repo_id)"

if [ -n "$HF_TOKEN_CFG" ]; then export HF_TOKEN="$HF_TOKEN_CFG"; fi
if [ -n "$WANDB_API_KEY_CFG" ]; then export WANDB_API_KEY="$WANDB_API_KEY_CFG"; fi
if [ -n "$HF_REPO_ID_CFG" ]; then export HF_REPO_ID="$HF_REPO_ID_CFG"; fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "오류: HF_TOKEN이 필요합니다. config.auth.hf_token 또는 환경변수로 설정하세요."
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$FINAL_OUTPUT_DIR"

echo "=== [1/5] uv 확인/설치 ==="
if ! command -v uv > /dev/null 2>&1; then
  python3 -m pip install --upgrade uv
fi

echo "=== [2/5] 의존성 동기화(uv) ==="
LOCK_ARG=""
if [ -f "$SCRIPT_DIR/uv.lock" ]; then
  LOCK_ARG="--frozen"
fi
uv sync $LOCK_ARG --extra train-linux --no-install-project

echo "=== [3/5] GPU 확인 ==="
uv run --no-sync python -c "
import torch
assert torch.cuda.is_available(), 'CUDA GPU가 감지되지 않습니다'
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
cap = torch.cuda.get_device_capability(0)
print(f'GPU: {name} ({vram:.0f}GB, sm_{cap[0]}{cap[1]})')
if cap[0] < 8:
    raise SystemExit('Ampere(sm80+) 이상이 필요합니다')
"

export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== [4/5] 데이터셋 다운로드 ==="
uv run --no-sync python "$SCRIPT_DIR/download_dataset.py" --config "$CONFIG_PATH"

echo "=== [5/5] 실행 스크립트 권한 ==="
chmod +x "$SCRIPT_DIR/run_predefined_finetune.sh" "$SCRIPT_DIR/run_hp_search.sh" "$SCRIPT_DIR/watch_golden.sh" 2> /dev/null || true

cat << EOT

✅ setup 완료

다음 실행:
  CONFIG_PATH=$CONFIG_PATH bash $SCRIPT_DIR/run_predefined_finetune.sh
  CONFIG_PATH=$CONFIG_PATH N_TRIALS=20 bash $SCRIPT_DIR/run_hp_search.sh

EOT
