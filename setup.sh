#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/config.json}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "오류: config 파일을 찾을 수 없습니다: $CONFIG_PATH"
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
elif isinstance(obj, list):
    print(' '.join(str(x) for x in obj))
else:
    print(obj)
PY
}

FINAL_OUTPUT_DIR="$(read_cfg paths.final_output_dir)"
DATA_DIR="$(read_cfg paths.data_dir)"
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

print_disk_usage() {
  echo "[disk] 사용량 요약"
  df -hP / "$HOME" "$SCRIPT_DIR" "$FINAL_OUTPUT_DIR" "$DATA_DIR" 2>/dev/null | awk 'NR==1 || !seen[$6]++'
}

print_top_usage() {
  local target="$1"
  if [ -d "$target" ]; then
    echo "[disk] 큰 디렉터리(top, $target)"
    du -xh -d 1 "$target" 2>/dev/null | sort -h | tail -n 15 || true
  fi
}

check_free_space_gb() {
  local min_gb="$1"
  local target="$2"
  local avail_kb
  avail_kb="$(df -Pk "$target" 2>/dev/null | awk 'NR==2 {print $4}')"
  if [ -z "$avail_kb" ]; then
    echo "경고: 디스크 여유 공간 확인 실패 (path=$target)"
    return 0
  fi
  local avail_gb=$((avail_kb / 1024 / 1024))
  if [ "$avail_gb" -lt "$min_gb" ]; then
    echo "오류: 디스크 공간 부족 (path=$target, free=${avail_gb}GB, required>=${min_gb}GB)"
    return 1
  fi
  echo "[disk] 여유 공간 확인: path=$target free=${avail_gb}GB (required>=${min_gb}GB)"
}

echo "=== [1/6] 기본 유틸리티 설치(apt) ==="
if [ "${INSTALL_BASE_UTILS:-1}" = "1" ] && command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  APT_PACKAGES="${BASIC_APT_PACKAGES:-ca-certificates curl git nano jq unzip zip procps less}"
  APT_PREFIX=""
  if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
      APT_PREFIX="sudo"
    else
      echo "오류: apt-get 실행에 root 권한이 필요하지만 sudo를 찾을 수 없습니다."
      exit 1
    fi
  fi
  echo "[apt] 설치 패키지: $APT_PACKAGES"
  $APT_PREFIX apt-get update -y
  # shellcheck disable=SC2086
  $APT_PREFIX apt-get install -y --no-install-recommends $APT_PACKAGES
  $APT_PREFIX rm -rf /var/lib/apt/lists/*
else
  echo "[apt] 설치 단계 스킵 (INSTALL_BASE_UTILS=${INSTALL_BASE_UTILS:-1}, apt-get 미탐지 가능)"
fi

echo "=== [2/6] uv 확인/설치 ==="
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --upgrade uv
fi

echo "=== [3/6] 의존성 동기화(uv) ==="
LOCK_ARG=""
if [ -f "$SCRIPT_DIR/uv.lock" ]; then
  LOCK_ARG="--frozen"
fi
UV_NO_CACHE="${UV_NO_CACHE:-1}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
print_disk_usage
check_free_space_gb "${MIN_FREE_GB_SYNC:-20}" "${UV_SPACE_CHECK_PATH:-$HOME}"
UV_VERBOSE_ARG=""
if [ "${UV_SYNC_VERBOSE:-1}" = "1" ]; then
  UV_VERBOSE_ARG="-v"
fi
UV_NO_CACHE_ARG=""
if [ "$UV_NO_CACHE" = "1" ]; then
  UV_NO_CACHE_ARG="--no-cache"
fi
echo "[uv] sync 시작: UV_LINK_MODE=$UV_LINK_MODE uv sync $LOCK_ARG --extra train-linux --no-install-project $UV_NO_CACHE_ARG $UV_VERBOSE_ARG"
time UV_LINK_MODE="$UV_LINK_MODE" uv sync $LOCK_ARG --extra train-linux --no-install-project $UV_NO_CACHE_ARG $UV_VERBOSE_ARG
print_disk_usage
print_top_usage "${UV_CACHE_DIR:-$HOME/.cache/uv}"

echo "=== [4/6] GPU 확인 ==="
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

echo "=== [5/6] 데이터셋 다운로드 ==="
check_free_space_gb "${MIN_FREE_GB_DATASET:-30}" "${DATA_SPACE_CHECK_PATH:-$DATA_DIR}"
uv run --no-sync python "$SCRIPT_DIR/script/download_dataset.py" --config "$CONFIG_PATH"
print_disk_usage
print_top_usage "${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== [6/6] 실행 스크립트 권한 ==="
chmod +x "$SCRIPT_DIR/finetune.sh" "$SCRIPT_DIR/search.sh" "$SCRIPT_DIR/golden.sh" 2>/dev/null || true

cat <<EOT

✅ setup 완료

다음 실행:
  CONFIG_PATH=$CONFIG_PATH bash $SCRIPT_DIR/finetune.sh
  CONFIG_PATH=$CONFIG_PATH N_TRIALS=20 bash $SCRIPT_DIR/search.sh

EOT
