# Golden Snapshot 업로드 가이드

학습이 비정상 종료되거나 자동 중단된 경우, `golden`에 보존된 best checkpoint를 HuggingFace Hub에 PEFT Adapter로 업로드하는 절차입니다.

## 1) 전제 조건

- `config.json`에 `paths.golden_dir`가 설정되어 있어야 함 (기본: `/workspace/_golden`)
- `auth.hf_token` 또는 환경변수 `HF_TOKEN`이 설정되어 있어야 함
- 업로드 대상 repo (`huggingface.hf_repo_id`)가 존재해야 함

## 2) best snapshot 위치 확인

```bash
GOLDEN_DIR="/workspace/_golden"
cat "$GOLDEN_DIR/best_source.txt"
ls -la "$GOLDEN_DIR/best_ckpt"
```

- `best_source.txt`: 원본 best checkpoint 경로
- `best_ckpt`: `golden.sh`가 복사해둔 안전한 백업 경로

## 3) 업로드 전 파일 확인

아래 파일이 `best_ckpt` 안에 있는지 확인합니다.

```bash
ls -la /workspace/_golden/best_ckpt
```

최소 권장:
- `adapter_config.json`
- `adapter_model.safetensors` (또는 유사 adapter weight 파일)
- tokenizer 관련 파일들 (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` 등)

## 4) HuggingFace 업로드

### 방법 A. Python one-liner (권장)

```bash
python3 - <<'PY'
import json, os
from huggingface_hub import HfApi, upload_folder

cfg = json.load(open("config.json", encoding="utf-8"))
repo_id = cfg.get("huggingface", {}).get("hf_repo_id", "").strip()
token = os.environ.get("HF_TOKEN") or cfg.get("auth", {}).get("hf_token", "").strip()
folder = os.path.join(cfg.get("paths", {}).get("golden_dir", "/workspace/_golden"), "best_ckpt")

if not repo_id:
    raise SystemExit("오류: huggingface.hf_repo_id가 비어 있습니다.")
if not token:
    raise SystemExit("오류: HF_TOKEN이 비어 있습니다.")
if not os.path.isdir(folder):
    raise SystemExit(f"오류: golden best_ckpt 경로가 없습니다: {folder}")

api = HfApi(token=token)
api.whoami()
upload_folder(
    folder_path=folder,
    repo_id=repo_id,
    path_in_repo=".",
    token=token,
)
print(f"업로드 완료: https://huggingface.co/{repo_id}")
PY
```

### 방법 B. huggingface-cli

```bash
export HF_TOKEN="<your_token>"
huggingface-cli upload "<your-repo-id>" /workspace/_golden/best_ckpt . --repo-type model
```

## 5) 업로드 후 확인

- Hub 파일 목록에 adapter/tokenizer 파일이 있는지 확인
- 모델 사용 테스트:

```python
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained("unsloth/Qwen3.5-9B")
model.load_adapter("<your-repo-id>")
FastVisionModel.for_inference(model)
```

## 6) (선택) Discord 완료 알림

```bash
python3 script/common/notify_cli.py \
  --config config.json \
  --title "Golden Snapshot 업로드 완료" \
  --exit-code 0 \
  --message "best_ckpt를 HuggingFace에 업로드했습니다."
```

## 운영 팁

- `golden.sh`를 기본 활성화(`DISABLE_GOLDEN_WATCH=0`)로 유지하면, 발산/중단 상황에서도 best snapshot 복구가 쉬워집니다.
- 디스크 폭주를 막으려면 `predefined_save_total_limit`, `predefined_save_only_model`, `predefined_save_steps/eval_steps`를 함께 조정하세요.
