# kor-pest-detection-training

해충 이미지 분류 VLM 학습 파이프라인입니다.
핵심 워크플로는 다음 2가지입니다.

1. 하이퍼파라미터 탐색(Optuna): `run_hp_search.sh`
2. 고정 하이퍼파라미터 파인튜닝: `run_predefined_finetune.sh`

## 엔트리포인트 역할

- `run_hp_search.sh`
  - `hp_search.py`를 실행해 다수 trial을 탐색합니다.
  - 실패 시 `common/notify_cli.py`를 통해 Discord 알림을 보냅니다.
- `run_predefined_finetune.sh`
  - `train.py`를 실행해 `config.json.hyperparameters` 값으로 단일 학습을 수행합니다.
  - 실패 시 Discord 알림을 보냅니다.

## 설정 파일

모든 실행 설정은 `config.json`에서 관리합니다(표준 JSON, 주석 미지원).

### `paths`

- `data_dir`: 데이터셋 다운로드/학습용 JSONL 및 이미지 경로
- `volume_dir`: 공용 볼륨 루트
- `output_dir`: HP 탐색 결과물(트라이얼 아웃풋/분석 파일)
- `best_model_dir`: HP 탐색 후 retrain 결과 저장 경로
- `db_path`: Optuna SQLite DB 파일 위치
- `log_file`: HP 탐색 로그 파일
- `preload_cache_dir`: 이미지 preload 캐시 경로
- `final_output_dir`: 고정 HP 파인튜닝 결과 저장 경로
- `golden_dir`: `watch_golden.sh` 백업 사이드카 출력 경로

### `runtime`

- `study_name`: Optuna study 이름
- `base_model`: 베이스 모델 ID
- `random_seed`: 전체 시드
- `default_n_trials`: HP 탐색 기본 trial 수
- `wandb_project`, `wandb_entity`: W&B 런 식별 정보
- `predefined_save_steps`: 고정 파인튜닝 체크포인트 저장 주기 기본값
- `predefined_eval_steps`: 고정 파인튜닝 평가 주기 기본값
- `predefined_logging_steps`: 고정 파인튜닝 로그/W&B 기록 주기 기본값
- `predefined_save_only_model`: 고정 파인튜닝 체크포인트에서 옵티마이저/스케줄러 저장 생략 여부(기본 `true`)
- `predefined_min_free_space_gb`: 경고를 띄울 최소 디스크 여유 공간(GB)

### `notifications`

- `discord_webhooks`: Discord webhook URL 배열
  - URL이 1개여도 배열로 유지해야 합니다.

## Discord 알림 명령줄 도구

- 기본 텍스트 알림(기존 동작):
  - `python3 common/notify_cli.py --config config.json --title '작업 실패' --exit-code 1 --message '학습이 중단되었습니다.'`
- JSON 페이로드 직접 전송:
  - `python3 common/notify_cli.py --config config.json --payload-json '{"content":"hello","embeds":[{"title":"run failed"}]}'`
- JSON 파일에서 페이로드 전송:
  - `python3 common/notify_cli.py --config config.json --payload-json-file ./discord_payload.json`

### `github`

- `repo`: Optuna DB 백업/릴리즈용 GitHub repo
- `backup_db_path_in_repo`: repo 내 DB 업로드 경로

### `auth`

- `hf_token`: Hugging Face 토큰
- `wandb_api_key`: W&B API 키
- `github_token`: GitHub 토큰

### `huggingface`

- `dataset_repo`: 다운로드할 dataset repo ID
- `hf_repo_id`: 학습 결과 업로드 대상 model repo ID

### `hyperparameters`

`train.py`가 직접 읽는 고정 학습 하이퍼파라미터 묶음입니다.
LoRA/optimizer/batch/sequence/seed 관련 값이 여기에 모여 있습니다.

## 기능별 필수 설정값

아래 항목은 "해당 기능을 쓰는 경우" 반드시 유효해야 합니다.

### 1) 하이퍼파라미터 탐색 (`run_hp_search.sh`)

- 필수:
  - `paths.data_dir` (학습 데이터 JSONL + 이미지 위치)
  - `paths.output_dir` (트라이얼 산출물/분석 결과 저장)
  - `paths.db_path` (Optuna SQLite DB 저장 경로)
  - `paths.log_file` (탐색 로그 파일 경로)
  - `runtime.study_name` (Optuna 스터디 식별자)
  - `runtime.base_model` (베이스 모델)
- 조건부 필수:
  - `auth.hf_token`: 비공개 HF 데이터셋/모델에 접근하는 경우
  - `auth.wandb_api_key`: W&B 로깅을 켜서 실행하는 경우
  - `notifications.discord_webhooks`: Discord 알림이 필요한 경우

### 2) 고정 HP 파인튜닝 (`run_predefined_finetune.sh`)

- 필수:
  - `paths.data_dir`
  - `paths.final_output_dir`
  - `runtime.base_model`
  - `hyperparameters.*` (학습 하이퍼파라미터 세트)
- 조건부 필수:
  - `auth.hf_token`: `setup_a6000.sh` 실행 시 필수, 또는 비공개 HF 리소스 접근 시 필요
  - `huggingface.hf_repo_id` + `auth.hf_token`: 학습 결과를 HF Hub에 업로드할 경우
  - `auth.wandb_api_key`: W&B 로깅 사용 시
  - `notifications.discord_webhooks`: 단계별 Discord 알림이 필요한 경우

### 3) DB 백업/릴리즈(GitHub)

- 필수(해당 기능 사용 시):
  - `github.repo`
  - `github.backup_db_path_in_repo`
  - `auth.github_token`

## 고정 파인튜닝 Step 설정 가이드 (W&B 그래프 밀도)

`train.py`는 아래 3개 step 주기를 사용합니다.

- `--logging-steps`: train loss/W&B 포인트 기록 주기 (기본 `10`)
- `--save-steps`: 체크포인트 저장 주기 (기본 `25`)
- `--eval-steps`: 평가 주기 (`0`이면 `save_steps`와 동일)
- `--save-only-model`: 체크포인트에서 옵티마이저/스케줄러 상태 저장 생략
- `--save-full-state`: 체크포인트에 옵티마이저/스케줄러 상태까지 포함 저장

`run_predefined_finetune.sh` 기본값은 `config.runtime.predefined_*`를 우선 사용합니다.
환경변수(`SAVE_STEPS`, `EVAL_STEPS`, `LOGGING_STEPS`)를 주면 config 값보다 우선합니다.
`DISABLE_GOLDEN_WATCH=1`을 주면 `watch_golden.sh` 사이드카를 시작하지 않습니다(추가 디스크 사용 방지).

값을 작게 하면(예: 10) 그래프 포인트(꼭지점)가 많아지고 추세를 세밀하게 볼 수 있지만, 로깅/평가/저장 오버헤드가 증가합니다.
값을 크게 하면(예: 50) 포인트는 적어지지만 학습 자체 오버헤드는 줄어듭니다.

예시:

- 촘촘한 로깅 + 적당한 평가 주기:
  - `EXTRA_ARGS="--logging-steps 10 --save-steps 50 --eval-steps 50" bash run_predefined_finetune.sh`

## 실행 순서

1. `config.example.json`을 참고해 `config.json` 작성
2. 의존성 설치
   - Windows(개발/테스트): `uv sync --frozen --extra dev --no-install-project`
   - Runpod Linux(실행): `bash setup_a6000.sh`
3. 목적에 맞게 실행
   - HP 탐색: `bash run_hp_search.sh`
   - 고정 HP 파인튜닝: `bash run_predefined_finetune.sh`
4. 모니터링
   - 학습 세션: `tmux -u attach -t <session>`
   - 골든 체크포인트: `tail -f /workspace/_golden/watcher.log` (경로는 config에 따라 달라짐)

## 결과물 위치

- HP 탐색 결과: `config.paths.output_dir`
- Optuna DB: `config.paths.db_path`
- 고정 HP 파인튜닝 LoRA: `config.paths.final_output_dir/lora`
- 고정 HP 평가 결과: `config.paths.final_output_dir/evaluation`
- 골든 백업: `config.paths.golden_dir/best_ckpt`

## 의존성 관리 정책 (Windows + Runpod)

- 표준: `pyproject.toml` + `uv.lock`
- Python 버전: `3.12.x` (프로젝트 기준)
- Linux 학습 전용 의존성: `extra` 그룹 `train-linux` (`unsloth[cu128-torch280]`)
- 개발/테스트 의존성: `extra` 그룹 `dev`

### 빠른 시작

- Windows PowerShell:
  - `py -m pip install -U uv`
  - `uv sync --frozen --extra dev --no-install-project`
  - `uv run python train.py --config config.json --epochs 1`
- Runpod Linux:
  - `python3 -m pip install -U uv` (없는 경우)
  - `uv sync --frozen --extra train-linux --no-install-project`
  - `bash run_predefined_finetune.sh`

### pip fallback

- `uv`를 사용할 수 없는 환경에서는 lock export 파일을 사용:
  - `pip install -r requirements-linux.lock.txt`
  - `pip install -r requirements-dev.lock.txt` (Windows 개발/테스트)
