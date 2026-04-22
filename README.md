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

모든 실행 설정은 `config.json`에서 관리합니다(JSONC 주석 허용).

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
