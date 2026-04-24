#!/usr/bin/env python3
"""
해충 탐지 비전 모델(Qwen3.5-9B LoRA) 하이퍼파라미터 탐색
=========================================================
Optuna + Unsloth + SFTTrainer + W&B + Discord 알림

사용 예시:
    bash setup_runpod.sh                        # 전체 자동 파이프라인
    python hp_search.py --proxy --n-trials 50  # 1단계: 빠른 선별
    python hp_search.py --n-trials 10 --retrain # 2단계: 전체 학습 + 재학습
    python hp_search.py --analyze              # 저장된 결과 분석

환경 변수(setup_runpod.sh에서 설정):
    WANDB_API_KEY          — Weights & Biases API 키
    DISCORD_WEBHOOK_URL    — 한국어 알림용 Discord 웹훅
    GITHUB_TOKEN           — GitHub PAT(필요 시)
    HF_TOKEN               — HuggingFace 토큰(비공개 데이터셋인 경우)
    HP_DATA_DIR            — 데이터셋 경로(기본값: /workspace/data)
    HP_VOLUME_DIR          — 영구 볼륨 경로(기본값: /workspace)
    HP_DB_PATH             — Optuna SQLite DB 경로
    HP_OUTPUT_DIR          — 트라이얼 출력 디렉터리
    HP_BEST_MODEL_DIR      — 최적 모델 저장 경로
    WANDB_PROJECT          — W&B 프로젝트명
    WANDB_ENTITY           — W&B 팀/사용자명
"""

import argparse
import json
import logging
import math
import os
import random
import shutil
import signal
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

import optuna
import torch
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from common.app_config import (
    apply_auth_environment,
    get_discord_webhooks,
    load_app_config,
)
from common.discord_utils import send_discord
from common.training_core import (
    build_sft_config_kwargs,
    clear_gpu_memory as core_clear_gpu_memory,
    evaluate_model as core_evaluate_model,
    get_line_count as core_get_line_count,
    get_max_data_fraction as core_get_max_data_fraction,
    load_dataset_from_jsonl as core_load_dataset_from_jsonl,
    load_model_with_retry as core_load_model_with_retry,
    PROMPTS,
    recommend_dataloader_num_workers,
    SYSTEM_MSG,
)
from common.wandb_utils import wandb_is_available as core_wandb_is_available

# Ampere GPU에서 TF32를 사용해 남은 FP32 연산(옵티마이저, 손실 계산)을 가속
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# 최대 이미지 크기 제한(공통 데이터 로더에 전달)
MAX_IMAGE_DIM = 768

# ═══════════════════════════════════════════════════════════════════════
# 1. 설정
# ═══════════════════════════════════════════════════════════════════════

DATA_DIR = os.environ.get("HP_DATA_DIR", "/workspace/data")
VOLUME_DIR = os.environ.get("HP_VOLUME_DIR", "/workspace")
OUTPUT_BASE = os.environ.get("HP_OUTPUT_DIR", f"{VOLUME_DIR}/hp_search_outputs")
BEST_MODEL_DIR = os.environ.get("HP_BEST_MODEL_DIR", f"{VOLUME_DIR}/best-pest-detector")
DB_PATH = os.environ.get("HP_DB_PATH", f"{VOLUME_DIR}/hp_search_results.db")
STORAGE_DB = f"sqlite:///{DB_PATH}"
LOG_FILE = os.environ.get("HP_LOG_FILE", f"{VOLUME_DIR}/hp_search.log")
PRELOAD_CACHE_DIR = os.environ.get(
    "HP_PRELOAD_CACHE_DIR", f"{VOLUME_DIR}/preload_cache"
)
_env_dl_workers = os.environ.get("HP_DATALOADER_NUM_WORKERS")
DATALOADER_NUM_WORKERS = int(_env_dl_workers) if _env_dl_workers is not None else None

STUDY_NAME = "pest-detection-hpsearch"
BASE_MODEL = "unsloth/Qwen3.5-9B"
RANDOM_SEED = 42
N_TRIALS_DEFAULT = 30

DISCORD_WEBHOOKS = []
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "pest-detection-hpsearch")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")

# GitHub 백업: 새 최고 성능 트라이얼이 나오면 Optuna DB 업로드
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("github_pat", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "pfox1995/pest-hyperparameter-search")
GITHUB_DB_BACKUP_PATH = "hp_search_results.db"

QUICK_DATA_FRACTION = 0.2
QUICK_EPOCHS = 1
PROXY_DATA_FRACTION = 0.05
PROXY_EPOCHS = 1
# 실제 소요 시간을 고려한 step 추정: 샘플링된 하이퍼파라미터와 무관하게
# 트라이얼당 약 30~40분에 맞추도록 계산
# 모델(유효 배치만으로 비례하지 않음):
#   step_sec = (OVERHEAD × grad_accum) + (PER_SAMPLE × eff_batch) × vision_mult
# - OVERHEAD: 마이크로배치당 커널 실행/콜레이터 준비(스텝마다 ga회 발생)
# - PER_SAMPLE: 샘플당 순전파/역전파 연산량(eff_batch에 비례)
# bs=1,ga=8 과 bs=4,ga=2 는 eff_batch=8로 같아도 실제 스텝 시간은 다름
# (전자는 스텝당 루프 횟수가 더 많음)
# 기준: bs=4, ga=4, vision=False에서 측정된 19초/step
PROXY_TARGET_MIN = 35  # 트라이얼당 목표 시간(분)
PROXY_STEP_OVERHEAD = 1.5  # 마이크로배치당 초(스텝당 ga배)
PROXY_STEP_PER_SAMPLE = 1.0  # 샘플당 초(스텝당 eff_batch배)
PROXY_VISION_MULT = 1.3  # finetune_vision=True일 때 시간 배수
PROXY_STEPS_FLOOR = 50  # 너무 낮아지지 않도록 하한
PROXY_STEPS_CEILING = 200  # 너무 길어지지 않도록 상한
PROXY_VAL_CAP = 150  # 트레이너 손실 평가용 검증셋 상한(전체는 수천 개)


def initialize_from_config(config_path: str) -> dict:
    """config.json을 로드하고 런타임 전역 설정값에 반영한다."""
    global DATA_DIR, VOLUME_DIR, OUTPUT_BASE, BEST_MODEL_DIR, DB_PATH, STORAGE_DB
    global LOG_FILE, PRELOAD_CACHE_DIR, STUDY_NAME, BASE_MODEL, RANDOM_SEED
    global N_TRIALS_DEFAULT, WANDB_PROJECT, WANDB_ENTITY, GITHUB_REPO
    global GITHUB_TOKEN, DISCORD_WEBHOOKS, GITHUB_DB_BACKUP_PATH
    global DATALOADER_NUM_WORKERS

    cfg = load_app_config(config_path)
    apply_auth_environment(cfg)

    paths = cfg.get("paths", {})
    runtime = cfg.get("runtime", {})
    github = cfg.get("github", {})

    DATA_DIR = paths.get("data_dir", DATA_DIR)
    VOLUME_DIR = paths.get("volume_dir", VOLUME_DIR)
    OUTPUT_BASE = paths.get("output_dir", OUTPUT_BASE)
    BEST_MODEL_DIR = paths.get("best_model_dir", BEST_MODEL_DIR)
    DB_PATH = paths.get("db_path", DB_PATH)
    LOG_FILE = paths.get("log_file", LOG_FILE)
    PRELOAD_CACHE_DIR = paths.get("preload_cache_dir", PRELOAD_CACHE_DIR)
    STORAGE_DB = f"sqlite:///{DB_PATH}"

    STUDY_NAME = runtime.get("study_name", STUDY_NAME)
    BASE_MODEL = runtime.get("base_model", BASE_MODEL)
    RANDOM_SEED = int(runtime.get("random_seed", RANDOM_SEED))
    N_TRIALS_DEFAULT = int(runtime.get("default_n_trials", N_TRIALS_DEFAULT))
    WANDB_PROJECT = runtime.get("wandb_project", WANDB_PROJECT)
    WANDB_ENTITY = runtime.get("wandb_entity", WANDB_ENTITY)
    DATALOADER_NUM_WORKERS = runtime.get(
        "dataloader_num_workers", DATALOADER_NUM_WORKERS
    )

    GITHUB_REPO = github.get("repo", GITHUB_REPO)
    GITHUB_DB_BACKUP_PATH = github.get("backup_db_path_in_repo", GITHUB_DB_BACKUP_PATH)
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("github_pat", "")

    DISCORD_WEBHOOKS = get_discord_webhooks(cfg)
    return cfg


def get_max_data_fraction(n_train_samples: int) -> float:
    return core_get_max_data_fraction(n_train_samples)


SEARCH_SPACE = {
    # 30트라이얼 분석 + Unsloth/Qwen 자료를 바탕으로 탐색 공간 축소
    # 범주형 후보 목록을 변경하면 새 DB 사용 권장
    "per_device_train_batch_size": [1, 2, 4],
    "gradient_accumulation_steps": [2, 4, 8],
    "learning_rate": (1e-5, 2e-4),  # Unsloth 권장 시작점은 2e-4
    "num_train_epochs": [2, 3, 5],
    "warmup_steps": [0, 10, 50, 100],
    "weight_decay": (0.0, 0.05),
    "lr_scheduler_type": ["linear", "cosine", "cosine_with_restarts"],
    "max_seq_length": [1024],  # 2048은 VRAM 낭비, 해충 라벨은 보통 2~6토큰
    "lora_r": [16, 32, 64],  # r=8은 9B VLM 기준 용량이 너무 작음
    "lora_alpha_ratio": [1.0, 2.0],  # 4.0은 높은 r에서 불안정
    "finetune_vision_layers": [True, False],
    "use_rslora": [True],  # 데이터/리서치 기준으로 항상 우세
    "crop_tight_prob": (0.4, 0.65),
}


def _get_line_count(path: str) -> int:
    return core_get_line_count(path)


OBJECTIVE_METRIC = "eval_loss"

# 볼륨이 없을 때 import 단계에서 크래시하지 않도록 로거는 main()에서 설정
logger = logging.getLogger(__name__)


class TrialExecutionError(Exception):
    """트라이얼 내부 실행 실패를 Optuna FAIL 상태로 남기기 위한 예외."""


# ═══════════════════════════════════════════════════════════════════════
# 2. Discord 알림(비동기)
# ═══════════════════════════════════════════════════════════════════════

KST = timezone(timedelta(hours=9))


def _kst_now():
    return datetime.now(KST).strftime("%m/%d %H:%M")


def discord_send(
    content: str | None = None, embed: dict[str, Any] | None = None
) -> None:
    """설정된 웹훅 목록으로 Discord 메시지를 비동기 전송한다."""
    send_discord(DISCORD_WEBHOOKS, content=content, embed=embed, timeout=10)


def discord_search_started(n_trials, mode):
    discord_send(
        embed={
            "embeds": [
                {
                    "author": {"name": "AI 모델 하이퍼-파라미터 탐색"},
                    "color": 3447003,
                    "title": "🔍 하이퍼-파라미터 검색 시작",
                    "description": f"- 모드: {mode}\n- 트라이얼 수: {str(n_trials)}\n- 모델: {BASE_MODEL}\n- 목적 함수: {OBJECTIVE_METRIC}",
                    "footer": {"text": f"시작: {_kst_now()} KST"},
                }
            ]
        }
    )


def discord_trial_complete(
    trial_num, value, params, duration_min, is_best, completed, total, eval_metrics=None
):
    color = 0x2ECC71 if is_best else 0x95A5A6
    title = (
        f"🏆 트라이얼 #{trial_num} — 새로운 최고 기록!"
        if is_best
        else f"✅ 트라이얼 #{trial_num} 완료"
    )
    bs = params.get("batch_size", 0) * params.get("grad_accum", 0)
    fields = [
        {"name": "Eval Loss", "value": f"`{value:.6f}`", "inline": True},
        {"name": "소요 시간", "value": f"{duration_min:.1f}분", "inline": True},
        {"name": "진행률", "value": f"{completed}/{total}", "inline": True},
    ]
    if eval_metrics:
        fields.extend(
            [
                {
                    "name": "정확도",
                    "value": f"`{eval_metrics.get('accuracy', 0):.4f}`",
                    "inline": True,
                },
                {
                    "name": "F1 (macro)",
                    "value": f"`{eval_metrics.get('f1_macro', 0):.4f}`",
                    "inline": True,
                },
                {
                    "name": "F1 (weighted)",
                    "value": f"`{eval_metrics.get('f1_weighted', 0):.4f}`",
                    "inline": True,
                },
                {
                    "name": "Precision",
                    "value": f"`{eval_metrics.get('precision_macro', 0):.4f}`",
                    "inline": True,
                },
                {
                    "name": "Recall",
                    "value": f"`{eval_metrics.get('recall_macro', 0):.4f}`",
                    "inline": True,
                },
            ]
        )
    fields.extend(
        [
            {
                "name": "학습률",
                "value": f"`{params.get('learning_rate', 0):.2e}`",
                "inline": True,
            },
            {"name": "LoRA r", "value": str(params.get("lora_r", "?")), "inline": True},
            {"name": "배치 크기", "value": str(bs), "inline": True},
        ]
    )
    discord_send(
        embed={
            "title": title,
            "color": color,
            "fields": fields,
            "footer": {"text": f"{_kst_now()} KST"},
        }
    )


def discord_trial_pruned(trial_num, reason):
    discord_send(
        embed={
            "title": f"✂️ 트라이얼 #{trial_num} 가지치기",
            "description": reason,
            "color": 0xE67E22,
            "footer": {"text": f"{_kst_now()} KST"},
        }
    )


def discord_trial_error(trial_num, error):
    discord_send(
        embed={
            "title": f"❌ 트라이얼 #{trial_num} 오류",
            "description": f"```{str(error)[:500]}```",
            "color": 0xE74C3C,
            "footer": {"text": f"{_kst_now()} KST"},
        }
    )


def discord_phase_complete(
    phase, best_value, best_params, completed, pruned, total_time_hr
):
    r = best_params.get("lora_r", "?")
    alpha_ratio = best_params.get("lora_alpha_ratio", 1)
    alpha = int(r * alpha_ratio) if isinstance(r, int) else "?"
    discord_send(
        embed={
            "title": f"🎯 {phase} 완료!",
            "color": 0x9B59B6,
            "fields": [
                {
                    "name": "최적 Eval Loss",
                    "value": f"`{best_value:.6f}`",
                    "inline": True,
                },
                {
                    "name": "총 소요 시간",
                    "value": f"{total_time_hr:.1f}시간",
                    "inline": True,
                },
                {
                    "name": "완료/가지치기",
                    "value": f"{completed}/{pruned}",
                    "inline": True,
                },
                {
                    "name": "최적 학습률",
                    "value": f"`{best_params.get('learning_rate', 0):.2e}`",
                    "inline": True,
                },
                {"name": "최적 LoRA", "value": f"r={r}, a={alpha}", "inline": True},
                {
                    "name": "비전 레이어",
                    "value": "O" if best_params.get("finetune_vision") else "X",
                    "inline": True,
                },
            ],
            "footer": {"text": f"완료: {_kst_now()} KST"},
        }
    )


def discord_retrain_complete(accuracy, model_path):
    discord_send(
        embed={
            "title": "🚀 최적 모델 학습 완료!",
            "description": f"최종 정확도: **{accuracy:.2%}**\n모델 경로: `{model_path}`",
            "color": 0x2ECC71,
            "footer": {"text": f"완료: {_kst_now()} KST"},
        }
    )


def discord_error(error):
    discord_send(
        embed={
            "title": "💥 치명적 오류 발생",
            "description": f"```{str(error)[:800]}```",
            "color": 0xE74C3C,
            "footer": {"text": f"{_kst_now()} KST"},
        }
    )


# 처리 가능한 시그널은 종료 전에 Discord 알림 전송
# SIGKILL/OOM-kill은 여기서 잡을 수 없고, bash 래퍼가
# ${PIPESTATUS[0]}로 감지해 별도 웹훅을 전송한다.
def _install_fatal_signal_handlers():
    def _handler(signum, _frame):
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = f"SIG{signum}"
        try:
            discord_error(f"프로세스가 시그널 {name}({signum})로 종료됨")
            time.sleep(1.5)  # 비동기 Discord 전송 스레드가 마무리될 시간 확보
        finally:
            os._exit(128 + signum)

    signals_to_handle: list[int] = [int(signal.SIGTERM)]
    for sig_name in ("SIGHUP", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signals_to_handle.append(int(sig))

    for sig in signals_to_handle:
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


# ═══════════════════════════════════════════════════════════════════════
# 3. W&B 연동
# ═══════════════════════════════════════════════════════════════════════


def wandb_is_available():
    return core_wandb_is_available(logger=logger)


def wandb_init_trial(trial_num, params):
    if not wandb_is_available():
        return None
    import wandb

    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY or None,
        name=f"trial-{trial_num:03d}",
        group=STUDY_NAME,
        config=params,
        reinit=True,
        tags=["hp-search", "pest-detection", "qwen3.5-9b"],
    )


def wandb_finish(run, exit_code=0):
    """W&B run을 안전하게 종료한다. run이 None이면 아무 작업도 하지 않는다."""
    if run is None:
        return
    try:
        import wandb

        wandb.finish(exit_code=exit_code)
    except Exception:
        pass


def wandb_log_best_summary(study):
    if not wandb_is_available():
        return
    import wandb

    best = study.best_trial
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY or None,
        name="best-params-summary",
        group=STUDY_NAME,
        reinit=True,
        tags=["summary"],
    )
    wandb.config.update(best.params)
    wandb.log(
        {
            "best_eval_loss": best.value,
            "best_train_loss": best.user_attrs.get("train_loss", 0),
            "best_trial_number": best.number,
        }
    )
    wandb.finish()


# ═══════════════════════════════════════════════════════════════════════
# 3b. GitHub DB 백업
# ═══════════════════════════════════════════════════════════════════════


def github_upload_db(reason: str = ""):
    """백업을 위해 Optuna SQLite DB를 GitHub에 업로드한다.

    GitHub Contents API(PUT)로 파일을 생성/업데이트하며,
    학습 블로킹을 피하기 위해 백그라운드 스레드에서 실행한다.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    if not os.path.exists(DB_PATH):
        return

    def _upload():
        import base64

        import requests as _req

        try:
            with open(DB_PATH, "rb") as f:
                content = base64.b64encode(f.read()).decode()

            url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_DB_BACKUP_PATH}"
            headers = {
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json",
            }

            # 현재 파일 SHA 조회(업데이트 시 필요)
            sha = None
            try:
                resp = _req.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    sha = resp.json().get("sha")
            except Exception:
                pass  # 파일이 아직 없거나 API 오류면 새로 생성

            timestamp = _kst_now()
            message = f"Update Optuna DB ({timestamp} KST)"
            if reason:
                message += f" — {reason}"

            payload = {"message": message, "content": content}
            if sha:
                payload["sha"] = sha

            resp = _req.put(url, json=payload, headers=headers, timeout=30)
            if resp.status_code in (200, 201):
                logger.info(f"Optuna DB → GitHub 업로드 완료 ({reason})")
            else:
                logger.warning(
                    f"GitHub 업로드 실패: {resp.status_code} {resp.text[:200]}"
                )
        except Exception as e:
            logger.warning(f"GitHub 업로드 오류: {e}")

    threading.Thread(target=_upload, daemon=True).start()


def github_create_release(tag: str, name: str, body: str, files: dict):
    """GitHub Release를 만들고 에셋(LoRA 어댑터, 평가 결과 등)을 업로드한다.

    Args:
        tag:   릴리스 태그, 예: "run-20260416"
        name:  릴리스 제목
        body:  릴리스 설명(markdown)
        files: 에셋 업로드용 {표시이름: 로컬경로} 딕셔너리.
               디렉터리는 업로드 전에 자동으로 tar.gz 압축.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        logger.warning("GitHub 토큰 또는 레포가 설정되지 않아 릴리스를 건너뜁니다.")
        return None
    import tarfile
    import tempfile

    import requests as _req

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    api = f"https://api.github.com/repos/{GITHUB_REPO}"

    # ─── 릴리스 생성 ────────────────────────────────────────────────
    logger.info(f"GitHub 릴리스 생성 중: {tag}")
    resp = _req.post(
        f"{api}/releases",
        headers=headers,
        json={
            "tag_name": tag,
            "name": name,
            "body": body,
            "draft": False,
        },
        timeout=30,
    )

    if resp.status_code == 422:
        # 태그가 이미 존재하면 기존 릴리스를 찾아 갱신
        logger.info(f"태그 {tag} 이미 존재, 기존 릴리스에 에셋 추가")
        resp = _req.get(f"{api}/releases/tags/{tag}", headers=headers, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"기존 릴리스 조회 실패: {resp.status_code}")
            return None
        release = resp.json()
        # 교체를 위해 기존 에셋 삭제
        for asset in release.get("assets", []):
            _req.delete(asset["url"], headers=headers, timeout=15)
    elif resp.status_code == 201:
        release = resp.json()
    else:
        logger.warning(f"릴리스 생성 실패: {resp.status_code} {resp.text[:300]}")
        return None

    upload_url = release["upload_url"].replace("{?name,label}", "")

    # ─── 에셋 업로드 ────────────────────────────────────────────────
    for display_name, local_path in files.items():
        if not os.path.exists(local_path):
            logger.warning(f"에셋 없음, 건너뜀: {local_path}")
            continue

        # 디렉터리는 업로드 전에 tar.gz로 압축
        if os.path.isdir(local_path):
            tmp_tar = os.path.join(tempfile.gettempdir(), f"{display_name}.tar.gz")
            logger.info(f"압축 중: {local_path} → {tmp_tar}")
            with tarfile.open(tmp_tar, "w:gz") as tar:
                tar.add(local_path, arcname=os.path.basename(local_path))
            local_path = tmp_tar
            display_name = f"{display_name}.tar.gz"

        file_size = os.path.getsize(local_path) / 1024**2
        logger.info(f"업로드 중: {display_name} ({file_size:.1f}MB)")

        with open(local_path, "rb") as f:
            resp = _req.post(
                f"{upload_url}?name={display_name}",
                headers={
                    "Authorization": f"token {GITHUB_TOKEN}",
                    "Content-Type": "application/octet-stream",
                },
                data=f,
                timeout=300,
            )
        if resp.status_code == 201:
            logger.info(f"  ✓ {display_name}")
        else:
            logger.warning(f"  ✗ {display_name}: {resp.status_code}")

    release_url = release.get("html_url", "")
    logger.info(f"GitHub 릴리스 완료: {release_url}")
    return release_url


def github_upload_results(eval_result=None):
    """최적 모델/평가 결과/HP 탐색 결과를 GitHub Release로 업로드한다."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return

    timestamp = datetime.now(KST).strftime("%Y%m%d-%H%M")
    tag = f"run-{timestamp}"

    # 릴리스 본문 생성
    body_parts = [f"## HP 탐색 결과 — {_kst_now()} KST\n"]
    if eval_result:
        body_parts.append(
            f"- **Accuracy**: {eval_result.get('accuracy', 0):.4f}\n"
            f"- **F1 (macro)**: {eval_result.get('f1_macro', 0):.4f}\n"
            f"- **F1 (weighted)**: {eval_result.get('f1_weighted', 0):.4f}\n"
            f"- **Precision**: {eval_result.get('precision_macro', 0):.4f}\n"
            f"- **Recall**: {eval_result.get('recall_macro', 0):.4f}\n"
        )
    body_parts.append(f"\n모델: `{BASE_MODEL}` + LoRA\n")

    # 업로드할 파일 수집
    files = {}
    lora_path = os.path.join(BEST_MODEL_DIR, "lora")
    if os.path.isdir(lora_path):
        files["lora-adapter"] = lora_path

    eval_dir = os.path.join(BEST_MODEL_DIR, "evaluation")
    if os.path.isdir(eval_dir):
        files["evaluation"] = eval_dir

    results_json = os.path.join(OUTPUT_BASE, "hp_search_results.json")
    if os.path.isfile(results_json):
        files["hp_search_results.json"] = results_json

    if os.path.exists(DB_PATH):
        files["hp_search_results.db"] = DB_PATH

    if not files:
        logger.warning("업로드할 파일이 없습니다.")
        return

    return github_create_release(
        tag=tag,
        name=f"HP 탐색 {_kst_now()} KST",
        body="".join(body_parts),
        files=files,
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. 데이터 로딩
# ═══════════════════════════════════════════════════════════════════════


def load_dataset_from_jsonl(split, tight_prob=0.5, fraction=1.0):
    return core_load_dataset_from_jsonl(
        data_dir=DATA_DIR,
        split=split,
        tight_prob=tight_prob,
        fraction=fraction,
        random_seed=RANDOM_SEED,
        preload_cache_dir=PRELOAD_CACHE_DIR,
        max_image_dim=MAX_IMAGE_DIM,
        lazy_dataset=os.environ.get("HP_LAZY_DATASET") == "1",
        logger=logger,
        system_msg=SYSTEM_MSG,
        prompts=PROMPTS,
    )


def estimate_proxy_max_steps(
    batch_size: int,
    grad_accum: int,
    finetune_vision: bool,
    target_min: float = PROXY_TARGET_MIN,
) -> int:
    """프록시 트라이얼의 실제 시간이 target_min 근처가 되도록 max_steps를 고른다.

    2요소 모델(마이크로배치 오버헤드와 샘플 연산량 분리)을 사용한다.
    bs=1,ga=8은 스텝당 8개 마이크로배치, bs=4,ga=2는 2개만 실행하므로
    eff_batch가 같아도 시간은 달라진다.

        step_sec = (OVERHEAD × grad_accum
                    + PER_SAMPLE × eff_batch) × vision_mult

    기준: bs=4, ga=4, vision=False에서 관측한 19초/step.
        예측: 1.5×4 + 1.0×16 = 22초
        (약 15% 과대추정으로 보수적이며, 목표시간 초과보다 미달 쪽으로 유도).
    """
    eff_batch = batch_size * grad_accum
    base_step_sec = PROXY_STEP_OVERHEAD * grad_accum + PROXY_STEP_PER_SAMPLE * eff_batch
    step_sec = base_step_sec * (PROXY_VISION_MULT if finetune_vision else 1.0)
    target_sec = target_min * 60
    raw_steps = int(target_sec / step_sec)
    return max(PROXY_STEPS_FLOOR, min(PROXY_STEPS_CEILING, raw_steps))


def cleanup_trainer(trainer: object) -> None:
    if hasattr(trainer, "data_collator"):
        setattr(trainer, "data_collator", None)

    optimizer = getattr(trainer, "optimizer", None)
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["params"] = []
    if hasattr(trainer, "optimizer"):
        setattr(trainer, "optimizer", None)

    if hasattr(trainer, "lr_scheduler"):
        setattr(trainer, "lr_scheduler", None)
    if hasattr(trainer, "model"):
        setattr(trainer, "model", None)
    if hasattr(trainer, "callback_handler"):
        setattr(trainer, "callback_handler", None)


# ═══════════════════════════════════════════════════════════════════════
# 5. GPU 메모리 관리
# ═══════════════════════════════════════════════════════════════════════


def clear_gpu_memory():
    core_clear_gpu_memory(logger=logger)


def load_model_with_retry(base_model, max_retries=2):
    return core_load_model_with_retry(
        base_model=base_model,
        max_retries=max_retries,
        logger=logger,
    )


# ═══════════════════════════════════════════════════════════════════════
# 6. 평가
# ═══════════════════════════════════════════════════════════════════════


def evaluate_model(
    model, tokenizer, val_dataset, max_samples=200, save_dir=None, trial_num=None
):
    return core_evaluate_model(
        model=model,
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        max_samples=max_samples,
        save_dir=save_dir,
        trial_num=trial_num,
        logger=logger,
        system_msg=SYSTEM_MSG,
    )


# ═══════════════════════════════════════════════════════════════════════
# 7. 목적 함수
# ═══════════════════════════════════════════════════════════════════════


def objective(trial: optuna.Trial, args) -> float:
    from transformers import TrainerCallback
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    trial_start = time.time()
    trial_dir = os.path.join(OUTPUT_BASE, f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    # finally 블록에서 안전하게 정리할 수 있도록 None으로 초기화
    model, tokenizer, trainer = None, None, None
    wandb_run = None
    train_dataset, val_dataset = None, None
    _wandb_exit_code = 0

    try:
        # ─── 하이퍼파라미터 샘플링 ────────────────────────────────────

        batch_size = trial.suggest_categorical(
            "batch_size", SEARCH_SPACE["per_device_train_batch_size"]
        )
        grad_accum = trial.suggest_categorical(
            "grad_accum", SEARCH_SPACE["gradient_accumulation_steps"]
        )
        lr = trial.suggest_float(
            "learning_rate", *SEARCH_SPACE["learning_rate"], log=True
        )

        num_epochs = trial.suggest_categorical(
            "num_epochs", SEARCH_SPACE["num_train_epochs"]
        )
        if args.proxy:
            num_epochs = PROXY_EPOCHS
        elif args.quick:
            num_epochs = QUICK_EPOCHS

        warmup = trial.suggest_categorical("warmup_steps", SEARCH_SPACE["warmup_steps"])
        wd = trial.suggest_float("weight_decay", *SEARCH_SPACE["weight_decay"])
        scheduler = trial.suggest_categorical(
            "lr_scheduler", SEARCH_SPACE["lr_scheduler_type"]
        )
        max_seq = trial.suggest_categorical(
            "max_seq_length", SEARCH_SPACE["max_seq_length"]
        )

        lora_r = trial.suggest_categorical("lora_r", SEARCH_SPACE["lora_r"])
        alpha_ratio = trial.suggest_categorical(
            "lora_alpha_ratio", SEARCH_SPACE["lora_alpha_ratio"]
        )
        lora_alpha = int(lora_r * alpha_ratio)

        ft_vision = trial.suggest_categorical(
            "finetune_vision", SEARCH_SPACE["finetune_vision_layers"]
        )
        use_rslora = trial.suggest_categorical("use_rslora", SEARCH_SPACE["use_rslora"])

        tight_prob = trial.suggest_float(
            "crop_tight_prob", *SEARCH_SPACE["crop_tight_prob"]
        )

        effective_batch = batch_size * grad_accum
        all_params = trial.params.copy()
        all_params["lora_alpha"] = lora_alpha
        all_params["effective_batch_size"] = effective_batch

        logger.info(
            f"\n{'=' * 60}\n"
            f"트라이얼 {trial.number} — 파라미터:\n"
            f"  batch={batch_size}, grad_accum={grad_accum} (유효={effective_batch})\n"
            f"  lr={lr:.2e}, epochs={num_epochs}, warmup={warmup}\n"
            f"  wd={wd:.4f}, scheduler={scheduler}, rslora={use_rslora}\n"
            f"  lora_r={lora_r}, lora_alpha={lora_alpha}, vision={ft_vision}\n"
            f"  crop_tight_prob={tight_prob:.2f}\n"
            f"  max_seq_length={max_seq}\n"
            f"{'=' * 60}"
        )

        # ─── W&B 초기화 ───────────────────────────────────────────────
        wandb_run = wandb_init_trial(trial.number, all_params)

        # ─── 데이터 로드 ───────────────────────────────────────────────

        if args.proxy:
            data_fraction = PROXY_DATA_FRACTION
        elif args.quick:
            data_fraction = QUICK_DATA_FRACTION
        else:
            data_fraction = 1.0

        # 시스템 RAM 기준으로 데이터 비율 상한 적용
        jsonl_path = os.path.join(DATA_DIR, "train.jsonl")
        n_train_total = _get_line_count(jsonl_path)
        ram_max_frac = get_max_data_fraction(n_train_total)
        if data_fraction > ram_max_frac:
            logger.info(
                f"RAM 제한: data_fraction {data_fraction:.0%} → "
                f"{ram_max_frac:.0%} ({n_train_total}개 샘플 기준)"
            )
            data_fraction = ram_max_frac

        # subset 자체는 RANDOM_SEED 기준으로 고정되고,
        # 트라이얼마다 crop/prompt/shuffle만 달라지도록 시드를 분리한다.
        random.seed(RANDOM_SEED + trial.number)

        train_dataset = load_dataset_from_jsonl(
            "train",
            tight_prob=tight_prob,
            fraction=data_fraction,
        )
        # val은 고정 시드 사용: 모든 트라이얼이 동일한 프롬프트/순서를 보도록 유지
        random.seed(RANDOM_SEED)
        val_dataset = load_dataset_from_jsonl(
            "validation",
            tight_prob=0.5,
            fraction=1.0,
        )
        # proxy 모드에서는 val 상한 적용: 약 150샘플의 trainer loss로도 순위화 가능.
        # 전체 val 평가는 트라이얼당 약 10~15분 추가 소요됨.
        if args.proxy and len(val_dataset) > PROXY_VAL_CAP:
            val_dataset = val_dataset[:PROXY_VAL_CAP]
        logger.info(
            f"데이터 로딩 완료 — train: {len(train_dataset)}, val: {len(val_dataset)}"
        )

        # ─── 목표 시간 기반 proxy max_steps 계산 ─────────────────────
        proxy_max_steps = estimate_proxy_max_steps(
            batch_size=batch_size,
            grad_accum=grad_accum,
            finetune_vision=ft_vision,
        )
        if args.proxy:
            logger.info(
                f"프록시 스텝 예측: eff_batch={batch_size * grad_accum}, "
                f"vision={ft_vision} -> max_steps={proxy_max_steps} "
                f"(목표 {PROXY_TARGET_MIN}분)"
            )

        # ─── 총 학습 스텝을 넘지 않도록 warmup 보정 ───────────────────
        if args.proxy:
            total_steps = proxy_max_steps
        else:
            steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * grad_accum))
            total_steps = steps_per_epoch * num_epochs
        if warmup > total_steps // 2:
            old_warmup = warmup
            warmup = max(0, total_steps // 4)
            logger.info(
                f"warmup {old_warmup} -> {warmup} (총 스텝 {total_steps}의 "
                f"절반 초과하여 자동 조정)"
            )

        # ─── VRAM 예산 점검 (A6000 = 48GB) ────────────────────────────
        # gradient checkpointing 기준 대략 추정:
        #   - 베이스 모델 bf16 가중치: 약 18GB
        #   - LoRA 파라미터용 adamw_8bit 옵티마이저: 약 5GB
        #   - grad ckpt 활성화 시 activation: seq=1024에서 배치 원소당 약 2.5GB
        #   - 비전 레이어 파인튜닝 오버헤드: 약 3GB
        # lora_r는 activation 크기를 키우지 않음(어댑터 가중치만 증가, <1GB)
        if torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            vision_overhead = 3.0 if ft_vision else 0.0
            activation_est = batch_size * (max_seq / 1024) * 2.5
            estimated_gb = 18.0 + 5.0 + vision_overhead + activation_est
            if estimated_gb > total_vram_gb * 0.92:
                logger.warning(
                    f"트라이얼 {trial.number} VRAM 예산 초과 예측: "
                    f"{estimated_gb:.1f}GB > {total_vram_gb:.1f}GB — 가지치기"
                )
                discord_trial_pruned(
                    trial.number,
                    f"VRAM 예산 초과 예측: {estimated_gb:.1f}GB "
                    f"(batch={batch_size}, seq={max_seq}, r={lora_r}, "
                    f"vision={ft_vision})",
                )
                raise optuna.TrialPruned()

        # ─── 모델 + LoRA 로드 ─────────────────────────────────────────

        clear_gpu_memory()

        model, tokenizer = load_model_with_retry(BASE_MODEL)

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=ft_vision,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=use_rslora,
            loftq_config=None,
        )

        FastVisionModel.for_training(model)
        logger.info("모델 로딩 및 LoRA 적용 완료")

        # ─── 가지치기 콜백 ─────────────────────────────────────────────

        class OptunaCallback(TrainerCallback):
            def __init__(self, _trial):
                self._trial = _trial

            def on_evaluate(self, args, state, control, metrics=None, **kw):
                if metrics and "eval_loss" in metrics:
                    step = state.global_step
                    self._trial.report(metrics["eval_loss"], step)
                    # 마지막 eval에서는 가지치기하지 않음.
                    # 학습이 거의 끝난 시점이라 중단보다 완료가 낫다.
                    is_last_step = step >= state.max_steps
                    if not is_last_step and self._trial.should_prune():
                        logger.info(
                            f"트라이얼 {self._trial.number} 스텝 {step}에서 가지치기"
                        )
                        raise optuna.TrialPruned()

        # ─── 학습 ─────────────────────────────────────────────────────

        report_to = "wandb" if wandb_is_available() else "none"
        dataloader_num_workers = recommend_dataloader_num_workers(
            configured=DATALOADER_NUM_WORKERS,
            logger=logger,
        )
        dataloader_persistent_workers = dataloader_num_workers > 0

        sft_config_base_kwargs: dict[str, Any] = {
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "warmup_steps": warmup,
            "num_train_epochs": num_epochs,
            # Proxy: 고정 연산 예산 사용.
            # NopPruner에서는 epoch-end eval이 불필요하므로 생략
            # (아래 explicit trainer.evaluate()와 중복).
            "max_steps": proxy_max_steps if args.proxy else -1,
            "learning_rate": lr,
            "bf16": True,
            "logging_steps": 20,
            "save_strategy": "no",
            "eval_strategy": "no" if args.proxy else "epoch",
            "optim": "adamw_8bit",
            "weight_decay": wd,
            "lr_scheduler_type": scheduler,
            "max_grad_norm": 1.0,
            "bf16_full_eval": True,
            "seed": RANDOM_SEED,
            "output_dir": trial_dir,
            "report_to": report_to,
            "remove_unused_columns": False,
            "dataset_text_field": "",
            "dataset_kwargs": {"skip_prepare_dataset": True},
        }
        sft_config_kwargs = build_sft_config_kwargs(
            base_kwargs=sft_config_base_kwargs,
            sft_config_cls=SFTConfig,
            seq_len=max_seq,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=dataloader_persistent_workers,
            logger=logger,
        )
        logger.info(
            "트라이얼 %d DataLoader 설정: workers=%d, persistent=%s",
            trial.number,
            dataloader_num_workers,
            dataloader_persistent_workers,
        )
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "tokenizer": tokenizer,
            "data_collator": UnslothVisionDataCollator(model, tokenizer),
            "train_dataset": train_dataset,
            "eval_dataset": val_dataset,
            "callbacks": [OptunaCallback(trial)],
            "args": SFTConfig(**sft_config_kwargs),
        }
        trainer = SFTTrainer(**trainer_kwargs)

        train_result = trainer.train()
        train_loss = train_result.training_loss

        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss", float("inf"))

        logger.info(
            f"트라이얼 {trial.number} — train_loss: {train_loss:.4f}, eval_loss: {eval_loss:.4f}"
        )

        # NaN 조기 감지: 이 경우 정확도 평가를 진행할 의미가 없음
        if math.isnan(train_loss) or math.isnan(eval_loss):
            raise ValueError(f"NaN detected: train={train_loss}, eval={eval_loss}")

        # ─── 전체 평가 ────────────────────────────────────────────────
        # proxy 모드에서는, 현재 최고 eval_loss와 같거나 더 좋을 때만
        # 비용이 큰 추론 기반 평가를 수행한다.
        # 순위화는 trainer eval_loss로 충분하고, 정확도 평가는 확인용이다.
        current_best = eval_loss
        try:
            current_best = trial.study.best_value
            is_promising = eval_loss <= current_best * 1.05
        except ValueError:
            is_promising = True  # 최초 완료 트라이얼

        if args.proxy and not is_promising:
            # 비용이 큰 추론 평가는 건너뛰고 eval_loss만 사용
            eval_result = {
                "accuracy": 0,
                "f1_macro": 0,
                "f1_weighted": 0,
                "precision_macro": 0,
                "recall_macro": 0,
                "total": 0,
                "per_class": {},
                "confusion_matrix": [],
                "confusion_matrix_path": None,
            }
            logger.info(
                f"트라이얼 {trial.number} — eval_loss {eval_loss:.4f} > "
                f"best*1.05 ({current_best * 1.05:.4f}), 정밀 평가 건너뜀"
            )
        else:
            FastVisionModel.for_inference(model)
            eval_samples = 50 if args.proxy else 200 if args.quick else 400
            eval_save_dir = os.path.join(OUTPUT_BASE, "evaluations")

            eval_result = evaluate_model(
                model,
                tokenizer,
                val_dataset,
                max_samples=eval_samples,
                save_dir=eval_save_dir,
                trial_num=trial.number,
            )

        accuracy = eval_result["accuracy"]
        f1_macro = eval_result["f1_macro"]
        f1_weighted = eval_result["f1_weighted"]

        logger.info(
            f"트라이얼 {trial.number} 평가 결과:\n"
            f"  정확도:       {accuracy:.4f}\n"
            f"  F1 (macro):   {f1_macro:.4f}\n"
            f"  F1 (weighted):{f1_weighted:.4f}\n"
            f"  Precision:    {eval_result['precision_macro']:.4f}\n"
            f"  Recall:       {eval_result['recall_macro']:.4f}\n"
            f"  평가 샘플:    {eval_result['total']}개"
        )

        # ─── 메타데이터 저장 ──────────────────────────────────────────

        duration_min = (time.time() - trial_start) / 60
        trial.set_user_attr("train_loss", train_loss)
        trial.set_user_attr("eval_loss", eval_loss)
        trial.set_user_attr("effective_batch_size", effective_batch)
        trial.set_user_attr("lora_alpha", lora_alpha)
        trial.set_user_attr("duration_min", duration_min)
        trial.set_user_attr("accuracy", accuracy)
        trial.set_user_attr("f1_macro", f1_macro)
        trial.set_user_attr("f1_weighted", f1_weighted)
        trial.set_user_attr("precision_macro", eval_result["precision_macro"])
        trial.set_user_attr("recall_macro", eval_result["recall_macro"])
        trial.set_user_attr("per_class", eval_result["per_class"])

        # ─── W&B 최종 지표 기록 ───────────────────────────────────────

        if wandb_run:
            import wandb

            wb_metrics = {
                "final/train_loss": train_loss,
                "final/eval_loss": eval_loss,
                "final/duration_min": duration_min,
                "final/accuracy": accuracy,
                "final/f1_macro": f1_macro,
                "final/f1_weighted": f1_weighted,
                "final/precision_macro": eval_result["precision_macro"],
                "final/recall_macro": eval_result["recall_macro"],
            }
            wandb.log(wb_metrics)
            # 혼동 행렬 이미지 업로드
            cm_path = eval_result.get("confusion_matrix_path")
            if cm_path and os.path.exists(cm_path):
                wandb.log({"confusion_matrix": wandb.Image(cm_path)})

        # ─── Discord 알림 ─────────────────────────────────────────────

        try:
            best_value = trial.study.best_value
            is_best = eval_loss <= best_value
        except ValueError:
            is_best = True  # 최초 완료 트라이얼

        completed_count = sum(
            1 for t in trial.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        discord_trial_complete(
            trial.number,
            eval_loss,
            trial.params,
            duration_min,
            is_best,
            completed_count,
            args.n_trials,
            eval_metrics=eval_result,
        )

        # ─── 새 최고 성능이면 GitHub DB 백업 ─────────────────────────
        if is_best:
            github_upload_db(
                f"trial {trial.number}: eval_loss={eval_loss:.6f}, acc={accuracy:.4f}"
            )

        # ─── 목적 함수 반환 ───────────────────────────────────────────

        # NaN/Inf 보호: Optuna는 NaN을 받으면 실패할 수 있음
        if math.isnan(eval_loss) or math.isinf(eval_loss):
            logger.warning(f"트라이얼 {trial.number} — eval_loss가 NaN/Inf, 가지치기")
            discord_trial_pruned(
                trial.number, f"eval_loss={eval_loss} (NaN/Inf) — 학습 발산"
            )
            raise optuna.TrialPruned()

        if OBJECTIVE_METRIC == "eval_loss":
            return eval_loss
        elif OBJECTIVE_METRIC == "accuracy":
            return -(accuracy or 0.0)
        elif OBJECTIVE_METRIC == "combined":
            return -(accuracy or 0.0) + 0.1 * eval_loss
        return eval_loss

    except torch.cuda.OutOfMemoryError:
        _wandb_exit_code = 1
        logger.warning(f"트라이얼 {trial.number} OOM 발생")
        discord_trial_pruned(
            trial.number,
            f"GPU 메모리 부족 (batch={trial.params.get('batch_size', '?')}, "
            f"r={trial.params.get('lora_r', '?')})",
        )
        raise optuna.TrialPruned()

    except optuna.TrialPruned:
        _wandb_exit_code = 1
        raise

    except Exception as e:
        _wandb_exit_code = 1
        logger.error(f"트라이얼 {trial.number} 실패: {e}", exc_info=True)
        discord_trial_error(trial.number, str(e)[:500])
        raise TrialExecutionError(f"trial {trial.number} failed") from e

    finally:
        # 모든 코드 경로에서 정리 보장: 순환 참조를 강하게 끊어
        # 트라이얼 간 GPU 메모리 누수를 방지
        if trainer is not None:
            cleanup_trainer(trainer)
        if model is not None:
            try:
                model.cpu()  # Move off GPU before deleting
            except Exception:
                pass  # Model may be in broken state after CUDA error
        del trainer, model, tokenizer, train_dataset, val_dataset
        wandb_finish(wandb_run, exit_code=_wandb_exit_code)
        clear_gpu_memory()
        shutil.rmtree(trial_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
# 8. 결과 분석
# ═══════════════════════════════════════════════════════════════════════


def analyze_study(study: optuna.Study):
    print("\n" + "=" * 70)
    print("  하이퍼파라미터 검색 결과")
    print("=" * 70)

    best = study.best_trial
    print(f"\n  최적 트라이얼: #{best.number}")
    print(f"  최적 값: {best.value:.6f}")
    print(f"  소요 시간: {best.user_attrs.get('duration_min', 0):.1f}분")
    print("\n  최적 파라미터:")
    for k, v in sorted(best.params.items()):
        print(f"    {k:30s} = {v}")

    lora_r = best.params["lora_r"]
    alpha_ratio = best.params["lora_alpha_ratio"]
    print(f"    {'lora_alpha (계산됨)':30s} = {int(lora_r * alpha_ratio)}")
    bs = best.params["batch_size"]
    ga = best.params["grad_accum"]
    print(f"    {'effective_batch_size':30s} = {bs * ga}")

    if "accuracy" in best.user_attrs:
        print(f"\n  최적 정확도: {best.user_attrs['accuracy']:.4f}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_with_value: list[tuple[float, optuna.trial.FrozenTrial]] = []
    for trial in completed:
        if trial.value is not None:
            completed_with_value.append((float(trial.value), trial))
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print("\n  트라이얼 요약:")
    print(f"    완료: {len(completed)}")
    print(f"    가지치기: {len(pruned)}")
    print(f"    실패: {len(failed)}")

    if completed_with_value:
        values = [value for value, _ in completed_with_value]
        print("\n  목적 함수 분포:")
        print(f"    최고: {min(values):.6f}")
        print(f"    최악: {max(values):.6f}")
        print(f"    중앙값: {sorted(values)[len(values) // 2]:.6f}")

    print("\n  상위 5개 트라이얼:")
    print(
        f"  {'#':>4} {'값':>10} {'학습률':>10} {'R':>4} {'배치':>4} {'에폭':>6} {'비전':>5}"
    )
    print(f"  {'-' * 50}")
    for trial_value, t in sorted(completed_with_value, key=lambda item: item[0])[:5]:
        p = t.params
        print(
            f"  {t.number:4d} {trial_value:10.6f} "
            f"{p['learning_rate']:10.2e} {p['lora_r']:4d} "
            f"{p['batch_size'] * p['grad_accum']:4d} "
            f"{p['num_epochs']:6d} {'O' if p['finetune_vision'] else 'X':>5}"
        )

    if len(completed) >= 5:
        try:
            importance = optuna.importance.get_param_importances(study)
            print("\n  파라미터 중요도:")
            for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
                bar = "#" * int(imp * 40)
                print(f"    {param:30s} {imp:.4f} {bar}")
        except Exception:
            print("\n  (파라미터 중요도 분석에 더 많은 트라이얼 필요)")

    # JSON 저장
    results = {
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "total_trials": len(study.trials),
        "completed": len(completed),
        "pruned": len(pruned),
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
                "user_attrs": t.user_attrs,
            }
            for t in completed
        ],
    }
    results_path = os.path.join(OUTPUT_BASE, "hp_search_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  결과 저장됨: {results_path}")

    lora_alpha_val = int(best.params["lora_r"] * best.params["lora_alpha_ratio"])
    print(f"""
  최적 파라미터 요약:
  ─────────────────────────────────────────────
  per_device_train_batch_size = {best.params["batch_size"]}
  gradient_accumulation_steps = {best.params["grad_accum"]}
  learning_rate = {best.params["learning_rate"]:.2e}
  num_train_epochs = {best.params["num_epochs"]}
  warmup_steps = {best.params["warmup_steps"]}
  weight_decay = {best.params["weight_decay"]:.4f}
  lr_scheduler_type = "{best.params["lr_scheduler"]}"
  max_seq_length = {best.params["max_seq_length"]}
  lora_r = {best.params["lora_r"]}
  lora_alpha = {lora_alpha_val}
  use_rslora = {best.params.get("use_rslora", False)}
  finetune_vision_layers = {best.params["finetune_vision"]}
  crop_tight_prob = {best.params["crop_tight_prob"]:.2f}
    """)
    print("=" * 70)
    return results


# ═══════════════════════════════════════════════════════════════════════
# 9. 최적 모델 재학습
# ═══════════════════════════════════════════════════════════════════════


def retrain_best(study: optuna.Study):
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    best = study.best_trial
    p = best.params

    logger.info("최적 파라미터로 재학습 시작...")
    discord_send("🔄 **최적 파라미터로 전체 데이터 재학습을 시작합니다...**")

    model, tokenizer, trainer = None, None, None
    train_dataset, val_dataset = None, None
    wandb_run = None

    try:
        random.seed(RANDOM_SEED)

        # 사용 가능한 RAM 기준으로 데이터 비율 상한 적용
        jsonl_path = os.path.join(DATA_DIR, "train.jsonl")
        n_train_total = _get_line_count(jsonl_path)
        retrain_frac = get_max_data_fraction(n_train_total)
        if retrain_frac < 1.0:
            logger.info(
                f"RAM 제한: retrain data_fraction 100% → "
                f"{retrain_frac:.0%} ({n_train_total}개 샘플 기준)"
            )

        train_dataset = load_dataset_from_jsonl(
            "train",
            tight_prob=p["crop_tight_prob"],
            fraction=retrain_frac,
        )
        val_dataset = load_dataset_from_jsonl("validation")

        clear_gpu_memory()

        model, tokenizer = load_model_with_retry(BASE_MODEL)

        lora_alpha = int(p["lora_r"] * p["lora_alpha_ratio"])
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=p["finetune_vision"],
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=p["lora_r"],
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=p.get("use_rslora", False),
        )
        FastVisionModel.for_training(model)

        wandb_available = wandb_is_available()
        if wandb_available:
            import wandb

            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY or None,
                name="retrain-best",
                group=STUDY_NAME,
                config=p,
                reinit=True,
                tags=["retrain", "best-model"],
            )

        report_to = "wandb" if wandb_available else "none"
        dataloader_num_workers = recommend_dataloader_num_workers(
            configured=DATALOADER_NUM_WORKERS,
            logger=logger,
        )
        dataloader_persistent_workers = dataloader_num_workers > 0

        retrain_sft_config_base_kwargs: dict[str, Any] = {
            "per_device_train_batch_size": p["batch_size"],
            "gradient_accumulation_steps": p["grad_accum"],
            "warmup_steps": p["warmup_steps"],
            "num_train_epochs": p["num_epochs"],
            "learning_rate": p["learning_rate"],
            "bf16": True,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "eval_strategy": "epoch",
            "optim": "adamw_8bit",
            "weight_decay": p["weight_decay"],
            "lr_scheduler_type": p["lr_scheduler"],
            "max_grad_norm": 1.0,
            "bf16_full_eval": True,
            "seed": RANDOM_SEED,
            "output_dir": BEST_MODEL_DIR,
            "report_to": report_to,
            "remove_unused_columns": False,
            "dataset_text_field": "",
            "dataset_kwargs": {"skip_prepare_dataset": True},
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
        }
        retrain_sft_config_kwargs = build_sft_config_kwargs(
            base_kwargs=retrain_sft_config_base_kwargs,
            sft_config_cls=SFTConfig,
            seq_len=p["max_seq_length"],
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=dataloader_persistent_workers,
            logger=logger,
        )
        logger.info(
            "재학습 DataLoader 설정: workers=%d, persistent=%s",
            dataloader_num_workers,
            dataloader_persistent_workers,
        )
        retrain_trainer_kwargs: dict[str, Any] = {
            "model": model,
            "tokenizer": tokenizer,
            "data_collator": UnslothVisionDataCollator(model, tokenizer),
            "train_dataset": train_dataset,
            "eval_dataset": val_dataset,
            "args": SFTConfig(**retrain_sft_config_kwargs),
        }
        trainer = SFTTrainer(**retrain_trainer_kwargs)

        trainer.train()

        lora_path = os.path.join(BEST_MODEL_DIR, "lora")
        model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)

        FastVisionModel.for_inference(model)
        acc_result = evaluate_model(
            model,
            tokenizer,
            val_dataset,
            max_samples=500,
            save_dir=os.path.join(BEST_MODEL_DIR, "evaluation"),
            trial_num="final",
        )
        logger.info(
            f"최종 평가 결과:\n"
            f"  정확도:         {acc_result['accuracy']:.4f}\n"
            f"  F1 (macro):     {acc_result['f1_macro']:.4f}\n"
            f"  F1 (weighted):  {acc_result['f1_weighted']:.4f}\n"
            f"  Precision:      {acc_result['precision_macro']:.4f}\n"
            f"  Recall:         {acc_result['recall_macro']:.4f}"
        )

        if wandb_run:
            import wandb

            wandb.log(
                {
                    "retrain/accuracy": acc_result["accuracy"],
                    "retrain/f1_macro": acc_result["f1_macro"],
                    "retrain/f1_weighted": acc_result["f1_weighted"],
                    "retrain/precision_macro": acc_result["precision_macro"],
                    "retrain/recall_macro": acc_result["recall_macro"],
                }
            )
            cm_path = acc_result.get("confusion_matrix_path")
            if cm_path and os.path.exists(cm_path):
                wandb.log({"retrain/confusion_matrix": wandb.Image(cm_path)})

        # 클래스별 지표를 JSON으로 저장
        metrics_path = os.path.join(BEST_MODEL_DIR, "evaluation", "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": acc_result["accuracy"],
                    "f1_macro": acc_result["f1_macro"],
                    "f1_weighted": acc_result["f1_weighted"],
                    "precision_macro": acc_result["precision_macro"],
                    "recall_macro": acc_result["recall_macro"],
                    "per_class": acc_result["per_class"],
                    "confusion_matrix": acc_result["confusion_matrix"],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        discord_retrain_complete(acc_result["accuracy"], lora_path)
        print(f"\n  최적 모델 저장됨: {lora_path}")
        print(f"  최종 정확도:     {acc_result['accuracy']:.2%}")
        print(f"  최종 F1 (macro): {acc_result['f1_macro']:.4f}")
        print(f"  혼동 행렬:       {acc_result.get('confusion_matrix_path', 'N/A')}")
        return acc_result

    finally:
        if trainer is not None:
            cleanup_trainer(trainer)
        if model is not None:
            try:
                model.cpu()
            except Exception:
                pass
        del trainer, model, tokenizer, train_dataset, val_dataset
        wandb_finish(wandb_run)
        clear_gpu_memory()


# ═══════════════════════════════════════════════════════════════════════
# 10. 메인
# ═══════════════════════════════════════════════════════════════════════


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="config.json")
    pre_args, _ = pre_parser.parse_known_args()
    initialize_from_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="해충 탐지 LoRA 파인튜닝 하이퍼파라미터 검색"
    )
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument(
        "--quick", action="store_true", help="빠른 모드: 1 에폭, 20%% 데이터"
    )
    parser.add_argument(
        "--proxy", action="store_true", help="프록시 모드: 1 에폭, 5%% 데이터"
    )
    parser.add_argument("--analyze", action="store_true", help="기존 결과 분석만 수행")
    parser.add_argument("--retrain", action="store_true", help="최적 파라미터로 재학습")
    parser.add_argument(
        "--metric", choices=["eval_loss", "accuracy", "combined"], default=None
    )
    args = parser.parse_args()
    initialize_from_config(args.config)

    if args.proxy:
        args.quick = False

    global OBJECTIVE_METRIC
    if args.metric:
        OBJECTIVE_METRIC = args.metric

    # ─── 로깅 설정(import 시 크래시 방지를 위해 지연 실행) ───────────
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )

    _install_fatal_signal_handlers()

    # ─── 분석 전용 모드 ───────────────────────────────────────────────

    if args.analyze:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_DB)
        analyze_study(study)
        wandb_log_best_summary(study)
        return

    # ─── Optuna가 DB를 열기 전에 오래된 RUNNING 트라이얼 정리 ─────────
    # 세션이 비정상 종료되면 RUNNING 상태가 영구히 남을 수 있음.
    # 이는 TPE 샘플링을 오염시키고 트라이얼 예산을 낭비하므로,
    # create_study() 전에 반드시 정리한다.
    import sqlite3

    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.execute(
                "UPDATE trials SET state = 'FAIL' WHERE state = 'RUNNING'"
            )
            if cur.rowcount > 0:
                conn.commit()
                logger.info(f"오래된 RUNNING 트라이얼 -> FAIL 전환: {cur.rowcount}건")
            conn.close()
        except Exception as e:
            logger.warning(f"오래된 트라이얼 정리 실패: {e}")

    # ─── 스터디 생성/로드 ────────────────────────────────────────────

    # proxy 모드(1 epoch)는 eval 지점이 1개라 가지치기 효율이 낮음.
    # 모든 트라이얼이 끝까지 수행되도록 NopPruner 사용.
    if args.proxy:
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=5,
            reduction_factor=3,
        )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_DB,
        direction="minimize",
        sampler=TPESampler(
            seed=RANDOM_SEED,
            n_startup_trials=3,
            multivariate=True,
        ),
        pruner=pruner,
        load_if_exists=True,
    )

    # ─── 스터디 상태 요약 로깅 ───────────────────────────────────────
    states = {}
    for t in study.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    if states:
        logger.info(f"스터디 상태: {states}")

    mode = "PROXY (5%)" if args.proxy else "QUICK (20%)" if args.quick else "FULL"

    logger.info(
        f"\n{'=' * 60}\n"
        f"  하이퍼파라미터 검색\n"
        f"  스터디: {STUDY_NAME}\n"
        f"  목적 함수: {OBJECTIVE_METRIC}\n"
        f"  트라이얼: {args.n_trials}\n"
        f"  모드: {mode}\n"
        f"  W&B: {'활성화' if wandb_is_available() else '비활성화'}\n"
        f"  Discord 알림: {'활성화' if DISCORD_WEBHOOKS else '비활성화'}\n"
        f"{'=' * 60}"
    )

    discord_search_started(args.n_trials, mode)

    # ─── 최적화 실행 ─────────────────────────────────────────────────

    search_start = time.time()

    try:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
            catch=(TrialExecutionError,),
        )
    except Exception:
        discord_error(traceback.format_exc())
        raise

    search_hours = (time.time() - search_start) / 3600

    # ─── 결과 분석 ───────────────────────────────────────────────────

    analyze_study(study)
    wandb_log_best_summary(study)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    discord_phase_complete(
        phase=f"{mode} 검색",
        best_value=study.best_trial.value,
        best_params=study.best_trial.params,
        completed=len(completed),
        pruned=len(pruned),
        total_time_hr=search_hours,
    )

    # 단계 완료 후 DB 업로드(모든 트라이얼 데이터 최종 백업)
    github_upload_db(
        f"{mode} 완료: {len(completed)}개 트라이얼, best={study.best_trial.value:.6f}"
    )

    # ─── 재학습 ───────────────────────────────────────────────────────

    if args.retrain:
        try:
            retrain_best(study)
        except Exception:
            discord_error(f"재학습 실패: {traceback.format_exc()}")
            raise

    # ─── 시각화 ───────────────────────────────────────────────────────

    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_slice,
        )

        for name, fn in [
            ("optimization_history", plot_optimization_history),
            ("param_importance", plot_param_importances),
            ("parallel_coordinate", plot_parallel_coordinate),
            ("param_slices", plot_slice),
        ]:
            fig = fn(study)
            fig.write_html(os.path.join(OUTPUT_BASE, f"{name}.html"))
        logger.info(f"시각화 저장됨: {OUTPUT_BASE}/")
    except ImportError:
        logger.info("pip install plotly kaleido 로 시각화 설치")

    logger.info("검색 완료!")


if __name__ == "__main__":
    main()
