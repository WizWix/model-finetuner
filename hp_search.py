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
import gc
import json
import logging
import os
import pickle
import random
import math
import shutil
import signal
import threading
import time
import traceback
from datetime import datetime, timezone, timedelta
from collections.abc import Iterable
from typing import Any, cast

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

import torch
from PIL import Image
from common.app_config import (
    apply_auth_environment,
    get_discord_webhooks,
    load_app_config,
)
from common.discord_utils import send_discord

# Ampere GPU에서 TF32를 사용해 남은 FP32 연산(옵티마이저, 손실 계산)을 가속
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# 대형 해충 이미지에서 DecompressionBombWarning을 억제하고
# 최대 이미지 크기를 제한해 OOM을 방지
Image.MAX_IMAGE_PIXELS = None
MAX_IMAGE_DIM = 768  # 이 값보다 큰 이미지는 리사이즈(모델 입력은 512px)

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

# RAM 기준 데이터 비율 상한
# 1024x1024 RGB 이미지는 메모리에서 약 3MB.
# 모델 로딩(임시 CPU 복사), 검증셋, 파이썬/OS 오버헤드를 위해 약 15GB 예약.
_RAM_RESERVE_GB = 15.0
_BYTES_PER_IMAGE = 3.0 * 1024**2  # 이미지 1장당 약 3MB


def initialize_from_config(config_path: str) -> dict:
    """config.json을 로드하고 런타임 전역 설정값에 반영한다."""
    global DATA_DIR, VOLUME_DIR, OUTPUT_BASE, BEST_MODEL_DIR, DB_PATH, STORAGE_DB
    global LOG_FILE, PRELOAD_CACHE_DIR, STUDY_NAME, BASE_MODEL, RANDOM_SEED
    global N_TRIALS_DEFAULT, WANDB_PROJECT, WANDB_ENTITY, GITHUB_REPO
    global GITHUB_TOKEN, DISCORD_WEBHOOKS, GITHUB_DB_BACKUP_PATH

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

    GITHUB_REPO = github.get("repo", GITHUB_REPO)
    GITHUB_DB_BACKUP_PATH = github.get("backup_db_path_in_repo", GITHUB_DB_BACKUP_PATH)
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("github_pat", "")

    DISCORD_WEBHOOKS = get_discord_webhooks(cfg)
    return cfg


def get_max_data_fraction(n_train_samples: int) -> float:
    """사용 가능한 RAM에 맞는 최대 데이터 비율을 계산한다."""
    try:
        total_bytes: int | None = None

        # POSIX 환경: sysconf가 존재하면 우선 사용
        sysconf = getattr(os, "sysconf", None)
        if callable(sysconf):
            page_size = int(sysconf("SC_PAGE_SIZE"))  # pyright: ignore[reportArgumentType]
            phys_pages = int(sysconf("SC_PHYS_PAGES"))  # pyright: ignore[reportArgumentType]
            total_bytes = page_size * phys_pages

        # Windows/기타 환경 fallback
        if total_bytes is None:
            import psutil

            total_bytes = int(psutil.virtual_memory().total)

        total_ram_gb = total_bytes / 1024**3
    except (ImportError, AttributeError, ValueError, OSError):
        return 1.0  # 감지 실패 시 제한 없음으로 간주

    usable_gb = total_ram_gb - _RAM_RESERVE_GB
    if usable_gb <= 0:
        return 0.1

    max_images = int(usable_gb * 1024**3 / _BYTES_PER_IMAGE)
    max_frac = min(1.0, max_images / max(1, n_train_samples))
    return max_frac


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

_line_count_cache = {}

# 트라이얼 간 이미지 디코딩 캐시. 키는 (split, fraction)
# 580샘플 기준 디코딩+LANCZOS 리사이즈에 트라이얼당 약 7분이 걸려
# 기존에는 매번 반복됐고, 캐시로 (split, fraction) 조합당 1회로 상쇄
# 각 항목: {"label": str, "full": PIL, "tight": PIL|None}
# tight crop은 bbox 기반으로 미리 계산해두어, 트라이얼마다 파일 재오픈 없이
# full/tight만 확률적으로 선택하면 되도록 구성
_PRELOADED_SAMPLES: dict = {}


def _get_line_count(path: str) -> int:
    """파일 줄 수를 계산하고 트라이얼 간 캐시한다."""
    if path not in _line_count_cache:
        with open(path, "r") as f:
            _line_count_cache[path] = sum(1 for _ in f)
    return _line_count_cache[path]


SYSTEM_MSG = (
    "당신은 작물 해충 식별 전문가입니다. "
    "사진을 보고 해충의 이름만 한국어로 답하세요. "
    '해충이 없으면 "정상"이라고만 답하세요. '
    "부가 설명 없이 이름만 출력하세요."
)

PROMPTS = [
    "이 사진에 있는 해충의 이름을 알려주세요.",
    "이 벌레는 무엇인가요?",
    "사진 속 해충을 식별해주세요.",
    "이 작물에 있는 해충의 종류가 무엇인가요?",
    "이 사진에서 어떤 해충이 보이나요?",
]

OBJECTIVE_METRIC = "eval_loss"

# 볼륨이 없을 때 import 단계에서 크래시하지 않도록 로거는 main()에서 설정
logger = logging.getLogger(__name__)


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
            "title": "🔍 하이퍼파라미터 검색 시작",
            "color": 0x3498DB,
            "fields": [
                {"name": "모드", "value": mode, "inline": True},
                {"name": "트라이얼 수", "value": str(n_trials), "inline": True},
                {"name": "모델", "value": BASE_MODEL, "inline": True},
                {"name": "목적 함수", "value": OBJECTIVE_METRIC, "inline": True},
            ],
            "footer": {"text": f"시작: {_kst_now()} KST"},
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
    return bool(os.environ.get("WANDB_API_KEY"))


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
    import requests as _req
    import tarfile
    import tempfile

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


def crop_to_bbox(img, bbox, padding_ratio=0.0):
    xtl, ytl = bbox["xtl"], bbox["ytl"]
    xbr, ybr = bbox["xbr"], bbox["ybr"]
    bw, bh = xbr - xtl, ybr - ytl
    pad_x, pad_y = int(bw * padding_ratio), int(bh * padding_ratio)
    x1 = max(0, xtl - pad_x)
    y1 = max(0, ytl - pad_y)
    x2 = min(img.width, xbr + pad_x)
    y2 = min(img.height, ybr + pad_y)
    # 잘못된 bbox로 0크기 crop이 생기는 경우 방지
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def cap_image_size(img):
    """가로/세로 중 하나라도 MAX_IMAGE_DIM을 넘으면 리사이즈한다.
    종횡비를 유지하며, RAM 해제를 위해 원본 이미지를 닫는다."""
    w, h = img.size
    if max(w, h) <= MAX_IMAGE_DIM:
        return img
    scale = MAX_IMAGE_DIM / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    lanczos = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    resized = img.resize((new_w, new_h), lanczos)
    img.close()  # 원본 대형 이미지 메모리 해제
    return resized


def find_label_json(split, class_name, img_filename):
    json_path = os.path.join(DATA_DIR, split, class_name, img_filename + ".json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return None  # 비어있거나 손상된 라벨 파일이면 bbox 생략
    for obj in data.get("annotations", {}).get("object", []):
        if obj["grow"] == 33 and obj.get("points"):
            return obj["points"][0]
    return None


def make_conversation(image, label):
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": random.choice(PROMPTS)},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": label}]},
        ]
    }


def _preload_samples(split: str, fraction: float) -> list:
    """트라이얼 간 재사용을 위해 고정 샘플 부분집합을 디코딩/캐시한다.

    (split, fraction) 조합마다 최초 1회만 실행하고 이후에는 즉시 캐시를 반환한다.
    부분집합 선택은 트라이얼 번호와 무관하게 고정 시드(RANDOM_SEED)를 사용하므로
    모든 트라이얼이 동일한 이미지 집합을 보게 되어 비교 가능성이 높아진다.
    트라이얼별 변동은 부분집합이 아니라 조립 단계의 crop/prompt/shuffle에서 발생한다.
    """
    key = (split, round(fraction, 4))
    if key in _PRELOADED_SAMPLES:
        return _PRELOADED_SAMPLES[key]

    # 디스크 캐시: 프로세스 재시작 후에도 유지.
    # 키를 (split, fraction, MAX_IMAGE_DIM)으로 두어 리사이즈 한도 변경 시 새 캐시를 생성.
    os.makedirs(PRELOAD_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(
        PRELOAD_CACHE_DIR,
        f"{split}_f{round(fraction, 4)}_d{MAX_IMAGE_DIM}.pkl",
    )
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                samples = pickle.load(f)
            _PRELOADED_SAMPLES[key] = samples
            logger.info(
                f"이미지 캐시 디스크에서 로드 — {cache_file}, {len(samples)}개 샘플"
            )
            return samples
        except Exception as e:
            logger.warning(f"디스크 캐시 로드 실패 ({cache_file}): {e} — 재생성")

    jsonl_path = os.path.join(DATA_DIR, f"{split}.jsonl")
    total_lines = _get_line_count(jsonl_path)

    # 부분집합 선택 전용 고정 시드 RNG(전역 random 상태에는 영향 없음)
    _rng = random.Random(RANDOM_SEED)
    if fraction < 1.0:
        keep = set(_rng.sample(range(total_lines), int(total_lines * fraction)))
    else:
        keep = None

    expected = len(keep) if keep is not None else total_lines
    logger.info(
        f"이미지 캐시 적재 시작 — split={split}, fraction={fraction}, "
        f"~{expected}개 디코딩 예정 (RAM 사용량 주의)"
    )

    samples = []
    _preload_t0 = time.time()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if keep is not None and i not in keep:
                continue

            record = json.loads(line)
            messages = record["messages"]
            label = messages[-1]["content"][0]["text"]

            img_rel_path = None
            for msg in messages:
                for content in msg["content"]:
                    if content["type"] == "image" and "image" in content:
                        img_rel_path = content["image"]
                        break

            if img_rel_path is None:
                continue

            img_rel_path = img_rel_path.replace("\\", "/")
            parts = img_rel_path.split("/")
            if len(parts) < 3:
                logger.warning(f"예상치 못한 이미지 경로: {img_rel_path}, 건너뜀")
                continue

            img_path = os.path.join(DATA_DIR, img_rel_path)
            if not os.path.exists(img_path):
                continue

            class_name, img_filename = parts[1], parts[2]

            # 원본 전체 프레임 이미지(1회 디코딩 + 1회 크기 제한)
            full_img = cap_image_size(Image.open(img_path).convert("RGB"))

            # bbox 좌표는 원본 기준이므로, 축소 전 원본에서 tight crop을 미리 계산
            # (픽셀 좌표 정확도 보장)
            tight_img = None
            if label != "정상":
                bbox = find_label_json(split, class_name, img_filename)
                if bbox:
                    orig = Image.open(img_path).convert("RGB")
                    tight_img = cap_image_size(
                        crop_to_bbox(orig, bbox, padding_ratio=0.0)
                    )
                    orig.close()

            samples.append(
                {
                    "label": label,
                    "full": full_img,
                    "tight": tight_img,
                }
            )

            if len(samples) % 1000 == 0:
                elapsed = time.time() - _preload_t0
                rate = len(samples) / max(elapsed, 1e-3)
                logger.info(
                    f"  preload 진행: {len(samples)}/{expected} "
                    f"({elapsed:.0f}s, {rate:.1f} img/s)"
                )

    _PRELOADED_SAMPLES[key] = samples
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"이미지 캐시 디스크 저장 — {cache_file}")
    except Exception as e:
        logger.warning(f"디스크 캐시 저장 실패 ({cache_file}): {e}")
    logger.info(
        f"이미지 캐시 적재 완료 — split={split}, fraction={fraction}, "
        f"{len(samples)}개 샘플 (이후 트라이얼에서 재사용)"
    )
    return samples


class LazyImageDataset:
    """__getitem__ 시점 지연 디코딩으로 1회성 실행의 ~20GB 프리로드 비용을 피한다.

    HP_LAZY_DATASET=1 로 활성화한다. eager 경로와 샘플링 의미론은 동일
    (동일한 fraction 부분집합, 동일한 tight/full 확률, 동일한 라벨)하며,
    차이는 이미지를 사전 디코딩하지 않고 접근 시점에 디코딩한다는 점이다.
    메모리 사용량은 O(전체 디코딩 이미지)에서 O(워커 수 × 배치)로 줄어든다.
    """

    def __init__(self, split, tight_prob, fraction):
        self.tight_prob = tight_prob
        self.samples = self._collect_metadata(split, fraction)
        logger.info(
            f"LazyImageDataset 생성 — split={split}, "
            f"{len(self.samples)}개 샘플 (지연 디코딩, 프리로드 생략)"
        )

    @staticmethod
    def _collect_metadata(split, fraction):
        jsonl_path = os.path.join(DATA_DIR, f"{split}.jsonl")
        total_lines = _get_line_count(jsonl_path)
        _rng = random.Random(RANDOM_SEED)
        if fraction < 1.0:
            keep = set(_rng.sample(range(total_lines), int(total_lines * fraction)))
        else:
            keep = None

        out = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if keep is not None and i not in keep:
                    continue
                record = json.loads(line)
                messages = record["messages"]
                label = messages[-1]["content"][0]["text"]

                img_rel = None
                for msg in messages:
                    for content in msg["content"]:
                        if content["type"] == "image" and "image" in content:
                            img_rel = content["image"]
                            break
                if img_rel is None:
                    continue

                img_rel = img_rel.replace("\\", "/")
                parts = img_rel.split("/")
                if len(parts) < 3:
                    continue

                img_path = os.path.join(DATA_DIR, img_rel)
                if not os.path.exists(img_path):
                    continue

                class_name, img_filename = parts[1], parts[2]
                bbox = None
                if label != "정상":
                    bbox = find_label_json(split, class_name, img_filename)

                out.append(
                    {
                        "label": label,
                        "img_path": img_path,
                        "bbox": bbox,
                    }
                )
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self.samples)))]
        meta = self.samples[idx]
        if meta["bbox"] is not None and random.random() < self.tight_prob:
            orig = Image.open(meta["img_path"]).convert("RGB")
            img = cap_image_size(crop_to_bbox(orig, meta["bbox"], padding_ratio=0.0))
            orig.close()
        else:
            img = cap_image_size(Image.open(meta["img_path"]).convert("RGB"))
        return make_conversation(img, meta["label"])


def load_dataset_from_jsonl(split, tight_prob=0.5, fraction=1.0):
    """캐시된 프리로드 샘플에서 트라이얼별 데이터셋 리스트를 구성한다.

    bbox가 있는 해충 이미지의 경우:
      - Tight crop(bbox만): 확률 = tight_prob
      - 원본 이미지(비-crop): 확률 = 1 - tight_prob

    "정상" 이미지와 bbox가 없는 이미지는 항상 원본을 사용한다.
    트라이얼별 변동은 호출 전에 설정한 전역 random.seed()로 제어한다.

    HP_LAZY_DATASET=1 이면 eagerly-preloaded 리스트 대신
    LazyImageDataset(__getitem__ 시 디코딩)을 반환한다.
    1회성 재학습에는 lazy가 유리하고, 다수 트라이얼 HP 탐색에는 eager가 유리하다.
    """
    if os.environ.get("HP_LAZY_DATASET") == "1":
        return LazyImageDataset(split, tight_prob, fraction)

    samples = _preload_samples(split, fraction)

    dataset = []
    for s in samples:
        if s["tight"] is None:
            img = s["full"]
        elif random.random() < tight_prob:
            img = s["tight"]
        else:
            img = s["full"]
        dataset.append(make_conversation(img, s["label"]))

    random.shuffle(dataset)
    return dataset


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


# ═══════════════════════════════════════════════════════════════════════
# 5. GPU 메모리 관리
# ═══════════════════════════════════════════════════════════════════════


def clear_gpu_memory():
    # 순환 참조 해제를 위해 gc를 여러 차례 수행
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"GPU 메모리 — allocated: {allocated:.1f}GB, "
            f"reserved: {reserved:.1f}GB | peak 초기화됨"
        )


def _has_meta_tensors(model) -> bool:
    """모델 파라미터 중 meta 디바이스에 남아 있는 항목이 있는지 확인한다."""
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            logger.warning(f"Meta tensor 발견: {name}")
            return True
    return False


def load_model_with_retry(base_model, max_retries=2):
    """meta 텐서 검증을 포함해 모델 로딩을 수행하고 실패 시 재시도한다.

    트라이얼이 누적되면 GPU 메모리 단편화로 가중치가 완전히 물질화되지 않아
    일부 텐서가 meta 디바이스에 남을 수 있다. 이 래퍼는 해당 상황을 감지하고
    공격적으로 정리한 뒤 다시 로드한다.
    """
    from unsloth import FastVisionModel

    last_err = None
    model = None
    tokenizer = None
    for attempt in range(1, max_retries + 1):
        try:
            model, tokenizer = FastVisionModel.from_pretrained(
                base_model,
                load_in_4bit=False,
                use_gradient_checkpointing="unsloth",
            )
            if _has_meta_tensors(model):
                raise RuntimeError(
                    "모델은 로드됐지만 meta tensor가 포함되어 있음: "
                    "가중치가 완전히 물질화되지 않았습니다"
                )
            return model, tokenizer
        except Exception as e:
            last_err = e
            logger.warning(f"모델 로딩 시도 {attempt}/{max_retries} 실패: {e}")
            # 부분적으로 로드된 객체를 최대한 강하게 해제
            try:
                del model
            except UnboundLocalError:
                pass
            try:
                del tokenizer
            except UnboundLocalError:
                pass
            model = None
            tokenizer = None
            for _ in range(5):
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
            # CUDA가 메모리를 충분히 회수할 짧은 대기
            time.sleep(2)

    raise RuntimeError(
        f"모델 로딩 {max_retries}회 시도 후 실패: {last_err}"
    ) from last_err


# ═══════════════════════════════════════════════════════════════════════
# 6. 평가
# ═══════════════════════════════════════════════════════════════════════


def evaluate_model(
    model, tokenizer, val_dataset, max_samples=200, save_dir=None, trial_num=None
):
    """추론을 수행하고 분류 지표 전체를 계산한다.

    반환값에는 정확도/정밀도/재현율/F1(매크로·가중), 클래스별 지표,
    혼동 행렬이 포함되며, CM 플롯은 PNG로 저장한다.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    samples = val_dataset[:max_samples]
    y_true, y_pred = [], []
    misclassifications = []

    for item in samples:
        messages = item["messages"]
        ground_truth = messages[-1]["content"][0]["text"]
        image = messages[1]["content"][0]["image"]

        infer_messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": messages[1]["content"][1]["text"]},
                ],
            },
        ]

        try:
            input_text = tokenizer.apply_chat_template(
                infer_messages, add_generation_prompt=True
            )
            inputs = tokenizer(
                image, input_text, add_special_tokens=False, return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    use_cache=True,
                )

            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            # GPU 메모리 누적 방지를 위해 CUDA 텐서를 즉시 해제
            del inputs, output_ids

            y_true.append(ground_truth)
            y_pred.append(generated)

            if generated != ground_truth:
                misclassifications.append(
                    {
                        "truth": ground_truth,
                        "predicted": generated,
                    }
                )

        except Exception as e:
            logger.warning(f"추론 오류: {e}")
            continue

    # 추론 루프에서 남은 CUDA 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(y_true) == 0:
        return {
            "accuracy": 0,
            "f1_macro": 0,
            "f1_weighted": 0,
            "precision_macro": 0,
            "recall_macro": 0,
            "total": 0,
        }

    # ─── 전체 클래스 라벨 집합 계산 ───────────────────────────────────
    all_labels = sorted(set(y_true + y_pred))

    # ─── 핵심 지표 계산 ───────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    # sklearn 타입 스텁 버전에 따라 average/zero_division 타입이 과도하게 좁게 잡힐 수 있다.
    zero_division = cast(Any, 0)
    avg_none = cast(Any, None)

    prec_macro = precision_score(
        y_true, y_pred, labels=all_labels, average="macro", zero_division=zero_division
    )
    rec_macro = recall_score(
        y_true, y_pred, labels=all_labels, average="macro", zero_division=zero_division
    )
    f1_macro = f1_score(
        y_true, y_pred, labels=all_labels, average="macro", zero_division=zero_division
    )
    prec_weighted = precision_score(
        y_true,
        y_pred,
        labels=all_labels,
        average="weighted",
        zero_division=zero_division,
    )
    rec_weighted = recall_score(
        y_true,
        y_pred,
        labels=all_labels,
        average="weighted",
        zero_division=zero_division,
    )
    f1_weighted = f1_score(
        y_true,
        y_pred,
        labels=all_labels,
        average="weighted",
        zero_division=zero_division,
    )

    # ─── 클래스별 지표 계산 ───────────────────────────────────────────
    prec_per = precision_score(
        y_true,
        y_pred,
        labels=all_labels,
        average=avg_none,
        zero_division=zero_division,
    )
    rec_per = recall_score(
        y_true,
        y_pred,
        labels=all_labels,
        average=avg_none,
        zero_division=zero_division,
    )
    f1_per = f1_score(
        y_true,
        y_pred,
        labels=all_labels,
        average=avg_none,
        zero_division=zero_division,
    )

    def _as_float_list(values: Any, expected_len: int) -> list[float]:
        # sklearn의 타입 스텁은 scalar/ndarray를 모두 허용해
        # 정적 분석기에서 인덱싱 경고가 날 수 있으므로 리스트로 정규화한다.
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            return [float(v) for v in values]
        return [float(values)] * expected_len

    prec_per_list = _as_float_list(prec_per, len(all_labels))
    rec_per_list = _as_float_list(rec_per, len(all_labels))
    f1_per_list = _as_float_list(f1_per, len(all_labels))

    per_class = {}
    for i, cls in enumerate(all_labels):
        per_class[cls] = {
            "precision": prec_per_list[i],
            "recall": rec_per_list[i],
            "f1": f1_per_list[i],
            "support": int(y_true.count(cls)),
        }

    # ─── 혼동 행렬 계산 ───────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_path = None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import font_manager

            # 한국어 표시 가능한 폰트를 우선 시도
            _korean_keywords = ["nanum", "malgun", "gothic", "gulim", "noto", "cjk"]

            def _find_korean_font():
                return [
                    f
                    for f in font_manager.findSystemFonts()
                    if any(k in f.lower() for k in _korean_keywords)
                ]

            korean_fonts = _find_korean_font()
            if not korean_fonts:
                # matplotlib 캐시 이후 폰트가 설치됐을 수 있으므로 재탐색
                font_manager.fontManager = font_manager.FontManager()
                korean_fonts = _find_korean_font()
            if korean_fonts:
                plt.rcParams["font.family"] = font_manager.FontProperties(
                    fname=korean_fonts[0]
                ).get_name()
            plt.rcParams["axes.unicode_minus"] = False

            # 가독성을 위해 라벨을 앞 4글자로 축약
            short = [lbl[:4] for lbl in all_labels]
            n = len(all_labels)

            fig_size = max(8, n * 0.6)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))

            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            fig.colorbar(im, ax=ax, shrink=0.8)

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(short, fontsize=7)
            ax.set_xlabel("예측 (Predicted)")
            ax.set_ylabel("실제 (Actual)")

            trial_label = f"Trial {trial_num}" if trial_num is not None else ""
            ax.set_title(
                f"Confusion Matrix {trial_label}\n"
                f"Acc={acc:.3f}  F1(macro)={f1_macro:.3f}"
            )

            # 셀 내부에 개수 표시
            thresh = cm.max() / 2
            for i in range(n):
                for j in range(n):
                    if cm[i, j] > 0:
                        ax.text(
                            j,
                            i,
                            str(cm[i, j]),
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="white" if cm[i, j] > thresh else "black",
                        )

            plt.tight_layout()
            cm_path = os.path.join(
                save_dir, f"confusion_matrix_trial_{trial_num or 'final'}.png"
            )
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)
            logger.info(f"혼동 행렬 저장됨: {cm_path}")

        except Exception as e:
            logger.warning(f"혼동 행렬 플롯 실패: {e}")

    # ─── 결과 딕셔너리 구성 ───────────────────────────────────────────
    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": cm_path,
        "total": len(y_true),
        "correct": int(acc * len(y_true)),
        "top_misclassifications": misclassifications[:20],
    }


# ═══════════════════════════════════════════════════════════════════════
# 7. 목적 함수
# ═══════════════════════════════════════════════════════════════════════


def objective(trial: optuna.Trial, args) -> float:
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl.trainer.sft_trainer import SFTTrainer
    from trl.trainer.sft_config import SFTConfig
    from transformers import TrainerCallback

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

        # ─── 데이터 로드(트라이얼별 시드로 독립 부분집합 구성) ───────

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

        # 트라이얼별 시드: 각 트라이얼이 서로 다른 부분집합/crop을 보도록 설정
        random.seed(RANDOM_SEED + trial.number)

        train_dataset = load_dataset_from_jsonl(
            "train",
            tight_prob=tight_prob,
            fraction=data_fraction,
        )
        # val은 고정 시드 사용: 모든 트라이얼이 동일한 프롬프트/순서를 보도록 유지
        random.seed(RANDOM_SEED)
        val_dataset = load_dataset_from_jsonl(
            "val",
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

        sft_config_kwargs: dict[str, Any] = {
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
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": True,
            "bf16_full_eval": True,
            "seed": RANDOM_SEED,
            "output_dir": trial_dir,
            "report_to": report_to,
            "remove_unused_columns": False,
            "dataset_text_field": "",
            "dataset_kwargs": {"skip_prepare_dataset": True},
            "max_seq_length": max_seq,
        }
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
        raise optuna.TrialPruned()

    finally:
        # 모든 코드 경로에서 정리 보장: 순환 참조를 강하게 끊어
        # 트라이얼 간 GPU 메모리 누수를 방지
        if trainer is not None:
            cast(Any, trainer).data_collator = None
            # optimizer -> param 참조를 끊음(주요 누수 원인)
            if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
                optimizer = cast(Any, trainer.optimizer)
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group["params"] = []
                trainer.optimizer = None
            if hasattr(trainer, "lr_scheduler"):
                trainer.lr_scheduler = None
            if hasattr(trainer, "model"):
                trainer.model = None
            cast(Any, trainer).callback_handler = None
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
    completed_with_value = [t for t in completed if t.value is not None]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print("\n  트라이얼 요약:")
    print(f"    완료: {len(completed)}")
    print(f"    가지치기: {len(pruned)}")
    print(f"    실패: {len(failed)}")

    if completed_with_value:
        values = [float(cast(float, t.value)) for t in completed_with_value]
        print("\n  목적 함수 분포:")
        print(f"    최고: {min(values):.6f}")
        print(f"    최악: {max(values):.6f}")
        print(f"    중앙값: {sorted(values)[len(values) // 2]:.6f}")

    print("\n  상위 5개 트라이얼:")
    print(
        f"  {'#':>4} {'값':>10} {'학습률':>10} {'R':>4} {'배치':>4} {'에폭':>6} {'비전':>5}"
    )
    print(f"  {'-' * 50}")
    for t in sorted(completed_with_value, key=lambda t: cast(float, t.value))[:5]:
        p = t.params
        trial_value = cast(float, t.value)
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
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl.trainer.sft_trainer import SFTTrainer
    from trl.trainer.sft_config import SFTConfig

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
        val_dataset = load_dataset_from_jsonl("val")

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

        if wandb_is_available():
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

        report_to = "wandb" if wandb_is_available() else "none"

        retrain_sft_config_kwargs: dict[str, Any] = {
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
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": True,
            "bf16_full_eval": True,
            "seed": RANDOM_SEED,
            "output_dir": BEST_MODEL_DIR,
            "report_to": report_to,
            "remove_unused_columns": False,
            "dataset_text_field": "",
            "dataset_kwargs": {"skip_prepare_dataset": True},
            "max_seq_length": p["max_seq_length"],
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
        }
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
            cast(Any, trainer).data_collator = None
            if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
                optimizer = cast(Any, trainer.optimizer)
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group["params"] = []
                trainer.optimizer = None
            if hasattr(trainer, "lr_scheduler"):
                trainer.lr_scheduler = None
            if hasattr(trainer, "model"):
                trainer.model = None
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
        "--proxy", action="store_true", help="프록시 모드: 1 에폭, 10%% 데이터"
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

    mode = "PROXY (10%)" if args.proxy else "QUICK (20%)" if args.quick else "FULL"

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
            plot_param_importances,
            plot_parallel_coordinate,
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
