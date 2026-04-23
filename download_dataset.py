#!/usr/bin/env python3
"""
레이트 리밋을 고려한 안정적인 HuggingFace 데이터셋 다운로더.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from common.app_config import apply_auth_environment, load_app_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF = 30
REQUIRED_FILES = ["train.jsonl", "validation.jsonl"]


def call_with_kwargs(func: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    return func(**kwargs)


def check_hf_transfer() -> None:
    enabled = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1"
    installed = importlib.util.find_spec("hf_transfer") is not None

    if installed and enabled:
        logger.info("hf_transfer 활성화됨")
    elif installed and not enabled:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        logger.info("hf_transfer 자동 활성화 완료")
    else:
        logger.info("hf_transfer 미설치 - 기본 다운로더 사용")


def verify_download(data_dir: str) -> bool:
    missing = [
        name
        for name in REQUIRED_FILES
        if not os.path.exists(os.path.join(data_dir, name))
    ]
    if missing:
        logger.error("필수 파일 누락: %s", missing)
        return False

    total_files = 0
    total_bytes = 0
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            path = os.path.join(root, name)
            total_files += 1
            total_bytes += os.path.getsize(path)

    total_gb = total_bytes / 1024**3
    logger.info("데이터셋 검증: 파일 %,d개, %.1fGB", total_files, total_gb)

    if total_gb < 15.0:
        logger.warning("데이터셋 크기가 예상보다 작음 (%.1fGB)", total_gb)
        return False

    for name in REQUIRED_FILES:
        path = os.path.join(data_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        logger.info("%s: 샘플 %,d개", name, n_lines)

    return True


def download_with_retry(repo_id: str, data_dir: str, token: str | None) -> bool:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    os.makedirs(data_dir, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "다운로드 시도 %d/%d: %s -> %s", attempt, MAX_RETRIES, repo_id, data_dir
            )
            params = inspect.signature(snapshot_download).parameters
            kwargs: dict[str, Any] = {"repo_id": repo_id}

            if "repo_type" in params:
                kwargs["repo_type"] = "dataset"
            if "local_dir" in params:
                kwargs["local_dir"] = data_dir
            if "force_download" in params:
                kwargs["force_download"] = False
            if "etag_timeout" in params:
                kwargs["etag_timeout"] = 30
            if "local_dir_use_symlinks" in params:
                kwargs["local_dir_use_symlinks"] = False
            if token:
                if "token" in params:
                    kwargs["token"] = token
                elif "use_auth_token" in params:
                    kwargs["use_auth_token"] = token

            call_with_kwargs(snapshot_download, kwargs)
            return True
        except HfHubHTTPError as exc:
            text = str(exc).lower()
            if "429" in text or "rate" in text:
                backoff = min(INITIAL_BACKOFF * (2 ** (attempt - 1)), 300)
                logger.warning("요청 제한 감지 - %d초 후 재시도", backoff)
                time.sleep(backoff)
            elif attempt < MAX_RETRIES:
                wait = attempt * 10
                logger.warning("HF 오류: %s (%d초 대기)", exc, wait)
                time.sleep(wait)
            else:
                raise
        except (ConnectionError, TimeoutError, OSError) as exc:
            if attempt < MAX_RETRIES:
                wait = attempt * 15
                logger.warning("네트워크 오류: %s (%d초 대기)", exc, wait)
                time.sleep(wait)
            else:
                raise
        except Exception as exc:
            text = str(exc).lower()
            if "429" in text or "rate" in text or "too many" in text:
                backoff = min(INITIAL_BACKOFF * (2 ** (attempt - 1)), 300)
                logger.warning("요청 제한 감지 - %d초 후 재시도", backoff)
                time.sleep(backoff)
            else:
                raise

    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    cfg = load_app_config(args.config)
    apply_auth_environment(cfg)

    data_dir = cfg["paths"]["data_dir"]
    repo_id = cfg["huggingface"]["dataset_repo"]
    token = os.environ.get("HF_TOKEN") or None
    sentinel = os.path.join(data_dir, ".download_complete")

    if args.verify:
        return 0 if verify_download(data_dir) else 1

    if os.path.exists(sentinel):
        logger.info("다운로드 완료 마커 발견 - 검증 후 스킵")
        if verify_download(data_dir):
            return 0
        os.remove(sentinel)
        logger.warning("검증 실패로 재다운로드 진행")

    check_hf_transfer()
    logger.info("데이터셋 저장소: %s", repo_id)
    logger.info("데이터 경로: %s", data_dir)

    start = time.time()
    ok = download_with_retry(repo_id=repo_id, data_dir=data_dir, token=token)
    if not ok:
        logger.error("다운로드 실패")
        return 1

    elapsed = (time.time() - start) / 60
    logger.info("다운로드 소요 시간: %.1f분", elapsed)

    if not verify_download(data_dir):
        logger.error("다운로드 후 검증 실패")
        return 1

    with open(sentinel, "w", encoding="utf-8") as f:
        f.write(f"downloaded at {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"elapsed: {elapsed:.1f} min\\n")
    logger.info("다운로드 + 검증 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
