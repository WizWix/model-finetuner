from __future__ import annotations

import importlib.util
import logging
import os


def wandb_is_available(*, logger: logging.Logger | None = None) -> bool:
    """환경/의존성 기준으로 W&B를 안전하게 활성화할 수 있는지 판단한다."""
    log = logger if logger is not None else logging.getLogger(__name__)

    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return False
    if not os.environ.get("WANDB_API_KEY"):
        return False
    if importlib.util.find_spec("wandb") is None:
        log.warning("WANDB_API_KEY는 설정됐지만 wandb 패키지가 설치되어 있지 않습니다.")
        return False
    return True
