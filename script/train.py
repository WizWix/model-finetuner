#!/usr/bin/env python3
"""고정 하이퍼파라미터 파인튜닝 진입점."""

from __future__ import annotations

# NOTE: Keep this before transformers/peft imports.
import unsloth  # noqa: F401  # isort: skip
import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import torch
from common.app_config import (
    apply_auth_environment,
    get_discord_webhooks,
    load_app_config,
)
from common.discord_utils import send_discord
from common.hf_upload import upload_finetune_output
from common.training_core import (
    build_sft_config_kwargs,
    clear_gpu_memory,
    evaluate_model,
    get_line_count,
    get_max_data_fraction,
    load_dataset_from_jsonl,
    load_model_with_retry,
    recommend_dataloader_num_workers,
)
from common.wandb_utils import wandb_is_available
from transformers import TrainerCallback

KST = timezone(timedelta(hours=9))
logger = logging.getLogger(__name__)


class DiscordTrainingMonitorCallback(TrainerCallback):
    def __init__(
        self,
        *,
        webhooks: list[str],
        heartbeat_steps: int,
        alert_cooldown_steps: int,
        loss_threshold: float,
        eval_loss_threshold: float,
        grad_norm_threshold: float,
        loss_jump_ratio: float,
        eval_loss_jump_ratio: float,
        auto_stop_enabled: bool,
        auto_stop_grad_norm_threshold: float,
        auto_stop_train_loss_threshold: float,
        auto_stop_train_loss_consecutive: int,
        auto_stop_eval_loss_threshold: float,
        auto_stop_eval_loss_best_ratio: float,
    ) -> None:
        self.webhooks = webhooks
        self.heartbeat_steps = max(0, int(heartbeat_steps))
        self.alert_cooldown_steps = max(1, int(alert_cooldown_steps))
        self.loss_threshold = float(loss_threshold)
        self.eval_loss_threshold = float(eval_loss_threshold)
        self.grad_norm_threshold = float(grad_norm_threshold)
        self.loss_jump_ratio = float(loss_jump_ratio)
        self.eval_loss_jump_ratio = float(eval_loss_jump_ratio)
        self.auto_stop_enabled = bool(auto_stop_enabled)
        self.auto_stop_grad_norm_threshold = float(auto_stop_grad_norm_threshold)
        self.auto_stop_train_loss_threshold = float(auto_stop_train_loss_threshold)
        self.auto_stop_train_loss_consecutive = max(
            1, int(auto_stop_train_loss_consecutive)
        )
        self.auto_stop_eval_loss_threshold = float(auto_stop_eval_loss_threshold)
        self.auto_stop_eval_loss_best_ratio = float(auto_stop_eval_loss_best_ratio)

        self.started_at: float | None = None
        self.total_steps: int | None = None
        self.last_heartbeat_step = 0
        self.last_alert_step_by_key: dict[str, int] = {}
        self.last_train_loss: float | None = None
        self.last_eval_loss: float | None = None
        self.best_eval_loss: float | None = None
        self.high_train_loss_streak = 0
        self.stop_triggered = False
        self.stop_reason: str | None = None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number

    @staticmethod
    def _looks_invalid_number(value: Any) -> bool:
        text = str(value).strip().lower()
        return text in {"nan", "inf", "+inf", "-inf"}

    def _send_embed(self, title: str, description: str, color: int) -> None:
        if not self.webhooks:
            return
        send_discord(
            self.webhooks,
            embed={
                "embeds": [
                    {
                        "author": {"name": "AI 모델 파인튜너"},
                        "title": title,
                        "description": description,
                        "color": color,
                    }
                ]
            },
        )

    def _format_progress(self, step: int) -> str:
        if self.total_steps is None or self.total_steps <= 0:
            return f"{step}/?"
        return f"{step}/{self.total_steps} ({step / self.total_steps * 100:.1f}%)"

    def _format_eta(self, step: int) -> str:
        if self.started_at is None or step <= 0 or self.total_steps is None:
            return "계산 불가"
        elapsed = time.time() - self.started_at
        sec_per_step = elapsed / step
        remain_steps = max(0, self.total_steps - step)
        eta_sec = int(remain_steps * sec_per_step)
        eta_h, eta_rem = divmod(eta_sec, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"

    def _should_alert(self, key: str, step: int) -> bool:
        last = self.last_alert_step_by_key.get(key, -(10**9))
        if step - last < self.alert_cooldown_steps:
            return False
        self.last_alert_step_by_key[key] = step
        return True

    def _trigger_stop(self, control, step: int, title: str, lines: list[str]):
        self.stop_triggered = True
        self.stop_reason = lines[0] if lines else title
        description = "\n".join(
            [f"- step: {self._format_progress(step)}"] + [f"- {line}" for line in lines]
        )
        self._send_embed(title, description, 15158332)
        control.should_training_stop = True
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.time()
        self.total_steps = int(getattr(state, "max_steps", 0) or 0)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.stop_triggered:
            control.should_training_stop = True
            return control
        if logs is None:
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        epoch = logs.get("epoch")
        loss = self._safe_float(logs.get("loss"))
        grad_norm = self._safe_float(logs.get("grad_norm"))
        lr = self._safe_float(logs.get("learning_rate"))
        raw_loss = logs.get("loss")
        raw_grad_norm = logs.get("grad_norm")

        if (
            self.heartbeat_steps > 0
            and step > 0
            and step - self.last_heartbeat_step >= self.heartbeat_steps
        ):
            self.last_heartbeat_step = step
            parts = [
                f"- 진행: {self._format_progress(step)}",
                f"- epoch: {epoch if epoch is not None else '?'}",
                f"- train_loss: {loss:.6f}" if loss is not None else "- train_loss: ?",
                f"- grad_norm: {grad_norm:.6f}"
                if grad_norm is not None
                else "- grad_norm: ?",
                f"- learning_rate: {lr:.3e}"
                if lr is not None
                else "- learning_rate: ?",
                f"- 남은 ETA: {self._format_eta(step)}",
            ]
            self._send_embed("📈 훈련 중간 상태", "\n".join(parts), 3447003)

        if loss is not None:
            if loss >= self.loss_threshold and self._should_alert("loss_abs", step):
                self._send_embed(
                    "⚠️ 이상 징후 감지 (train_loss)",
                    "\n".join(
                        [
                            f"- step: {self._format_progress(step)}",
                            f"- epoch: {epoch if epoch is not None else '?'}",
                            f"- train_loss: {loss:.6f}",
                            f"- 기준 임계치: {self.loss_threshold:.6f}",
                        ]
                    ),
                    15277667,
                )
            if (
                self.last_train_loss is not None
                and self.last_train_loss > 0
                and loss >= self.last_train_loss * self.loss_jump_ratio
                and self._should_alert("loss_jump", step)
            ):
                self._send_embed(
                    "⚠️ 이상 징후 감지 (train_loss 급등)",
                    "\n".join(
                        [
                            f"- step: {self._format_progress(step)}",
                            f"- 이전 loss: {self.last_train_loss:.6f}",
                            f"- 현재 loss: {loss:.6f}",
                            f"- 급등 배수: x{loss / self.last_train_loss:.2f}",
                        ]
                    ),
                    15277667,
                )
            self.last_train_loss = loss
        elif self._looks_invalid_number(raw_loss) and self._should_alert(
            "loss_nan", step
        ):
            self._send_embed(
                "⚠️ 이상 징후 감지 (train_loss NaN/Inf)",
                f"- step: {self._format_progress(step)}\n- train_loss: {raw_loss}",
                15277667,
            )
            if self.auto_stop_enabled:
                return self._trigger_stop(
                    control,
                    step,
                    "🛑 자동 중단 (NaN/Inf)",
                    [f"train_loss={raw_loss}"],
                )

        if (
            grad_norm is not None
            and grad_norm >= self.grad_norm_threshold
            and self._should_alert("grad_norm", step)
        ):
            self._send_embed(
                "⚠️ 이상 징후 감지 (grad_norm)",
                "\n".join(
                    [
                        f"- step: {self._format_progress(step)}",
                        f"- epoch: {epoch if epoch is not None else '?'}",
                        f"- grad_norm: {grad_norm:.6f}",
                        f"- 기준 임계치: {self.grad_norm_threshold:.6f}",
                    ]
                ),
                15277667,
            )
        elif self._looks_invalid_number(raw_grad_norm) and self._should_alert(
            "grad_nan", step
        ):
            self._send_embed(
                "⚠️ 이상 징후 감지 (grad_norm NaN/Inf)",
                f"- step: {self._format_progress(step)}\n- grad_norm: {raw_grad_norm}",
                15277667,
            )
            if self.auto_stop_enabled:
                return self._trigger_stop(
                    control,
                    step,
                    "🛑 자동 중단 (NaN/Inf)",
                    [f"grad_norm={raw_grad_norm}"],
                )

        if self.auto_stop_enabled:
            if (
                loss is not None
                and grad_norm is not None
                and grad_norm >= self.auto_stop_grad_norm_threshold
                and loss >= 0.5
            ):
                return self._trigger_stop(
                    control,
                    step,
                    "🛑 자동 중단 (grad_norm 급등)",
                    [
                        (
                            "grad_norm="
                            f"{grad_norm:.6f} >= {self.auto_stop_grad_norm_threshold:.6f}"
                        ),
                        f"train_loss={loss:.6f} (>= 0.5)",
                        f"epoch={epoch if epoch is not None else '?'}",
                    ],
                )

            if loss is not None and loss >= self.auto_stop_train_loss_threshold:
                self.high_train_loss_streak += 1
            elif loss is not None:
                self.high_train_loss_streak = 0
            if self.high_train_loss_streak >= self.auto_stop_train_loss_consecutive:
                return self._trigger_stop(
                    control,
                    step,
                    "🛑 자동 중단 (train_loss 연속 급등)",
                    [
                        (
                            "train_loss가 연속 "
                            f"{self.high_train_loss_streak}회 "
                            f"{self.auto_stop_train_loss_threshold:.6f} 이상"
                        ),
                        f"마지막 train_loss={loss:.6f}"
                        if loss is not None
                        else "마지막 train_loss=?",
                    ],
                )

        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.stop_triggered:
            control.should_training_stop = True
            return control
        if metrics is None:
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        eval_loss = self._safe_float(metrics.get("eval_loss"))
        raw_eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            if self._looks_invalid_number(raw_eval_loss) and self._should_alert(
                "eval_nan", step
            ):
                self._send_embed(
                    "⚠️ 이상 징후 감지 (eval_loss NaN/Inf)",
                    f"- step: {self._format_progress(step)}\n- eval_loss: {raw_eval_loss}",
                    15277667,
                )
                if self.auto_stop_enabled:
                    return self._trigger_stop(
                        control,
                        step,
                        "🛑 자동 중단 (NaN/Inf)",
                        [f"eval_loss={raw_eval_loss}"],
                    )
            return control

        if eval_loss >= self.eval_loss_threshold and self._should_alert(
            "eval_loss_abs", step
        ):
            self._send_embed(
                "⚠️ 이상 징후 감지 (eval_loss)",
                "\n".join(
                    [
                        f"- step: {self._format_progress(step)}",
                        f"- eval_loss: {eval_loss:.6f}",
                        f"- 기준 임계치: {self.eval_loss_threshold:.6f}",
                    ]
                ),
                15277667,
            )

        if (
            self.last_eval_loss is not None
            and self.last_eval_loss > 0
            and eval_loss >= self.last_eval_loss * self.eval_loss_jump_ratio
            and self._should_alert("eval_loss_jump", step)
        ):
            self._send_embed(
                "⚠️ 이상 징후 감지 (eval_loss 급등)",
                "\n".join(
                    [
                        f"- step: {self._format_progress(step)}",
                        f"- 이전 eval_loss: {self.last_eval_loss:.6f}",
                        f"- 현재 eval_loss: {eval_loss:.6f}",
                        f"- 급등 배수: x{eval_loss / self.last_eval_loss:.2f}",
                    ]
                ),
                15277667,
            )

        if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss

        if self.auto_stop_enabled:
            if eval_loss >= self.auto_stop_eval_loss_threshold:
                return self._trigger_stop(
                    control,
                    step,
                    "🛑 자동 중단 (eval_loss 임계치 초과)",
                    [
                        (
                            f"eval_loss={eval_loss:.6f} >= "
                            f"{self.auto_stop_eval_loss_threshold:.6f}"
                        )
                    ],
                )
            if (
                self.best_eval_loss is not None
                and self.best_eval_loss > 0
                and eval_loss
                >= self.best_eval_loss * self.auto_stop_eval_loss_best_ratio
            ):
                return self._trigger_stop(
                    control,
                    step,
                    "🛑 자동 중단 (eval_loss 급격 악화)",
                    [
                        f"best_eval_loss={self.best_eval_loss:.6f}",
                        f"current_eval_loss={eval_loss:.6f}",
                        f"악화 배수=x{eval_loss / self.best_eval_loss:.2f}",
                    ],
                )

        self.last_eval_loss = eval_loss
        return control


def get_free_space_gb(path: str) -> float:
    """경로가 위치한 디스크의 가용 공간(GB)을 반환한다. 확인 실패 시 -1."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except OSError:
        return -1.0


def build_readme(
    base_model: str,
    hp: dict[str, Any],
    epochs: int,
    elapsed_min: float,
    eval_result: dict | None,
) -> str:
    accuracy = eval_result.get("accuracy") if eval_result else None
    f1_macro = eval_result.get("f1_macro") if eval_result else None
    f1_weighted = eval_result.get("f1_weighted") if eval_result else None
    precision_macro = eval_result.get("precision_macro") if eval_result else None
    recall_macro = eval_result.get("recall_macro") if eval_result else None
    val_total = eval_result.get("total", "?") if eval_result else "?"

    def _fmt_ratio(v: Any) -> str:
        return f"{float(v):.4f}" if v is not None else "N/A"

    def _fmt_pct(v: Any) -> str:
        return f"{float(v) * 100:.2f}%" if v is not None else "N/A"

    return f"""---
base_model: {base_model}
library_name: peft
language:
  - ko
pipeline_tag: image-text-to-text
tags:
  - agriculture
  - image-classification
  - korean
  - lora
  - peft
  - pest-detection
  - qwen
  - qwen3
  - vision-language
datasets:
  - Himedia-AI-01/kor-pest-detection-webp
metrics:
  - accuracy
  - f1
  - precision
  - recall
model-index:
  - name: kor-pest-detector
    results:
      - task:
          type: image-text-to-text
          name: Korean Pest Classification
        dataset:
          name: Himedia-AI-01/kor-pest-detection-webp
          type: image-classification
        metrics:
          - type: accuracy
            value: {_fmt_ratio(accuracy)}
            name: Accuracy
          - type: f1
            value: {_fmt_ratio(f1_macro)}
            name: F1 (macro)
          - type: f1
            value: {_fmt_ratio(f1_weighted)}
            name: F1 (weighted)
          - type: precision
            value: {_fmt_ratio(precision_macro)}
            name: Precision (macro)
          - type: recall
            value: {_fmt_ratio(recall_macro)}
            name: Recall (macro)
---

# 해충 탐지 VLM - Qwen3.5-9B LoRA

[unsloth/Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B)를 파인튜닝한 비전-언어 PEFT 기반 LoRA 어댑터입니다.<br>
작물 사진에서 한국 농작물 해충 19종을 식별합니다.<br>
제공된 잎, 과실, 식물 전체 사진에 감지된 해충이 있을 시 해충의 한국어 이름을 출력하고, 해충이 감지되지 않으면 `정상`을 출력합니다.

* 19개 클래스 분류기: 해충 18종 + '정상' (해충 없음)
* 베이스 모델: `{base_model}` (비전-언어, 하이브리드 Linear + Self Attention)
* 어댑터 유형: LoRA (PEFT), Rank {hp["lora_r"]}, Alpha {hp["lora_alpha"]}
* 언어: 한국어

## ⚠ 배포 전에 반드시 읽어야 할 단 한 가지

**이 LoRA는 GGUF / llama.cpp / Ollama 경로로 배포할 수 없습니다.**

현재 학습 스크립트의 LoRA 타깃 모듈에는 Qwen3.5 `linear_attn` 계열
(`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`)이 포함될 수 있습니다.
이 경우 merge/GGUF 변환 경로에서 LoRA 델타가 안정적으로 보존되지 않아
출력이 붕괴될 수 있습니다.

권장 배포 경로:
- `unsloth.FastVisionModel.from_pretrained`
- `peft.PeftModel.from_pretrained`
- `FastVisionModel.for_inference(model)`
- 런타임 LoRA 유지 (병합 없음, GGUF 변환 없음)

## 학습 설정

<details>
<summary>열기/접기</summary>

| 하이퍼파라미터 | 값 |
|---|---|
| LoRA Rank | {hp["lora_r"]} |
| LoRA alpha | {hp["lora_alpha"]} |
| rsLoRA 사용 | {hp["use_rslora"]} |
| 비전 레이어 파인튜닝 | {hp["finetune_vision"]} |
| 학습률 | {hp["learning_rate"]:.6f} |
| 웜업 비율 | {hp["warmup_ratio"]} |
| 가중치 감쇠 | {hp["weight_decay"]:.6f} |
| LR 스케줄러 | {hp["lr_scheduler_type"]} |
| 옵티마이저 | {hp["optim"]} |
| 디바이스당 배치 | {hp["per_device_train_batch_size"]} |
| 그래디언트 누적 | {hp["gradient_accumulation_steps"]} |
| 유효 배치 | {hp["per_device_train_batch_size"] * hp["gradient_accumulation_steps"]} |
| 최대 시퀀스 길이 | {hp["max_seq_length"]} |
| 에폭 수 | {epochs} |
| Tight crop 확률 | {hp["crop_tight_prob"]:.4f} |
| 정밀도 형식 | bf16 |
| 그래디언트 체크포인팅 | True |
| 학습 시간 | {elapsed_min:.0f}분 |

</details>

## 평가 결과

| 지표 | 값 |
|---|---|
| 정확도 (Accuracy) | {_fmt_pct(accuracy)} |
| 정밀도 (Precision, Macro) | {_fmt_pct(precision_macro)} |
| 재현율 (Recall, Macro) | {_fmt_pct(recall_macro)} |
| F1 (Macro) | {_fmt_pct(f1_macro)} |
| F1 (Weighted) | {_fmt_pct(f1_weighted)} |
| 검증 샘플 수 | {val_total} |

혼동 행렬 및 클래스별 지표는 `evaluation/` 폴더를 확인하세요.

## 사용 예시

### 빠른 시작 (권장 추론 경로: Unsloth + Runtime PEFT)

```python
import torch
from unsloth import FastVisionModel
from peft import PeftModel
from PIL import Image

BASE = "{base_model}"
ADAPTER = "<this-repo-id>"

SYSTEM_MSG = (
    "당신은 작물 해충 식별 전문가입니다. "
    "사진을 보고 해충의 이름만 한국어로 답하세요. "
    '해충이 없으면 "정상"이라고만 답하세요. '
    "부가 설명 없이 이름만 출력하세요."
)

model, tokenizer = FastVisionModel.from_pretrained(BASE, load_in_4bit=False)
model = PeftModel.from_pretrained(model, ADAPTER)
FastVisionModel.for_inference(model)
model.eval()

image = Image.open("pest.jpg").convert("RGB")
messages = [
    {{"role": "system", "content": [{{"type": "text", "text": SYSTEM_MSG}}]}},
    {{"role": "user", "content": [
        {{"type": "image", "image": image}},
        {{"type": "text", "text": "이 사진에 있는 해충의 이름을 알려주세요."}},
    ]}},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(image, text, add_special_tokens=False, return_tensors="pt").to("cuda")

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=10,
        use_cache=True,
        stop_strings=["\\n"],
        tokenizer=tokenizer.tokenizer,
    )

prediction = tokenizer.decode(
    out[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
).strip()
print(prediction)
```

### 주의: 병합/GGUF 변환 경로 비권장

```python
# 아래 경로는 현재 타깃 모듈 구성에서는 비권장
# model = model.merge_and_unload()
# model.save_pretrained("./merged")
# 이후 GGUF 변환, llama.cpp/Ollama 배포
```

## 라이선스

베이스 모델 및 데이터셋의 라이선스를 따릅니다.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="고정 HP 파인튜닝")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=0,
        help="0이면 save_steps 값을 사용합니다.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="학습 로그/W&B 기록 주기(step 단위).",
    )
    parser.add_argument("--eval-samples", type=int, default=-1)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--hf-repo", default="")
    parser.add_argument(
        "--save-only-model",
        dest="save_only_model",
        action="store_true",
        default=None,
        help="중간 체크포인트에 옵티마이저/스케줄러 상태를 저장하지 않습니다.",
    )
    parser.add_argument(
        "--save-full-state",
        dest="save_only_model",
        action="store_false",
        help="중간 체크포인트에 옵티마이저/스케줄러 상태를 포함해 저장합니다.",
    )
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs는 1 이상이어야 합니다.")
    if args.save_steps <= 0:
        raise ValueError("--save-steps는 1 이상이어야 합니다.")
    if args.eval_steps < 0:
        raise ValueError("--eval-steps는 0 이상이어야 합니다.")
    if args.logging_steps <= 0:
        raise ValueError("--logging-steps는 1 이상이어야 합니다.")

    cfg = load_app_config(args.config)
    apply_auth_environment(cfg)

    paths = cfg.get("paths", {})
    runtime = cfg.get("runtime", {})
    hp = cfg.get("hyperparameters", {})
    base_model = runtime.get("base_model", "unsloth/Qwen3.5-9B")
    data_dir = paths.get("data_dir", "/workspace/data")
    preload_cache_dir = paths.get("preload_cache_dir", "/workspace/preload_cache")
    output_dir = args.output_dir or paths.get(
        "final_output_dir", "/workspace/best-pest-detector"
    )
    wandb_project = runtime.get("wandb_project", "pest-detection-hpsearch")
    wandb_entity = runtime.get("wandb_entity", "")
    save_only_model_default = bool(runtime.get("predefined_save_only_model", True))
    save_only_model = (
        save_only_model_default
        if args.save_only_model is None
        else args.save_only_model
    )
    min_free_space_gb = float(runtime.get("predefined_min_free_space_gb", 8))
    save_total_limit = int(runtime.get("predefined_save_total_limit", 2))
    dataloader_num_workers = recommend_dataloader_num_workers(
        configured=runtime.get("dataloader_num_workers"),
        logger=logger,
    )
    dataloader_persistent_workers = dataloader_num_workers > 0
    discord_heartbeat_steps = int(runtime.get("discord_heartbeat_steps", 100))
    discord_alert_cooldown_steps = int(runtime.get("discord_alert_cooldown_steps", 50))
    anomaly_loss_threshold = float(runtime.get("anomaly_loss_threshold", 5.0))
    anomaly_eval_loss_threshold = float(runtime.get("anomaly_eval_loss_threshold", 2.0))
    anomaly_grad_norm_threshold = float(
        runtime.get("anomaly_grad_norm_threshold", 1000.0)
    )
    anomaly_loss_jump_ratio = float(runtime.get("anomaly_loss_jump_ratio", 3.0))
    anomaly_eval_loss_jump_ratio = float(
        runtime.get("anomaly_eval_loss_jump_ratio", 2.0)
    )
    auto_stop_enabled = bool(runtime.get("auto_stop_enabled", True))
    auto_stop_grad_norm_threshold = float(
        runtime.get("auto_stop_grad_norm_threshold", 1e7)
    )
    auto_stop_train_loss_threshold = float(
        runtime.get("auto_stop_train_loss_threshold", 2.0)
    )
    auto_stop_train_loss_consecutive = int(
        runtime.get("auto_stop_train_loss_consecutive", 2)
    )
    auto_stop_eval_loss_threshold = float(
        runtime.get("auto_stop_eval_loss_threshold", 1.0)
    )
    auto_stop_eval_loss_best_ratio = float(
        runtime.get("auto_stop_eval_loss_best_ratio", 8.0)
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    webhooks = get_discord_webhooks(cfg)

    random.seed(hp["seed"])
    torch.manual_seed(hp["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hp["seed"])

    assert torch.cuda.is_available(), "CUDA GPU가 필요합니다"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    capability = torch.cuda.get_device_capability(0)
    logger.info(
        "GPU: %s | VRAM: %.0fGB | compute_capability: sm_%s%s",
        gpu_name,
        vram_gb,
        capability[0],
        capability[1],
    )

    os.makedirs(output_dir, exist_ok=True)
    free_space_gb = get_free_space_gb(output_dir)
    if 0 <= free_space_gb < min_free_space_gb:
        logger.warning(
            "출력 경로 디스크 여유 공간이 낮습니다: %.2fGB (< %.2fGB, output_dir=%s)",
            free_space_gb,
            min_free_space_gb,
            output_dir,
        )

    logger.info("데이터셋 로딩 중...")
    send_discord(
        webhooks,
        embed={
            "embeds": [
                {
                    "author": {"name": "AI 모델 파인튜너"},
                    "color": 3066993,
                    "title": "데이터셋 로딩 중...",
                }
            ]
        },
    )
    n_train_total = get_line_count(os.path.join(data_dir, "train.jsonl"))
    train_frac = get_max_data_fraction(n_train_total)
    train_dataset = load_dataset_from_jsonl(
        data_dir=data_dir,
        split="train",
        tight_prob=hp["crop_tight_prob"],
        fraction=train_frac,
        random_seed=hp["seed"],
        preload_cache_dir=preload_cache_dir,
        max_image_dim=768,
        lazy_dataset=True,
        logger=logger,
    )
    val_dataset = load_dataset_from_jsonl(
        data_dir=data_dir,
        split="validation",
        random_seed=hp["seed"],
        preload_cache_dir=preload_cache_dir,
        max_image_dim=768,
        lazy_dataset=True,
        logger=logger,
    )
    n_val = len(val_dataset)
    eval_samples = n_val if args.eval_samples < 0 else min(args.eval_samples, n_val)
    send_discord(
        webhooks,
        embed={
            "embeds": [
                {
                    "author": {"name": "AI 모델 파인튜너"},
                    "color": 3066993,
                    "title": "📦 데이터셋 로딩 완료",
                    "description": f"- 학습 샘플: {len(train_dataset)}\n- 검증 샘플: {n_val}\n- 평가 샘플: {eval_samples}",
                }
            ]
        },
    )

    clear_gpu_memory(logger=logger)
    logger.info("모델 로딩: %s", base_model)
    model, tokenizer = load_model_with_retry(base_model=base_model, logger=logger)

    from unsloth import FastVisionModel
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=hp["finetune_vision"],
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        bias="none",
        random_state=hp["data_seed"],
        use_rslora=hp["use_rslora"],
    )
    FastVisionModel.for_training(model)

    wandb_run = None
    report_to = "none"
    if not args.no_wandb and wandb_is_available(logger=logger):
        import wandb

        run_name = f"predefined-{datetime.now(KST).strftime('%m%d-%H%M')}"
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity or None,
            name=run_name,
            config={**hp, "num_train_epochs": args.epochs},
            tags=["predefined-finetune"],
            reinit=True,
        )
        report_to = "wandb"

    eval_steps = args.save_steps if args.eval_steps == 0 else args.eval_steps

    sft_config_base_kwargs: dict[str, Any] = {
        "per_device_train_batch_size": hp["per_device_train_batch_size"],
        "gradient_accumulation_steps": hp["gradient_accumulation_steps"],
        "per_device_eval_batch_size": 2,
        "num_train_epochs": args.epochs,
        "learning_rate": hp["learning_rate"],
        "weight_decay": hp["weight_decay"],
        "lr_scheduler_type": hp["lr_scheduler_type"],
        "warmup_ratio": hp["warmup_ratio"],
        "max_grad_norm": hp["max_grad_norm"],
        "adam_beta1": hp["adam_beta1"],
        "adam_beta2": hp["adam_beta2"],
        "adam_epsilon": hp["adam_epsilon"],
        "optim": hp["optim"],
        "bf16": True,
        "bf16_full_eval": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": save_total_limit,
        "eval_strategy": ("steps" if not args.no_eval else "no"),
        "eval_steps": eval_steps,
        "eval_accumulation_steps": 2,
        "load_best_model_at_end": (not args.no_eval),
        "metric_for_best_model": "eval_loss" if not args.no_eval else None,
        "greater_is_better": False,
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "seed": hp["seed"],
        "data_seed": hp["data_seed"],
        "output_dir": output_dir,
        "report_to": report_to,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }
    sft_config_kwargs = build_sft_config_kwargs(
        base_kwargs=sft_config_base_kwargs,
        sft_config_cls=SFTConfig,
        seq_len=hp["max_seq_length"],
        save_only_model=save_only_model,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=dataloader_persistent_workers,
        logger=logger,
    )
    sft_args = SFTConfig(**sft_config_kwargs)

    monitor_callback = DiscordTrainingMonitorCallback(
        webhooks=webhooks,
        heartbeat_steps=discord_heartbeat_steps,
        alert_cooldown_steps=discord_alert_cooldown_steps,
        loss_threshold=anomaly_loss_threshold,
        eval_loss_threshold=anomaly_eval_loss_threshold,
        grad_norm_threshold=anomaly_grad_norm_threshold,
        loss_jump_ratio=anomaly_loss_jump_ratio,
        eval_loss_jump_ratio=anomaly_eval_loss_jump_ratio,
        auto_stop_enabled=auto_stop_enabled,
        auto_stop_grad_norm_threshold=auto_stop_grad_norm_threshold,
        auto_stop_train_loss_threshold=auto_stop_train_loss_threshold,
        auto_stop_train_loss_consecutive=auto_stop_train_loss_consecutive,
        auto_stop_eval_loss_threshold=auto_stop_eval_loss_threshold,
        auto_stop_eval_loss_best_ratio=auto_stop_eval_loss_best_ratio,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "processing_class": tokenizer,
        "data_collator": UnslothVisionDataCollator(model, tokenizer),
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset if not args.no_eval else None,
        "args": sft_args,
        "callbacks": [monitor_callback],
    }
    trainer = SFTTrainer(**trainer_kwargs)
    send_discord(
        webhooks,
        embed={
            "embeds": [
                {
                    "author": {"name": "AI 모델 파인튜너"},
                    "color": 3066993,
                    "title": "🧱 모델/트레이너 준비 완료",
                    "description": f"- 모델: {base_model}\n- 출력 경로: {output_dir}\n- save/eval/logging steps: {args.save_steps}/{eval_steps}/{args.logging_steps}\n- save_total_limit: {save_total_limit}\n- save_only_model: {save_only_model}\n- dataloader_num_workers: {dataloader_num_workers}\n- dataloader_persistent_workers: {dataloader_persistent_workers}\n- Discord heartbeat_steps: {discord_heartbeat_steps}\n- auto_stop_enabled: {auto_stop_enabled}",
                }
            ]
        },
    )

    send_discord(
        webhooks,
        embed={
            "embeds": [
                {
                    "author": {"name": "AI 모델 파인튜너"},
                    "color": 3066993,
                    "title": "🚀 고정 HP 파인튜닝 시작",
                    "description": f"- 에폭: {args.epochs}\n- LR: {hp['learning_rate']:.2e}\n- LoRA r={hp['lora_r']}, alpha={hp['lora_alpha']}\n- 학습 샘플: {len(train_dataset)}\n- 검증 샘플: {n_val}\n- 평가 샘플 수: {eval_samples}\n- save/eval/logging steps: {args.save_steps}/{eval_steps}/{args.logging_steps}\n- save_only_model: {save_only_model}",
                }
            ]
        },
    )

    t0 = time.time()
    send_discord(
        webhooks,
        embed={
            "embeds": [
                {
                    "author": {"name": "AI 모델 파인튜너"},
                    "color": 3066993,
                    "title": "🏋️ 학습 루프 시작",
                }
            ]
        },
    )
    try:
        trainer.train()
    except RuntimeError as exc:
        error_text = str(exc)
        if "inline_container.cc" in error_text and "unexpected pos" in error_text:
            free_space_gb = get_free_space_gb(output_dir)
            hint = (
                "체크포인트 저장 중 파일 쓰기가 중단되었습니다. "
                "디스크 공간 부족 또는 파일시스템 I/O 오류 가능성이 큽니다."
            )
            logger.error(
                "%s output_dir=%s free_space_gb=%.2f save_only_model=%s",
                hint,
                output_dir,
                free_space_gb,
                save_only_model,
            )
            send_discord(
                webhooks,
                embed={
                    "embeds": [
                        {
                            "author": {"name": "AI 모델 파인튜너"},
                            "color": 15277667,
                            "title": "❌ 체크포인트 저장 실패",
                            "description": f"- 원인 추정: 디스크 공간 부족 또는 파일시스템 I/O 오류\n- output_dir: `{output_dir}`\n- free_space_gb: {free_space_gb:.2f}\n- save_only_model: {save_only_model}",
                        }
                    ]
                },
            )
            raise RuntimeError(
                f"{hint} output_dir={output_dir}, free_space_gb={free_space_gb:.2f}, save_only_model={save_only_model}"
            ) from exc
        raise
    elapsed_min = (time.time() - t0) / 60
    if monitor_callback.stop_triggered:
        send_discord(
            webhooks,
            embed={
                "embeds": [
                    {
                        "author": {"name": "AI 모델 파인튜너"},
                        "title": "🛑 학습 루프 조기 종료",
                        "color": 15158332,
                        "description": (
                            f"- 소요 시간: {elapsed_min:.1f}분\n"
                            f"- 중단 사유: {monitor_callback.stop_reason or '자동 중단 조건 충족'}"
                        ),
                    }
                ]
            },
        )
    else:
        send_discord(
            webhooks,
            embed={
                "embeds": [
                    {
                        "author": {"name": "AI 모델 파인튜너"},
                        "title": "✅ 학습 루프 완료",
                        "color": 3066993,
                        "description": f"- 소요 시간: {elapsed_min:.1f}분",
                    }
                ]
            },
        )

    lora_dir = os.path.join(output_dir, "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    eval_result = None
    if not args.no_eval:
        FastVisionModel.for_inference(model)
        eval_dir = os.path.join(output_dir, "evaluation")
        eval_result = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            val_dataset=val_dataset,
            max_samples=eval_samples,
            save_dir=eval_dir,
            trial_num="predefined-finetune",
            logger=logger,
        )
        send_discord(
            webhooks,
            embed={
                "embeds": [
                    {
                        "author": {"name": "AI 모델 파인튜너"},
                        "color": 3066993,
                        "title": "🧪 평가 완료",
                        "description": f"- Accuracy: {eval_result.get('accuracy', 0.0):.4f}\n- Precision macro: {eval_result.get('precision_macro', 0.0):.4f}\n- Recall macro: {eval_result.get('recall_macro', 0.0):.4f}\n- F1 (macro): {eval_result.get('f1_macro', 0.0):.4f}\n- F1 (weighted): {eval_result.get('f1_weighted', 0.0):.4f}",
                    }
                ]
            },
        )
        with open(os.path.join(eval_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False, default=str)
    else:
        send_discord(
            webhooks,
            embed={
                "embeds": [
                    {
                        "author": {"name": "AI 모델 파인튜너"},
                        "color": 15844367,
                        "title": "🧪 평가 스킵됨",
                        "description": "`--no-eval` 사용됨",
                    }
                ]
            },
        )

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(build_readme(base_model, hp, args.epochs, elapsed_min, eval_result))

    hf_repo = args.hf_repo or cfg.get("huggingface", {}).get("hf_repo_id", "")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_repo and hf_token:
        try:
            upload_result = upload_finetune_output(
                output_dir=output_dir,
                repo_id=hf_repo,
                token=hf_token,
                private=True,
                create_repo_if_missing=True,
            )
            send_discord(
                webhooks,
                embed={
                    "embeds": [
                        {
                            "author": {"name": "AI 모델 파인튜너"},
                            "color": 3066993,
                            "title": "📤 HuggingFace 업로드 완료",
                            "description": (
                                f"- Repo: [`{upload_result['repo_id']}`]"
                                f"({upload_result['repo_url']})\n"
                                f"- evaluation 업로드: {upload_result['eval_uploaded']}\n"
                                f"- README 업로드: {upload_result['readme_uploaded']}"
                            ),
                        }
                    ]
                },
            )
        except Exception as exc:
            logger.exception("HF 업로드 실패: %s", exc)
            send_discord(
                webhooks,
                embed={
                    "embeds": [
                        {
                            "author": {"name": "AI 모델 파인튜너"},
                            "color": 15277667,
                            "title": "⚠️ HuggingFace 업로드 실패",
                            "description": f"- Repo: [`{hf_repo}`](https://huggingface.co/{hf_repo})\n- 오류: {exc}",
                        }
                    ]
                },
            )

    if wandb_run:
        wandb_run.finish()

    logger.info("완료: %s", lora_dir)
    send_discord(
        webhooks,
        embed={
            "embeds": [
                {
                    "author": {"name": "AI 모델 파인튜너"},
                    "color": 3066993,
                    "title": "🎉 고정 하이퍼-파라미터 파인튜닝 전체 완료",
                    "description": f"- LoRA 출력: `{lora_dir}`\n- 소요 시간: {elapsed_min:.1f}분",
                }
            ]
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
