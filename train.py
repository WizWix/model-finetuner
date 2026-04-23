#!/usr/bin/env python3
"""고정 하이퍼파라미터 파인튜닝 진입점."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
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
from common.training_core import (
    clear_gpu_memory,
    evaluate_model,
    get_line_count,
    get_max_data_fraction,
    load_dataset_from_jsonl,
    load_model_with_retry,
)
from common.wandb_utils import wandb_is_available

KST = timezone(timedelta(hours=9))
logger = logging.getLogger(__name__)


def build_readme(
    base_model: str,
    hp: dict[str, Any],
    epochs: int,
    elapsed_min: float,
    eval_result: dict | None,
) -> str:
    metrics_section = ""
    if eval_result:
        metrics_section = f"""## 평가 결과

| 지표 | 값 |
|---|---|
| 정확도 | {eval_result["accuracy"]:.4f} |
| F1 (macro) | {eval_result["f1_macro"]:.4f} |
| F1 (weighted) | {eval_result["f1_weighted"]:.4f} |
| 정밀도 (macro) | {eval_result["precision_macro"]:.4f} |
| 재현율 (macro) | {eval_result["recall_macro"]:.4f} |
| 검증 샘플 수 | {eval_result.get("total", "?")} |

혼동 행렬 및 클래스별 지표는 `evaluation/` 폴더를 확인하세요.
"""
    return f"""---
base_model: {base_model}
library_name: peft
tags:
- lora
- vision-language
- pest-detection
- unsloth
---

# 해충 탐지 VLM - Qwen3.5-9B LoRA

한국어 19개 해충 분류를 위해 파인튜닝된 LoRA 어댑터입니다.

## 학습 설정

| 하이퍼파라미터 | 값 |
|---|---|
| LoRA 랭크 | {hp["lora_r"]} |
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

{metrics_section}
## 사용 예시

```python
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained("{base_model}")
model.load_adapter("<this-repo-id>")
FastVisionModel.for_inference(model)
```
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
        split="val",
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

    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer
    from unsloth import FastVisionModel
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

    sft_config_kwargs: dict[str, Any] = {
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
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
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
        "max_seq_length": hp["max_seq_length"],
    }
    sft_args = SFTConfig(**sft_config_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "processing_class": tokenizer,
        "data_collator": UnslothVisionDataCollator(model, tokenizer),
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset if not args.no_eval else None,
        "args": sft_args,
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
                    "description": f"- 모델: {base_model}\n- 출력 경로: {output_dir}\n- save/eval/logging steps: {args.save_steps}/{eval_steps}/{args.logging_steps}",
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
                    "description": f"- 에폭: {args.epochs}\n- LR: {hp['learning_rate']:.2e}\n- LoRA r={hp['lora_r']}, alpha={hp['lora_alpha']}\n- 학습 샘플: {len(train_dataset)}\n- 검증 샘플: {n_val}\n- 평가 샘플 수: {eval_samples}\n- save/eval/logging steps: {args.save_steps}/{eval_steps}/{args.logging_steps}",
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
    trainer.train()
    elapsed_min = (time.time() - t0) / 60
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
            model.push_to_hub(hf_repo, token=hf_token, private=True)
            tokenizer.push_to_hub(hf_repo, token=hf_token, private=True)
            from huggingface_hub import upload_file, upload_folder

            eval_dir = os.path.join(output_dir, "evaluation")
            if os.path.isdir(eval_dir):
                upload_folder(
                    folder_path=eval_dir,
                    path_in_repo="evaluation",
                    repo_id=hf_repo,
                    token=hf_token,
                )
            upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=hf_repo,
                token=hf_token,
            )
            send_discord(
                webhooks,
                embed={
                    "embeds": [
                        {
                            "author": {"name": "AI 모델 파인튜너"},
                            "color": 3066993,
                            "title": "📤 HuggingFace 업로드 완료",
                            "description": f"- Repo: [`{hf_repo}`](https://huggingface.co/{hf_repo})",
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
