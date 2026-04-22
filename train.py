#!/usr/bin/env python3
"""고정 하이퍼파라미터 파인튜닝 진입점."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import torch

from common.app_config import apply_auth_environment, load_app_config

KST = timezone(timedelta(hours=9))


def build_readme(
    base_model: str, hp: dict, epochs: int, elapsed_min: float, eval_result: dict | None
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
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="config.json")
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_app_config(pre_args.config)
    apply_auth_environment(cfg)

    paths = cfg.get("paths", {})
    hp = cfg.get("hyperparameters", {})

    parser = argparse.ArgumentParser(description="고정 HP 파인튜닝")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default=paths.get("final_output_dir", "/workspace/best-pest-detector"),
    )
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--eval-samples", type=int, default=-1)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--hf-repo", default=cfg.get("huggingface", {}).get("hf_repo_id", "")
    )
    args = parser.parse_args()

    cfg = load_app_config(args.config)
    apply_auth_environment(cfg)

    import hp_search

    hp_search.initialize_from_config(args.config)
    hp = cfg.get("hyperparameters", {})

    # 원본 run.sh 동작을 유지하기 위해 고정 파인튜닝은 lazy dataset을 기본 사용한다.
    os.environ["HP_LAZY_DATASET"] = "1"

    random.seed(hp["seed"])
    torch.manual_seed(hp["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hp["seed"])

    assert torch.cuda.is_available(), "CUDA GPU가 필요합니다"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    capability = torch.cuda.get_device_capability(0)
    hp_search.logger.info(
        "GPU: %s | VRAM: %.0fGB | compute_capability: sm_%s%s",
        gpu_name,
        vram_gb,
        capability[0],
        capability[1],
    )

    os.makedirs(args.output_dir, exist_ok=True)

    hp_search.logger.info("데이터셋 로딩 중...")
    n_train_total = hp_search._get_line_count(
        os.path.join(hp_search.DATA_DIR, "train.jsonl")
    )
    train_frac = hp_search.get_max_data_fraction(n_train_total)
    train_dataset = hp_search.load_dataset_from_jsonl(
        "train",
        tight_prob=hp["crop_tight_prob"],
        fraction=train_frac,
    )
    val_dataset = hp_search.load_dataset_from_jsonl("val")
    n_val = len(val_dataset)
    eval_samples = n_val if args.eval_samples < 0 else min(args.eval_samples, n_val)

    hp_search.clear_gpu_memory()
    hp_search.logger.info("모델 로딩: %s", hp_search.BASE_MODEL)
    model, tokenizer = hp_search.load_model_with_retry(hp_search.BASE_MODEL)

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

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
    if not args.no_wandb and hp_search.wandb_is_available():
        import wandb

        run_name = f"predefined-{datetime.now(KST).strftime('%m%d-%H%M')}"
        wandb_run = wandb.init(
            project=hp_search.WANDB_PROJECT,
            entity=hp_search.WANDB_ENTITY or None,
            name=run_name,
            config={**hp, "num_train_epochs": args.epochs},
            tags=["predefined-finetune"],
            reinit=True,
        )
        report_to = "wandb"

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
        "eval_steps": args.save_steps,
        "eval_accumulation_steps": 2,
        "load_best_model_at_end": (not args.no_eval),
        "metric_for_best_model": "eval_loss" if not args.no_eval else None,
        "greater_is_better": False,
        "logging_steps": 20,
        "logging_strategy": "steps",
        "seed": hp["seed"],
        "data_seed": hp["data_seed"],
        "output_dir": args.output_dir,
        "report_to": report_to,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_seq_length": hp["max_seq_length"],
    }
    sft_args = SFTConfig(**sft_config_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer,
        "data_collator": UnslothVisionDataCollator(model, tokenizer),
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset if not args.no_eval else None,
        "args": sft_args,
    }
    trainer = SFTTrainer(**trainer_kwargs)

    hp_search.discord_send(
        f"🚀 **고정 HP 파인튜닝 시작**\\n"
        f"• 에폭: {args.epochs}\\n"
        f"• LR: {hp['learning_rate']:.2e}\\n"
        f"• LoRA r={hp['lora_r']}, α={hp['lora_alpha']}\\n"
        f"• 학습 샘플: {len(train_dataset)}\\n"
        f"• 검증 샘플: {n_val}\\n"
        f"• 평가 샘플 수: {eval_samples}"
    )

    t0 = time.time()
    trainer.train()
    elapsed_min = (time.time() - t0) / 60

    lora_dir = os.path.join(args.output_dir, "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    eval_result = None
    if not args.no_eval:
        FastVisionModel.for_inference(model)
        eval_dir = os.path.join(args.output_dir, "evaluation")
        eval_result = hp_search.evaluate_model(
            model,
            tokenizer,
            val_dataset,
            max_samples=eval_samples,
            save_dir=eval_dir,
            trial_num="predefined-finetune",
        )
        with open(os.path.join(eval_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False, default=str)

    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            build_readme(
                hp_search.BASE_MODEL, hp, args.epochs, elapsed_min, eval_result
            )
        )

    hf_repo = args.hf_repo or cfg.get("huggingface", {}).get("hf_repo_id", "")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_repo and hf_token:
        try:
            model.push_to_hub(hf_repo, token=hf_token, private=True)
            tokenizer.push_to_hub(hf_repo, token=hf_token, private=True)
            from huggingface_hub import upload_file, upload_folder

            eval_dir = os.path.join(args.output_dir, "evaluation")
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
        except Exception as exc:
            hp_search.logger.exception("HF 업로드 실패: %s", exc)

    if wandb_run:
        wandb_run.finish()

    hp_search.logger.info("완료: %s", lora_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
