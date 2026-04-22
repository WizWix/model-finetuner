from __future__ import annotations

import os
from typing import Any

from .jsonc_utils import load_jsonc


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _default_config() -> dict[str, Any]:
    return {
        "paths": {
            "data_dir": "/workspace/data",
            "volume_dir": "/workspace",
            "output_dir": "/workspace/hp_search_outputs",
            "best_model_dir": "/workspace/best-pest-detector",
            "db_path": "/workspace/hp_search_results.db",
            "log_file": "/workspace/hp_search.log",
            "preload_cache_dir": "/workspace/preload_cache",
            "final_output_dir": "/workspace/best-pest-detector",
            "golden_dir": "/workspace/_golden",
        },
        "runtime": {
            "study_name": "pest-detection-hpsearch",
            "base_model": "unsloth/Qwen3.5-9B",
            "random_seed": 42,
            "default_n_trials": 30,
            "wandb_project": "pest-detection-hpsearch",
            "wandb_entity": "",
        },
        "notifications": {"discord_webhooks": []},
        "github": {
            "repo": "pfox1995/pest-hyperparameter-search",
            "backup_db_path_in_repo": "hp_search_results.db",
        },
        "auth": {
            "hf_token": "",
            "wandb_api_key": "",
            "github_token": "",
        },
        "huggingface": {
            "dataset_repo": "Himedia-AI-01/pest-detection-korean",
            "hf_repo_id": "",
        },
        "hyperparameters": {
            "lora_r": 64,
            "lora_alpha": 128,
            "use_rslora": True,
            "lora_dropout": 0.0,
            "finetune_vision": False,
            "learning_rate": 0.00011645105452323228,
            "weight_decay": 0.013802470048539942,
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.03,
            "max_grad_norm": 1.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "optim": "adamw_torch",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 1024,
            "crop_tight_prob": 0.4560998466814129,
            "seed": 42,
            "data_seed": 3407,
        },
    }


def load_app_config(config_path: str = "config.json") -> dict[str, Any]:
    cfg = _default_config()
    if os.path.exists(config_path):
        loaded = load_jsonc(config_path)
        cfg = _deep_merge(cfg, loaded)
    return cfg


def get_discord_webhooks(config: dict[str, Any]) -> list[str]:
    webhooks = config.get("notifications", {}).get("discord_webhooks", []) or []
    return [w for w in webhooks if isinstance(w, str) and w.strip()]


def apply_auth_environment(config: dict[str, Any]) -> None:
    auth = config.get("auth", {})
    hf = auth.get("hf_token")
    wandb = auth.get("wandb_api_key")
    github = auth.get("github_token")

    if hf:
        os.environ["HF_TOKEN"] = hf
    if wandb:
        os.environ["WANDB_API_KEY"] = wandb
    if github:
        os.environ["GITHUB_TOKEN"] = github
        os.environ["github_pat"] = github
