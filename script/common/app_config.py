from __future__ import annotations

import json
import os
from typing import Any


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
            "best_model_dir": "/workspace/best-pest-detector",
            "data_dir": "/workspace/data",
            "db_path": "/workspace/hp_search_results.db",
            "final_output_dir": "/workspace/best-pest-detector",
            "golden_dir": "/workspace/_golden",
            "log_file": "/workspace/hp_search.log",
            "output_dir": "/workspace/hp_search_outputs",
            "preload_cache_dir": "/workspace/preload_cache",
            "volume_dir": "/workspace",
        },
        "runtime": {
            "anomaly_eval_loss_jump_ratio": 2.0,
            "anomaly_eval_loss_threshold": 2.0,
            "anomaly_grad_norm_threshold": 10000.0,
            "anomaly_loss_jump_ratio": 3.0,
            "anomaly_loss_threshold": 5.0,
            "auto_stop_enabled": True,
            "auto_stop_eval_loss_best_ratio": 8.0,
            "auto_stop_eval_loss_threshold": 1.0,
            "auto_stop_grad_norm_threshold": 1e7,
            "auto_stop_train_loss_consecutive": 2,
            "auto_stop_train_loss_threshold": 2.0,
            "base_model": "unsloth/Qwen3.5-9B",
            "dataloader_num_workers": 8,
            "default_n_trials": 30,
            "discord_alert_cooldown_steps": 50,
            "discord_heartbeat_steps": 100,
            "predefined_eval_steps": 120,
            "predefined_logging_steps": 10,
            "predefined_min_free_space_gb": 20,
            "predefined_save_only_model": True,
            "predefined_save_steps": 120,
            "predefined_save_total_limit": 2,
            "random_seed": 42,
            "study_name": "wandb-study-name",
            "wandb_entity": "team-name",
            "wandb_project": "wandb-project-name",
        },
        "notifications": {"discord_webhooks": ["https://discord.com/api/webhooks/"]},
        "github": {
            "backup_db_path_in_repo": "hp_search_results.db",
            "repo": "your-name/result-save-repo",
        },
        "auth": {
            "github_token": "",
            "hf_token": "",
            "wandb_api_key": "",
        },
        "huggingface": {
            "dataset_repo": "your-name/dataset-repo",
            "hf_repo_id": "your-name/model-adapter-repo",
        },
        "hyperparameters": {
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "crop_tight_prob": 0.4560998466814129,
            "data_seed": 3407,
            "finetune_vision": False,
            "gradient_accumulation_steps": 8,
            "learning_rate": 0.00011645105452323228,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "lora_r": 64,
            "lr_scheduler_type": "linear",
            "max_grad_norm": 1.0,
            "max_seq_length": 1024,
            "optim": "adamw_torch",
            "per_device_train_batch_size": 1,
            "seed": 42,
            "use_rslora": True,
            "warmup_ratio": 0.03,
            "weight_decay": 0.013802470048539942,
        },
    }


def load_app_config(config_path: str = "config.json") -> dict[str, Any]:
    cfg = _default_config()
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
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
