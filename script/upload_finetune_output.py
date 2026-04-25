#!/usr/bin/env python3
"""고정 HP 파인튜닝 산출물을 HuggingFace Hub로 재업로드한다."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from common.app_config import apply_auth_environment, get_discord_webhooks, load_app_config
from common.discord_utils import send_discord
from common.hf_upload import upload_finetune_output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="finetune 출력 디렉터리를 HuggingFace Hub로 업로드"
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument(
        "--public",
        action="store_true",
        help="repo 생성 시 public으로 생성합니다.",
    )
    parser.add_argument(
        "--no-create-repo",
        action="store_true",
        help="repo가 없을 때 자동 생성하지 않습니다.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_app_config(args.config)
    apply_auth_environment(cfg)
    webhooks = get_discord_webhooks(cfg)

    output_dir = args.output_dir or cfg.get("paths", {}).get(
        "final_output_dir", "/workspace/best-pest-detector"
    )
    repo_id = (args.hf_repo or cfg.get("huggingface", {}).get("hf_repo_id", "")).strip()
    token = (os.environ.get("HF_TOKEN") or "").strip()

    if not repo_id:
        raise ValueError("업로드 대상 repo_id가 비어 있습니다. (--hf-repo 또는 config.huggingface.hf_repo_id)")
    if not token:
        raise ValueError("HF_TOKEN이 비어 있습니다. (환경변수 또는 config.auth.hf_token)")

    send_discord(
        webhooks,
        embed={
            "embeds": [
                {
                    "author": {"name": "AI 모델 파인튜너"},
                    "color": 3447003,
                    "title": "📤 수동 재업로드 시작",
                    "description": f"- output_dir: `{output_dir}`\n- repo: `{repo_id}`",
                }
            ]
        },
    )

    try:
        result = upload_finetune_output(
            output_dir=output_dir,
            repo_id=repo_id,
            token=token,
            private=(not args.public),
            create_repo_if_missing=(not args.no_create_repo),
        )
    except Exception as exc:
        logging.exception("업로드 실패: %s", exc)
        send_discord(
            webhooks,
            embed={
                "embeds": [
                    {
                        "author": {"name": "AI 모델 파인튜너"},
                        "color": 15158332,
                        "title": "⚠️ 수동 재업로드 실패",
                        "description": f"- repo: `{repo_id}`\n- 오류: {exc}",
                    }
                ]
            },
        )
        return 1

    logging.info("업로드 완료: %s", result["repo_url"])
    send_discord(
        webhooks,
        embed={
            "embeds": [
                {
                    "author": {"name": "AI 모델 파인튜너"},
                    "color": 3066993,
                    "title": "✅ 수동 재업로드 완료",
                    "description": (
                        f"- Repo: [`{result['repo_id']}`]({result['repo_url']})\n"
                        f"- evaluation 업로드: {result['eval_uploaded']}\n"
                        f"- README 업로드: {result['readme_uploaded']}"
                    ),
                }
            ]
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
