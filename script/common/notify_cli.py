#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.app_config import get_discord_webhooks, load_app_config
from common.discord_utils import send_discord


def main() -> int:
    parser = argparse.ArgumentParser(description="Discord로 실패 알림을 전송합니다")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--title", default="작업 실패")
    parser.add_argument("--exit-code", type=int, default=1)
    message_group = parser.add_mutually_exclusive_group(required=True)
    message_group.add_argument("--message")
    message_group.add_argument(
        "--payload-json",
        help="Discord 웹훅 페이로드 JSON 문자열(객체)",
    )
    message_group.add_argument(
        "--payload-json-file",
        help="Discord 웹훅 페이로드 JSON 파일 경로(객체)",
    )
    args = parser.parse_args()

    cfg = load_app_config(args.config)
    webhooks = get_discord_webhooks(cfg)
    if not webhooks:
        return 0

    payload: dict[str, Any] | None = None
    if args.payload_json is not None:
        payload = json.loads(args.payload_json)
    elif args.payload_json_file is not None:
        payload_text = Path(args.payload_json_file).read_text(encoding="utf-8")
        payload = json.loads(payload_text)

    if payload is not None:
        if not isinstance(payload, dict):
            raise ValueError("페이로드 JSON은 JSON 객체여야 합니다.")
        send_discord(webhooks=webhooks, payload=payload, wait=True)
        return 0

    host = socket.gethostname()
    content = (
        f"💥 **{args.title}**\\n"
        f"호스트: `{host}`\\n"
        f"exit_code: `{args.exit_code}`\\n"
        f"{args.message}"
    )
    send_discord(webhooks=webhooks, content=content, wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
