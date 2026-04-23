from __future__ import annotations

import logging
import threading
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _redact_webhook(url: str) -> str:
    parts = url.rstrip("/").split("/")
    if len(parts) >= 1:
        parts[-1] = "***"
    return "/".join(parts)


def send_discord(
    webhooks: list[str],
    content: str | None = None,
    embed: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 10,
) -> None:
    if not webhooks:
        return

    outgoing_payload: dict[str, Any] = {}
    if payload is not None:
        outgoing_payload = dict(payload)
    else:
        if content:
            outgoing_payload["content"] = content
        if embed:
            outgoing_payload["embeds"] = [embed]

    if not outgoing_payload:
        return

    def _send_one(url: str) -> None:
        safe_url = _redact_webhook(url)
        try:
            resp = requests.post(url, json=outgoing_payload, timeout=timeout)
            if resp.status_code >= 400:
                body = (resp.text or "").replace("\n", " ").strip()
                logger.warning(
                    "Discord 알림 실패 (status=%s, webhook=%s, body=%s)",
                    resp.status_code,
                    safe_url,
                    body[:300],
                )
        except Exception as exc:
            logger.warning("Discord 알림 실패 (webhook=%s): %s", safe_url, exc)

    for webhook in webhooks:
        threading.Thread(target=_send_one, args=(webhook,), daemon=True).start()
