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
    wait: bool = False,
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
            embeds = embed.get("embeds")
            if isinstance(embeds, list):
                outgoing_payload["embeds"] = embeds
            else:
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

    threads: list[threading.Thread] = []
    for webhook in webhooks:
        t = threading.Thread(target=_send_one, args=(webhook,), daemon=not wait)
        t.start()
        threads.append(t)

    if wait:
        for t in threads:
            t.join(timeout=timeout + 1)
