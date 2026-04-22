from __future__ import annotations

import logging
import threading
from typing import Any

import requests

logger = logging.getLogger(__name__)


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
        try:
            requests.post(url, json=outgoing_payload, timeout=timeout)
        except Exception as exc:
            logger.warning("Discord 알림 실패 (%s): %s", url, exc)

    for webhook in webhooks:
        threading.Thread(target=_send_one, args=(webhook,), daemon=True).start()
