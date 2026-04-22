import json
import re
from typing import Any

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_jsonc(text: str) -> str:
    """JSONC 문자열에서 // 및 /* */ 주석을 제거한다."""
    text = _BLOCK_COMMENT_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        new_line = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == '"' and not escaped:
                in_string = not in_string
                new_line.append(ch)
            elif (
                ch == "/" and not in_string and i + 1 < len(line) and line[i + 1] == "/"
            ):
                break
            else:
                new_line.append(ch)

            escaped = ch == "\\" and not escaped
            if ch != "\\":
                escaped = False
            i += 1
        lines.append("".join(new_line))
    return "\n".join(lines)


def load_jsonc(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return json.loads(strip_jsonc(text))
