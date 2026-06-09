import json
import os
import time
from typing import Any

import dashscope
from dashscope import MultiModalConversation
from dotenv import load_dotenv


# DashScope returns these when the service is temporarily overloaded; retry them.
TRANSIENT_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 5


def normalize_api_key(raw_key: str | None) -> str:
    if not raw_key:
        return ""
    key = raw_key.strip().strip('"').strip("'")
    if key and not key.startswith("sk-"):
        key = f"sk-{key}"
    return key


def configure_dashscope() -> None:
    load_dotenv(override=True)
    api_key = normalize_api_key(os.getenv("DASHSCOPE_API_KEY"))
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is missing. Check your .env file.")

    dashscope.api_key = api_key
    dashscope.base_http_api_url = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/api/v1"
    ).strip()


def call_qwen(model: str, content: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    configure_dashscope()
    messages = [{"role": "user", "content": content}]

    response = None
    for attempt in range(MAX_RETRIES):
        response = MultiModalConversation.call(model=model, messages=messages)
        status = response.get("status_code")
        if status == 200:
            break
        if status in TRANSIENT_STATUS and attempt < MAX_RETRIES - 1:
            wait = 2 ** attempt * 5  # 5, 10, 20, 40 s
            print(f"  [retry {attempt + 1}/{MAX_RETRIES}] status {status}, waiting {wait}s...")
            time.sleep(wait)
            continue
        raise RuntimeError(json.dumps(dict(response), ensure_ascii=False, indent=2))

    if response is None or response.get("status_code") != 200:
        raise RuntimeError(json.dumps(dict(response or {}), ensure_ascii=False, indent=2))

    output = response.get("output", {})
    choices = output.get("choices", [])
    texts: list[str] = []
    if choices:
        message_content = choices[0].get("message", {}).get("content", [])
        for item in message_content:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])

    return "\n".join(texts).strip(), dict(response)


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction for model outputs that should be JSON."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(stripped[start : end + 1])

    raise ValueError(f"Could not parse JSON object from model output:\n{text}")
