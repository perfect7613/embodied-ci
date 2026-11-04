"""Baseten Model APIs (OpenAI-compatible). All LLM calls go through `chat`."""

from __future__ import annotations

import base64
import io
import json
import os
import time

from openai import OpenAI

MODEL = os.environ.get("BASETEN_MODEL", "meta-models/muse-spark-open-1026")
_client = None
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def client() -> OpenAI:
    global _client
    if _client is None:
        if not os.environ.get("BASETEN_API_KEY"):
            from dotenv import load_dotenv

            load_dotenv()
        _client = OpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url="https://inference.baseten.co/v1")
    return _client


def image_part(img) -> dict:
    """numpy HxWx3 uint8 or PNG bytes -> OpenAI image content part."""
    if not isinstance(img, (bytes, bytearray)):
        import imageio.v3 as iio

        buf = io.BytesIO()
        iio.imwrite(buf, img, extension=".png")
        img = buf.getvalue()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()}}


def chat(messages: list[dict], max_tokens: int = 4096, json_mode: bool = False, retries: int = 2) -> tuple[str, int]:
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    for attempt in range(retries + 1):
        try:
            r = client().chat.completions.create(model=MODEL, messages=messages, max_tokens=max_tokens, **kw)
            u = r.usage
            USAGE["calls"] += 1
            USAGE["prompt_tokens"] += u.prompt_tokens
            USAGE["completion_tokens"] += u.completion_tokens
            return r.choices[0].message.content or "", u.total_tokens
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def ask(prompt: str, data=None, images: list | None = None, json_out: bool = False, max_tokens: int = 4096):
    """One-shot call. `data` is JSON-serialised into the prompt; `images` are frames/PNGs."""
    content: list[dict] = [{"type": "text", "text": prompt + ("\n\nDATA:\n" + json.dumps(data, default=str)[:200_000] if data is not None else "")}]
    for im in images or []:
        content.append(image_part(im))
    text, _ = chat([{"role": "user", "content": content}], max_tokens=max_tokens, json_mode=json_out)
    if json_out:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            s, e = text.find("{"), text.rfind("}")
            return json.loads(text[s : e + 1])
    return text
