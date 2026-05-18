import asyncio
import json
import re

import httpx

from . import config
from .prompts import SYSTEM_PROMPT, build_user_prompt


async def analyze(
    metadata: dict,
    transcript_text: str,
    heatmap_text: str,
    max_clips: int = config.DEFAULT_MAX_CLIPS,
    min_duration: int = config.DEFAULT_MIN_DURATION,
    max_duration: int = config.DEFAULT_MAX_DURATION,
    buffer: int = config.DEFAULT_BUFFER,
) -> dict:
    if not config.KIE_AI_API_KEY:
        raise ValueError("KIE_AI_API_KEY belum diset di .env")

    user_prompt = build_user_prompt(
        metadata, transcript_text, heatmap_text, max_clips, min_duration, max_duration, buffer
    )

    payload = {
        "model": "gemini-3-flash",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ],
        "stream": False,
    }

    last_error = None
    for attempt in range(config.MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                resp = await client.post(
                    config.KIE_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {config.KIE_AI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

            if resp.status_code == 429:
                delay = config.RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)
                continue

            if resp.status_code != 200:
                raise ValueError(f"kie.ai error {resp.status_code}: {resp.text[:400]}")

            data = resp.json()
            raw = data["choices"][0]["message"]["content"] or ""
            return _parse(raw)

        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Response kie.ai tidak terduga: {e}. Data: {str(data)[:400]}")
        except ValueError:
            raise
        except Exception as e:
            last_error = e
            if attempt < config.MAX_RETRIES - 1:
                await asyncio.sleep(config.RETRY_DELAY)

    raise ValueError(f"Gagal setelah {config.MAX_RETRIES} percobaan: {last_error}")


def _parse(raw: str) -> dict:
    raw = raw.strip()
    # strip markdown codeblock
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini tidak menghasilkan JSON valid: {e}. Raw: {raw[:500]}")
