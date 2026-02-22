from __future__ import annotations

import json
import os
import urllib.request


class ResponsesClient:
    def __init__(self, base_url: str, model: str, reasoning_effort: str = "high"):
        self.base_url = base_url.rstrip("/") + "/responses"
        self.model = model
        self.reasoning_effort = reasoning_effort

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "reasoning": {"effort": self.reasoning_effort},
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(self.base_url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode("utf-8")

        parsed = json.loads(body)
        text = self._extract_text(parsed)
        if not text:
            raise RuntimeError(f"No text found in response payload: {parsed}")
        return text

    @staticmethod
    def _extract_text(payload: dict) -> str:
        if payload.get("output_text"):
            return payload["output_text"]

        chunks: list[str] = []
        for item in payload.get("output", []):
            for c in item.get("content", []):
                if c.get("type") in {"output_text", "text"}:
                    t = c.get("text", "")
                    if t:
                        chunks.append(t)
        return "\n".join(chunks).strip()
