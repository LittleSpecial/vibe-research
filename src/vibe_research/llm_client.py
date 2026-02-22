from __future__ import annotations

import json
import os
import urllib.request
import urllib.error


class ResponsesClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        reasoning_effort: str = "high",
        max_output_tokens: int = 1400,
    ):
        self.base_url = base_url.rstrip("/") + "/responses"
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "instructions": system_prompt,
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "stream": True,
            "input": [
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(self.base_url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                text, raw_stream = self._read_sse_response(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e

        if not text:
            # Fallback for non-stream payloads.
            parsed = json.loads(raw_stream or "{}")
            text = self._extract_text(parsed)
        if not text:
            raise RuntimeError("No text found in response payload")
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

    @staticmethod
    def _extract_stream_text(body: str) -> str:
        """
        Parse SSE body from codex-lb Responses endpoint.
        """
        chunks: list[str] = []
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line.startswith("data: "):
                continue
            data_part = line[6:].strip()
            if not data_part:
                continue
            try:
                event = json.loads(data_part)
            except json.JSONDecodeError:
                continue

            typ = event.get("type", "")
            if typ == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    chunks.append(delta)
            elif typ == "response.output_text.done" and not chunks:
                text = event.get("text", "")
                if text:
                    chunks.append(text)

        return "".join(chunks).strip()

    @staticmethod
    def _read_sse_response(resp) -> tuple[str, str]:
        """
        Consume SSE lines until response.completed (or EOF).
        Returns: (concatenated_text, raw_stream_text)
        """
        chunks: list[str] = []
        raw_lines: list[str] = []

        while True:
            line_b = resp.readline()
            if not line_b:
                break
            line = line_b.decode("utf-8", errors="replace").strip()
            raw_lines.append(line)
            if not line.startswith("data: "):
                continue

            data_part = line[6:].strip()
            if not data_part:
                continue

            try:
                event = json.loads(data_part)
            except json.JSONDecodeError:
                continue

            typ = event.get("type", "")
            if typ == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    chunks.append(delta)
            elif typ == "response.output_text.done" and not chunks:
                done_text = event.get("text", "")
                if done_text:
                    chunks.append(done_text)
            elif typ == "response.completed":
                break

        return "".join(chunks).strip(), "\n".join(raw_lines)
