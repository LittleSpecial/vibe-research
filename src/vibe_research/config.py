from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib


@dataclass
class Settings:
    raw: dict

    @property
    def model(self) -> dict:
        return self.raw.get("model", {})

    @property
    def research(self) -> dict:
        return self.raw.get("research", {})

    @property
    def remote(self) -> dict:
        return self.raw.get("remote", {})

    @property
    def agents(self) -> dict:
        return self.raw.get("agents", {})

    def provider(self) -> dict:
        key = self.model.get("model_provider")
        providers = self.raw.get("model_providers", {})
        if not key or key not in providers:
            raise KeyError(f"model_provider '{key}' not found in model_providers")
        return providers[key]


def load_settings(path: str | Path) -> Settings:
    p = Path(path)
    with p.open("rb") as f:
        raw = tomllib.load(f)
    return Settings(raw=raw)
