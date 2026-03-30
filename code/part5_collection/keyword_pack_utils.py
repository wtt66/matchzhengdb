from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_KEYWORD_PACK = Path(__file__).resolve().with_name("fuzhou_fragrance_keyword_pack.json")


def load_keyword_pack(path: str | Path | None = None) -> dict[str, Any]:
    pack_path = Path(path) if path else DEFAULT_KEYWORD_PACK
    return json.loads(pack_path.read_text(encoding="utf-8"))


def get_pain_point_taxonomy(pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return dict(pack.get("pain_points", {}))


def get_implicit_signal_taxonomy(pack: dict[str, Any]) -> dict[str, list[str]]:
    return dict(pack.get("implicit_signals", {}))


def get_stopwords(pack: dict[str, Any]) -> set[str]:
    return set(pack.get("stopwords", []))


def get_custom_terms(pack: dict[str, Any]) -> list[str]:
    return list(pack.get("custom_terms", []))


def get_recommended_queries(pack: dict[str, Any]) -> dict[str, list[str]]:
    return dict(pack.get("recommended_queries", {}))
