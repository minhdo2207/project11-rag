"""Configuration management from YAML.

Philosophy: All tunable parameters (storage backend, model choice, feature toggles)
live in config.yaml; code only reads. This enables:
  - Reproducible experiments (config → results mapping)
  - Zero-code configuration changes
  - Easy ablation studies (toggle layers on/off)

Example config.yaml structure:
  storage:
    backend: memory  # or 'file', 'qdrant'
    path: ./data
  embedding:
    model: bge-m3
  pipeline:
    mode: rag  # or 'zero-shot'
    retrieval: hybrid  # or 'vector', 'bm25'
    rerank: true
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "config.yaml"


@dataclass
class Config:
    """Parsed configuration with dotted-key access.

    Stores raw YAML structure (nested dicts) and provides convenient
    getters for deeply-nested keys using dot notation.

    Example:
        cfg = load_config()
        backend = cfg.get("storage.backend")  # "memory"
        retrieval_mode = cfg.get("pipeline.retrieval", default="vector")
    """

    data: dict[str, Any] = field(default_factory=dict)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve config value by dotted path (e.g. 'storage.backend.path').

        Args:
            dotted_key: Dot-separated path into nested dict structure.
            default: Fallback value if key not found.

        Returns:
            Config value, or default if key does not exist.
        """
        node: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def backend(self) -> str:
        """Storage backend name: 'memory', 'file', or 'qdrant'."""
        return self.get("storage.backend", "memory")

    @property
    def mode(self) -> str:
        """Pipeline mode: 'rag' or 'zero-shot'."""
        return self.get("pipeline.mode", "rag")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and parse a YAML configuration file.

    Args:
        path: Path to config.yaml (default: 'config.yaml' in cwd).

    Returns:
        Config object with dotted-key access.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config is not a top-level mapping (dict).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            "Config must be a top-level mapping (key: value). "
            f"Got {type(raw).__name__} instead."
        )
    return Config(raw)
