#!/usr/bin/env python3
r"""
embedding_cache.py – Precomputed embedding storage for search groups.

This module owns all saving and loading of precomputed embedding matrices.
The search index builds these matrices once and saves them here.
The search engine loads them at startup, avoiding on‑the‑fly embedding calls.

Groups are stored under:
    Explore/data/embeddings/<group_name>.npz

Each .npz contains:
    terms_json   : JSON string of the term list (strings)
    embeddings   : normalized embedding matrix (float32)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
EXPLORE_DIR = Path(__file__).parent
EMBEDDINGS_DIR = EXPLORE_DIR / "data" / "embeddings"

# Valid group names
VALID_GROUPS = {
    "component_attribute",
    "component_value",
    "token_attribute",
    "token_value",
    "token_object",
    "token_action",
    "verb",
    "object_names",
    "action_names",
}


class EmbeddingCache:
    """Handles saving and loading of precomputed embedding groups."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or EMBEDDINGS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _group_path(self, group_name: str) -> Path:
        """Return the .npz file path for a group."""
        if group_name not in VALID_GROUPS:
            raise ValueError(f"Unknown embedding group: {group_name}")
        return self.base_dir / f"{group_name}.npz"

    def save_group(self, group_name: str, terms: List[str], embeddings: np.ndarray):
        """
        Save one group's terms and embedding matrix.

        Args:
            group_name: One of the valid group names.
            terms: List of term strings, in same order as embedding rows.
            embeddings: Normalized embedding matrix (num_terms x dim).
        """
        if group_name not in VALID_GROUPS:
            raise ValueError(f"Unknown embedding group: {group_name}")

        if not isinstance(terms, list):
            raise TypeError("terms must be a list of strings")

        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D matrix")

        if len(terms) != embeddings.shape[0]:
            raise ValueError(
                f"Number of terms ({len(terms)}) does not match "
                f"embedding rows ({embeddings.shape[0]})"
            )

        terms_json = json.dumps(terms, ensure_ascii=False)
        path = self._group_path(group_name)

        np.savez_compressed(
            path,
            terms_json=terms_json,
            embeddings=embeddings,
        )

    def load_group(self, group_name: str) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Load one group's terms and embedding matrix.

        Returns:
            (terms, embeddings) if the file exists and is valid, else None.
        """
        if group_name not in VALID_GROUPS:
            raise ValueError(f"Unknown embedding group: {group_name}")

        path = self._group_path(group_name)
        if not path.exists():
            return None

        try:
            data = np.load(path, allow_pickle=False)
            terms_json = str(data["terms_json"])
            embeddings = np.asarray(data["embeddings"], dtype="float32")

            terms = json.loads(terms_json)
            if not isinstance(terms, list):
                return None

            if len(terms) != embeddings.shape[0]:
                return None

            return terms, embeddings

        except Exception:
            return None

    def has_cache(self, group_name: Optional[str] = None) -> bool:
        """
        Return True if a group's cache exists, or if all group caches exist
        when group_name is None.
        """
        if group_name is not None:
            return self._group_path(group_name).exists()

        return all(self._group_path(name).exists() for name in VALID_GROUPS)

    def clear_cache(self, group_name: Optional[str] = None):
        """
        Remove one group's cache file, or all group files if group_name is None.
        """
        if group_name is not None:
            path = self._group_path(group_name)
            if path.exists():
                path.unlink()
            return

        for name in VALID_GROUPS:
            path = self._group_path(name)
            if path.exists():
                path.unlink()

    def list_cached_groups(self) -> List[str]:
        """Return names of groups that currently have cache files."""
        return [
            name for name in VALID_GROUPS
            if self._group_path(name).exists()
        ]