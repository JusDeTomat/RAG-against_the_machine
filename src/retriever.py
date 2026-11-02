import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.indexer import tokenize
from src.models import MinimalSource


class Retriever:

    def __init__(self, processed_dir: str = "data/processed") -> None:
        folder = Path(processed_dir)
        with open(folder / "chunks.json", encoding="utf-8") as f:
            self.chunks: list[dict[str, Any]] = json.load(f)
        with open(folder / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

    def search(self, query: str, k: int = 10) -> list[MinimalSource]:
        tokens = tokenize(query)
        if not tokens or k <= 0:
            return []
        scores = self.bm25.get_scores(tokens)
        best = np.argsort(scores)[::-1][:k]
        return [
            MinimalSource(
                file_path=self.chunks[i]["file_path"],
                first_character_index=self.chunks[i]["first_character_index"],
                last_character_index=self.chunks[i]["last_character_index"],
            )
            for i in best
        ]


def search(query: str, k: int = 10,
           processed_dir: str = "data/processed") -> None:
    if not str(query).strip():
        print("Error: empty query")
        return
    if k <= 0:
        print("Error: k must be > 0")
        return
    try:
        retriever = Retriever(processed_dir)
    except (OSError, json.JSONDecodeError, pickle.UnpicklingError) as e:
        print(f"Error: cannot load the index ({e}). Run 'index' first.")
        return
    for s in retriever.search(str(query), k):
        print(f"{s.file_path} "
              f"[{s.first_character_index}:{s.last_character_index}]")
