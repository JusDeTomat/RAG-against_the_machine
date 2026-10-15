import json
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.chuncking import Chunk
from src.search_file import take_all_files_paths


def tokenize(text: str) -> list[str]:
    """Découpe un texte en tokens, en séparant aussi snake_case et camelCase.

    "max_chunk_size" donne : max_chunk_size, max, chunk, size
    """
    tokens: list[str] = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
        tokens.append(word.lower())
        parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+",
                           word.replace("_", " "))
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
    return tokens


def index(max_chunk_size: int = 2000,
          raw_dir: str = "data/raw",
          output_dir: str = "data/processed") -> None:
    """Chunke le corpus, construit l'index BM25 et le sauvegarde."""
    if not Path(raw_dir).is_dir():
        print(f"Error: folder not found: {raw_dir}")
        return
    try:
        chunker = Chunk(chunk_size=max_chunk_size,
                        chunk_overlap=min(200, max_chunk_size // 10))
    except ValueError as e:
        print(f"Error: {e}")
        return

    chunks: list[dict[str, Any]] = []
    for file in tqdm(take_all_files_paths(raw_dir), desc="Chunking"):
        chunks.extend(chunker.create_chunk(file))

    tokenized = [tokenize(c["text"]) for c in tqdm(chunks, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    with open(out / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"Ingestion complete! Indexed {len(chunks)} chunks under {out}/")
