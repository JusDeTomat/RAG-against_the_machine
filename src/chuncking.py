from pathlib import Path
from typing import Any

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

MAX_CHUNK_SIZE = 2000


class Chunk:

    def __init__(self, chunk_size: int = MAX_CHUNK_SIZE,
                 chunk_overlap: int = 200) -> None:
        if chunk_size > MAX_CHUNK_SIZE:
            raise ValueError(f"chunk_size must be <= {MAX_CHUNK_SIZE}")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitters = {
            ".py": RecursiveCharacterTextSplitter.from_language(
                Language.PYTHON,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            ),
            ".md": RecursiveCharacterTextSplitter.from_language(
                Language.MARKDOWN,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            ),
        }

    def read_file(self, path_file: str) -> str:
        with open(path_file, encoding="utf-8", newline="") as f:
            return f.read()

    def create_chunk(self, path_file: str) -> list[dict[str, Any]]:
        splitter = self.splitters.get(Path(path_file).suffix)
        if splitter is None:
            return []
        try:
            content = self.read_file(path_file)
        except (OSError, UnicodeDecodeError):
            return []

        docs = splitter.create_documents([content])
        chunks: list[dict[str, Any]] = []
        for doc in docs:
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            chunks.append({
                "file_path": path_file,
                "first_character_index": start,
                "last_character_index": end,
                "text": doc.page_content,
            })
        return chunks
