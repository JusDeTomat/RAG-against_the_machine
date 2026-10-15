import fire

from src.indexer import index
from src.retriever import search

if __name__ == "__main__":
    fire.Fire({"index": index, "search": search})
