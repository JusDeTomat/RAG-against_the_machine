import fire

from src.dataset import evaluate, search_dataset
from src.generator import answer, answer_dataset
from src.indexer import index
from src.retriever import search

if __name__ == "__main__":
    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "answer": answer,
        "answer_dataset": answer_dataset,
        "evaluate": evaluate,
    })