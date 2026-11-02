import pickle
from pathlib import Path

from tqdm import tqdm

from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.retriever import Retriever

IOU_THRESHOLD = 0.05


def load_dataset(path: str) -> RagDataset | None:
    try:
        with open(path, encoding="utf-8") as f:
            return RagDataset.model_validate_json(f.read())
    except (OSError, ValueError) as e:
        print(f"Error: cannot read dataset '{path}' ({e})")
        return None


def search_dataset(dataset_path: str,
                   k: int = 10,
                   save_directory: str = "data/output/search_results",
                   processed_dir: str = "data/processed") -> None:
    if k <= 0:
        print("Error: k must be > 0")
        return
    dataset = load_dataset(dataset_path)
    if dataset is None:
        return
    try:
        retriever = Retriever(processed_dir)
    except (OSError, ValueError, pickle.UnpicklingError) as e:
        print(f"Error: cannot load the index ({e}). Run 'index' first.")
        return

    results: list[MinimalSearchResults] = []
    for q in tqdm(dataset.rag_questions, desc="Searching"):
        results.append(MinimalSearchResults(
            question_id=q.question_id,
            question=q.question,
            retrieved_sources=retriever.search(q.question, k),
        ))

    output = StudentSearchResults(search_results=results, k=k)
    out_file = Path(save_directory) / Path(dataset_path).name
    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(output.model_dump_json(indent=2),
                            encoding="utf-8")
    except OSError as e:
        print(f"Error: cannot write '{out_file}' ({e})")
        return
    print(f"Saved student_search_results to {out_file}")


def _iou(a: MinimalSource, b: MinimalSource) -> float:
    if a.file_path != b.file_path:
        return 0.0
    inter = max(0, min(a.last_character_index, b.last_character_index)
                - max(a.first_character_index, b.first_character_index))
    union = ((a.last_character_index - a.first_character_index)
             + (b.last_character_index - b.first_character_index) - inter)
    return inter / union if union > 0 else 0.0


def _recall_at_k(retrieved: dict[str, list[MinimalSource]],
                 truth: dict[str, list[MinimalSource]],
                 k: int) -> float:
    scores: list[float] = []
    for question_id, expected in truth.items():
        if not expected:
            continue
        top = retrieved.get(question_id, [])[:k]
        found = sum(
            1 for source in expected
            if any(_iou(source, r) >= IOU_THRESHOLD for r in top)
        )
        scores.append(found / len(expected))
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    try:
        with open(student_search_results_path, encoding="utf-8") as f:
            student = StudentSearchResults.model_validate_json(f.read())
    except (OSError, ValueError) as e:
        print(f"Error: cannot read results '{student_search_results_path}' "
              f"({e})")
        return
    dataset = load_dataset(dataset_path)
    if dataset is None:
        return

    truth = {q.question_id: q.sources for q in dataset.rag_questions
             if isinstance(q, AnsweredQuestion)}
    if not truth:
        print("Error: the dataset has no answered questions (no sources)")
        return
    retrieved = {r.question_id: r.retrieved_sources
                 for r in student.search_results}

    print(f"Questions evaluated: {len(truth)}")
    for k in (1, 3, 5, 10):
        if k <= student.k:
            value = _recall_at_k(retrieved, truth, k)
            print(f"Recall@{k}: {value:.3f} ({value * 100:.1f}%)")