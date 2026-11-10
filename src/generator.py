import json
import pickle
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retriever import Retriever

MODEL_NAME = "Qwen/Qwen3-0.6B"
SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase. "
    "Use only the provided context. "
    "The context may mix documentation and example code: "
    "prefer the documentation when they differ. "
    "Answer in a few sentences, citing the concrete options, commands "
    "or parameters that appear in the context. "
    "If the context does not contain the answer, say so briefly."
)


class Generator:

    def __init__(self, model_name: str = MODEL_NAME,
                 max_context_tokens: int = 3000,
                 max_new_tokens: int = 256) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self._files: dict[str, str] = {}

    def read_snippet(self, source: MinimalSource) -> str:
        path = source.file_path
        if path not in self._files:
            try:
                with open(path, encoding="utf-8", newline="") as f:
                    self._files[path] = f.read()
            except (OSError, UnicodeDecodeError):
                self._files[path] = ""
        return self._files[path][source.first_character_index:
                                 source.last_character_index]

    def build_context(self, sources: list[MinimalSource]) -> str:
        parts: list[str] = []
        budget = self.max_context_tokens
        for source in sources:
            text = self.read_snippet(source)
            if not text:
                continue
            ids = self.tokenizer.encode(text)
            if len(ids) > budget:
                text = str(self.tokenizer.decode(ids[:budget]))
            parts.append(f"[{source.file_path}]\n{text}")
            budget -= min(len(ids), budget)
            if budget <= 0:
                break
        return "\n\n".join(parts)

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        import torch

        context = self.build_context(sources)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": (f"Question: {question}\n\n"
                         f"Context:\n{context}\n\n"
                         f"Using the context above, answer this question: "
                         f"{question}")},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return str(self.tokenizer.decode(new_tokens,
                                         skip_special_tokens=True)).strip()


def answer(query: str, k: int = 5,
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
    sources = retriever.search(str(query), k)
    try:
        generator = Generator()
        text = generator.answer(str(query), sources)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Error: generation failed ({e})")
        return
    print("\nAnswer:")
    print(text)
    print("\nSources:")
    for s in sources:
        print(f"  {s.file_path} "
              f"[{s.first_character_index}:{s.last_character_index}]")


def answer_dataset(student_search_results_path: str,
                   save_directory: str =
                   "data/output/search_results_and_answer") -> None:
    try:
        with open(student_search_results_path, encoding="utf-8") as f:
            student = StudentSearchResults.model_validate_json(f.read())
    except (OSError, ValueError) as e:
        print(f"Error: cannot read results "
              f"'{student_search_results_path}' ({e})")
        return
    try:
        generator = Generator()
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Error: cannot load the model ({e})")
        return

    print(f"Loaded {len(student.search_results)} questions")
    answers: list[MinimalAnswer] = []
    for result in tqdm(student.search_results, desc="Answering"):
        try:
            text = generator.answer(result.question,
                                    result.retrieved_sources)
        except (RuntimeError, ValueError) as e:
            text = f"Error while generating the answer: {e}"
        answers.append(MinimalAnswer(
            question_id=result.question_id,
            question=result.question,
            retrieved_sources=result.retrieved_sources,
            answer=text,
        ))

    output = StudentSearchResultsAndAnswer(search_results=answers,
                                           k=student.k)
    out_file = Path(save_directory) / Path(student_search_results_path).name
    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(output.model_dump_json(indent=2),
                            encoding="utf-8")
    except OSError as e:
        print(f"Error: cannot write '{out_file}' ({e})")
        return
    print(f"Saved student_search_results_and_answer to {out_file}")
