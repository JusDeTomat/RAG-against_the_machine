*This project has been created as part of the 42 curriculum by <login1>.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about the
vLLM codebase (v0.10.1). It indexes the repository, retrieves the most
relevant snippets for a question, and generates a grounded answer with
`Qwen/Qwen3-0.6B`. Retrieval quality is measured with recall@k.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
make install            # uv sync
```

Place the vLLM sources in `data/raw/vllm-0.10.1/` and the question datasets in
`data/datasets/`. Then:

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How to configure the OpenAI server?" --k 5
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
uv run python -m src answer "How to configure the OpenAI server?" --k 5
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Other Makefile rules: `run`, `debug`, `clean`, `lint`, `lint-strict`.

## System architecture

TODO: describe the pipeline (indexing -> retrieval -> augmentation ->
generation) and the role of each module in `src/`:
`chuncking.py`, `indexer.py`, `retriever.py`, `dataset.py`, `generator.py`,
`models.py`, `__main__.py`.

## Chunking strategy

TODO: explain the two strategies (Python code, Markdown), the 2000-character
limit, the overlap, and how `first_character_index` / `last_character_index`
are computed.

## Retrieval method

TODO: explain BM25 (library, parameters), the tokenization (identifiers split
on snake_case and camelCase) and how results are ranked.

## Performance analysis

TODO: fill with your own measurements.

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---------|----------|----------|----------|-----------|
| docs    | TODO     | TODO     | TODO     | TODO      |
| code    | TODO     | TODO     | TODO     | TODO      |

Indexing time: TODO. Retrieval time for 200 questions: TODO.
Effect of `--max_chunk_size` on recall: TODO.

## Design decisions

TODO: explain your key choices and why.

## Challenges faced

TODO: difficulties encountered and how you solved them.

## Example usage

See the commands in the Instructions section.

## Resources

- vLLM documentation: https://docs.vllm.ai
- LangChain text splitters documentation
- rank_bm25 library
- Qwen3 model card on Hugging Face

How AI was used: TODO (state precisely for which tasks and which parts of the
project, and what you verified and rewrote yourself).