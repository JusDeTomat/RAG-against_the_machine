*This project has been created as part of the 42 curriculum by mbichet.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about the
vLLM codebase (v0.10.1). A language model only knows what it was trained on,
so instead of putting the codebase *into* the model, the system lets the model
look it up at answer time:

1. **Indexing**: the repository is split into chunks and a BM25 index is built.
2. **Retrieving**: for a question, the most relevant chunks are found.
3. **Augmenting**: their text is placed in the prompt of the model.
4. **Generating**: `Qwen/Qwen3-0.6B`, running locally on CPU, writes a grounded
   answer.

Retrieval quality is measured with recall@k. The system is judged on whether
it retrieves the right source locations (file path + character range) and on
whether its answers are grounded in them.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/). About 2 GB of
free disk space are needed for the model weights.

```bash
make install
make install-dataset
```

Expected layout (the `data/` folder is not versioned):

```
data/raw/vllm-0.10.1/                       indexed sources
data/processed/                             index built by `index`
data/datasets/UnansweredQuestions/*.json    questions
data/datasets/AnsweredQuestions/*.json      questions + ground truth
data/output/search_results/<scope>/         output of `search_dataset`
data/output/search_results_and_answer/<scope>/   output of `answer_dataset`
```

If your home directory is small, point the Hugging Face cache to a bigger
location before the first run, for example
`export HF_HOME=/path/with/space/hf`.

Commands (all run through `uv run python -m src <command>`, or
`make run ARGS="<command> ..."`):

| Command | Purpose |
|---------|---------|
| `index --max_chunk_size 2000` | chunk `data/raw/` and build the index in `data/processed/` |
| `search <query> --k 5` | print the top-k sources of one query |
| `search_dataset --dataset_path P --k 10 --save_directory D` | search a whole dataset, write a `StudentSearchResults` JSON |
| `answer <query> --k 5` | retrieve, then generate an answer for one query |
| `answer_dataset --student_search_results_path P --save_directory D` | generate answers for a whole dataset |
| `evaluate --student_search_results_path P --dataset_path G` | compute recall@1/3/5/10 against a ground truth |

Other Makefile rules: `run`, `debug` (pdb), `clean`, `fclean`, `lint`,
`lint-strict`.

## System architecture

```
data/raw  ->  Chunk (chuncking.py)  ->  tokenize + BM25Okapi (indexer.py)
                                              |
                                      data/processed/
                                      chunks.json + bm25.pkl
                                              |
question -> tokenize -> Retriever.search (retriever.py) -> top-k sources
                                              |
              Generator (generator.py): re-read the source text,
              build the prompt, run Qwen3-0.6B -> answer
```

| Module | Role |
|--------|------|
| `chuncking.py` | `Chunk`: splits a file into chunks and records their character range |
| `search_file.py` | lists every file of a folder |
| `indexer.py` | `tokenize` and the `index` command (chunks + BM25) |
| `retriever.py` | `Retriever` (loads the index, ranks chunks) and the `search` command |
| `dataset.py` | `search_dataset`, `evaluate`, IoU and recall@k |
| `generator.py` | `Generator` (Qwen3-0.6B) and the `answer` / `answer_dataset` commands |
| `models.py` | pydantic models exchanged between the stages |
| `__main__.py` | Python Fire command-line interface |

A retrieved source is only a `MinimalSource` (`file_path`,
`first_character_index`, `last_character_index`). The text is not stored in the
output: the generator re-reads it from the original file with the character
range, so `answer_dataset` only needs the search results file.

## Chunking strategy

Two strategies are implemented, selected by file extension, both built on
LangChain's `RecursiveCharacterTextSplitter`:

- **Python (`.py`)**: `Language.PYTHON` separators, so that cuts happen
  preferably between classes and functions, then between lines.
- **Markdown (`.md`)**: `Language.MARKDOWN` separators, so that cuts happen
  preferably at headings, code blocks and paragraphs.

Other file types are ignored.

- `--max_chunk_size` defaults to 2000 characters and cannot exceed it (the
  grader rejects longer sources). The overlap is `min(200, size // 10)`, so
  that an idea cut between two chunks stays readable in both.
- `add_start_index=True` gives the position of each chunk in its file; the end
  is `start + len(chunk)` (end exclusive).
- Files are read with `newline=""`, so `\r\n` is not converted to `\n`.
  Otherwise the indexes would be shifted compared to the real file.
- `file_path` is kept exactly as found under `data/raw/`, because the grader
  compares paths verbatim.

## Retrieval method

- **Algorithm**: BM25 (`BM25Okapi` from `rank_bm25`), a classic lexical
  ranking. Scores are computed against every chunk and the `k` best are
  returned with `numpy.argsort`. The index is persisted with `pickle`.
- **Tokenization** (same function for chunks and questions): words and
  identifiers are lowercased; `snake_case` and `camelCase` identifiers are
  kept whole **and** split into parts (`max_chunk_size` gives `max_chunk_size`,
  `max`, `chunk`, `size`). A question can therefore quote an identifier or
  paraphrase it. Common English stop words are removed.
- **File path indexing**: the tokens of the file path are added to the tokens
  of each chunk, because the file name often contains the words of the
  question (for example `docs/serving/openai_compatible_server.md`).
- **Parameters**: `k1` and `b` are options of `index` (defaults 1.5 and 0.75).
  TODO: write the values you finally chose.

## Answer generation

- Model: `Qwen/Qwen3-0.6B` through `transformers`, on CPU, no external API.
- The text of the retrieved sources is concatenated (each one preceded by its
  path) within a budget of 3000 tokens.
- The prompt asks the model to use only the context, to prefer the
  documentation over example code when they differ, and to say so when the
  context does not contain the answer. Thinking mode is disabled
  (`enable_thinking=False`) and decoding is greedy (`do_sample=False`, at most
  256 new tokens), so answers are deterministic.

## Performance analysis

TODO: fill this section with your own measurements (the targets come from the
subject).

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Target @5 |
|---------|----------|----------|----------|-----------|-----------|
| docs    | 67 %     | 80 %     | 84 %     | 89 %      | 80 %      |
| code    | 47,5 %     | 74,7 %     | 78,8 %     |78,8 %      | 50 %      |


## Design decisions

- **BM25 rather than TF-IDF**: it takes the length of the chunks into account.
  Its `b` parameter turned out to matter here, because short example scripts
  were ranked above long documentation pages.
- **Identifiers split and kept whole**: a question may quote an identifier
  verbatim or describe it with natural words.
- **Character ranges instead of storing the text**: the output format only
  needs `file_path` and the range, and the text can always be re-read.
- **Only `.py` and `.md` files are indexed**: the questions are about the
  documentation and the code. TODO: confirm by testing other extensions.
- **Everything local**: the model runs on CPU with no API, as required.
- **Errors never crash the CLI**: empty query, `k=0`, missing file or index and
  invalid JSON print a message instead of a traceback.

## Challenges faced

- **Wrong file ranked first**: for the question about configuring the OpenAI
  server, short client examples (`examples/online_serving/openai_*.py`) ranked
  above the documentation page. Lowering `b` and indexing the file path moved
  the documentation up in the ranking. TODO: add the measured effect on recall.
- **Character offsets**: indexes were only reliable once files were read with
  `newline=""` and checked with an assertion
  (`content[start:end] == chunk`).
- **Disk space**: the model weights (about 1.5 GB) did not fit in a small home
  directory and the download failed with a misleading error. Solved by moving
  the Hugging Face cache with `HF_HOME`.
- **Static typing**: `mypy` rejects the `generate` call of `transformers` for
  a typing reason inside the library. Solved by annotating the model as `Any`.
- TODO: add your own difficulties.

## Example usage

```bash
make install
make install-dataset
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

## Resources

- vLLM documentation: https://docs.vllm.ai
- LangChain text splitters: https://python.langchain.com/docs/concepts/text_splitters/
- BM25 (Okapi BM25 ranking function): https://en.wikipedia.org/wiki/Okapi_BM25
- `rank_bm25` library: https://github.com/dorianbrown/rank_bm25
- Qwen3 model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
- Python Fire: https://github.com/google/python-fire
- Pydantic: https://docs.pydantic.dev

### Use of AI

An AI assistant (Claude) was used during this project for:

- explaining the RAG pipeline and the LangChain / BM25 / Transformers APIs;
- create the Readme
- debugging (typing errors reported by `mypy`, disk space, command-line
  arguments) and suggesting ways to improve recall (`b`, stop words, file
  path tokens).
