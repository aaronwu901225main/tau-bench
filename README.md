# tau-bench handover fork

This repository is a handover-ready fork of `sierra-research/tau-bench`. It
keeps the original benchmark code and commit history, plus local changes for
Traditional Chinese evaluation, resumable runs, token-budgeted OpenAI user
simulation, and recorded experiment outputs.

Upstream warning: the original tau-bench tasks are no longer the newest task
set. Sierra Research now points users to
[tau2-bench / tau3-bench](https://github.com/sierra-research/tau2-bench) for the
latest fixed tasks and newer domains. This fork is preserved for the local
airline/retail experiments already run here.

## What this fork adds

- `--locale en|zh-TW` for English and Traditional Chinese benchmark runs.
- File-based Traditional Chinese data under `tau_bench/locales/zh-TW/`.
- Localized user simulator and Act/ReAct prompts in
  `tau_bench/locales/zh-TW/shared/`.
- `--resume` support that loads existing checkpoint JSON files and skips
  completed `(task_id, trial)` pairs.
- Stable checkpoint filenames in resume mode.
- LiteLLM retry handling for retryable API errors and known header parsing
  failures.
- `run_with_budget.py` and `token_budget_manager.py` for OpenAI API key
  rotation and per-key token limits.
- Recorded result CSVs and checkpoint trajectories under `results/`.

## Environment setup

Use Python 3.10 or newer. The current workspace was checked with Python 3.11.7.
Use a clean virtual environment; the shared user site on this machine contains
other ML projects with incompatible dependency pins.

```bash
cd tau-bench
conda create -n tau-bench python=3.11
conda activate tau-bench
python -m pip install --upgrade pip
python -m pip install -e .
```

There is no separate `requirements.txt`. Runtime dependencies are declared in
`pyproject.toml`. The dependency list was migrated from `setup.py` and extended
after scanning the current source imports:

- Existing package requirements kept: `openai`, `mistralai`, `anthropic`,
  `google-generativeai`, `tenacity`, `termcolor`, `numpy`, `litellm`
- Missing direct imports now included: `pydantic`, `requests`, `tiktoken`,
  `tqdm`
- Runtime dependency observed through LiteLLM import: `tokenizers`

## API keys

Set the key for whichever provider you run through LiteLLM.

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export MISTRAL_API_KEY=...
```

For token-budgeted OpenAI user simulation:

```bash
export OPENAI_API_KEYS=sk-key-1,sk-key-2
export TOKEN_LIMIT_PER_KEY=2500000
export TAU_RUN_MAX_ATTEMPTS=3
```

Set `TOKEN_LIMIT_PER_KEY=-1` to keep token accounting enabled without enforcing
a per-key token limit.

`run_with_budget.py` also accepts a `.openai_api_keys` file in the repository
root, one key per line. The token usage state is written to
`.token_usage.json`, which is ignored by git.

## Run benchmark

Basic English retail run:

```bash
python run.py \
  --agent-strategy tool-calling \
  --env retail \
  --model gpt-4o \
  --model-provider openai \
  --user-model gpt-5-mini \
  --user-model-provider openai \
  --user-strategy llm \
  --temperature 1 \
  --locale en \
  --max-concurrency 10
```

Traditional Chinese retail run:

```bash
python run.py \
  --agent-strategy tool-calling \
  --env retail \
  --model gpt-4o \
  --model-provider openai \
  --user-model gpt-5-mini \
  --user-model-provider openai \
  --user-strategy llm \
  --temperature 1 \
  --locale zh-TW \
  --max-concurrency 10
```

Airline uses the same flags with `--env airline`.

Run only selected tasks:

```bash
python run.py \
  --agent-strategy tool-calling \
  --env retail \
  --model gpt-4o \
  --model-provider openai \
  --user-model gpt-5-mini \
  --user-model-provider openai \
  --user-strategy llm \
  --temperature 1 \
  --locale zh-TW \
  --task-ids 2 4 6
```

Resume an interrupted run and skip completed tasks:

```bash
python run.py \
  --agent-strategy tool-calling \
  --env retail \
  --model gpt-4o \
  --model-provider openai \
  --user-model gpt-5-mini \
  --user-model-provider openai \
  --user-strategy llm \
  --temperature 1 \
  --locale zh-TW \
  --log-dir results/my-run \
  --resume
```

Use the budget manager wrapper when using multiple OpenAI user-simulator keys:

```bash
python run_with_budget.py \
  --agent-strategy tool-calling \
  --env retail \
  --model gpt-4o \
  --model-provider openai \
  --user-model gpt-5-mini \
  --user-model-provider openai \
  --user-strategy llm \
  --temperature 1 \
  --locale zh-TW \
  --max-concurrency 10 \
  --resume
```

Note: `gpt-5-mini` is guarded in `run.py` and `tau_bench/run.py`; it must run
with `--temperature 1`.

## Output layout

Benchmark runs write checkpoint JSON files to `--log-dir`:

```text
results/<model>/<env>_<locale>_<strategy>/
```

Each checkpoint item includes:

- `task_id`
- `reward`
- `info`
- `traj`
- `trial`

The committed summary CSVs are:

- `results/Gpt-oss-20b.csv`
- `results/Llama-3.1-8B-Instruct.csv`
- `results/Llama-xLAM-2-8b-fc-r.csv`
- `results/Gemma-4-26B-A4B-it.csv`
- `results/Gemma-4-31B-it.csv`
- `results/Qwen3.5-35B-A3B.csv`

## Auto error identification

```bash
python auto_error_identification.py \
  --env retail \
  --platform openai \
  --model gpt-4o \
  --results-path results/path/to/checkpoint.json \
  --max-concurrency 16 \
  --output-path error-analysis.json \
  --max-num-failed-results 10
```

This feature uses an LLM and should be treated as an assisted analysis tool,
not ground truth.

## Important files

- `run.py`: command-line parser and benchmark entrypoint.
- `tau_bench/run.py`: main benchmark loop, checkpointing, resume behavior, and
  pass^k metric display.
- `tau_bench/envs/`: retail and airline environment definitions, tools, tasks,
  and user simulator.
- `tau_bench/localization.py`: applies file-based locale overrides.
- `tau_bench/locales/zh-TW/`: Traditional Chinese localized tasks, tools,
  wiki, rules, and shared prompts.
- `tau_bench/litellm_retry.py`: retry wrapper for LiteLLM calls.
- `run_with_budget.py`: wrapper around `run.py` with token budget management.
- `token_budget_manager.py`: LiteLLM callback for OpenAI token accounting and
  key rotation.
- `tau.slurm.example`: editable Slurm/vLLM batch-run template.
- `scripts/translate_file_locale.py`: translation utility used to generate or
  repair localized files.
- `results/`: committed summary CSVs and checkpoint trajectories.
- `HANDOVER.md`: project handover notes and completed work summary.

## Original citation

```bibtex
@misc{yao2024tau,
      title={$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains},
      author={Shunyu Yao and Noah Shinn and Pedram Razavi and Karthik Narasimhan},
      year={2024},
      eprint={2406.12045},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2406.12045},
}
@misc{barres2025tau2,
      title={$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment},
      author={Victor Barres and Honghua Dong and Soham Ray and Xujie Si and Karthik Narasimhan},
      year={2025},
      eprint={2506.07982},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.07982},
}
```
