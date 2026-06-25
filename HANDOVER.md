# tau-bench 專案交接文件

## 交接範圍

本專案是 `sierra-research/tau-bench` 的本地 fork，保留 upstream 完整 git
history，並加入本地繁體中文 benchmark、斷點續跑、token 預算管理與多模型實驗結果。

目前 remote 設定：

```text
origin   git@github.com:aaronwu901225main/tau-bench.git
upstream https://github.com/sierra-research/tau-bench.git
```

目前分支：

```text
main -> origin/main
```

## 最近本地修改脈絡

本 fork 從 upstream `59a200c` 之後加入本地 commit：

- `d65e850 初測英文版成功`
  - 新增 `run_with_budget.py`、`token_budget_manager.py`
  - 新增 LiteLLM retry wrapper
  - 調整 benchmark run loop、temperature、checkpoint 與初步英文測試流程
- `fe39c08 繁體中文化與部分測試`
  - 新增 `--locale`
  - 新增 `tau_bench/localization.py`
  - 新增 `tau_bench/locales/zh-TW/` 的 airline/retail 任務、工具、wiki、rules、shared prompts
  - 新增多個 gpt-oss、Llama 系列英文/繁中 checkpoint 與 summary CSV
- `b4c4f19 Gemma result`
  - 新增 Gemma-4-31B-it 與部分 gpt-oss / Qwen3.5 繁中結果
  - 將 `results/gpt-oss-20b.csv` rename 成 `results/Gpt-oss-20b.csv`
- `a061b81 Gemma 4 all and Qwen3.5 result`
  - 新增 Gemma-4-26B-A4B-it、Gemma-4-31B-it、Qwen3.5-35B-A3B 的英文/繁中 airline/retail checkpoint 與 summary CSV

本次交接整理新增或調整：

- 新增 `pyproject.toml`，把 packaging metadata 從 `setup.py` 移到 PEP 621 格式
- 刪除舊 `setup.py`，避免 dependency metadata 有兩份來源
- 更新 `MANIFEST.in`，補上 locale shared prompt 的 `*.txt`
- 重寫 `README.md` 的環境安裝、API key、執行、resume、budget wrapper 與結果位置說明
- 新增本文件 `HANDOVER.md`

## 已完成項目

- 保留完整 git history 與 upstream remote。
- 完成 English / Traditional Chinese (`en`, `zh-TW`) locale 參數化。
- 完成繁中 airline/retail 任務、工具 schema、wiki、rules、user simulator prompts、Act/ReAct prompts。
- 完成 benchmark resume 行為：可從既有 checkpoint JSON 載入已完成 `(task_id, trial)`，並跳過已完成項目。
- 完成 checkpoint naming：resume 模式使用穩定檔名，非 resume 模式使用 timestamp。
- 完成 LiteLLM retry wrapper：處理 rate limit、timeout、暫時性 5xx、header contamination 等可重試錯誤。
- 完成 OpenAI token budget manager：支援多把 `OPENAI_API_KEYS` 輪替、每日 token limit、`.token_usage.json` 持久化。
- 完成多模型結果整理：`gpt-oss-20b`、`Llama-3.1-8B-Instruct`、`Llama-xLAM-2-8b-fc-r`、`Gemma-4-26B-A4B-it`、`Gemma-4-31B-it`、`Qwen3.5-35B-A3B`。
- 完成 dependency audit，補齊 `setup.py` 原本沒列出的直接 import：`pydantic`、`requests`、`tiktoken`、`tqdm`。
- 驗證 CLI 載入時發現目前 LiteLLM 版本還需要 `tokenizers`，已明列於 `pyproject.toml`，避免半安裝環境缺 transitive dependency。

## 重要檔案

- `README.md`
  - 接手者的主要安裝與執行入口。
- `pyproject.toml`
  - Python packaging 與 runtime dependencies 的單一來源。
- `MANIFEST.in`
  - source distribution package data 規則；目前包含 `*.json`、`*.md`、`*.txt`。
- `run.py`
  - CLI 參數解析；包含 `--locale`、`--resume`、`--task-ids`、`--temperature` 等。
- `tau_bench/run.py`
  - benchmark 主流程、parallel task execution、checkpoint 寫入、resume 載入、pass^k metrics。
- `tau_bench/types.py`
  - Pydantic data models；需要 `pydantic>=2.5`。
- `tau_bench/envs/`
  - airline/retail domain、tool implementations、task data、user simulator。
- `tau_bench/localization.py`
  - 套用 file-based locale override 的邏輯。
- `tau_bench/locales/zh-TW/`
  - 繁中任務、工具、wiki、rules、shared prompts。
- `tau_bench/litellm_retry.py`
  - LiteLLM completion retry helper。
- `run_with_budget.py`
  - 具 token 預算管理的 benchmark wrapper。
- `token_budget_manager.py`
  - LiteLLM callback，追蹤 OpenAI token 用量與 API key 輪替。
- `scripts/translate_file_locale.py`
  - 本地化檔案翻譯/修補工具。
- `results/`
  - 已提交的 summary CSV 與 checkpoint trajectories。
- `auto_error_identification.py`
  - LLM 輔助錯誤歸因工具。

## 環境與安裝

建議 Python 3.10+；目前工作區使用 Python 3.11.7 驗證。請使用乾淨 virtual
environment，避免共用 user site 中其他 ML 專案的 dependency pins 互相干擾。

```bash
cd tau-bench
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

本專案不再使用 `setup.py` 或 `requirements.txt` 作為 dependency 來源。請以
`pyproject.toml` 為準。

核心 runtime dependencies：

```text
anthropic
google-generativeai
litellm
mistralai
numpy
openai
pydantic
requests
tenacity
termcolor
tiktoken
tokenizers
tqdm
```

## 常用執行方式

英文 retail：

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

繁中 retail：

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

斷點續跑：

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

token budget wrapper：

```bash
export OPENAI_API_KEYS=sk-key-1,sk-key-2
export TOKEN_LIMIT_PER_KEY=2500000

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

注意：目前 `gpt-5-mini` 在程式中被限制必須使用 `temperature=1`。

## 結果檔案

已提交 summary CSV：

- `results/Gpt-oss-20b.csv`
- `results/Llama-3.1-8B-Instruct.csv`
- `results/Llama-xLAM-2-8b-fc-r.csv`
- `results/Gemma-4-26B-A4B-it.csv`
- `results/Gemma-4-31B-it.csv`
- `results/Qwen3.5-35B-A3B.csv`

checkpoint trajectory 位於：

```text
results/<model>/<env>_<locale>_<strategy>/
```

checkpoint JSON 每筆包含 `task_id`、`reward`、`info`、`traj`、`trial`。

## 交接前檢查清單

1. 執行 `git status --short`，確認哪些變更要一起交接。
2. 執行 `git log --oneline -n 10`，確認接手者看到最新整理 commit。
3. 執行 `python -m pip install -e .`，確認 `pyproject.toml` 可安裝。
4. 執行 `python -m compileall run.py run_with_budget.py token_budget_manager.py tau_bench`，確認語法可編譯。
5. 若要交付 GitHub，commit README / HANDOVER / pyproject / MANIFEST 變更後 push 到 `origin/main`。
6. 若使用 zip/tar 移交，請改用 `git bundle` 或直接移交包含 `.git/` 的完整 repo。

## 已知注意事項

- Upstream README 已提醒原始 airline/retail tasks 不是最新版本；若目標是最新 tau benchmark，應另外評估 tau2/tau3。
- `results/` 內 checkpoint JSON 很大，但目前是本地實驗證據的一部分，交接時不要任意刪除。
- `.token_usage.json`、`.openai_api_keys`、log 檔與 API key 檔不應提交；`.gitignore` 已涵蓋主要本地產物。
- auto error identification 是 LLM 輔助分析，不應當成 deterministic ground truth。
- 若切換模型 provider，請確認 LiteLLM 對應的 provider 名稱與環境變數。
