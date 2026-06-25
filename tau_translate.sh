#!/usr/bin/env bash

set -eo pipefail

if [[ "${1:-}" == "--bg" ]]; then
  mkdir -p logs
  TS=$(date +%Y%m%d_%H%M%S)
  LOG_FILE="logs/tau_translate_${TS}.log"
  nohup bash "$0" --run >"${LOG_FILE}" 2>&1 &
  PID=$!
  echo "${PID}" > logs/tau_translate.pid
  echo "✅ 翻譯流程已在背景啟動"
  echo "PID: ${PID}"
  echo "LOG: ${LOG_FILE}"
  exit 0
fi

if [[ "${1:-}" != "--run" && -n "${1:-}" ]]; then
  echo "用法:"
  echo "  bash tau_translate.sh        # 前景執行"
  echo "  bash tau_translate.sh --bg   # 背景執行"
  exit 1
fi

# conda activate 腳本可能引用未設定變數，activate 完再開啟 nounset
set -u

# ── 設定 ────────────────────────────────────────────────────
WORKSPACE=""
TAU_DIR="${WORKSPACE}/tau-bench"
TRANSLATE_MODEL="gpt-5-mini"
TARGET_LOCALE="zh-TW"

# ── OpenAI API keys（翻譯用）────────────────────────────────
# 多把 key 用逗號分隔
# 快達上限時自動切換下一把，全部用完時中止
OPENAI_API_KEYS=""
export TOKEN_LIMIT_PER_KEY=2500000

cd "${TAU_DIR}"
export PYTHONPATH="${TAU_DIR}:${PYTHONPATH:-}"
export TRANSLATE_MODEL="${TRANSLATE_MODEL:-gpt-5-mini}"
export MIN_TOKENS="${MIN_TOKENS:-50000}"

echo "🌐 開始產生 locale: ${TARGET_LOCALE}（補翻未完成條目）"

# 執行翻譯腳本
python -u scripts/translate_file_locale.py \
  --locale "${TARGET_LOCALE}" \
  --envs retail airline

echo "✅ 翻譯完成"
echo "📁 輸出目錄: ${TAU_DIR}/tau_bench/locales/${TARGET_LOCALE}/<env>/"
