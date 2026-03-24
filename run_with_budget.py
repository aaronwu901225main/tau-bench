#!/usr/bin/env python3
"""
帶有 Token 預算管理的 tau-bench 執行器。
在 run.py 之上加入 API key 輪替與 token 用量追蹤。
"""

import sys
import signal
import time
import os

# 在所有 litellm import 之前先設定 budget manager
from token_budget_manager import setup_budget_manager

# 初始化 budget manager（會自動從 OPENAI_API_KEYS 環境變數讀取 key）
budget_manager = setup_budget_manager()

# 設定 litellm 的重試機制，避免 RateLimitError 直接導致失敗
import litellm
litellm.num_retries = 5              # 最多重試 5 次
litellm.retry_after = 2              # 重試間隔基礎秒數（會指數退避）
litellm.request_timeout = 120        # 單次請求超時 120 秒

# 處理 SIGTERM 以便優雅退出
def handle_sigterm(signum, frame):
    print(budget_manager.get_summary())
    print("\n⛔ 收到終止信號，結束執行。")
    if budget_manager.is_exhausted:
        sys.exit(42)
    sys.exit(1)

signal.signal(signal.SIGTERM, handle_sigterm)

# 引入原本的 run.py 邏輯
from tau_bench.types import RunConfig
from tau_bench.run import run
from run import parse_args
from tau_bench.litellm_retry import is_retryable_litellm_error


def main():
    try:
        config = parse_args()
        max_run_attempts = max(1, int(os.getenv("TAU_RUN_MAX_ATTEMPTS", "3")))
        attempt = 1
        while True:
            try:
                results = run(config)
                break
            except SystemExit:
                raise
            except Exception as run_exc:
                if attempt >= max_run_attempts or not is_retryable_litellm_error(run_exc):
                    raise
                sleep_seconds = min(30, 2 ** (attempt - 1))
                print(
                    f"[RunRetry] attempt {attempt}/{max_run_attempts} failed with "
                    f"{run_exc.__class__.__name__}: {run_exc}. "
                    f"retrying full run in {sleep_seconds}s..."
                )
                time.sleep(sleep_seconds)
                attempt += 1
    except SystemExit as e:
        print(budget_manager.get_summary())
        # 如果是 key 用完導致的 SIGTERM，用 exit code 42 標記
        if budget_manager.is_exhausted:
            sys.exit(42)
        raise
    except Exception as e:
        print(budget_manager.get_summary())
        print(f"\n❌ 執行錯誤: {e}")
        # 區分 key 用完 (42) 和一般錯誤 (1)
        if budget_manager.is_exhausted:
            sys.exit(42)
        sys.exit(1)

    # 正常結束時印出摘要
    print(budget_manager.get_summary())


if __name__ == "__main__":
    main()
