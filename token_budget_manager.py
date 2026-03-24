"""
OpenAI API Key 輪替與 Token 預算管理器。
透過 litellm callback 追蹤每個 API key 的 token 使用量，
在達到每日上限時自動切換到下一個 key，
所有 key 額度用完時中止程式以避免計費。

支援持久化：將用量儲存到 JSON 檔案，下次啟動時恢復，
避免因 job 重啟而重置已使用的額度。
"""

import os
import sys
import json
import signal
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

from litellm.integrations.custom_logger import CustomLogger

# 持久化檔案位置
DEFAULT_USAGE_FILE = os.path.join(os.path.dirname(__file__), ".token_usage.json")
DEFAULT_TOKEN_LIMIT_PER_KEY = 2_500_000


class TokenBudgetManager(CustomLogger):
    """litellm callback，追蹤 OpenAI token 用量並自動輪替 API key。"""

    def __init__(
        self,
        api_keys: List[str],
        token_limit_per_key: int = DEFAULT_TOKEN_LIMIT_PER_KEY,
        usage_file: str = DEFAULT_USAGE_FILE,
    ) -> None:
        super().__init__()
        if not api_keys:
            raise ValueError("至少需要提供一個 API key")
        self.api_keys = api_keys
        self.token_limit = token_limit_per_key
        self.usage_file = usage_file
        self.current_key_idx = 0
        self.usage_per_key: dict[str, int] = {k: 0 for k in api_keys}
        self.total_openai_tokens = 0
        self._lock = threading.Lock()
        self._exhausted = False

        # 從檔案恢復上次的用量
        self._load_usage()

        # 找到第一把還有額度的 key
        while (self.current_key_idx < len(self.api_keys) and
               self.usage_per_key[self.api_keys[self.current_key_idx]] >= self.token_limit):
            self.current_key_idx += 1

        if self.current_key_idx >= len(self.api_keys):
            self._exhausted = True
            print("🚨 所有 API key 的 token 額度已在上次執行中用完！")
        else:
            # 設定第一把可用的 key
            os.environ["OPENAI_API_KEY"] = self.api_keys[self.current_key_idx]
            self._print_status("初始化完成")

    @property
    def is_exhausted(self) -> bool:
        return self._exhausted

    def _mask_key(self, key: str) -> str:
        if len(key) <= 12:
            return key[:4] + "****"
        return key[:8] + "..." + key[-4:]

    @staticmethod
    def _budget_date() -> str:
        """取得目前的『預算日期』。以每天早上 8:00 為分界：
        - 08:00 之後 → 今天
        - 08:00 之前 → 昨天（仍屬於前一天的額度週期）
        OpenAI 免費額度大約在每日 08:00 UTC+8 前後重置。
        """
        now = datetime.now()
        if now.hour < 8:
            return (now - timedelta(days=1)).date().isoformat()
        return now.date().isoformat()

    def _load_usage(self) -> None:
        """從 JSON 檔案恢復上次的 token 用量。若預算日期不同則自動重置（每日 08:00 為分界）。"""
        if not os.path.exists(self.usage_file):
            return
        try:
            with open(self.usage_file, "r") as f:
                data = json.load(f)

            # 檢查日期：如果儲存日期不是目前的預算日期，代表新的額度週期
            saved_date = data.get("_date", None)
            budget_today = self._budget_date()
            if saved_date is not None and saved_date != budget_today:
                print(f"[TokenBudget] 🔄 偵測到預算日期變更 ({saved_date} → {budget_today})，重置所有 key 額度")
                # 清除舊的用量檔案，從零開始
                self._save_usage()
                return
            if saved_date is None:
                # 舊格式沒有日期欄位，無法判斷是否同一天
                # 如果所有 key 都已超過上限，很可能是舊的一天，安全起見重置
                all_over = all(data.get(k, 0) >= self.token_limit for k in self.api_keys if k in data)
                if all_over and len([k for k in self.api_keys if k in data]) > 0:
                    print(f"[TokenBudget] 🔄 用量紀錄無日期且所有 key 已滿，視為新的一天，重置額度")
                    self._save_usage()
                    return

            restored = 0
            for key in self.api_keys:
                if key in data:
                    self.usage_per_key[key] = data[key]
                    self.total_openai_tokens += data[key]
                    restored += 1
            if restored > 0:
                print(f"[TokenBudget] 📂 從 {self.usage_file} 恢復了 {restored} 把 key 的用量紀錄")
                print(f"[TokenBudget]    上次總消耗: {self.total_openai_tokens:,} tokens")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[TokenBudget] ⚠️  讀取用量檔案失敗 ({e})，從零開始計算")

    def _save_usage(self) -> None:
        """將目前的 token 用量儲存到 JSON 檔案（包含日期記錄）。"""
        try:
            save_data = dict(self.usage_per_key)
            save_data["_date"] = self._budget_date()
            with open(self.usage_file, "w") as f:
                json.dump(save_data, f, indent=2)
        except OSError as e:
            print(f"[TokenBudget] ⚠️  儲存用量檔案失敗: {e}")

    def _print_status(self, event: str = "") -> None:
        key = self.api_keys[self.current_key_idx]
        used = self.usage_per_key.get(key, 0)
        remaining = self.token_limit - used
        print(
            f"[TokenBudget] {event} | "
            f"Key #{self.current_key_idx + 1}/{len(self.api_keys)} "
            f"({self._mask_key(key)}) | "
            f"已用: {used:,} / {self.token_limit:,} tokens | "
            f"剩餘: {remaining:,} tokens"
        )

    def _rotate_key(self) -> None:
        """切換到下一個 API key，若全部用完則中止。"""
        old_idx = self.current_key_idx
        self.current_key_idx += 1
        self._save_usage()  # 切換前先存檔

        if self.current_key_idx >= len(self.api_keys):
            self._exhausted = True
            print("\n" + "=" * 60)
            print("🚨 所有 API key 的 token 額度已用完！")
            print(f"   共使用 {len(self.api_keys)} 把 key，"
                  f"總計消耗 {self.total_openai_tokens:,} tokens")
            print("   中止測試以避免超額計費。")
            print("=" * 60 + "\n")
            # 發送 SIGTERM 給自己的 process group，讓 shell trap 可以清理 vLLM
            os.kill(os.getpid(), signal.SIGTERM)
            return

        new_key = self.api_keys[self.current_key_idx]
        os.environ["OPENAI_API_KEY"] = new_key
        print(f"\n🔄 API Key #{old_idx + 1} 額度已達上限，切換到 Key #{self.current_key_idx + 1}")
        self._print_status("已切換")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """litellm 成功回呼：追蹤 token 使用量。"""
        # 只追蹤 OpenAI 的呼叫（user simulator）
        model = kwargs.get("model", "")
        custom_provider = kwargs.get("custom_llm_provider", "")

        # 跳過非 OpenAI 的呼叫（本地 vLLM 模型）
        if custom_provider and custom_provider != "openai":
            return

        # 也檢查 model 名稱，排除本地模型
        if "/" in model and not model.startswith("gpt"):
            return

        with self._lock:
            usage = getattr(response_obj, "usage", None)
            if usage is None:
                return

            total_tokens = getattr(usage, "total_tokens", 0) or 0
            if total_tokens == 0:
                return

            current_key = self.api_keys[self.current_key_idx]
            self.usage_per_key[current_key] += total_tokens
            self.total_openai_tokens += total_tokens

            # 每 10k tokens 印一次狀態，並存檔
            if self.usage_per_key[current_key] % 10_000 < total_tokens:
                self._print_status("使用中")
                self._save_usage()

            # 檢查是否超過額度
            if self.usage_per_key[current_key] >= self.token_limit:
                self._rotate_key()

    def get_summary(self) -> str:
        """取得所有 key 的使用摘要，並存檔。"""
        self._save_usage()
        lines = ["\n📊 OpenAI API Token 使用摘要:"]
        lines.append("-" * 50)
        for i, key in enumerate(self.api_keys):
            used = self.usage_per_key[key]
            marker = " ◀ 目前" if i == self.current_key_idx else ""
            lines.append(
                f"  Key #{i + 1} ({self._mask_key(key)}): "
                f"{used:,} / {self.token_limit:,} tokens{marker}"
            )
        lines.append(f"  總計: {self.total_openai_tokens:,} tokens")
        lines.append("-" * 50)
        return "\n".join(lines)


def setup_budget_manager(
    api_keys: Optional[List[str]] = None,
    token_limit: Optional[int] = None,
) -> TokenBudgetManager:
    """
    初始化並註冊 TokenBudgetManager 到 litellm callbacks。

    API keys 來源（優先順序）：
    1. 直接傳入 api_keys 參數
    2. 環境變數 OPENAI_API_KEYS（逗號分隔）
    3. 檔案 .openai_api_keys（每行一個 key）

    Returns:
        TokenBudgetManager 實例
    """
    import litellm

    if token_limit is None:
        token_limit_env = os.environ.get("TOKEN_LIMIT_PER_KEY", "").strip()
        if token_limit_env:
            try:
                token_limit = int(token_limit_env)
            except ValueError as exc:
                raise ValueError(
                    f"TOKEN_LIMIT_PER_KEY 必須是正整數，收到: {token_limit_env}"
                ) from exc
        else:
            token_limit = DEFAULT_TOKEN_LIMIT_PER_KEY

    if token_limit <= 0:
        raise ValueError(f"token_limit 必須是正整數，收到: {token_limit}")

    if api_keys is None:
        # 嘗試從環境變數讀取
        keys_env = os.environ.get("OPENAI_API_KEYS", "")
        if keys_env:
            api_keys = [k.strip() for k in keys_env.split(",") if k.strip()]

    if not api_keys:
        # 嘗試從檔案讀取
        keys_file = os.path.join(os.path.dirname(__file__), ".openai_api_keys")
        if os.path.exists(keys_file):
            with open(keys_file, "r") as f:
                api_keys = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not api_keys:
        # 最後 fallback: 用現有的 OPENAI_API_KEY
        single_key = os.environ.get("OPENAI_API_KEY", "")
        if single_key:
            api_keys = [single_key]
            print("[TokenBudget] ⚠️  只找到 1 把 key，建議設定 OPENAI_API_KEYS 環境變數提供多把 key")
        else:
            raise ValueError(
                "找不到 API key。請透過以下方式之一提供：\n"
                "  1. OPENAI_API_KEYS 環境變數（逗號分隔）\n"
                "  2. .openai_api_keys 檔案（每行一個 key）\n"
                "  3. OPENAI_API_KEY 環境變數"
            )

    manager = TokenBudgetManager(api_keys=api_keys, token_limit_per_key=token_limit)
    litellm.callbacks = [manager]
    print(f"[TokenBudget] 已註冊 {len(api_keys)} 把 API key，每把額度 {token_limit:,} tokens")
    return manager
