# Copyright Sierra

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tau_bench.envs.base import Env
from tau_bench.types import Task

LOCALES_DIR = Path(__file__).resolve().parent / "locales"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_file_based_locale(env_name: str, locale: str) -> Path | None:
    locale_root = LOCALES_DIR / locale / env_name
    if not locale_root.exists():
        print(f"[Locale] ⚠️ 找不到逐檔 locale 目錄: {locale_root}，將退回英文")
        return None

    required_files = ["wiki.md", "rules.json", "tools.json", "meta.json"]
    missing = [name for name in required_files if not (locale_root / name).exists()]
    if missing:
        print(
            f"[Locale] ⚠️ locale 目錄缺少必要檔案 ({', '.join(missing)}): {locale_root}，將退回英文"
        )
        return None
    return locale_root


def _resolve_tasks_file(env_name: str, task_split: str, locale_root: Path) -> Path | None:
    if env_name == "airline":
        candidate = locale_root / "tasks_test.json"
    elif env_name == "retail":
        candidate = locale_root / f"tasks_{task_split}.json"
    else:
        return None

    if not candidate.exists():
        print(f"[Locale] ⚠️ 找不到任務翻譯檔案: {candidate}，將退回英文")
        return None
    return candidate


def _merge_translated_tools(
    original_tools_info: list[dict[str, Any]],
    translated_tools_info: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    translated_by_name: dict[str, dict[str, Any]] = {}
    for tool in translated_tools_info:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str):
            translated_by_name[name] = tool

    merged: list[dict[str, Any]] = []
    for original_tool in original_tools_info:
        function = original_tool.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name in translated_by_name:
            merged.append(translated_by_name[name])
        else:
            merged.append(original_tool)
    return merged


def apply_locale_to_env(env: Env, env_name: str, locale: str) -> Env:
    env.locale = locale
    if locale == "en":
        return env

    locale_root = _load_file_based_locale(env_name=env_name, locale=locale)
    if locale_root is None:
        return env

    task_split = getattr(env, "task_split", "test")
    tasks_file = _resolve_tasks_file(
        env_name=env_name,
        task_split=task_split,
        locale_root=locale_root,
    )
    if tasks_file is None:
        return env

    wiki_path = locale_root / "wiki.md"
    rules_path = locale_root / "rules.json"
    tools_path = locale_root / "tools.json"

    env.wiki = wiki_path.read_text(encoding="utf-8")
    rules = _read_json(rules_path)
    if isinstance(rules, list):
        env.rules = [rule for rule in rules if isinstance(rule, str)]
    elif isinstance(rules, dict) and isinstance(rules.get("rules"), list):
        env.rules = [rule for rule in rules["rules"] if isinstance(rule, str)]

    translated_tasks_raw = _read_json(tasks_file)
    if not isinstance(translated_tasks_raw, list):
        print(f"[Locale] ⚠️ 任務翻譯格式錯誤: {tasks_file}，將退回英文")
        return env
    env.tasks = [Task.model_validate(item) for item in translated_tasks_raw]

    translated_tools_raw = _read_json(tools_path)
    if isinstance(translated_tools_raw, list):
        env.tools_info = _merge_translated_tools(env.tools_info, translated_tools_raw)

    if not env.tasks:
        print(f"[Locale] ⚠️ 任務翻譯清單為空: {tasks_file}，將退回英文")
        return env

    if env.task_index < 0 or env.task_index >= len(env.tasks):
        env.task_index = min(max(env.task_index, 0), len(env.tasks) - 1)
    env.task = env.tasks[env.task_index]

    print(f"[Locale] 已套用逐檔本地化: env={env_name}, locale={locale}, split={task_split}")
    return env
