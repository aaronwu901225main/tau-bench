#!/usr/bin/env python3
"""
Translate tau-bench source assets into file-based locale packs.

This script follows a per-file translation workflow (no glossary table replacement):
- wiki.md
- rules
- tools descriptions
- tasks_<split>.json
- shared prompts used by user simulator and react/act agent

It supports:
- OpenAI API key rotation via OPENAI_API_KEYS
- per-key daily token budget tracking
- checkpoint resume for task files
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from openai import APIError, AuthenticationError, OpenAI, RateLimitError


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tau_bench.agents.chat_react_agent import ACT_INSTRUCTION, REACT_INSTRUCTION
from tau_bench.envs.user import (
    DEFAULT_LLM_USER_SYSTEM_PROMPT,
    DEFAULT_REACT_USER_SYSTEM_PROMPT,
    DEFAULT_REFLECT_PROMPT,
    DEFAULT_VERIFY_PROMPT,
)


USAGE_FILE = ROOT_DIR / "scripts" / "translation_api_usage_daily.json"


CORE_TERMINOLOGY = {
    "agent": "客服",
    "user": "使用者",
    "customer": "顧客",
    "order": "訂單",
    "reservation": "預訂",
    "refund": "退款",
    "return": "退貨",
    "exchange": "換貨",
    "payment method": "付款方式",
    "pending": "待處理",
    "delivered": "已送達",
    "cancelled": "已取消",
}


RETAIL_TERMINOLOGY = {
    **CORE_TERMINOLOGY,
    "product": "商品",
    "item": "品項",
    "shipping address": "收件地址",
    "gift card": "禮品卡",
    "credit card": "信用卡",
    "paypal": "PayPal",
}


AIRLINE_TERMINOLOGY = {
    **CORE_TERMINOLOGY,
    "flight": "航班",
    "one way": "單程",
    "round trip": "來回",
    "passenger": "乘客",
    "baggage": "行李",
    "basic economy": "basic economy",
    "economy": "economy",
    "business": "business",
    "travel insurance": "旅遊保險",
}


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - zh
    return max(1, int(zh / 2 + other / 4))


def _today_key() -> str:
    now = dt.datetime.now()
    if now.hour < 8:
        return (now.date() - dt.timedelta(days=1)).isoformat()
    return now.date().isoformat()


def _load_usage(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_usage(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_add_usage(path: Path, key: str, day_key: str, tokens: int) -> int:
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            usage = _load_usage(path)
            usage.setdefault(day_key, {})
            new_total = int(usage[day_key].get(key, 0)) + int(tokens)
            usage[day_key][key] = new_total
            _save_usage(path, usage)
            return new_total
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def _parse_keys() -> List[str]:
    keys_env = os.getenv("OPENAI_API_KEYS") or os.getenv("OPENAI_API_KEY", "")
    keys = [item.strip() for item in keys_env.split(",") if item.strip()]
    if not keys:
        raise RuntimeError("No OpenAI API keys found in OPENAI_API_KEYS/OPENAI_API_KEY")
    return keys


def get_remaining_tokens() -> Dict[str, int]:
    day_key = _today_key()
    usage = _load_usage(USAGE_FILE)
    limit = int(os.getenv("TOKEN_LIMIT_PER_KEY", os.getenv("API_DAILY_LIMIT_TOKENS", "2500000")))
    keys = _parse_keys()
    day_usage = usage.get(day_key, {})
    return {
        f"key_{idx + 1}_***{k[-4:]}": max(0, limit - int(day_usage.get(k, 0)))
        for idx, k in enumerate(keys)
    }


def get_total_remaining_tokens() -> int:
    return sum(get_remaining_tokens().values())


def _select_key(keys: List[str], usage: Dict[str, Any], day_key: str, limit: int, margin: int, tried: set[str]) -> str:
    day_usage = usage.get(day_key, {})
    candidates = [k for k in keys if k not in tried]
    if not candidates:
        candidates = keys[:]
    candidates.sort(key=lambda k: int(day_usage.get(k, 0)))
    for key in candidates:
        if int(day_usage.get(key, 0)) < max(0, limit - margin):
            return key
    return candidates[0]


def _build_translation_prompt(context: str, terminology: Dict[str, str], preserve: Iterable[str]) -> str:
    lines = [
        "You are a professional translator.",
        "Translate English text to Traditional Chinese (zh-TW).",
        f"Context: {context}",
        "Rules:",
        "1) Preserve markdown structure, list structure, punctuation style, and line breaks.",
        "2) Do NOT translate code symbols, function names, JSON keys, argument keys, IDs, flight/order/user/product identifiers, airport codes, dates, ###STOP###, ###TRANSFER###.",
        "3) Keep placeholders and template variables unchanged.",
        "4) Use consistent terminology. Do not use random synonyms for the same concept.",
        "5) Output translated text only. No notes or explanations.",
    ]
    preserve_list = [item for item in preserve if item]
    if preserve_list:
        lines.append("6) Preserve these exact tokens/labels verbatim: " + ", ".join(sorted(set(preserve_list))))
    if terminology:
        lines.append("Terminology mapping (must follow):")
        for src, dst in terminology.items():
            lines.append(f"- {src} => {dst}")
    return "\n".join(lines)


def translate_text(
    text: str,
    *,
    context: str,
    terminology: Dict[str, str],
    preserve_tokens: Iterable[str] = (),
    max_attempts: int = 4,
) -> str:
    if not text or not text.strip():
        return text

    model = os.getenv("TRANSLATE_MODEL", "gpt-5-mini")
    limit = int(os.getenv("TOKEN_LIMIT_PER_KEY", os.getenv("API_DAILY_LIMIT_TOKENS", "2500000")))
    margin = int(os.getenv("API_ROTATE_MARGIN", "25000"))
    day_key = _today_key()

    keys = _parse_keys()
    usage = _load_usage(USAGE_FILE)
    usage.setdefault(day_key, {})
    tried: set[str] = set()
    last_error: Exception | None = None

    prompt = _build_translation_prompt(
        context=context,
        terminology=terminology,
        preserve=preserve_tokens,
    )

    for _ in range(max_attempts * max(1, len(keys))):
        active_key = _select_key(keys, usage, day_key, limit, margin, tried)
        client = OpenAI(api_key=active_key)

        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=1,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            usage_obj = getattr(resp, "usage", None)
            if usage_obj and getattr(usage_obj, "total_tokens", None):
                used_tokens = int(usage_obj.total_tokens)
            else:
                used_tokens = _estimate_tokens(text) + _estimate_tokens(content)
            _atomic_add_usage(USAGE_FILE, active_key, day_key, used_tokens)
            return content

        except AuthenticationError as exc:
            _atomic_add_usage(USAGE_FILE, active_key, day_key, limit)
            tried.add(active_key)
            last_error = exc
        except RateLimitError as exc:
            tried.add(active_key)
            last_error = exc
        except APIError as exc:
            tried.add(active_key)
            last_error = exc

    raise RuntimeError(f"Translation failed after retries: {last_error}")


def _task_to_dict(task_obj: Any) -> Dict[str, Any]:
    if hasattr(task_obj, "model_dump"):
        return task_obj.model_dump()
    if hasattr(task_obj, "dict"):
        return task_obj.dict()
    raise TypeError("Unsupported task object type")


def _is_non_translatable_output(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    if re.fullmatch(r"[0-9.,\-+/%: ]+", s):
        return True
    if re.fullmatch(r"#?[A-Z0-9_\-]{3,}", s):
        return True
    if re.fullmatch(r"[a-z]+_[0-9]+", s):
        return True
    return False


def _translate_task(task_obj: Any, env_name: str, terminology: Dict[str, str]) -> Dict[str, Any]:
    task = _task_to_dict(task_obj)
    translated = copy.deepcopy(task)

    translated["instruction"] = translate_text(
        task.get("instruction", ""),
        context=f"tau-bench {env_name} task instruction",
        terminology=terminology,
        preserve_tokens=("###STOP###", "###TRANSFER###"),
    )

    outputs = task.get("outputs", [])
    translated_outputs: List[str] = []
    for output in outputs:
        if not isinstance(output, str) or _is_non_translatable_output(output):
            translated_outputs.append(output)
            continue
        translated_outputs.append(
            translate_text(
                output,
                context=f"tau-bench {env_name} expected answer string for reward matching",
                terminology=terminology,
                preserve_tokens=("###STOP###", "###TRANSFER###"),
            )
        )
    translated["outputs"] = translated_outputs
    return translated


def _translate_tool_schema(tool_info: Dict[str, Any], env_name: str, terminology: Dict[str, str]) -> Dict[str, Any]:
    translated = copy.deepcopy(tool_info)
    function = translated.get("function")
    if not isinstance(function, dict):
        return translated

    desc = function.get("description")
    if isinstance(desc, str) and desc.strip():
        function["description"] = translate_text(
            desc,
            context=f"tau-bench {env_name} tool description",
            terminology=terminology,
            preserve_tokens=("###STOP###", "###TRANSFER###"),
        )

    params = function.get("parameters")
    if isinstance(params, dict):
        props = params.get("properties")
        if isinstance(props, dict):
            for _, prop in props.items():
                if not isinstance(prop, dict):
                    continue
                prop_desc = prop.get("description")
                if isinstance(prop_desc, str) and prop_desc.strip():
                    prop["description"] = translate_text(
                        prop_desc,
                        context=f"tau-bench {env_name} tool parameter description",
                        terminology=terminology,
                        preserve_tokens=("###STOP###", "###TRANSFER###"),
                    )
                items = prop.get("items")
                if isinstance(items, dict):
                    item_desc = items.get("description")
                    if isinstance(item_desc, str) and item_desc.strip():
                        items["description"] = translate_text(
                            item_desc,
                            context=f"tau-bench {env_name} tool array item description",
                            terminology=terminology,
                            preserve_tokens=("###STOP###", "###TRANSFER###"),
                        )
    return translated


def _load_env_assets(env_name: str) -> Dict[str, Any]:
    if env_name == "retail":
        from tau_bench.envs.retail.rules import RULES
        from tau_bench.envs.retail.tools import ALL_TOOLS
        from tau_bench.envs.retail.wiki import WIKI
        from tau_bench.envs.retail.tasks_test import TASKS_TEST
        from tau_bench.envs.retail.tasks_train import TASKS_TRAIN
        from tau_bench.envs.retail.tasks_dev import TASKS_DEV

        tasks_by_split = {
            "test": TASKS_TEST,
            "train": TASKS_TRAIN,
            "dev": TASKS_DEV,
        }
        terminology = RETAIL_TERMINOLOGY
    elif env_name == "airline":
        from tau_bench.envs.airline.rules import RULES
        from tau_bench.envs.airline.tools import ALL_TOOLS
        from tau_bench.envs.airline.wiki import WIKI
        from tau_bench.envs.airline.tasks_test import TASKS

        tasks_by_split = {
            "test": TASKS,
        }
        terminology = AIRLINE_TERMINOLOGY
    else:
        raise ValueError(f"Unsupported env: {env_name}")

    tools_info = [tool.get_info() for tool in ALL_TOOLS]
    return {
        "rules": RULES,
        "wiki": WIKI,
        "tools_info": tools_info,
        "tasks_by_split": tasks_by_split,
        "terminology": terminology,
    }


def _translate_tasks_with_checkpoint(
    *,
    tasks: List[Any],
    env_name: str,
    split: str,
    terminology: Dict[str, str],
    output_path: Path,
    max_tasks: int | None,
    dry_run: bool,
) -> None:
    checkpoint_path = Path(str(output_path) + ".checkpoint.json")
    checkpoint = {
        "translated_indices": [],
        "translated_tasks": {},
    }
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    translated_indices = set(checkpoint.get("translated_indices", []))
    translated_tasks: Dict[str, Any] = checkpoint.get("translated_tasks", {})

    todo = [idx for idx in range(len(tasks)) if idx not in translated_indices]
    if max_tasks is not None:
        todo = todo[:max_tasks]

    if dry_run:
        print(f"[DRY RUN] {env_name}/{split}: {len(todo)} tasks to translate")
        return

    for pos, task_index in enumerate(todo, start=1):
        remaining = get_total_remaining_tokens()
        min_tokens = int(os.getenv("MIN_TOKENS", "50000"))
        if remaining < min_tokens:
            print(
                f"[STOP] {env_name}/{split}: remaining tokens {remaining:,} below MIN_TOKENS {min_tokens:,}"
            )
            break

        print(f"[TASK] {env_name}/{split} {pos}/{len(todo)} index={task_index} remaining={remaining:,}")
        translated = _translate_task(tasks[task_index], env_name=env_name, terminology=terminology)
        translated_indices.add(task_index)
        translated_tasks[str(task_index)] = translated

        checkpoint_data = {
            "translated_indices": sorted(translated_indices),
            "translated_tasks": translated_tasks,
        }
        tmp_path = Path(str(checkpoint_path) + ".tmp")
        tmp_path.write_text(json.dumps(checkpoint_data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, checkpoint_path)

    if len(translated_indices) == len(tasks):
        final_tasks: List[Any] = []
        for idx, task_obj in enumerate(tasks):
            key = str(idx)
            if key in translated_tasks:
                final_tasks.append(translated_tasks[key])
            else:
                final_tasks.append(_task_to_dict(task_obj))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(final_tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] wrote {output_path}")


def _translate_shared_prompts(locale_root: Path, dry_run: bool) -> None:
    shared_dir = locale_root / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    shared_assets = {
        "llm_user_system_prompt.txt": (
            DEFAULT_LLM_USER_SYSTEM_PROMPT,
            "tau-bench user simulator system prompt",
            CORE_TERMINOLOGY,
            ("{instruction_display}", "###STOP###"),
        ),
        "react_user_system_prompt.txt": (
            DEFAULT_REACT_USER_SYSTEM_PROMPT,
            "tau-bench react user simulator system prompt",
            CORE_TERMINOLOGY,
            ("{instruction_display}", "Thought:", "User Response:", "###STOP###"),
        ),
        "verify_prompt.txt": (
            DEFAULT_VERIFY_PROMPT,
            "tau-bench verify prompt",
            CORE_TERMINOLOGY,
            ("{transcript}", "{response}", "Classification:"),
        ),
        "reflect_prompt.txt": (
            DEFAULT_REFLECT_PROMPT,
            "tau-bench reflection prompt",
            CORE_TERMINOLOGY,
            ("{transcript}", "{response}", "Reflection:", "Response:"),
        ),
        "react_instruction.txt": (
            REACT_INSTRUCTION,
            "tau-bench react agent instruction",
            CORE_TERMINOLOGY,
            ("Thought:", "Action:", "respond", "content"),
        ),
        "act_instruction.txt": (
            ACT_INSTRUCTION,
            "tau-bench act agent instruction",
            CORE_TERMINOLOGY,
            ("Action:", "respond", "content"),
        ),
    }

    for file_name, payload in shared_assets.items():
        src, context, terminology, preserve = payload
        out_path = shared_dir / file_name
        if out_path.exists():
            print(f"[SKIP] shared prompt exists: {out_path}")
            continue
        if dry_run:
            print(f"[DRY RUN] shared prompt: {out_path}")
            continue
        translated = translate_text(
            src,
            context=context,
            terminology=terminology,
            preserve_tokens=preserve,
        )
        out_path.write_text(translated, encoding="utf-8")
        print(f"[DONE] wrote {out_path}")


def _write_meta(meta_path: Path, env_name: str, locale: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY RUN] meta: {meta_path}")
        return
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "locale": locale,
        "env": env_name,
        "model": os.getenv("TRANSLATE_MODEL", "gpt-5-mini"),
        "generated_at": dt.datetime.now().isoformat(),
        "source": "tau-bench",
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def translate_env(env_name: str, locale_root: Path, max_tasks: int | None, dry_run: bool) -> None:
    assets = _load_env_assets(env_name)
    terminology = assets["terminology"]

    env_dir = locale_root / env_name
    env_dir.mkdir(parents=True, exist_ok=True)

    _write_meta(env_dir / "meta.json", env_name=env_name, locale=locale_root.name, dry_run=dry_run)

    wiki_out = env_dir / "wiki.md"
    if not wiki_out.exists():
        if dry_run:
            print(f"[DRY RUN] wiki: {wiki_out}")
        else:
            wiki_out.write_text(
                translate_text(
                    assets["wiki"],
                    context=f"tau-bench {env_name} policy/wiki markdown",
                    terminology=terminology,
                    preserve_tokens=("###STOP###", "###TRANSFER###"),
                ),
                encoding="utf-8",
            )
            print(f"[DONE] wrote {wiki_out}")
    else:
        print(f"[SKIP] wiki exists: {wiki_out}")

    rules_out = env_dir / "rules.json"
    if not rules_out.exists():
        if dry_run:
            print(f"[DRY RUN] rules: {rules_out}")
        else:
            translated_rules = [
                translate_text(
                    rule,
                    context=f"tau-bench {env_name} behavior rule",
                    terminology=terminology,
                    preserve_tokens=("###STOP###", "###TRANSFER###"),
                )
                for rule in assets["rules"]
            ]
            rules_out.write_text(json.dumps(translated_rules, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[DONE] wrote {rules_out}")
    else:
        print(f"[SKIP] rules exists: {rules_out}")

    tools_out = env_dir / "tools.json"
    if not tools_out.exists():
        if dry_run:
            print(f"[DRY RUN] tools: {tools_out}")
        else:
            translated_tools = [
                _translate_tool_schema(tool_info=item, env_name=env_name, terminology=terminology)
                for item in assets["tools_info"]
            ]
            tools_out.write_text(json.dumps(translated_tools, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[DONE] wrote {tools_out}")
    else:
        print(f"[SKIP] tools exists: {tools_out}")

    for split, tasks in assets["tasks_by_split"].items():
        out_path = env_dir / f"tasks_{split}.json"
        if out_path.exists():
            print(f"[SKIP] tasks exists: {out_path}")
            continue
        _translate_tasks_with_checkpoint(
            tasks=tasks,
            env_name=env_name,
            split=split,
            terminology=terminology,
            output_path=out_path,
            max_tasks=max_tasks,
            dry_run=dry_run,
        )


def print_status(locale_root: Path, envs: List[str]) -> None:
    print("=" * 60)
    print(f"Locale root: {locale_root}")
    print("Remaining tokens:")
    for key, remain in get_remaining_tokens().items():
        print(f"  {key}: {remain:,}")
    print(f"  total: {get_total_remaining_tokens():,}")
    print("\nFiles status:")
    for env_name in envs:
        env_dir = locale_root / env_name
        print(f"\n[{env_name}] {env_dir}")
        for name in ["meta.json", "wiki.md", "rules.json", "tools.json", "tasks_test.json", "tasks_train.json", "tasks_dev.json"]:
            p = env_dir / name
            if p.exists():
                print(f"  OK   {name}")
            else:
                print(f"  MISS {name}")
    shared_dir = locale_root / "shared"
    print(f"\n[shared] {shared_dir}")
    for name in [
        "llm_user_system_prompt.txt",
        "react_user_system_prompt.txt",
        "verify_prompt.txt",
        "reflect_prompt.txt",
        "react_instruction.txt",
        "act_instruction.txt",
    ]:
        p = shared_dir / name
        if p.exists():
            print(f"  OK   {name}")
        else:
            print(f"  MISS {name}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate tau-bench into locale files")
    parser.add_argument("--locale", type=str, default="zh-TW")
    parser.add_argument("--envs", nargs="+", default=["retail", "airline"], choices=["retail", "airline"])
    parser.add_argument("--status", action="store_true", help="Show progress/token status and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without calling APIs")
    parser.add_argument("--max-tasks", type=int, default=None, help="Translate at most N tasks per split in this run")
    parser.add_argument("--skip-shared", action="store_true", help="Skip shared prompt translation")
    args = parser.parse_args()

    locale_root = ROOT_DIR / "tau_bench" / "locales" / args.locale

    if args.status:
        print_status(locale_root=locale_root, envs=args.envs)
        return

    remaining = get_total_remaining_tokens()
    min_tokens = int(os.getenv("MIN_TOKENS", "50000"))
    if remaining < min_tokens and not args.dry_run:
        raise RuntimeError(
            f"Not enough remaining tokens to start: {remaining:,} < {min_tokens:,}."
        )

    locale_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_shared:
        _translate_shared_prompts(locale_root=locale_root, dry_run=args.dry_run)

    for env_name in args.envs:
        translate_env(
            env_name=env_name,
            locale_root=locale_root,
            max_tasks=args.max_tasks,
            dry_run=args.dry_run,
        )

    print_status(locale_root=locale_root, envs=args.envs)


if __name__ == "__main__":
    main()
