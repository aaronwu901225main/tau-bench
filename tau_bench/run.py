# Copyright Sierra

import os
import json
import random
import traceback
from math import comb
import multiprocessing
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult, RunConfig
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def is_gpt_5_mini(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return normalized == "gpt-5-mini" or normalized.endswith("/gpt-5-mini")


def run(config: RunConfig) -> List[EnvRunResult]:
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert config.model_provider in provider_list, "Invalid model provider"
    assert config.user_model_provider in provider_list, "Invalid user model provider"
    assert config.agent_strategy in ["tool-calling", "act", "react", "few-shot"], "Invalid agent strategy"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"
    if is_gpt_5_mini(config.model):
        assert config.temperature == 1.0, "gpt-5-mini requires temperature=1.0"

    random.seed(config.seed)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    # Checkpoint 路徑：resume 模式用穩定名稱，否則用時間戳
    ckpt_base = f"{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}_range_{config.start_index}-{config.end_index}_user-{config.user_model}-{config.user_strategy}_locale-{config.locale}"
    if config.resume:
        ckpt_path = f"{config.log_dir}/{ckpt_base}_checkpoint.json"
    else:
        time_str = datetime.now().strftime("%m%d%H%M%S")
        ckpt_path = f"{config.log_dir}/{ckpt_base}_{time_str}.json"

    # Resume: 掃描 log_dir 中所有 JSON 檔，載入已完成的 (task_id, trial) 結果
    existing_results: List[EnvRunResult] = []
    completed_pairs: set = set()  # {(task_id, trial)}

    if config.resume and os.path.exists(config.log_dir):
        for fname in sorted(os.listdir(config.log_dir)):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(config.log_dir, fname)
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    continue
                for item in data:
                    if not (isinstance(item, dict) and 'task_id' in item and 'trial' in item and 'reward' in item):
                        continue
                    pair = (item['task_id'], item['trial'])
                    if pair not in completed_pairs:
                        existing_results.append(EnvRunResult(**item))
                        completed_pairs.add(pair)
            except Exception:
                pass
        if existing_results:
            n_tasks = len(set(r.task_id for r in existing_results))
            n_trials_done = len(set(r.trial for r in existing_results))
            print(f"📂 Resume: 載入 {len(existing_results)} 筆歷史結果 ({n_tasks} tasks × {n_trials_done} trials)")
            # 預寫入穩定 checkpoint，讓後續增量保存能包含完整歷史
            with open(ckpt_path, "w") as f:
                json.dump([r.model_dump() for r in existing_results], f, indent=2)

            # 預先檢查是否所有 (task_id, trial) 都已完成，若是則跳過 env/agent 初始化
            if n_trials_done >= config.num_trials:
                # 取得 task 數量（不需要載入整個 env）
                from tau_bench.envs import get_env as _get_env_for_count
                _count_env = _get_env_for_count(
                    config.env,
                    user_strategy="human",  # 使用 human 策略避免初始化 LLM
                    user_model=config.user_model,
                    user_provider=config.user_model_provider,
                    task_split=config.task_split,
                    locale=config.locale,
                )
                _end = len(_count_env.tasks) if config.end_index == -1 else min(config.end_index, len(_count_env.tasks))
                _all_pairs = set()
                for trial_i in range(config.num_trials):
                    if config.task_ids and len(config.task_ids) > 0:
                        _idxs = config.task_ids
                    else:
                        _idxs = list(range(config.start_index, _end))
                    for idx in _idxs:
                        _all_pairs.add((idx, trial_i))
                _remaining = _all_pairs - completed_pairs
                if not _remaining:
                    print(f"\n🎉 所有 {config.num_trials} 個 trial 的 {len(_all_pairs) // config.num_trials} 個 task 已全部完成，無需重新執行")
                    display_metrics(existing_results)
                    with open(ckpt_path, "w") as f:
                        json.dump([r.model_dump() for r in existing_results], f, indent=2)
                        print(f"\n📄 Results saved to {ckpt_path}\n")
                    return existing_results
                else:
                    print(f"📋 尚有 {len(_remaining)} 個 (task, trial) 組合待完成")

    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
        locale=config.locale,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
        locale=config.locale,
    )
    end_index = (
        len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = list(existing_results)
    lock = multiprocessing.Lock()
    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
    )
    all_skipped = True
    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))
        if config.shuffle:
            random.shuffle(idxs)

        # Resume: 跳過此 trial 中已完成的 task
        if completed_pairs:
            orig_len = len(idxs)
            idxs = [idx for idx in idxs if (idx, i) not in completed_pairs]
            skipped = orig_len - len(idxs)
            if skipped > 0:
                print(f"⏭️  Trial {i}: 跳過 {skipped} 個已完成 task，剩餘 {len(idxs)} 個")
        if not idxs:
            print(f"✅ Trial {i}: 全部已完成，跳過")
            continue
        all_skipped = False

        def _run(idx: int) -> EnvRunResult:
            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
                locale=config.locale,
            )

            print(f"Running task {idx}")
            try:
                res = agent.solve(
                    env=isolated_env,
                    task_index=idx,
                )
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )
            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                result.info,
            )
            print("-----")
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)
            return result

        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            res = list(executor.map(_run, idxs))
            results.extend(res)

    if all_skipped and existing_results:
        print(f"\n🎉 所有 {config.num_trials} 個 trial 的全部 task 已完成，無需重新執行")

    display_metrics(results)

    with open(ckpt_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")
    return results


def agent_factory(
    tools_info: List[Dict[str, Any]], wiki, config: RunConfig, locale: str = "en"
) -> Agent:
    if config.agent_strategy == "tool-calling":
        # native tool calling
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "act":
        # `act` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
            locale=locale,
        )
    elif config.agent_strategy == "react":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
            locale=locale,
        )
    elif config.agent_strategy == "few-shot":
        from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent
        assert config.few_shot_displays_path is not None, "Few shot displays path is required for few-shot agent strategy"
        with open(config.few_shot_displays_path, "r") as f:
            few_shot_displays = [json.loads(line)["messages_display"] for line in f]

        return FewShotToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            few_shot_displays=few_shot_displays,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")


def display_metrics(results: List[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    if not results:
        print("⚠️ 沒有結果可顯示")
        return

    num_trials = len(set([r.trial for r in results]))

    # 檢查每個 task 是否有相同數量的 trial（不均勻時警告）
    trials_per_task: dict[int, set] = {}
    for r in results:
        trials_per_task.setdefault(r.task_id, set()).add(r.trial)
    trial_counts = [len(t) for t in trials_per_task.values()]
    min_tc, max_tc = min(trial_counts), max(trial_counts)
    if min_tc != max_tc:
        print(f"⚠️ 部分 task 的 trial 數量不一致 (min={min_tc}, max={max_tc})，pass^k 為近似值")
        print(f"   完整 trial 數={num_trials}，共 {len(trials_per_task)} 個 task")

    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print(f"📊 Total results: {len(results)} ({len(c_per_task_id)} tasks × {num_trials} trials)")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
