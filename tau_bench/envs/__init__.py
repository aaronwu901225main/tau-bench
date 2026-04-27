# Copyright Sierra

from typing import Optional, Union
from tau_bench.envs.base import Env
from tau_bench.envs.user import UserStrategy
from tau_bench.localization import apply_locale_to_env


def get_env(
    env_name: str,
    user_strategy: Union[str, UserStrategy],
    user_model: str,
    task_split: str,
    user_provider: Optional[str] = None,
    task_index: Optional[int] = None,
    locale: str = "en",
) -> Env:
    if env_name == "retail":
        from tau_bench.envs.retail import MockRetailDomainEnv

        env = MockRetailDomainEnv(
            user_strategy=user_strategy,
            user_model=user_model,
            task_split=task_split,
            user_provider=user_provider,
            task_index=task_index,
            locale=locale,
        )
        return apply_locale_to_env(env=env, env_name=env_name, locale=locale)
    elif env_name == "airline":
        from tau_bench.envs.airline import MockAirlineDomainEnv

        env = MockAirlineDomainEnv(
            user_strategy=user_strategy,
            user_model=user_model,
            task_split=task_split,
            user_provider=user_provider,
            task_index=task_index,
            locale=locale,
        )
        return apply_locale_to_env(env=env, env_name=env_name, locale=locale)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
