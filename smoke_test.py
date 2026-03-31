#!/usr/bin/env python3
"""Minimal smoke test for the legacy Unity soccer environment."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the Unity environment, print behavior/observation details, "
            "step a few actions, and exit with a success or failure code."
        )
    )
    parser.add_argument(
        "--env-path",
        default="CoE202",
        help="Path to the Unity environment binary or stem. Default: CoE202",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of environment steps to attempt. Default: 10",
    )
    return parser.parse_args()


def fail(message: str, code: int = 1) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


def resolve_env_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    search_order = [candidate]

    if candidate.suffix == "":
        search_order.append(candidate.with_suffix(".exe"))

    for path in search_order:
        if path.exists():
            return path

    searched = ", ".join(str(path) for path in search_order)
    raise FileNotFoundError(
        f"Unity environment not found. Checked: {searched}"
    )


def import_mlagents() -> tuple[Any, Any | None]:
    try:
        from mlagents_envs.environment import UnityEnvironment
    except ImportError as exc:
        raise ImportError(
            "mlagents_envs is not installed. Install the legacy runtime "
            "dependencies before running the smoke test."
        ) from exc

    try:
        from mlagents_envs.base_env import ActionTuple
    except ImportError:
        ActionTuple = None

    return UnityEnvironment, ActionTuple


def require_numpy() -> Any:
    if np is None:
        raise ImportError(
            "numpy is not installed. Install the legacy runtime dependencies "
            "before running the smoke test."
        )
    return np


def behavior_agent_count(decision_steps: Any, terminal_steps: Any) -> int:
    if len(decision_steps) > 0:
        return len(decision_steps)
    return len(terminal_steps)


def observation_shapes(decision_steps: Any, terminal_steps: Any) -> list[tuple[int, ...]]:
    if len(decision_steps) > 0:
        obs_blocks = decision_steps.obs
    else:
        obs_blocks = terminal_steps.obs
    return [tuple(obs.shape) for obs in obs_blocks]


def action_payload(spec: Any, agent_count: int, action_tuple_cls: Any | None) -> tuple[Any, tuple[int, ...], str]:
    np_module = require_numpy()
    action_spec = getattr(spec, "action_spec", None)

    if action_spec is not None:
        continuous_size = int(getattr(action_spec, "continuous_size", 0) or 0)
        discrete_branches = tuple(getattr(action_spec, "discrete_branches", ()) or ())

        if continuous_size > 0:
            actions = np_module.zeros((agent_count, continuous_size), dtype=np_module.float32)
            if action_tuple_cls is not None:
                payload = action_tuple_cls(continuous=actions)
            else:
                payload = actions
            return payload, actions.shape, "continuous"

        if discrete_branches:
            actions = np_module.zeros((agent_count, len(discrete_branches)), dtype=np_module.int32)
            if action_tuple_cls is not None:
                payload = action_tuple_cls(discrete=actions)
            else:
                payload = actions
            return payload, actions.shape, "discrete"

    # Repo-specific fallback: sample_code.py and main1.py both assume 2 players
    # per behavior and three action values per player.
    actions = np_module.zeros((agent_count, 3), dtype=np_module.float32)
    return actions, actions.shape, "repo-fallback"


def decision_rewards(decision_steps: Any) -> list[float]:
    np_module = require_numpy()
    rewards = getattr(decision_steps, "reward", None)
    if rewards is None:
        return []
    return np_module.asarray(rewards).astype(float).tolist()


def terminal_rewards(terminal_steps: Any) -> list[float]:
    np_module = require_numpy()
    rewards = getattr(terminal_steps, "reward", None)
    if rewards is None:
        return []
    return np_module.asarray(rewards).astype(float).tolist()


def run_smoke_test(env_path: Path, steps: int) -> int:
    if steps < 1:
        return fail("--steps must be at least 1", code=2)

    print(f"Resolved environment path: {env_path}", flush=True)
    print(f"Host platform: {platform.system()} {platform.release()}", flush=True)
    if env_path.suffix.lower() == ".exe" and platform.system() != "Windows":
        print(
            "NOTE: resolved a Windows executable on a non-Windows host. "
            "This is expected to fail unless you are running in a compatible "
            "Windows environment.",
            file=sys.stderr,
        )

    env = None
    try:
        require_numpy()
        UnityEnvironment, ActionTuple = import_mlagents()
        env = UnityEnvironment(file_name=str(env_path))
        env.reset()
        behavior_names = list(env.behavior_specs)
        if not behavior_names:
            return fail("No behaviors were exposed after reset.", code=3)

        print(f"Discovered behaviors ({len(behavior_names)}): {behavior_names}", flush=True)

        for behavior_name in behavior_names:
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            count = behavior_agent_count(decision_steps, terminal_steps)
            shapes = observation_shapes(decision_steps, terminal_steps)
            print(
                f"Behavior {behavior_name}: agents={count}, "
                f"observation_shapes={shapes}",
                flush=True,
            )

        for step_index in range(steps):
            action_shapes: dict[str, tuple[int, ...]] = {}
            action_modes: dict[str, str] = {}

            for behavior_name, spec in env.behavior_specs.items():
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                agent_count = behavior_agent_count(decision_steps, terminal_steps)
                if agent_count == 0:
                    continue

                payload, shape, mode = action_payload(spec, agent_count, ActionTuple)
                env.set_actions(behavior_name, payload)
                action_shapes[behavior_name] = shape
                action_modes[behavior_name] = mode

            env.step()

            print(f"Step {step_index + 1}/{steps}", flush=True)
            for behavior_name in behavior_names:
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                print(
                    f"  {behavior_name}: "
                    f"action_mode={action_modes.get(behavior_name, 'none')}, "
                    f"action_shape={action_shapes.get(behavior_name)}, "
                    f"decision_rewards={decision_rewards(decision_steps)}, "
                    f"terminal_rewards={terminal_rewards(terminal_steps)}",
                    flush=True,
                )

        print(
            f"SUCCESS: smoke test completed {steps} step(s) without launch or "
            "step failures.",
            flush=True,
        )
        return 0
    except FileNotFoundError as exc:
        return fail(str(exc), code=2)
    except ImportError as exc:
        return fail(str(exc), code=3)
    except Exception as exc:  # pragma: no cover - legacy runtime safety net
        return fail(f"Smoke test failed during Unity runtime interaction: {exc}", code=4)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main() -> int:
    args = parse_args()

    try:
        env_path = resolve_env_path(args.env_path)
    except FileNotFoundError as exc:
        return fail(str(exc), code=2)

    return run_smoke_test(env_path, args.steps)


if __name__ == "__main__":
    sys.exit(main())
