Project Revival Checklist
=========================

This document captures the main things that would need attention before trying to run or modernize the project again.

It is intentionally practical. The goal is not to redesign the repo yet, but to identify the shortest path back to a runnable baseline.


Current Situation
-----------------

From the files in this repository, the project appears to be:

- a Windows-only compiled Unity environment,
- driven by a Python script that expects ML-Agents,
- trained with PyTorch,
- using TensorFlow only for random opponent actions.

There is enough here to study the project, but not enough to rerun it cleanly on the current machine without extra setup.


Priority 1: Hard Blockers
-------------------------

These are the main issues that can stop the project before training even begins.

1. Platform mismatch

- The available simulator is `CoE202.exe`, which is a Windows executable.
- The current machine is macOS.
- Without a Windows environment or a separate macOS Unity build, the Python code cannot launch the simulator.

2. Missing Unity source project

- The repo contains the built game, not the Unity project source.
- That means you cannot rebuild the environment, change the scene, or export a macOS binary from this repository alone.

3. Likely ML-Agents version mismatch

- The Unity timer metadata points to Unity `2019.4.5f1` and ML-Agents `1.0.4`.
- The repo also includes `unityagents-0.4.0-py3-none-any.whl`.
- The Python code imports `mlagents_envs.environment`, not `unityagents`.

This strongly suggests the bundled wheel may be stale or unrelated to the final working setup.

4. No reproducible Python environment definition

- There is no `requirements.txt`, `environment.yml`, `pyproject.toml`, or lock file.
- Recreating the original Python environment will require manual dependency pinning.


Priority 2: Code-Level Risks
----------------------------

These issues are likely to cause incorrect behavior or runtime failures even after the environment launches.

1. `main1.py` runs immediately on import

- The script performs environment setup at the top level.
- It also calls `ppo_train()` directly near the bottom.
- There is no `if __name__ == "__main__":` guard.

This makes it harder to test or reuse parts of the code safely.

2. The evaluation loop appears broken

- The post-training section uses `while True` with no obvious termination path.
- It also treats `Agent.act()` results inconsistently.
- Reward array creation in that section differs from the training section and looks malformed.

Treat the evaluation block as unfinished.

3. The actor uses an unusual action/log-probability combination

- `ActorModel.forward()` returns `torch.sign(self.fc_action(x))` as the action.
- In the same method, it creates a categorical distribution from `softmax(self.fc_action(x))`.
- It then computes the log-probability of a newly sampled action from that distribution.

That means the returned action vector and the returned log-probability do not clearly correspond to the same decision.

4. The `action` argument in `ActorModel.forward()` is not actually used

- The method signature accepts `action=None`.
- The method does not use the provided action when computing the log-probability.

This is a major warning sign for PPO correctness.

5. Reward-shaping logic contains suspicious comparisons

- In the late-step penalty branch, the code compares some observation arrays to themselves instead of comparing old vs new values.
- A self-comparison like `cur_obs_s1 == cur_obs_s1` is always true.

This likely makes the anti-stalling penalty trigger incorrectly once that branch is reached.

6. Timing logic looks inconsistent

- `t0` starts as `perf_counter()`.
- `t1` is computed as elapsed time from `t0`.
- Later, `t0` is overwritten with `t1`, which changes the meaning of the variable mid-loop.

This makes the time-based logic hard to trust.

7. Only one side is really being optimized

- The learned team stores experiences and receives PPO updates.
- The opponent side uses random actions during training.

This is not symmetric multi-agent self-play in the current code.


Priority 3: Readability And Maintenance Problems
------------------------------------------------

These do not always break execution, but they make the project harder to reason about.

1. Team and role naming are mixed together.
2. There are many commented-out experiments left in the main script.
3. Training and evaluation live in one long file.
4. There is only one git commit in the original project history.
5. TensorFlow is only being used to generate random actions.


Suggested Recovery Plan
-----------------------

If the goal is to revive the project safely, the lowest-risk order is:

1. Get the environment running again on a compatible platform.
2. Confirm the exact ML-Agents Python package version that can still talk to this Unity build.
3. Make a non-training smoke test that only resets the environment, reads observations, sends fixed actions, and exits.
4. Separate training from evaluation.
5. Fix the actor/log-probability path before trusting any PPO results.
6. Replace TensorFlow random actions with NumPy or PyTorch random generation.
7. Only after that, attempt real training.


Smallest Useful Milestones
--------------------------

A realistic path back to progress is:

Milestone 1

- Run the Unity environment.
- Connect from Python.
- Print behavior names, observation shapes, and rewards.

Milestone 2

- Send fixed actions for a few steps without crashing.
- Confirm that the expected two-team, two-role layout is real.

Milestone 3

- Load checkpoints and run inference only.
- Do not train yet.

Milestone 4

- Repair the training math and evaluation loop.
- Then try a short training run.


What To Preserve
----------------

Even though the code is rough, these ideas are still worth preserving:

- role-based sharing between goalies and strikers,
- centralized critics that see all player states,
- saved checkpoints for both roles,
- the `sample_code.py` observation-debugging approach.


Bottom Line
-----------

This repo is understandable and partially reusable, but it is not currently a plug-and-play training project. The fastest path forward is:

- recover a compatible runtime environment first,
- treat the existing code as a useful prototype,
- then repair the action path and evaluation logic before trusting new results.
