Runtime Recovery Guide
======================

This document describes the current runtime reality of the project and the
lowest-risk path to getting it runnable again.


Current Facts
-------------

- The repository contains a compiled Unity environment as `CoE202.exe`.
- The repository does not contain the original Unity source project.
- The current machine is macOS, but the bundled Unity environment is Windows-only.
- The Python side is legacy code that expects Unity ML-Agents plus PyTorch.
- The repo also contains an old `unityagents-0.4.0-py3-none-any.whl`, but the
  training script imports `mlagents_envs`, so compatibility must be verified
  instead of assumed.


What "Runnable Enough To Trust" Means
-------------------------------------

For this recovery pass, the goal is not to resume training immediately.

The first meaningful milestone is a smoke test that can:

1. launch the Unity environment,
2. reset the environment,
3. list the exposed behaviors,
4. print initial observation shapes,
5. send a few actions,
6. step the environment successfully,
7. print rewards,
8. close cleanly.

If that works, the repo is "runnable enough to trust" as a baseline.


Short-Term Runtime Path
-----------------------

The short-term path should assume Windows access in one of these forms:

- a native Windows machine,
- a Windows VM,
- a remote Windows box,
- any other environment that can directly run `CoE202.exe`.

This is the only practical path available from the files currently present in
the repository.


Long-Term Runtime Path
----------------------

Long-term macOS support is not blocked by Python alone. It is blocked by the
missing Unity build target.

To run this project natively on macOS, at least one of these would be required:

- the original Unity project so a macOS player can be exported,
- a previously exported macOS build of the environment,
- a replacement environment compatible with the existing Python code.

None of those artifacts are currently present in this repo.


What We Can Safely Do Now
-------------------------

Without changing the training stack yet, the repo can be improved by adding:

- a standalone smoke-test script,
- a legacy environment manifest for Python dependencies,
- a compatibility note that records which ML-Agents package/runtime
  combinations were tried and whether they passed.

This keeps the recovery work low-risk and avoids touching PPO logic before the
runtime path is verified.


What We Should Not Assume
-------------------------

- Do not assume the old wheel in the repo is the correct runtime dependency.
- Do not assume the current Python environment can talk to the Unity build.
- Do not assume macOS can run the bundled executable.
- Do not assume `main1.py` is the right first entrypoint for validation.


Recommended Recovery Order
--------------------------

1. Document the runtime constraints.
2. Add a sidecar smoke test for Unity connectivity.
3. Probe for a working Python package combination on Windows.
4. Lock the first passing dependency set into a legacy manifest.
5. Only then evaluate inference or training behavior.


Bottom Line
-----------

This repo can be revived, but the recovery path should begin with environment
compatibility, not with training refactors. The Windows Unity build is the
short-term source of truth, and native macOS execution remains blocked until a
macOS-compatible Unity build exists.
