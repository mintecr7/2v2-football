Compatibility Probe Note
========================

This document defines how runtime compatibility is judged for the current
recovery pass and how a passing dependency combination should be recorded.


Probe Goal
----------

The goal is to discover the first Python package combination that can talk to
the existing Unity build closely enough to support a smoke test.

This is not a model-quality benchmark. It is only a runtime validation step.


Success Criteria
----------------

A package/runtime combination counts as passing only if all of the following
are true:

1. `smoke_test.py` launches the Unity environment.
2. The environment resets successfully.
3. At least one behavior is exposed after reset.
4. Initial observation shapes are printed.
5. At least one `set_actions` + `step` cycle succeeds.
6. The script exits with status code `0`.


Probe Procedure
---------------

1. Use Windows or another environment that can actually run `CoE202.exe`.
2. Create a fresh virtual environment for the probe.
3. Install candidate packages from `requirements-legacy.txt`.
4. Run:
   `python smoke_test.py --env-path CoE202 --steps 1`
5. If that fails after launch, iterate on package compatibility.
6. When the smoke test passes, freeze the exact package versions and replace
   the broad entries in `requirements-legacy.txt` with exact pins.


How To Classify Failures
------------------------

- Missing file: the Unity build path is wrong or unavailable.
- Missing dependency: the local Python environment is incomplete.
- Protocol mismatch: Python packages can import, but the environment cannot be
  reset or stepped successfully.
- Action path failure: the environment launches, but `set_actions` or `step`
  raises an error.
- Platform mismatch: the host cannot run the bundled Windows executable.


Current Status
--------------

No passing compatibility baseline has been recorded yet.

The local checks performed during this recovery pass confirmed:

- the repo can resolve `CoE202` to `CoE202.exe`,
- the smoke test fails fast on a bad path,
- the current macOS environment is still blocked before Unity launch because
  the required legacy Python dependencies are not installed.


Baseline Rule
-------------

The first package combination that passes the smoke test on a Windows-capable
host becomes the locked legacy baseline for this repository.
