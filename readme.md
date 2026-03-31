2v2 Football Reinforcement Learning Project
===========================================

This repository is a class project that combines:

- a compiled Unity soccer environment,
- Python scripts that talk to the Unity environment through ML-Agents,
- PyTorch models for policy/value learning,
- saved model checkpoints from an earlier training run.

The repository is not a full Unity source project. The Unity side is already built and packaged as an executable.


What Is In This Repo
--------------------

The main pieces are:

- `CoE202.exe`: the compiled Unity simulator.
- `CoE202_Data/`: Unity build data and ML-Agents timer metadata.
- `main1.py`: the main training script. It also runs an evaluation block after training.
- `smoke_test.py`: a minimal runtime smoke test for launching the Unity environment and stepping a few actions.
- `sample_code.py`: a small environment-inspection script used to poke at observations/actions.
- `agent.py`: action-selection wrapper around the actor model plus rollout memory.
- `model.py`: actor and critic neural networks.
- `memory.py`: simple rollout storage.
- `optimizer.py`: PPO-style optimization logic.
- `requirements-legacy.txt`: provisional dependency manifest for the runtime recovery pass.
- `checkpoint_*.pth`: saved PyTorch weights for goalie and striker actor/critic models.
- `unityagents-0.4.0-py3-none-any.whl`: an old wheel bundled with the project.
- `docs/runtime-recovery.md`: the current runtime/platform recovery guide.
- `docs/compatibility-probe.md`: the success criteria and status note for locking a working dependency baseline.


High-Level Architecture
-----------------------

The project uses Unity ML-Agents to run a 2v2 soccer simulation.

At a high level:

1. Python launches the Unity environment.
2. Unity exposes two behavior groups.
3. Each behavior group appears to represent one team.
4. Inside each team, agent index `0` is treated as the goalie and agent index `1` as the striker.
5. Python builds one shared policy for all goalies and one shared policy for all strikers.
6. During training, one team is controlled by the learned policies and the other team is mostly random.
7. Rollouts are stored in memory and used for a PPO-style update.
8. Updated weights are written back to the checkpoint files.

Important naming note:

- In `main1.py`, variables named `g_brain_name` and `b_brain_name` are confusing.
- They are closer to "team A behavior" and "team B behavior" than "goalie brain" and "blue brain".
- Goalie versus striker is determined by the row inside each behavior's observations/actions.


How The Main Script Works
-------------------------

`main1.py` is the center of the project.

Startup flow:

1. Create the Unity environment with `UnityEnvironment(file_name="CoE202")`.
2. Reset the environment and grab the two behavior names.
3. Read observations from both behaviors.
4. Build per-agent states by concatenating two observation tensors.
5. Infer state sizes and define action sizes.
6. Create actor/critic networks for the goalie role and striker role.
7. Load saved checkpoints if they exist.
8. Wrap the models in `Agent` and `Optimizer` helper classes.

Training flow (`ppo_train()`):

1. Reset the environment at the start of each episode.
2. Rebuild current states for both teams.
3. Ask the learned goalie and striker policies for actions for team 0.
4. Generate random actions for team 1 using TensorFlow.
5. Send actions to Unity with `env.set_actions(...)`.
6. Step the simulation with `env.step()`.
7. Read the new observations and rewards.
8. Store experiences in memory for the learned goalie and striker agents.
9. End the episode when a score-related reward or terminal-like condition is detected.
10. Run the PPO-style update from `optimizer.py`.
11. Save the updated checkpoints.

Evaluation flow:

- After training, `main1.py` enters a second loop intended to play evaluation episodes.
- That block looks unfinished and should be treated with caution before reuse.


File Guide
----------

`main1.py`

- Owns the end-to-end workflow.
- Launches Unity.
- Builds observations.
- Selects actions.
- Stores rewards and trajectories.
- Calls the optimizer.
- Saves checkpoints.

`agent.py`

- Defines the `Agent` helper.
- Uses the actor model to produce an action and log probability.
- Stores step data in `Memory`.

`model.py`

- Defines `ActorModel` and `CriticModel`.
- The actor is a small feed-forward network with two hidden layers.
- The critic is another feed-forward network that predicts a scalar value.
- Both models support `load()` and `checkpoint()`.

`memory.py`

- Stores experience tuples:
  `actor_state`, `critic_state`, `action`, `log_prob`, `reward`.
- Returns stacked NumPy arrays for optimization.

`optimizer.py`

- Implements a PPO-like update loop.
- Computes discounted returns.
- Computes advantages from critic values.
- Applies the clipped PPO objective plus a value loss and entropy term.

`sample_code.py`

- Used for observation debugging.
- Splits the front and back ray-sensor observations into readable chunks.
- Sends fixed actions to both teams while printing sensor values.


Observation Layout
------------------

The state for each player is built by concatenating two observation arrays:

- `decision_steps_*.obs[0][player_index, :]`
- `decision_steps_*.obs[1][player_index, :]`

`sample_code.py` suggests that:

- the front sensor has `33` slices of `8` values each,
- the back sensor has `9` slices of `8` values each,
- together this produces a `336`-value state per agent.

That matches the intended idea in `main1.py`, where each player's state is built from the two sensor blocks.


Model/Training Design
---------------------

The project uses role-based parameter sharing:

- one actor/critic pair for all goalies,
- one actor/critic pair for all strikers.

This means:

- both teams' goalies share the same goalie model,
- both teams' strikers share the same striker model.

In practice, the current training script mainly learns from one team's stored experiences while the opponent behaves randomly.


Saved Checkpoints
-----------------

These files are loaded at startup if present:

- `checkpoint_goalie_actor.pth`
- `checkpoint_goalie_critic.pth`
- `checkpoint_striker_actor.pth`
- `checkpoint_striker_critic.pth`

If you run `main1.py`, it will attempt to overwrite those checkpoint files during training.


Known Caveats
-------------

This repo is useful as a snapshot of the project, but it has several rough edges:

- The Unity environment is only present as a Windows executable (`CoE202.exe`).
- The repository does not include the Unity source project.
- The Python code mixes old/new ML-Agents ideas and may not run cleanly today without version pinning.
- `tensorflow` is imported only to create random opponent actions.
- The actor implementation is unusual: it returns `torch.sign(...)` of the action head while also building a categorical distribution from softmax logits.
- The evaluation section in `main1.py` appears incomplete.
- The repo contains only one git commit, so there is no development history to reconstruct intent from.


Runtime Recovery Starting Point
-------------------------------

If you want to validate the environment before touching training code, start
with the smoke test:

- `python smoke_test.py --env-path CoE202 --steps 10`

That script is intended to:

- resolve the Unity environment path,
- print behavior names and observation shapes,
- send a few actions,
- print rewards,
- exit cleanly on success or with an explicit error code on failure.

Important platform note:

- The bundled simulator is `CoE202.exe`, so native macOS execution is blocked
  unless you recover the original Unity project or a separate macOS build.
- Short-term runtime validation should assume Windows access or a Windows VM.


Practical Notes If You Want To Revive It
----------------------------------------

You will likely need to do some cleanup before rerunning training:

1. Run the Unity environment on Windows or obtain a macOS build.
2. Recreate a Python environment with compatible versions of:
   `numpy`, `torch`, `tensorflow`, and `mlagents_envs`.
3. Check ML-Agents protocol compatibility between the Python package and the Unity build.
4. Review `main1.py` before training, especially the action formatting, reward logic, and evaluation loop.
5. Decide whether to keep the current "learn against a random opponent" setup or convert it to self-play.


Short Summary
-------------

This project is a Unity soccer simulator plus Python PPO-style training code. The code uses shared models for the goalie role and striker role, stores rollouts, updates actor/critic networks, and saves checkpoints. The repo still explains the original project well enough to study, but it will need modernization before it can be run reliably again.
