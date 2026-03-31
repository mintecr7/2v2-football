`main1.py` Walkthrough
======================

This document explains the structure of `main1.py` in plain English.

The script is the orchestration layer for the whole project. It:

- launches the Unity environment,
- discovers the ML-Agents behaviors,
- builds state vectors,
- creates the PyTorch models,
- trains the learned team against a random opponent,
- saves checkpoints,
- then attempts to run an evaluation loop.


The Most Important Mental Model
-------------------------------

There are two separate concepts in the script:

- Unity behaviors, which appear to represent the two teams,
- player roles, which are goalie and striker.

The variable names are confusing here:

- `g_brain_name` is not really "the goalie brain",
- `b_brain_name` is not really "the striker brain".

Instead, each behavior contains two players:

- row `0` is treated as the goalie,
- row `1` is treated as the striker.

That means the script learns by role, not by team:

- one shared goalie policy is reused for both teams' goalies,
- one shared striker policy is reused for both teams' strikers.


Section 1: Environment Startup
------------------------------

The setup phase runs at import time near the top of the file.

Key steps:

1. Create the Unity environment with `UnityEnvironment(file_name="CoE202")`.
2. Reset the environment.
3. Read the available behavior names from `env.behavior_specs`.
4. Assume there are exactly two behaviors.
5. Assume each behavior contains exactly two controlled agents.

This means `main1.py` is not written as a reusable module. As soon as the file is run, it starts building the environment immediately.


Section 2: State Construction
-----------------------------

The script reads observations from both Unity behaviors using:

- `decision_steps_p, terminal_steps_p = env.get_steps(g_brain_name)`
- `decision_steps_b, terminal_steps_b = env.get_steps(b_brain_name)`

For each player, it concatenates two observation arrays:

- `obs[0][player_index, :]`
- `obs[1][player_index, :]`

The resulting state layout is:

- `cur_obs_g1`: team 0 goalie,
- `cur_obs_g2`: team 1 goalie,
- `cur_obs_s1`: team 0 striker,
- `cur_obs_s2`: team 1 striker.

Then it stacks them into two role-based matrices:

- `cur_obs_g`: both goalies,
- `cur_obs_s`: both strikers.

This is why `goalie_states` and `striker_states` each have shape `(2, state_size)`.


Section 3: Hyperparameters And Models
-------------------------------------

After the state sizes are inferred, the script defines PPO-style hyperparameters such as:

- `N_STEP`,
- `BATCH_SIZE`,
- `GAMMA`,
- `EPSILON`,
- `ENTROPY_WEIGHT`,
- role-specific learning rates.

Then it creates four networks:

- goalie actor,
- goalie critic,
- striker actor,
- striker critic.

The actor networks only see their own role-specific state vector.

The critic networks are larger because each critic gets a concatenation of all four players' states. This is a centralized-critic pattern:

- goalie critic input = goalie 0 + striker 0 + goalie 1 + striker 1,
- striker critic input = striker 0 + goalie 0 + striker 1 + goalie 1.

After creation, the script loads the saved checkpoint files if they exist.


Section 4: Agent And Optimizer Wrappers
---------------------------------------

Two helper objects are created for the learned team:

- `goalie_0 = Agent(...)`
- `striker_0 = Agent(...)`

These wrap:

- the actor model,
- rollout memory,
- basic action selection.

Two `Optimizer` instances are also created:

- one for the goalie networks,
- one for the striker networks.

Only these two learned-role agents accumulate memory and receive gradient updates.


Section 5: What `ppo_train()` Actually Does
-------------------------------------------

`ppo_train()` is the real training loop.

For every episode it:

1. Resets the Unity environment.
2. Rebuilds the initial states for all four players.
3. Resets episode scores.
4. Enters a `while True` loop for per-step interaction.

Inside each environment step:

1. The learned goalie policy chooses an action for team 0's goalie.
2. The learned striker policy chooses an action for team 0's striker.
3. The opponent team actions are generated randomly with TensorFlow.
4. The actions are sent to Unity with two `env.set_actions(...)` calls.
5. The simulation advances with `env.step()`.
6. New observations and rewards are read back.
7. Role-specific rewards are extracted from the returned arrays.
8. The learned team's experiences are appended to memory.
9. If a score or terminal-like condition is seen, the episode ends.

At the end of the episode:

1. The goalie optimizer updates the goalie actor/critic using the stored rollout.
2. The striker optimizer updates the striker actor/critic using the stored rollout.
3. All four checkpoint files are overwritten.
4. Rolling score statistics are printed.


Section 6: How Experience Storage Works
---------------------------------------

For each step, the script stores two experiences:

- one for the learned goalie,
- one for the learned striker.

Each stored experience contains:

- `actor_state`: the local state for the acting role,
- `critic_state`: the concatenated global state for all players,
- `action`,
- `log_prob`,
- `reward`.

This mirrors the design in `memory.py` and `optimizer.py`.


Section 7: What The PPO Update Is Doing
---------------------------------------

When `goalie_optimizer.learn(goalie_0.memory)` or `striker_optimizer.learn(striker_0.memory)` runs, the optimizer:

1. pulls all stored experiences out of memory,
2. computes discounted returns,
3. estimates values with the critic,
4. computes normalized advantages,
5. recomputes action log-probabilities,
6. applies a clipped PPO-style policy loss,
7. adds critic MSE loss,
8. subtracts an entropy bonus,
9. takes an optimizer step.

This is the main learning update for the project.


Section 8: Why The Opponent Looks Strange
-----------------------------------------

Even though the code creates shared goalie/striker models, the training loop only learns from one side's stored experiences.

The opponent is not trained in the same loop. Instead, its actions are generated randomly with TensorFlow.

So the project is closer to:

- train one team,
- against a random-action opponent,
- while sharing parameters by role.

It is not true self-play in the current form.


Section 9: The Evaluation Block
-------------------------------

After `ppo_train()` returns, the script immediately enters another loop labeled as testing.

This block appears intended to:

- run 50 post-training episodes,
- let both teams use the learned policies,
- print win statistics.

However, the section looks unfinished:

- it has no clear episode termination path inside the `while True`,
- some action handling differs from the training section,
- reward array creation is inconsistent with the training code.

It should be read as a rough experiment, not a polished evaluation pipeline.


Section 10: Things To Keep In Mind While Reading
------------------------------------------------

A few design choices make the file harder to read than it needs to be:

- the whole script runs top-to-bottom with global state,
- there is no `if __name__ == "__main__":` guard,
- team names and role names are mixed together,
- training and evaluation are both embedded directly in the same script,
- there are several commented-out experiments left in place.

If you ever refactor this project, `main1.py` is the first file to break apart into smaller functions.


Suggested Reading Order
-----------------------

If you want to revisit the code with minimal confusion, read in this order:

1. `readme.md`
2. `sample_code.py`
3. `model.py`
4. `agent.py`
5. `memory.py`
6. `optimizer.py`
7. `main1.py`

That order lets you understand the observation shape and network logic before reading the long training script.
