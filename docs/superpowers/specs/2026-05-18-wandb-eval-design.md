# wandb Logging & Model Evaluation Design

**Date:** 2026-05-18  
**Project:** orbit_war (Kaggle orbit_wars competition)  
**Scope:** Add wandb experiment tracking, game-performance evaluation, and reward-component logging to the MaskablePPO training pipeline.

---

## Goal

Log training health and game performance to wandb during training runs, evaluated every 50k timesteps, graphed in the wandb cloud dashboard. API key loaded from `.env`.

---

## Architecture

### Files Changed

| File | Change |
|------|--------|
| `spaces/CosmosEnvironment.py` | Enrich `step()` info dict with reward components and game stats |
| `agent/wandb_eval_callback.py` | New file: `WandbEvalCallback` |
| `agent/ppo.py` | Wire callback, load `.env`, create eval env |
| `requirements.txt` | Add `wandb`, `python-dotenv` |

### Data Flow

```
PPOAgent.learn()
  └── MaskablePPO.learn(callback=WandbEvalCallback)
        ├── every rollout  → _on_rollout_end() → wandb.log(train metrics)
        └── every 50_000 steps → _run_eval()
              └── 10 full episodes vs n_nearest_planet opponent
                    └── wandb.log(eval metrics)
```

### wandb Initialization

- `wandb.init(project="orbit-war", config=PPOAgent.HYPERPARAMS | {"total_timesteps": ..., "seed": ...})`
- API key: loaded from `.env` via `python-dotenv` before `wandb.init()`
- `wandb.finish()` called in `_on_training_end()`

---

## Metrics

### Training Metrics (every rollout, ~2048 steps)

Sourced from `self.logger.name_to_value` (SB3 internal logger):

| wandb key | SB3 key |
|-----------|---------|
| `train/value_loss` | `train/value_loss` |
| `train/policy_loss` | `train/policy_gradient_loss` |
| `train/entropy_loss` | `train/entropy_loss` |
| `train/explained_variance` | `train/explained_variance` |
| `train/clip_fraction` | `train/clip_fraction` |
| `train/approx_kl` | `train/approx_kl` |
| `time/total_timesteps` | `time/total_timesteps` |

### Eval Metrics (every 50k steps, 10 episodes)

Run 10 full episodes vs `n_nearest_planet` on a separate eval env instance:

| wandb key | Description |
|-----------|-------------|
| `eval/win_rate` | Fraction of episodes won |
| `eval/draw_rate` | Fraction drawn |
| `eval/loss_rate` | Fraction lost |
| `eval/mean_episode_length` | Avg steps per episode |
| `eval/mean_my_ships_end` | Avg agent ships at episode end |
| `eval/mean_enemy_ships_end` | Avg opponent ships at episode end |
| `eval/mean_my_planets_end` | Avg planets owned at episode end |
| `eval/mean_enemy_planets_end` | Avg opponent planets at episode end |

### Reward Breakdown (from `step()` info, aggregated per rollout)

| wandb key | Source |
|-----------|--------|
| `reward/mean_terminal` | `info["terminal_reward"]` averaged over rollout |
| `reward/mean_shaped` | `info["shaped_reward"]` averaged over rollout |

---

## Component Design

### `WandbEvalCallback` (`agent/wandb_eval_callback.py`)

```
WandbEvalCallback(BaseCallback)
  __init__(eval_env, eval_freq=50_000, n_eval_episodes=10, project="orbit-war", run_config=None)
  _on_training_start()   → wandb.init(project, config=run_config)
  _on_rollout_end()      → wandb.log(train metrics from logger.name_to_value)
  _on_step()             → accumulate info["terminal_reward"], info["shaped_reward"]
                           if num_timesteps % eval_freq == 0: _run_eval()
                           return True
  _run_eval()            → run n_eval_episodes on eval_env, collect stats, wandb.log(eval metrics)
  _on_training_end()     → wandb.finish()
```

Eval loop uses `model.predict(obs, action_masks=masks, deterministic=True)` — same inference path as production agent.

### `CosmosEnvironment.step()` info enrichment

Current return:
```python
{"status": state.status}
```

New return:
```python
{
    "status": state.status,
    "terminal_reward": terminal_reward,
    "shaped_reward": shaped_reward,
    "my_ships": <total ships owned by agent at this step>,
    "enemy_ships": <total ships owned by opponent>,
    "my_planets": <planets owned by agent>,
    "enemy_planets": <planets owned by opponent>,
}
```

Ship and planet counts computed from `self._last_obs` using existing `_count_ships()` helper and inline planet counting.

### `PPOAgent` changes (`agent/ppo.py`)

- `__init__`: load `.env` via `python-dotenv`; create `self._eval_env` (fresh `CosmosEnvironment` instance, wrapped with `ActionMasker`, wrapped in `DummyVecEnv`)
- `learn()`: replace `_CsvLoggerCallback` with `WandbEvalCallback(self._eval_env, ...)`; pass `run_config = {"total_timesteps": total_timesteps, "seed": self._seed, **self.HYPERPARAMS}`
- Keep `config.json` save as backup

---

## Dependencies

```
wandb>=0.17
python-dotenv>=1.0
```

`.env` file (already exists, not committed):
```
WANDB_API_KEY=<key>
```

---

## Out of Scope

- Hyperparameter tuning (log first, tune after seeing curves)
- TensorBoard (wandb only)
- Multiple opponent types in eval (only `n_nearest_planet` for now)
- Action distribution histograms (future iteration)
