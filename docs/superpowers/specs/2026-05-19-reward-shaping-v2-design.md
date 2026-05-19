# Reward Shaping v2 Design

**Date:** 2026-05-19  
**Project:** orbit_war (Kaggle orbit_wars competition)  
**Scope:** Replace `_compute_reward()` in `CosmosEnvironment` with a production-aware multi-component reward that incentivizes planet expansion over ship hoarding.

---

## Problem

Current reward:
```python
return float(planet_ships + fleet_ships + n_planets * 10)
```
- Treats all planets equal regardless of production value
- Ship hoarding (1:1 weight) is safer than expansion (ships lost in transit)
- Agent wins vs easy opponents (n_nearest_planet) but hoards vs harder ones
- Does not encode the actual win condition: highest ship count at time's up = production dominance

---

## Win Condition Insight

From `docs/agents.md`: **win = highest ship count (planets + fleets) when time runs out**.  
Production = ships generated per turn = compounding advantage.  
An agent with 2× production will have ~2× ships at game end.  
Therefore: production dominance is the primary strategic signal, not static ship count.

---

## Architecture

**One file, one function:**

| File | Change |
|------|--------|
| `spaces/CosmosEnvironment.py` | Replace `_compute_reward()` body only |

No other files changed. `WandbEvalCallback` already logs `reward/mean_shaped` per rollout — new reward automatically visible in wandb dashboard without further changes.

Terminal reward (kaggle win/loss signal) unchanged.

---

## New Reward Formula

```python
def _compute_reward(self, obs, prev_obs):
    if prev_obs is None:
        return 0.0
    player = obs["player"]

    def score(o):
        my_ships = (
            sum(p["ships"] for p in o["planets"] if p["owner"] == player) +
            sum(f["ships"] for f in o["fleets"]  if f["owner"] == player)
        )
        my_prod    = sum(p["production"] for p in o["planets"] if p["owner"] == player)
        enemy_prod = sum(p["production"] for p in o["planets"]
                         if p["owner"] not in (player, -1))
        return float(my_ships * 0.5 + my_prod * 30.0 + (my_prod - enemy_prod) * 15.0)

    return (score(obs) - score(prev_obs)) / 1000.0
```

---

## Weight Rationale

| Component | Weight | Reason |
|-----------|--------|--------|
| `my_ships * 0.5` | 0.5 | Ships still matter for terminal outcome but are a means, not the end |
| `my_prod * 30.0` | 30 | Production = future ships every turn; 30× reflects compounding value |
| `(my_prod - enemy_prod) * 15.0` | 15 | Relative dominance: rewards attacking enemy planets, not just neutral ones |

---

## Scaling Analysis

- Typical game: ~40 planets, total production pool ~100–200
- `my_prod` at peak ~100 → `my_prod * 30 = 3000`
- Relative term swings ±1500
- `my_ships` may reach 500 → `* 0.5 = 250`
- Score range: ~±4500; step delta when capturing a planet: ~50–200
- After `/1000`: shaped reward ~0.05–0.2 per step
- Terminal reward dominates at ±1 (win/loss) — correct priority

---

## Tuning Notes

Weights (0.5 / 30 / 15) are starting values. Adjust based on wandb `reward/mean_shaped` curves:

- Agent still hoards → raise `my_prod` weight
- Agent over-expands and collapses → lower relative dominance term (15)
- Reward too noisy → increase `/1000` divisor

---

## Out of Scope

- Changes to observation features, action space, or opponents
- Curriculum learning or self-play
- Per-planet distance/threat weighting
