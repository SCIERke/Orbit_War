# Reward Shaping v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `_compute_reward()` in `CosmosEnvironment` with a production-aware multi-component reward that makes owning high-production planets the primary signal, eliminating ship-hoarding behavior.

**Architecture:** Single function change inside `spaces/CosmosEnvironment.py`. New `score()` inner function weights production at 30×, relative production dominance at 15×, and raw ships at 0.5×. Delta of scores divided by 1000 is the shaped reward per step.

**Tech Stack:** Python, gymnasium, stable-baselines3, kaggle_environments (orbit_wars)

---

## File Map

| File | Action |
|------|--------|
| `spaces/CosmosEnvironment.py` | Modify `_compute_reward()` lines 191–202 |
| `tests/test_cosmos_environment.py` | Add `TestComputeRewardV2` test class |

---

### Task 1: Write failing tests for the new reward formula

**Files:**
- Modify: `tests/test_cosmos_environment.py`

- [ ] **Step 1: Add `TestComputeRewardV2` class to `tests/test_cosmos_environment.py`**

Append this class at the bottom of the file (before `if __name__ == "__main__"`):

```python
class TestComputeRewardV2(unittest.TestCase):
    def _env(self):
        return CosmosEnvironment([])

    def _obs(self, planets, fleets=None, player=0):
        return {
            "player": player,
            "planets": planets,
            "fleets": fleets or [],
        }

    def test_returns_zero_when_prev_obs_is_none(self):
        env = self._env()
        obs = self._obs([{"owner": 0, "ships": 10, "production": 5}])
        self.assertEqual(env._compute_reward(obs, None), 0.0)

    def test_production_gain_yields_positive_reward(self):
        # Agent captures a planet (production=5), all else equal
        env = self._env()
        prev = self._obs([
            {"owner": -1, "ships": 0, "production": 5},  # neutral, not counted
        ])
        curr = self._obs([
            {"owner": 0, "ships": 0, "production": 5},  # now owned by player 0
        ])
        reward = env._compute_reward(curr, prev)
        # my_prod goes 0->5, enemy_prod stays 0
        # score delta = (5*30 + 5*15) = 225, /1000 = 0.225
        self.assertAlmostEqual(reward, 0.225, places=5)

    def test_enemy_production_gain_yields_negative_reward(self):
        # Opponent captures a planet
        env = self._env()
        prev = self._obs([
            {"owner": -1, "ships": 0, "production": 4},
        ])
        curr = self._obs([
            {"owner": 1, "ships": 0, "production": 4},  # enemy captured it
        ])
        reward = env._compute_reward(curr, prev)
        # my_prod stays 0, enemy_prod goes 0->4
        # score delta = (0 - 4*15) = -60, /1000 = -0.060
        self.assertAlmostEqual(reward, -0.060, places=5)

    def test_neutral_planets_not_counted_as_enemy(self):
        # Neutral planets (owner=-1) must not be counted in enemy_prod
        env = self._env()
        obs = self._obs([
            {"owner": 0,  "ships": 10, "production": 3},
            {"owner": -1, "ships": 5,  "production": 10},  # neutral
        ])
        prev = self._obs([
            {"owner": 0,  "ships": 10, "production": 3},
            {"owner": -1, "ships": 5,  "production": 10},
        ])
        # no delta — reward should be 0
        self.assertAlmostEqual(env._compute_reward(obs, prev), 0.0, places=5)

    def test_ship_accumulation_yields_small_positive_reward(self):
        # Ships produced on owned planet, no production change
        env = self._env()
        prev = self._obs([{"owner": 0, "ships": 10, "production": 2}])
        curr = self._obs([{"owner": 0, "ships": 15, "production": 2}])
        # my_ships delta=5, my_prod delta=0, enemy_prod delta=0
        # score delta = 5*0.5 = 2.5, /1000 = 0.0025
        reward = env._compute_reward(curr, prev)
        self.assertAlmostEqual(reward, 0.0025, places=5)

    def test_production_reward_larger_than_ship_hoard_reward(self):
        # Capturing a production-4 planet should give more reward
        # than accumulating 100 ships
        env = self._env()
        prev_capture = self._obs([{"owner": -1, "ships": 0, "production": 4}])
        curr_capture = self._obs([{"owner": 0,  "ships": 0, "production": 4}])
        capture_reward = env._compute_reward(curr_capture, prev_capture)

        prev_hoard = self._obs([{"owner": 0, "ships": 0,   "production": 0}])
        curr_hoard = self._obs([{"owner": 0, "ships": 100, "production": 0}])
        hoard_reward = env._compute_reward(curr_hoard, prev_hoard)

        self.assertGreater(capture_reward, hoard_reward)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/schierke/Desktop/kaggle/orbit_war
python -m pytest tests/test_cosmos_environment.py::TestComputeRewardV2 -v
```

Expected: FAIL — current `_compute_reward` uses `n_planets * 10` not production, so `test_production_gain_yields_positive_reward` and others will fail.

---

### Task 2: Implement the new `_compute_reward`

**Files:**
- Modify: `spaces/CosmosEnvironment.py` lines 191–202

- [ ] **Step 1: Replace `_compute_reward` body**

Find and replace the entire `_compute_reward` method (currently at lines ~191–202):

```python
    def _compute_reward(self, obs: Dict[str, Any], prev_obs: Optional[Dict[str, Any]]) -> float:
        if prev_obs is None:
            return 0.0
        player = obs["player"]

        def score(o: Dict[str, Any]) -> float:
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

- [ ] **Step 2: Run all tests**

```bash
python -m pytest tests/test_cosmos_environment.py -v
```

Expected: All tests PASS including the new `TestComputeRewardV2` class.

- [ ] **Step 3: Commit**

```bash
git add spaces/CosmosEnvironment.py tests/test_cosmos_environment.py
git commit -m "feat: reward shaping v2 — production-weighted multi-component reward

- my_ships * 0.5 (means not end)
- my_prod * 30 (compounding future ships)
- (my_prod - enemy_prod) * 15 (relative dominance)
Fixes ship-hoarding behavior vs harder opponents.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Tuning Reference (post-training)

After a training run, check wandb `reward/mean_shaped`:

| Symptom | Adjustment |
|---------|-----------|
| Agent still hoards ships | Increase `my_prod` weight (30 → 40) |
| Agent over-expands and collapses | Decrease relative dominance weight (15 → 8) |
| Reward signal too noisy | Increase divisor (1000 → 2000) |
| Reward too sparse (near zero most steps) | Decrease divisor (1000 → 500) |
