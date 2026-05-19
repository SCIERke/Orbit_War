import unittest

from lib.planet import MyPlanet
from spaces.CosmosEnvironment import CosmosEnvironment


class TestCountShips(unittest.TestCase):
    def _make_env(self) -> CosmosEnvironment:
        return CosmosEnvironment([])

    def test_returns_zero_when_no_owned_assets(self):
        env = self._make_env()
        obs = {
            "player": 1,
            "planets": [{"owner": 2, "ships": 100}],
            "fleets": [{"owner": 2, "ships": 50}],
        }
        self.assertEqual(env._count_ships(player_id=1, obs=obs), 0)

    def test_sums_ships_on_owned_planets(self):
        env = self._make_env()
        obs = {
            "player": 0,
            "planets": [
                {"owner": 0, "ships": 7},
                {"owner": 1, "ships": 999},
                {"owner": 0, "ships": 3},
            ],
            "fleets": [],
        }
        self.assertEqual(env._count_ships(player_id=0, obs=obs), 10)

    def test_includes_owned_fleets(self):
        env = self._make_env()
        obs = {
            "player": 2,
            "planets": [{"owner": 2, "ships": 4}],
            "fleets": [
                {"owner": 2, "ships": 6},
                {"owner": 0, "ships": 200},
            ],
        }
        self.assertEqual(env._count_ships(player_id=2, obs=obs), 10)

    def test_counts_planets_and_fleets_together(self):
        env = self._make_env()
        obs = {
            "player": 1,
            "planets": [
                {"owner": 1, "ships": 10},
                {"owner": -1, "ships": 5},
            ],
            "fleets": [{"owner": 1, "ships": 3}],
        }
        self.assertEqual(env._count_ships(player_id=1, obs=obs), 13)


class _DummyObs:
    def __init__(self):
        self.step = 0
        self.player = 0
        self.angular_velocity = 0.0
        self.planets = [
            [0, 0, 50.0, 50.0, 1.0, 50, 1],
            [1, 1, 60.0, 50.0, 1.0, 50, 1],
        ]
        self.fleets = []


class _DummyState:
    def __init__(self):
        self.observation = _DummyObs()
        self.reward = 0
        self.status = "ACTIVE"


class _DummyKaggleEnv:
    def __init__(self):
        self.state = [_DummyState(), _DummyState()]
        self.last_actions = None
        self.configuration = type("cfg", (), {"shipSpeed": 6.0})()

    def reset(self, num_agents=2):
        pass

    def step(self, actions):
        self.last_actions = actions
        self.state[0].reward = 0
        self.state[0].status = "ACTIVE"


def _make_step_env():
    kaggle_env = _DummyKaggleEnv()
    env = CosmosEnvironment([], kaggle_env=kaggle_env, num_agents=2, agent_index=0)
    env._last_obs = {
        "step": 0,
        "player": 0,
        "angular_velocity": 0.0,
        "planets": [
            {"id": 0, "owner": 0, "x": 50.0, "y": 50.0, "radius": 1.0, "ships": 50, "production": 1},
            {"id": 1, "owner": 1, "x": 60.0, "y": 50.0, "radius": 1.0, "ships": 50, "production": 1},
        ],
        "fleets": [],
    }
    return env, kaggle_env


class TestStepKaggleWrapper(unittest.TestCase):
    def test_sends_correct_source_planet_id(self):
        import numpy as np
        env, kaggle_env = _make_step_env()
        env.step({"source": 0, "target": 1, "fraction": np.array([1.0])})
        sent_move = kaggle_env.last_actions[0][0]
        self.assertEqual(sent_move[0], 0)

    def test_computes_angle_towards_target(self):
        # planet 0 at (50,50), planet 1 at (60,50) — angle should be 0.0 (due east)
        import numpy as np
        env, kaggle_env = _make_step_env()
        env.step({"source": 0, "target": 1, "fraction": np.array([1.0])})
        sent_move = kaggle_env.last_actions[0][0]
        self.assertAlmostEqual(sent_move[1], 0.0, places=5)

    def test_ship_count_uses_fraction(self):
        import numpy as np
        env, kaggle_env = _make_step_env()
        env.step({"source": 0, "target": 1, "fraction": np.array([0.5])})
        sent_move = kaggle_env.last_actions[0][0]
        self.assertEqual(sent_move[2], 25)  # 50 ships * 0.5


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
        env = self._env()
        prev = self._obs([{"owner": -1, "ships": 0, "production": 5}])
        curr = self._obs([{"owner": 0,  "ships": 0, "production": 5}])
        reward = env._compute_reward(curr, prev)
        # my_prod 0->5, enemy_prod stays 0
        # score delta = 5*30 + 5*15 = 225, /1000 = 0.225
        self.assertAlmostEqual(reward, 0.225, places=5)

    def test_enemy_production_gain_yields_negative_reward(self):
        env = self._env()
        prev = self._obs([{"owner": -1, "ships": 0, "production": 4}])
        curr = self._obs([{"owner": 1,  "ships": 0, "production": 4}])
        reward = env._compute_reward(curr, prev)
        # my_prod stays 0, enemy_prod 0->4
        # score delta = 0 - 4*15 = -60, /1000 = -0.060
        self.assertAlmostEqual(reward, -0.060, places=5)

    def test_neutral_planets_not_counted_as_enemy(self):
        env = self._env()
        obs  = self._obs([{"owner": 0, "ships": 10, "production": 3},
                          {"owner": -1, "ships": 5, "production": 10}])
        prev = self._obs([{"owner": 0, "ships": 10, "production": 3},
                          {"owner": -1, "ships": 5, "production": 10}])
        self.assertAlmostEqual(env._compute_reward(obs, prev), 0.0, places=5)

    def test_ship_accumulation_yields_small_positive_reward(self):
        env = self._env()
        prev = self._obs([{"owner": 0, "ships": 10, "production": 2}])
        curr = self._obs([{"owner": 0, "ships": 15, "production": 2}])
        # ships delta=5 * 0.5 = 2.5, /1000 = 0.0025
        self.assertAlmostEqual(env._compute_reward(curr, prev), 0.0025, places=5)

    def test_production_reward_larger_than_ship_hoard_reward(self):
        env = self._env()
        prev_cap  = self._obs([{"owner": -1, "ships": 0, "production": 4}])
        curr_cap  = self._obs([{"owner": 0,  "ships": 0, "production": 4}])
        capture_reward = env._compute_reward(curr_cap, prev_cap)

        prev_hoard = self._obs([{"owner": 0, "ships": 0,   "production": 0}])
        curr_hoard = self._obs([{"owner": 0, "ships": 100, "production": 0}])
        hoard_reward = env._compute_reward(curr_hoard, prev_hoard)

        self.assertGreater(capture_reward, hoard_reward)


if __name__ == "__main__":
    unittest.main()
