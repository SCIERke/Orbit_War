from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Any
import numpy as np
import json
import os
import time


def _mask_fn(env) -> np.ndarray:
    return env.action_masks()


class PPOAgent:
    HYPERPARAMS = {
        "learning_rate": 1e-4,
        "clip_range": 0.2,
        "n_steps": 2048,
        "batch_size": 64,
        "ent_coef": 0.01,
    }

    def __init__(self, seed: int = 42):
        from dotenv import load_dotenv
        load_dotenv()
        from spaces.CosmosEnvironment import CosmosEnvironment
        from agent.n_nearest_planet import agent as nearest_planet_agent
        self._seed = seed
        env = DummyVecEnv([lambda: ActionMasker(
            CosmosEnvironment.from_orbit_wars(opponent_agent=nearest_planet_agent), _mask_fn
        )])
        self.model = MaskablePPO(
            "MlpPolicy", env, verbose=1,
            seed=seed,
            **self.HYPERPARAMS,
        )
        self._eval_env = DummyVecEnv([lambda: ActionMasker(
            CosmosEnvironment.from_orbit_wars(opponent_agent=nearest_planet_agent), _mask_fn
        )])
        self._permanent_planet_ids: set = set()
        self._comet_ids: set = set()

    def learn(self, total_timesteps: int = 10_000, run_dir: str = "runs") -> str:
        from agent.wandb_eval_callback import WandbEvalCallback
        tag = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(run_dir, tag)
        os.makedirs(out, exist_ok=True)

        run_config = {"total_timesteps": total_timesteps, "seed": self._seed, **self.HYPERPARAMS}
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(run_config, f, indent=2)

        cb = WandbEvalCallback(
            eval_env=self._eval_env,
            eval_freq=50_000,
            n_eval_episodes=10,
            project="orbit-war",
            run_config=run_config,
        )
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=cb)

        model_path = os.path.join(out, "model")
        self.model.save(model_path)
        print(f"Run saved → {out}")
        return out

    def load(self, path: str) -> None:
        self.model = MaskablePPO.load(path, env=self.model.get_env())

    def reset_game(self) -> None:
        """Call at the start of each new game when using agent() directly."""
        self._permanent_planet_ids = set()
        self._comet_ids = set()

    def agent(self, obs: Any, config: Any = None) -> Any:
        from spaces.CosmosEnvironment import CosmosEnvironment, N_PLANETS_MAX
        from lib.ship import MyFleet

        obs_dict = CosmosEnvironment._kaggle_observation_as_dict(obs)

        current_ids = {p["id"] for p in obs_dict["planets"]}
        if not self._permanent_planet_ids:
            # first call in this game — establish permanent set, no comets yet
            self._permanent_planet_ids = current_ids
            self._comet_ids = set()
        else:
            self._comet_ids |= current_ids - self._permanent_planet_ids

        obs_arr = CosmosEnvironment._obs_to_array(obs_dict, comet_ids=self._comet_ids)

        # build mask inline — same logic as action_masks()
        planets_sorted = sorted(obs_dict["planets"], key=lambda p: p["id"])
        player = obs_dict["player"]
        n = min(len(planets_sorted), N_PLANETS_MAX)
        src_mask = [
            i < n and planets_sorted[i]["owner"] == player and planets_sorted[i]["ships"] > 0
            for i in range(N_PLANETS_MAX)
        ]
        if not any(src_mask):
            src_mask = [i < n for i in range(N_PLANETS_MAX)]
        masks = np.array(
            src_mask +
            [i < n for i in range(N_PLANETS_MAX)] +
            [True] * 101,
            dtype=bool,
        )

        action, _ = self.model.predict(obs_arr, action_masks=masks, deterministic=True)

        source_idx    = int(action[0])
        target_idx    = int(action[1])
        ship_fraction = int(action[2]) / 100.0

        if not planets_sorted:
            return []

        source = planets_sorted[source_idx]
        target = planets_sorted[target_idx]

        n_ships          = max(1, int(ship_fraction * source["ships"]))
        angular_velocity = float(obs_dict.get("angular_velocity", 0.0))
        current_turn     = int(obs_dict.get("step", 0))
        max_ship_speed   = getattr(config, "shipSpeed", 6.0) if config else 6.0

        mine_planet   = CosmosEnvironment._planet_from_obs_dict(source, angular_velocity)
        target_planet = CosmosEnvironment._planet_from_obs_dict(target, angular_velocity)
        fleet = MyFleet.from_planet(mine_planet, n_ships)
        angle = fleet._shoot_at_planet(target_planet, current_turn, max_ship_speed=max_ship_speed)

        return [[source["id"], float(angle), n_ships]]
