import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Dict, List, Optional


class WandbEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: Any,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 10,
        project: str = "orbit-war",
        run_name: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
        curriculum_opponent: Optional[Any] = None,
        curriculum_container: Optional[Any] = None,
        curriculum_threshold: float = 0.6,
        curriculum_min_phase0_steps: int = 750_000,
    ):
        super().__init__()
        self._eval_env = eval_env
        self._eval_freq = eval_freq
        self._n_eval_episodes = n_eval_episodes
        self._project = project
        self._run_name = run_name
        self._run_config = run_config or {}
        self._terminal_rewards: List[float] = []
        self._shaped_rewards: List[float] = []
        self._last_eval_step = 0
        self._curriculum_opponent = curriculum_opponent
        self._curriculum_container = curriculum_container
        self._curriculum_threshold = curriculum_threshold
        self._curriculum_min_phase0_steps = curriculum_min_phase0_steps
        self._curriculum_phase = 0  # 0=passive, 1=n_nearest_planet

    def _on_training_start(self) -> None:
        wandb.init(project=self._project, name=self._run_name, config=self._run_config, reinit=True)

    def _on_rollout_end(self) -> None:
        log = self.logger.name_to_value
        metrics: Dict[str, float] = {}
        for wandb_key, sb3_key in [
            ("train/value_loss",         "train/value_loss"),
            ("train/policy_loss",        "train/policy_gradient_loss"),
            ("train/entropy_loss",       "train/entropy_loss"),
            ("train/explained_variance", "train/explained_variance"),
            ("train/clip_fraction",      "train/clip_fraction"),
            ("train/approx_kl",          "train/approx_kl"),
        ]:
            val = log.get(sb3_key)
            if val is not None:
                metrics[wandb_key] = float(val)

        if self._terminal_rewards:
            metrics["reward/mean_terminal"] = float(np.mean(self._terminal_rewards))
            metrics["reward/mean_shaped"]   = float(np.mean(self._shaped_rewards))
            self._terminal_rewards.clear()
            self._shaped_rewards.clear()

        metrics["curriculum/phase"] = float(self._curriculum_phase)

        if metrics:
            wandb.log(metrics, step=self.num_timesteps)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "terminal_reward" in info:
                self._terminal_rewards.append(float(info["terminal_reward"]))
                self._shaped_rewards.append(float(info["shaped_reward"]))

        if self.num_timesteps - self._last_eval_step >= self._eval_freq:
            self._last_eval_step = self.num_timesteps
            win_rate = self._run_eval()
            self._maybe_advance_curriculum(win_rate)
        return True

    def _maybe_advance_curriculum(self, win_rate: float) -> None:
        if self._curriculum_phase == 0 and self._curriculum_opponent is not None:
            if win_rate >= self._curriculum_threshold and self.num_timesteps >= self._curriculum_min_phase0_steps:
                self._curriculum_phase = 1
                if self._curriculum_container is not None:
                    self._curriculum_container.set(self._curriculum_opponent)
                wandb.log({
                    "curriculum/phase": 1.0,
                    "curriculum/switched_at_timestep": float(self.num_timesteps),
                }, step=self.num_timesteps)
                print(f"\n[Curriculum] Phase 1: switched to hard opponent at {self.num_timesteps} steps "
                      f"(win_rate={win_rate:.2f} >= {self._curriculum_threshold})\n")

    def _run_eval(self) -> float:
        wins = draws = losses = 0
        ep_lengths: List[int] = []
        my_ships_list: List[float] = []
        enemy_ships_list: List[float] = []
        my_planets_list: List[float] = []
        enemy_planets_list: List[float] = []

        for _ in range(self._n_eval_episodes):
            obs = self._eval_env.reset()
            done = False
            ep_len = 0
            last_info: Dict[str, Any] = {}

            while not done:
                masks = self._eval_env.env_method("action_masks")[0]
                action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
                obs, _, dones, infos = self._eval_env.step(action)
                done = bool(dones[0])
                ep_len += 1
                if infos:
                    last_info = infos[0]

            ep_lengths.append(ep_len)
            tr = last_info.get("terminal_reward", 0.0)
            if tr > 0:
                wins += 1
            elif tr < 0:
                losses += 1
            else:
                draws += 1

            my_ships_list.append(float(last_info.get("my_ships", 0)))
            enemy_ships_list.append(float(last_info.get("enemy_ships", 0)))
            my_planets_list.append(float(last_info.get("my_planets", 0)))
            enemy_planets_list.append(float(last_info.get("enemy_planets", 0)))

        n = self._n_eval_episodes
        win_rate = wins / n
        wandb.log({
            "eval/win_rate":               win_rate,
            "eval/draw_rate":              draws / n,
            "eval/loss_rate":              losses / n,
            "eval/mean_episode_length":    float(np.mean(ep_lengths)),
            "eval/mean_my_ships_end":      float(np.mean(my_ships_list)),
            "eval/mean_enemy_ships_end":   float(np.mean(enemy_ships_list)),
            "eval/mean_my_planets_end":    float(np.mean(my_planets_list)),
            "eval/mean_enemy_planets_end": float(np.mean(enemy_planets_list)),
        }, step=self.num_timesteps)
        return win_rate

    def _on_training_end(self) -> None:
        wandb.finish()
