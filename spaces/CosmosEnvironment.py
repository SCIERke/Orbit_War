import gymnasium as gym
import numpy as np
from lib.planet import MyPlanet
from typing import Any, Dict, List, Optional, Tuple, Union
from lib.ship import MyFleet

try:
    from kaggle_environments import make as kaggle_make
except ImportError:
    kaggle_make = None

N_PLANETS_MAX = 60  # 40 permanent + 20 comets
N_PLANET_FEATS = 6
OBS_SIZE = N_PLANETS_MAX * N_PLANET_FEATS + 6
ACTION_MASK_SIZE = N_PLANETS_MAX + N_PLANETS_MAX + 101


class CosmosEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "html", "ipython", "json"]}

    def __init__(
        self,
        Planets: List[MyPlanet],
        *,
        kaggle_env: Optional[Any] = None,
        num_agents: int = 2,
        agent_index: int = 0,
        render_mode: Optional[str] = "human",
        opponent_agent: Optional[Any] = None,
    ):
        self.planets = Planets
        self._kaggle = kaggle_env
        self._num_agents = num_agents
        self._agent_index = agent_index
        self._last_obs: Optional[Dict[str, Any]] = None
        self._prev_obs: Optional[Dict[str, Any]] = None
        self._permanent_planet_ids: set = set()  # planet IDs present at episode start
        self._comet_ids: set = set()             # IDs that spawned mid-episode
        self.render_mode = render_mode
        self._opponent_agent = opponent_agent

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([N_PLANETS_MAX, N_PLANETS_MAX, 101])

    @classmethod
    def from_orbit_wars(
        cls,
        *,
        configuration: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        num_agents: int = 2,
        agent_index: int = 0,
        opponent_agent: Optional[Any] = None,
    ) -> "CosmosEnvironment":
        if kaggle_make is None:
            raise ImportError("Install kaggle-environments to use from_orbit_wars()")
        kaggle_env = kaggle_make("orbit_wars", configuration=configuration or {}, debug=debug)
        return cls(
            [],
            kaggle_env=kaggle_env,
            num_agents=num_agents,
            agent_index=agent_index,
            opponent_agent=opponent_agent,
        )

    # ------------------------------------------------------------------
    # Action masking (MaskablePPO)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        if self._last_obs:
            planets = sorted(self._last_obs["planets"], key=lambda p: p["id"])
            player  = self._last_obs["player"]
            n       = min(len(planets), N_PLANETS_MAX)
            # source: only planets owned by this player with ships > 0
            src_mask = [
                i < n and planets[i]["owner"] == player and planets[i]["ships"] > 0
                for i in range(N_PLANETS_MAX)
            ]
            # fallback: if player owns nothing, allow all valid planets
            if not any(src_mask):
                src_mask = [i < n for i in range(N_PLANETS_MAX)]
        else:
            n        = N_PLANETS_MAX
            src_mask = [True] * N_PLANETS_MAX

        tgt_mask      = [i < n for i in range(N_PLANETS_MAX)]
        fraction_mask = [True] * 101
        return np.array(src_mask + tgt_mask + fraction_mask, dtype=bool)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kaggle_observation_as_dict(raw: Any) -> Dict[str, Any]:
        planets = []
        for p in raw.planets:
            planets.append({
                "id":         int(p[0]),
                "owner":      int(p[1]),
                "x":          float(p[2]),
                "y":          float(p[3]),
                "radius":     float(p[4]),
                "ships":      int(p[5]),
                "production": int(p[6]),
            })
        fleets = []
        for f in raw.fleets:
            fleets.append({
                "id":             int(f[0]),
                "owner":          int(f[1]),
                "x":              float(f[2]),
                "y":              float(f[3]),
                "angle":          float(f[4]),
                "from_planet_id": int(f[5]),
                "ships":          int(f[6]),
            })
        return {
            "step":             int(raw.step),
            "player":           int(raw.player),
            "angular_velocity": float(raw.angular_velocity),
            "planets":          planets,
            "fleets":           fleets,
        }

    @staticmethod
    def _obs_to_array(obs: Dict[str, Any], comet_ids: Optional[set] = None) -> np.ndarray:
        from game_types.planet import PlanetType
        player = obs["player"]
        angular_velocity = float(obs.get("angular_velocity", 0.0))
        planets = sorted(obs["planets"], key=lambda p: p["id"])

        planet_feats: List[float] = []
        for p in planets[:N_PLANETS_MAX]:
            if p["owner"] == player:
                owner_flag = 1.0
            elif p["owner"] == -1:
                owner_flag = 0.0
            else:
                owner_flag = -1.0

            planet_obj = MyPlanet(
                id=int(p["id"]), owner=int(p["owner"]),
                x=float(p["x"]), y=float(p["y"]),
                radius=float(p["radius"]), ships=int(p["ships"]),
                production=int(p["production"]),
                angular_velocity=angular_velocity,
            )
            is_orbital = 1.0 if planet_obj.planet_type == PlanetType.ORBITAL else 0.0
            is_comet   = 1.0 if (comet_ids and p["id"] in comet_ids) else 0.0
            # 0.0=static, 0.5=orbital permanent, 1.0=comet
            planet_type_feat = is_comet if is_comet else is_orbital * 0.5

            planet_feats += [
                owner_flag,
                p["ships"] / 500.0,
                p["production"] / 10.0,
                p["x"] / 100.0,
                p["y"] / 100.0,
                planet_type_feat,
            ]
        while len(planet_feats) < N_PLANETS_MAX * N_PLANET_FEATS:
            planet_feats += [0.0] * N_PLANET_FEATS

        my_transit    = sum(f["ships"] for f in obs["fleets"] if f["owner"] == player)
        enemy_transit = sum(f["ships"] for f in obs["fleets"] if f["owner"] != player)
        my_planets    = sum(1 for p in planets if p["owner"] == player)
        enemy_planets = sum(1 for p in planets if p["owner"] not in (player, -1))
        my_prod       = sum(p["production"] for p in planets if p["owner"] == player)

        global_feats = [
            my_transit  / 1000.0,
            enemy_transit / 1000.0,
            my_planets    / max(len(planets), 1),
            enemy_planets / max(len(planets), 1),
            obs.get("step", 0) / 400.0,
            my_prod       / 50.0,
        ]

        return np.clip(
            np.array(planet_feats + global_feats, dtype=np.float32), -1.0, 1.0
        )

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Move resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _planet_from_obs_dict(planet: Dict[str, Any], angular_velocity: float) -> MyPlanet:
        return MyPlanet(
            id=int(planet["id"]),
            owner=int(planet["owner"]),
            x=float(planet["x"]),
            y=float(planet["y"]),
            radius=float(planet["radius"]),
            ships=int(planet["ships"]),
            production=int(planet["production"]),
            angular_velocity=float(angular_velocity),
        )

    def _resolve_move(
        self,
        config: Any,
        action: Union[Dict[str, Any], np.ndarray, List[int], Tuple[int, int, int]],
    ) -> List[Union[int, float]]:
        if isinstance(action, dict):
            source_idx = int(action["source"])
            target_idx = int(action["target"])
            ship_fraction = float(np.asarray(action["fraction"]).flat[0])
        else:
            action_arr = np.asarray(action).reshape(-1)
            if action_arr.size != 3:
                raise ValueError(f"Expected action with 3 values, got shape {action_arr.shape}")
            source_idx = int(action_arr[0])
            target_idx = int(action_arr[1])
            fraction_bucket = int(action_arr[2])
            ship_fraction = fraction_bucket / 100.0
        ship_fraction = float(np.clip(ship_fraction, 0.0, 1.0))

        max_ship_speed = getattr(config, "shipSpeed", 6.0) if config is not None else 6.0

        planets: List[Dict[str, Any]] = []
        angular_velocity = 0.0
        current_turn = 0
        if self._last_obs is not None:
            planets          = sorted(self._last_obs["planets"], key=lambda p: p["id"])
            angular_velocity = float(self._last_obs.get("angular_velocity", 0.0))
            current_turn     = int(self._last_obs.get("step", 0))

        if not planets:
            return [0, 0.0, 1]

        source_idx  = int(np.clip(source_idx, 0, len(planets) - 1))
        target_idx  = int(np.clip(target_idx, 0, len(planets) - 1))
        source_data = planets[source_idx]
        target_data = planets[target_idx]

        n_ships = max(1, int(ship_fraction * source_data["ships"]))

        mine_planet   = self._planet_from_obs_dict(source_data, angular_velocity)
        target_planet = self._planet_from_obs_dict(target_data, angular_velocity)

        fleet = MyFleet.from_planet(mine_planet, n_ships)
        direction_angle = fleet._shoot_at_planet(target_planet, current_turn, max_ship_speed=max_ship_speed)

        return [source_data["id"], float(direction_angle), n_ships]

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._prev_obs = None
        self._permanent_planet_ids = set()
        self._comet_ids = set()

        if self._kaggle is not None:
            actual_seed = seed if seed is not None else int(np.random.randint(0, 2**31))
            self._kaggle.info.pop("seed", None)
            self._kaggle.configuration.seed = actual_seed
            self._kaggle.reset(num_agents=self._num_agents)
            raw_obs = self._kaggle.state[self._agent_index].observation
            self._last_obs = self._kaggle_observation_as_dict(raw_obs)
            self._permanent_planet_ids = {p["id"] for p in self._last_obs["planets"]}
            return self._obs_to_array(self._last_obs, comet_ids=set()), {}

        self._last_obs = None
        return np.zeros(OBS_SIZE, dtype=np.float32), {}

    def step(
        self,
        action: Union[Dict[str, Any], np.ndarray, List[int], Tuple[int, int, int]],
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._kaggle is not None:
            config = self._kaggle.configuration

            move = self._resolve_move(config, action)
            actions = [[] for _ in range(self._num_agents)]
            actions[self._agent_index] = [move]

            if self._opponent_agent is not None:
                opp_idx = 1 - self._agent_index
                opp_raw = self._kaggle.state[opp_idx].observation
                try:
                    opp_action = self._opponent_agent(opp_raw, config)
                    if opp_action:
                        actions[opp_idx] = opp_action
                except Exception:
                    pass

            self._prev_obs = self._last_obs
            self._kaggle.step(actions)
            raw_obs = self._kaggle.state[self._agent_index].observation
            self._last_obs = self._kaggle_observation_as_dict(raw_obs)

            state      = self._kaggle.state[self._agent_index]
            terminated = state.status != "ACTIVE"
            terminal_reward = 0.0 if state.reward is None else float(state.reward)
            shaped_reward   = self._compute_reward(self._last_obs, self._prev_obs)
            reward = terminal_reward + shaped_reward

            current_ids = {p["id"] for p in self._last_obs["planets"]}
            self._comet_ids |= current_ids - self._permanent_planet_ids
            obs_arr = self._obs_to_array(self._last_obs, comet_ids=self._comet_ids)

            player = self._last_obs["player"]
            opp = 1 - player
            planets = self._last_obs["planets"]
            info = {
                "status": state.status,
                "terminal_reward": terminal_reward,
                "shaped_reward": shaped_reward,
                "my_ships": self._count_ships(player, self._last_obs),
                "enemy_ships": self._count_ships(opp, self._last_obs),
                "my_planets": sum(1 for p in planets if p["owner"] == player),
                "enemy_planets": sum(1 for p in planets if p["owner"] == opp),
            }
            return obs_arr, reward, terminated, False, info

        return np.zeros(OBS_SIZE, dtype=np.float32), 0.0, False, False, {}

    def render(self) -> Optional[str]:
        if self._kaggle is None:
            return None
        return self._kaggle.render(mode=self.render_mode)

    def set_opponent_agent(self, agent) -> None:
        self._opponent_agent = agent

    def _count_ships(self, player_id: int, obs: Dict[str, Any]) -> int:
        total = sum(p["ships"] for p in obs["planets"] if p["owner"] == player_id)
        total += sum(f["ships"] for f in obs["fleets"]  if f["owner"] == player_id)
        return total
