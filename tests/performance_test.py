from agent.ppo import PPOAgent
from agent.n_nearest_planet import agent as nearest_planet_agent
from kaggle_environments import make
import glob
import os


def run_performance_test(model_path: str = None, n_games: int = 20) -> float:
    ppo = PPOAgent()
    if model_path is None:
        models = sorted(glob.glob("runs/*/model.zip"))
        if not models:
            print("No saved model found.")
            return 0.0
        model_path = models[-1].replace(".zip", "")
    print(f"Loading {model_path}")
    ppo.load(model_path)
    run_dir = os.path.dirname(model_path)

    wins = 0
    for i in range(n_games):
        ppo.reset_game()
        env = make("orbit_wars", configuration={"seed": i}, debug=False)
        env.run([ppo.agent, nearest_planet_agent])
        final = env.steps[-1]
        my_reward = final[0]["reward"]
        opp_reward = final[1]["reward"]
        won = my_reward is not None and (opp_reward is None or my_reward > opp_reward)
        wins += int(won)
        print(f"  Game {i+1}/{n_games}: {'WIN' if won else 'LOSS'}  "
              f"(reward={my_reward}, opp={opp_reward})")

    win_rate = wins / n_games
    print(f"\nWin rate vs n_nearest_planet: {win_rate:.0%} ({wins}/{n_games})")

    try:
        ppo.reset_game()
        env_html = make("orbit_wars", configuration={"seed": 42}, debug=False)
        env_html.run([ppo.agent, nearest_planet_agent])
        html = env_html.render(mode="html", width=800, height=600)
        out_path = os.path.join(run_dir, "output.html")
        with open(out_path, "w") as f:
            f.write(html)
        print(f"Saved {out_path}")
    except Exception:
        pass

    return win_rate


if __name__ == "__main__":
    run_performance_test()
