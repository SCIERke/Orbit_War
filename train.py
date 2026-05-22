from agent.ppo import PPOAgent
import os

if __name__ == "__main__":
    ppo_agent = PPOAgent(seed=42)
    run_path = ppo_agent.learn(total_timesteps=3_000_000, run_dir="runs")

    print("\n--- Post-train performance test ---")
    from tests.performance_test import run_performance_test
    run_performance_test(model_path=os.path.join(run_path, "model"), n_games=20)