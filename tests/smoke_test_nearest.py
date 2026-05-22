"""
Smoke test for n_nearest_planet agent.

Checks:
  1. Agent fires moves from turn 0
  2. Agent expands to >1 planet by step 50
  3. Agent expands to >3 planets by step 150
  4. Agent wins or draws vs passive opponent
  5. Both players expand when agent plays vs itself
  6. HTML render succeeds
"""

from agent.n_nearest_planet import agent as nearest_planet_agent
from kaggle_environments import make

PASS = "PASS"
FAIL = "FAIL"


def _run_vs_passive(seed: int = 0):
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run([nearest_planet_agent, lambda obs, cfg: []])
    return env


def _planet_counts(env, player: int):
    counts = []
    for step_state in env.steps:
        obs = step_state[player]["observation"]
        counts.append(sum(1 for p in obs.planets if p[1] == player))
    return counts


def check(label: str, condition: bool) -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    return condition


def test_fires_moves() -> bool:
    print("Test: agent fires moves from turn 0")
    env = make("orbit_wars", debug=True)
    fired = [False]

    def capture_agent(obs, cfg=None):
        result = nearest_planet_agent(obs, cfg)
        if result:
            fired[0] = True
        return result

    env.run([capture_agent, lambda o, c: []])
    return check("move fired during game", fired[0])


def test_expands_early() -> bool:
    print("Test: agent owns >1 planet by step 50")
    env = _run_vs_passive()
    counts = _planet_counts(env, player=0)
    step50 = counts[50] if len(counts) > 50 else counts[-1]
    return check(f"planets at step 50 = {step50} (need >1)", step50 > 1)


def test_expands_mid() -> bool:
    print("Test: agent owns >3 planets by step 150")
    env = _run_vs_passive()
    counts = _planet_counts(env, player=0)
    step150 = counts[150] if len(counts) > 150 else counts[-1]
    return check(f"planets at step 150 = {step150} (need >3)", step150 > 3)


def test_wins_vs_passive() -> bool:
    print("Test: agent wins or draws vs passive opponent")
    env = _run_vs_passive()
    reward = env.steps[-1][0]["reward"]
    return check(f"reward = {reward} (need >= 0)", reward is not None and reward >= 0)


def test_vs_self() -> bool:
    print("Test: both players expand when playing vs each other (step 100)")
    env = make("orbit_wars", debug=True)
    env.run([nearest_planet_agent, nearest_planet_agent])
    mid = env.steps[min(100, len(env.steps) - 1)]
    p0 = sum(1 for p in mid[0]["observation"].planets if p[1] == 0)
    p1 = sum(1 for p in mid[1]["observation"].planets if p[1] == 1)
    return check(f"P0={p0} planets, P1={p1} planets (need >1 each)", p0 > 1 and p1 > 1)


def test_html_output() -> bool:
    print("Test: HTML render produces output")
    env = _run_vs_passive()
    try:
        html = env.render(mode="html", width=800, height=600)
        with open("output_nearest.html", "w") as f:
            f.write(html)
        return check(f"output_nearest.html written ({len(html)} bytes)", len(html) > 100)
    except Exception as e:
        return check(f"render failed: {e}", False)


if __name__ == "__main__":
    results = [
        test_fires_moves(),
        test_expands_early(),
        test_expands_mid(),
        test_wins_vs_passive(),
        test_vs_self(),
        test_html_output(),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    if passed < total:
        raise SystemExit(1)
