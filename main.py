from kaggle_environments import make
from agent.n_nearest_planet import agent

def main():
    env = make("orbit_wars", debug=True)
    env.run([agent, "random"])
    
    html_output = env.render(mode="html")
    
    with open("output.html", "w") as f:
        f.write(html_output)
    
    
if __name__ == "__main__":
    main()