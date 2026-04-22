"""Entry point for multi-agent training experiments."""

import argparse
import yaml
from pathlib import Path
from src.envs.coop_nav import CoopNavEnv
from src.agents.ppo_agent import PPOAgent


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.agents:
        cfg["env"]["num_agents"] = args.agents
    if args.steps:
        cfg["training"]["total_steps"] = args.steps

    env = CoopNavEnv(num_agents=cfg["env"]["num_agents"],
                     grid_size=cfg["env"]["grid_size"])

    agents = [PPOAgent(obs_dim=env.obs_dim,
                       act_dim=env.act_dim,
                       lr=cfg["training"]["lr"])
              for _ in range(cfg["env"]["num_agents"])]

    print(f"Starting training: {cfg['env']['num_agents']} agents, "
          f"{cfg['training']['total_steps']:,} steps")

    step = 0
    obs = env.reset()
    while step < cfg["training"]["total_steps"]:
        actions = [agent.act(o) for agent, o in zip(agents, obs)]
        obs, rewards, done, _ = env.step(actions)
        for agent, reward in zip(agents, rewards):
            agent.store(reward)
        if done:
            obs = env.reset()
            for agent in agents:
                agent.update()
        step += 1
        if step % 10000 == 0:
            print(f"  step {step:>8,} | mean_reward={sum(rewards)/len(rewards):.3f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
