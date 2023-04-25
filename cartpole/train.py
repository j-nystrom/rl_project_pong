
import argparse
import gymnasium as gym
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--env", choices=["CartPole-v1"], default="CartPole-v1")
parser.add_argument(
    "--evaluate_freq",
    type=int,
    default=25,
    help="How often to run evaluation.",
    nargs="?",
)
parser.add_argument(
    "--evaluation_episodes",
    type=int,
    default=5,
    help="Number of evaluation episodes.",
    nargs="?",
)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {"CartPole-v1": config.CartPole}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks, incl. target network
    # They will be initialized with random weights
    dqn = DQN(env_config=env_config).to(device)
    target_dqn = DQN(env_config=env_config).to(device)

    # Create replay memory
    memory = ReplayMemory(env_config["memory_size"])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config["lr"])

    # Keep track of the best evaluation mean return achieved so far
    best_mean_return = -float("Inf")

    # Track number of steps across all episodes
    steps = 0

    for episode in range(env_config["n_episodes"]):
        terminated = False
        obs, info = env.reset()

        # Get first observation from environment
        obs = preprocess(obs, env=args.env).unsqueeze(0)

        while not terminated:
            steps += 1

            # Get action to take
            # Note: Added steps as argument, for annealing
            action = dqn.act(obs, steps).item()

            # Act in the true environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Preprocess incoming observation and push to replay memory
            if not terminated:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
                action = torch.tensor(action, device=device).int().unsqueeze(0)
                reward = torch.tensor(reward, device=device).float().unsqueeze(0)
                memory.push(obs, action, next_obs, reward)
                obs = next_obs

            # Run optimize() function every env_config["train_frequency"] steps
            if steps % env_config["train_frequency"] == 0:
                loss = optimize(dqn, target_dqn, memory, optimizer)

            # Update the target network weights every
            # env_config["target_update_frequency"] steps
            if steps % env_config["target_update_frequency"] == 0:
                target_dqn.fc1.weight = dqn.fc1.weight
                target_dqn.fc2.weight = dqn.fc2.weight

        # Evaluate the current agent
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(
                dqn, env, env_config, args, n_episodes=args.evaluation_episodes
            )
            print(f'Episode {episode + 1} / {env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print("Best performance so far! Saving model.")
                torch.save(dqn, f"models/{args.env}_best.pt")

    # Close environment after training is completed.
    env.close()
