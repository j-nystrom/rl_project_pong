
import argparse
import gymnasium as gym
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from gymnasium.wrappers import AtariPreprocessing 
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    choices=["CartPole-v1", "ALE/Pong-v5"],
    default="CartPole-v1",
)
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
ENV_CONFIGS = {"CartPole-v1": config.CartPole, "ALE/Pong-v5": config.Pong}


if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config
    env = gym.make(args.env, render_mode="human")
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
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

    # Loop through all episodes
    for episode in range(env_config["n_episodes"]):
        
        terminated = False
        obs, info = env.reset()
        
        ###############################################################
        obs_stack_size= 1 #not clear if this is like a malloc or what?
        ###############################################################

        # Get first observation from environment
        obs = preprocess(obs, env=args.env).unsqueeze(0)
        
        #Initialize the stacks
        obs_stack = torch.cat(obs_stack_size * [obs]).unsqueeze(0).to(device)
        next_obs_stack = torch.cat(obs_stack_size * [obs]).unsqueeze(0).to(device)
        
        #Put first obervation into the stacks
        obs_stack = torch.cat((obs_stack[:, 1:, ...], obs.unsqueeze(1)), dim=1).to(device)
        next_obs_stack = torch.cat((obs_stack[:, 1:, ...], obs.unsqueeze(1)), dim=1).to(device)
        
        while not terminated:
            steps += 1
            
            # Get action to take
            action = dqn.act(obs, steps)

            # Act in the true environment
            next_obs, reward, terminated, truncated, _ = env.step(action.item())

            # Preprocess incoming observation and push to replay memory
            # Next observation appended will depend on terminated or not
            if not terminated:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
                next_obs_stack = torch.cat((next_obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)
            else:
                next_obs = None
                ##############################################################
                #should check behaviour on the stack when nothing else happens
                ##############################################################
                next_obs_stack = None
            reward = torch.tensor(reward, device=device).float().unsqueeze(0)

            # Store transition in memory, move to next transition
            memory.push(obs_stack, action, next_obs_stack, reward)
            obs = next_obs
            obs_stack = next_obs_stack

            # Run optimize() function every env_config["train_frequency"] steps
            if steps % env_config["train_frequency"] == 0:
                loss = optimize(dqn, target_dqn, memory, optimizer)

            # Update the target network weights every
            # env_config["target_update_frequency"] steps
            if steps % env_config["target_update_frequency"] == 0:
                state_dict = dqn.state_dict()
                target_state_dict = target_dqn.state_dict()
                for key in state_dict:
                    target_state_dict[key] = state_dict[key]
                target_dqn.load_state_dict(target_state_dict)

        # Evaluate the current agent
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(
                dqn, env, env_config, args, n_episodes=args.evaluation_episodes
            )
            print(f'Episode {episode + 1} / {env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print("Best performance so far! Saving model.")
                torch.save(dqn, f"models/{args.env}_best.pt")

    # Close environment after training is completed
    env.close()
