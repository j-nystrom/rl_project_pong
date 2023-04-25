import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, observation, steps=0, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""

        # Calculate action values based on Q-value function
        action_values = self.forward(observation)

        if exploit:
            strategy = "exploit"
        else:
            # Calculate epsilon based on annealing strategy and current step
            eps = max(self.eps_start - (self.eps_start / self.anneal_length) * steps,
                      self.eps_end
                      )

            # Sample strategy using epsilon-greedy approach
            strategy = np.random.choice(["exploit", "explore"], p=[1 - eps, eps])

        # Select action values based on above
        if strategy == "exploit":
            action = torch.argmax(action_values, dim=1, keepdim=True)

        else:
            action = np.random.choice(self.n_actions)
            action = torch.tensor(action, device=device).int()

        return action


def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the
    Q-network."""

    # If we don't have enough transitions stored yet, we don't train
    if len(memory) < dqn.batch_size:
        return

    # If enough transitions, sample batch from memory
    batch = memory.sample(dqn.batch_size)

    # Create 4 separate tensors for observations, actions, next observations,
    # rewards and move to GPU if available
    observations = torch.cat(batch[0]).to(device)
    # print(f"Observation tensor: \n {observations}")
    actions = torch.cat(batch[1]).unsqueeze(1).to(device)
    # print(f"Action tensor: \n {actions}")
    next_observations = torch.cat(batch[2]).to(device)
    # print(f"Next observation tensor: \n {next_observations}")
    rewards = torch.cat(batch[3]).unsqueeze(1).to(device)
    # print(f"Reward tensor: \n {rewards}")

    # Question: Investigate handling of terminal transitions in step above?
    # These will never be stored in replay memory? See train.py
    q_values = dqn.forward(observations).gather(1, actions)
    # print(f"Q values \n {q_values} \n")

    # Compute the Q-value targets
    # Question: How to do this only for non-terminal transitions?
    # These will never be stored in replay memory? See train.py
    target_action_val = target_dqn.forward(next_observations)
    max_target_action_val, _ = torch.max(target_action_val, dim=1)
    max_target_action_val = max_target_action_val.unsqueeze(1)
    # print(f"Max act val \n {max_target_action_val}")
    q_value_targets = rewards + target_dqn.gamma * max_target_action_val
    # print(f"Q target values \n {q_value_targets}")

    # Compute the loss with current weights
    loss = F.mse_loss(q_values, q_value_targets.squeeze())
    # print(f"Loss: {loss}")

    # Perform gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
