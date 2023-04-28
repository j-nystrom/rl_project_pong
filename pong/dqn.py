
import random
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
        """Store transition in replay memory."""

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

        # Save hyperparameters needed in the DQN class
        self.obs_stack_size = env_config["obs_stack_size"]
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        # Set up the convolutional network architecture
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4), padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, observation, steps=0, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""

        # NOTE: The relevant actions are [2, 3] (down / up), so only those should
        # be considered

        # Use epsilon-greedy to decide between greedy and random action
        sample = random.random()
        eps = max(
            self.eps_start - (self.eps_start / self.anneal_length) * steps,
            self.eps_end
            )

        # Generate action according to random choice above
        if sample > eps or exploit:
            with torch.no_grad():  # Context manager to speed up computation
                action = torch.argmax(self(observation), dim=1).long()
        else:
            action = torch.tensor(
                random.choice(range(self.n_actions)),
                device=device,
            ).long().unsqueeze(0)

        return action


def optimize(dqn, target_dqn, memory, optimizer):
    """
    This function samples a batch from the replay buffer and optimizes the
    Q-network.
    """

    # If we don't have enough transitions stored yet, we don't train
    if len(memory) < dqn.batch_size:
        return

    # If enough transitions, sample batch from memory
    batch = memory.sample(dqn.batch_size)

    # Create 4 separate tensors for observations, actions, next observations,
    # rewards and move to GPU if available
    observations = torch.cat(batch[0]).to(device)
    actions = torch.cat(batch[1]).to(device)
    rewards = torch.cat(batch[3]).to(device)
    # For next observations, need to handle terminal states as special case
    non_terminal_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch[2])), device=device, dtype=torch.bool
    )
    non_terminal_next_obs = torch.cat([s for s in batch[2] if s is not None])

    # Compute Q-values for observations using the policy network
    # This is Q(s, a; theta_i), and uses the most updated weights
    q_values = dqn.forward(observations).gather(1, actions.unsqueeze(1))

    # Compute the Q-value targets for next obs, using target network
    # This is y_i = E[r + gamma * max_a Q(s', a'; theta_i-1], and uses "old" weights
    # from the policy network
    # For terminal states, the action value is 0
    target_action_val = torch.zeros(target_dqn.batch_size, device=device)
    with torch.no_grad():  # Context manager to speed up computation
        target_action_val[non_terminal_mask] = target_dqn.forward(
            non_terminal_next_obs
        ).max(1)[0]
    q_value_targets = rewards + target_dqn.gamma * target_action_val

    # Compute the loss with current weights
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
