
import torch
from gymnasium.wrappers import AtariPreprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(obs, env):
    """Performs necessary observation preprocessing."""

    if env in ["CartPole-v1"]:
        return torch.tensor(obs, device=device).float()
    elif env in ["ALE/Pong-v5"]:
        env = AtariPreprocessing(
            env,
            screen_size=84,
            grayscale_obs=True,
            frame_skip=1,
            noop_max=30,
        )
        return torch.tensor(obs, device=device).float()
    else:
        raise ValueError(
            "Please add necessary observation preprocessing instructions to "
            "preprocess() in utils.py."
        )
