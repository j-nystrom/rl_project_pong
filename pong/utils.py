
import torch
from gymnasium.wrappers import AtariPreprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(obs, env):
    """Performs necessary observation preprocessing."""

    if env in ["CartPole-v1"]:
        return torch.tensor(obs, device=device).float()
    elif env in ["ALE/Pong-v5"]:
        #Removed the Atari Preprocessing because it was a wrapper
        #for the environment and it is not needed to be applied
        #for each observation. Now it is in Train
        ###############################################
        #Dividing by 255 to normalize obs between 0-1
        ###############################################
        return torch.tensor(obs/255, device=device).float()
    else:
        raise ValueError(
            "Please add necessary observation preprocessing instructions to "
            "preprocess() in utils.py."
        )
