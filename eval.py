import torch
from models.model import ActorCritic
from environment.env import create_train_env
from utils.constants import WORLD, STAGE
import numpy as np
from utils.utils import save
from utils.constants import *

device = 'cpu'

def eval():
    """
    Evaluates the trained model on the environment and runs it in evaluation mode.

    This function loads a pre-trained model, interacts with the environment by performing actions 
    based on the model's policy, and monitors the environment until the environment is completed.
    """
    env, num_states, num_actions = create_train_env( render = True)

    model = ActorCritic(num_states, num_actions).to(device)

    model.load_state_dict(torch.load("checkpoints/curiosity/sparse/1_1/save_0/a3c_episode_31000.pt"))
    model.eval()
    
    state, _ = env.reset()
    state  =  torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
    done = True

    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
            c_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
            state, _ = env.reset()
            state  =  torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        state = state.to(device)
        logits, value, h_0, c_0 = model(state, h_0, c_0)
        policy = torch.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, done, _, info = env.step(action)
        state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
        if info["flag_get"]:
            print("World {} stage {} completed".format(WORLD, STAGE))
            break


if __name__ == "__main__":
    eval()

