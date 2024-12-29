import torch
from models.model import ActorCritic
from environment.env import create_train_env
from utils.constants import WORLD, STAGE
import numpy as np
from torch.distributions import Categorical
from utils.utils import save
from utils.constants import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

def eval():
    # torch.manual_seed(123)
    env, num_states, num_actions = create_train_env( render = True)

    model = ActorCritic(num_states, num_actions).to(device)

    model.load_state_dict(torch.load("checkpoints/a3c_1_1_episode_28000.pt"))
    model.eval()
    
    state, _ = env.reset()
    state  =  torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
    done = True

    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
            c_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
            state, _ = env.reset()
            #print("again")
            state  =  torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        state = state.to(device)
        #print(state)
        logits, value, h_0, c_0 = model(state, h_0, c_0)
        policy = torch.softmax(logits, dim=1)
        #print(policy)
        action = torch.argmax(policy).item()
        
        action = int(action)
        #print(action)
        
        # action_probs = torch.softmax(logits, dim=-1)
        # m = Categorical(action_probs)
        # action = m.sample().item()
        state, reward, done, _, info = env.step(action)
        #save(state)
        state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
        if info["flag_get"]:
            print("World {} stage {} completed".format(WORLD, STAGE))
            break


if __name__ == "__main__":
    eval()

