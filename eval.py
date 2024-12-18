import torch
from  model import ActorCritic
from env import create_train_env
from constants import WORLD, STAGE
import numpy as np

def eval():
    torch.manual_seed(123)
    env, num_states, num_actions = create_train_env(render = True)

    model = ActorCritic(num_states, num_actions)

    model.load_state_dict(torch.load("checkpoints/a3c_1_1_episode_20.pt"))
    model.eval()
    
    state, _ = env.reset()
    state  =  torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
    done = True

    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            env.reset()

        logits, value, h_0, c_0 = model(state, h_0, c_0)
        policy = torch.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, done, _, info = env.step(action)
        state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
        env.render()
        if info["flag_get"]:
            print("World {} stage {} completed".format(WORLD, STAGE))
            break


if __name__ == "__main__":
    eval()

