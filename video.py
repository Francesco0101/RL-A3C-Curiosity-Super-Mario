import torch
from models.model import ActorCritic
from environment.env import create_train_env
from utils.constants import WORLD, STAGE
import numpy as np
from utils.constants import *
import imageio


def eval():
    """
    Evaluates the trained model on the environment and saves a video of the evaluation.

    This function uses the `RecordVideo` wrapper to save a video of the agent's interaction
    with the environment during evaluation.
    """
    
    device = 'cpu'
    env, num_states, num_actions = create_train_env(render=False)
    video = []
    
    model = ActorCritic(num_states, num_actions).to(device)
    model.load_state_dict(torch.load("checkpoints/curiosity/dense/1_1/save_0/a3c_episode_29000.pt"))
    model.eval()
    
    state, _ = env.reset()
    state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
    done = True
    frames = 0
    while True:
        
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
            c_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
            state, _ = env.reset()
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
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

        frame = env.render()
        video.append(frame.copy())
        state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
        #Vedere se rimuovere
        if frames >300:
            break
        frames += 1
        if info.get("flag_get"):
            print("World {} stage {} completed".format(WORLD, STAGE))
            break
    env.close()
    imageio.mimsave("gif/mario_1_2_curiosity_dense.mp4", video, fps=30, )


if __name__ == "__main__":
    eval()
