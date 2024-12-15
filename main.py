import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as _mp
import gym
import numpy as np
from torch.distributions import Categorical
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import time, datetime
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info

# Define the Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(12800, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = x.squeeze(-1)
        x = self.common(x)
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

# Worker Process
def worker(global_model, optimizer, env_name, global_episode, max_episodes):
    name = _mp.current_process().name
    print(f"Worker {name} started")
    local_model = ActorCritic(global_model.state_dict()['common.0.weight'].shape[1], global_model.state_dict()['actor.bias'].shape[0]).to(device)
    local_model.load_state_dict(global_model.state_dict())
    env = gym_super_mario_bros.make(env_name , render_mode='human', apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    local_episode = 0
    while global_episode.value < max_episodes:
        print(f"Worker {name} started episode {local_episode}")
        state, _ = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []
        local_episode += 1

        # Rollout loop
        while not done:
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
            action_probs, value = local_model(state)
            action_probs = torch.softmax(action_probs, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()

            next_state, reward, done, _ , _= env.step(action.item())
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)

            state = next_state

        # Compute returns and advantages
        returns = []
        R = 0 if done else local_model(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))[1].item()
        for reward in reversed(rewards):
            R = reward + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values = torch.cat(values).squeeze(-1).to(device)

        # Loss calculation
        log_probs = torch.cat(log_probs)
        advantage = returns - values.detach()  # Detach values to avoid backprop through critic
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.MSELoss()(values, returns)  # Both now have shape [30]
        loss = actor_loss + critic_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            global_param._grad = local_param.grad
        optimizer.step()

        local_model.load_state_dict(global_model.state_dict())

        # Update global episode counter
        with global_episode.get_lock():
            global_episode.value += 1

        print(f"Worker {name} finished episode {local_episode}, Global Episode: {global_episode.value}, Reward Episode: {R}, Loss: {loss.item():.4f}")

# Main A3C Training Function
def main():
    eval = False
    if(eval == False):
        mp = _mp.get_context('spawn') # Create a new context for multiprocessing --> without, deadlock

        env_name = "SuperMarioBros-1-1-v0"
        env = gym_super_mario_bros.make(env_name , render_mode='human', apply_api_compatibility=True)
        env = JoypadSpace(env, RIGHT_ONLY)
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4)
        input_dim = env.observation_space.shape[0]
        print("input_dim: ", input_dim)
        action_dim = env.action_space.n

        # Create global model and optimizer
        global_model = ActorCritic(input_dim, action_dim).to(device)
        global_model.share_memory()
        optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-4)
        print("Global model created")
        print(global_model)

        # Multiprocessing variables
        max_episodes = 100
        print("cpu: ", mp.cpu_count())
        num_workers = 3
        global_episode = mp.Value('i', 0)

        # Start workers
        workers = []
        for _ in range(num_workers):
            worker_process = mp.Process(target=worker, args=(global_model, optimizer, env_name, global_episode, max_episodes))
            workers.append(worker_process)
            worker_process.start()
        
        print("Training started...")
        for worker_process in workers:
            worker_process.join()

        print("Training complete!")

    print("Testing policy...")
    #do test on 10 episodes rendering the env and printing the total reward
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='human', apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    state, _  = env.reset()
    env.render()
    done = False
    reward_ep = 0
    total_reward = 0
    for _ in range(10):
        while not done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs, _ = global_model(state)
            action = torch.argmax(action_probs, dim=-1)
            next_state, reward, done, _ , _= env.step(action.item())
            reward_ep += reward
            state = next_state
        print(f"Episode reward: {reward_ep}")
        total_reward += reward_ep
    total_reward /= 10
    print(f"Average total reward: {total_reward}")


    


if __name__ == "__main__":
    main()
