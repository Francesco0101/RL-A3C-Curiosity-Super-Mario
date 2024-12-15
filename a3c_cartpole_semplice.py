import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gym
import numpy as np
from torch.distributions import Categorical

# Define the Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.common(x)
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

# Worker Process
def worker(global_model, optimizer, env_name, global_episode, max_episodes):
    local_model = ActorCritic(global_model.state_dict()['common.0.weight'].shape[1], global_model.state_dict()['actor.bias'].shape[0])
    local_model.load_state_dict(global_model.state_dict())
    env = gym.make(env_name)
    local_episode = 0
    name = mp.current_process().name

    while global_episode.value < max_episodes:
        state, _ = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []
        local_episode += 1

        # Rollout loop
        while not done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
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
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(values).squeeze(-1)

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

        env_name = "CartPole-v1"
        env = gym.make(env_name , render_mode='human')
        input_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Create global model and optimizer
        global_model = ActorCritic(input_dim, action_dim)
        global_model.share_memory()
        optimizer = optim.Adam(global_model.parameters(), lr=1e-4)

        # Multiprocessing variables
        max_episodes = 3000
        print("cpu: ", mp.cpu_count())
        num_workers = 3
        global_episode = mp.Value('i', 0)

        # Start workers
        workers = []
        for _ in range(num_workers):
            worker_process = mp.Process(target=worker, args=(global_model, optimizer, env_name, global_episode, max_episodes))
            workers.append(worker_process)
            worker_process.start()

        for worker_process in workers:
            worker_process.join()

        print("Training complete!")

    print("Testing policy...")
    #do test on 10 episodes rendering the env and printing the total reward
    env = gym.make("CartPole-v1", render_mode='human')
    state, _  = env.reset()
    env.render()
    done = False
    reward = 0
    total_reward = 0
    for _ in range(10):
        while not done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs, _ = global_model(state)
            action = torch.argmax(action_probs, dim=-1)
            next_state, reward, done, _ , _= env.step(action.item())
            total_reward += reward
            state = next_state
        print(f"Total reward: {reward}")
        total_reward += reward
    total_reward /= 10
    print(f"Average total reward: {total_reward}")


    


if __name__ == "__main__":
    main()
