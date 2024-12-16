import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as _mp
import gym
import numpy as np
from torch.distributions import Categorical
from env import create_train_env
from model import ActorCritic
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Worker Process
def worker(global_model, optimizer, global_episode, max_episodes):
    name = _mp.current_process().name
    print(f"Worker {name} started")
    local_model = ActorCritic(global_model.state_dict()['common.0.weight'].shape[1], global_model.state_dict()['actor.bias'].shape[0]).to(device)
    local_model.load_state_dict(global_model.state_dict())
    # local_model.train()
    env, _, _ = create_train_env(1, 1, RIGHT_ONLY)
    local_episode = 0
   
    while global_episode.value < max_episodes:
        local_steps = 0
        print(f"Worker {name} started episode {local_episode}")
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        entropies = []
        local_episode += 1
        #initialize hidden states
        h_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
        c_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
        done = False
        local_model.load_state_dict(global_model.state_dict())
        # Rollout loop
        while not done:
            if local_steps == 50:
                break
            local_steps += 1
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
            logits, value, h_0 , c_0 = local_model(state, h_0, c_0) 
            # print("Critic value (before storing):", value)
            action_probs = torch.softmax(logits, dim=-1)
            log_action_probs = torch.log_softmax(logits, dim=1)
            entropy = -(action_probs * log_action_probs).sum(1, keepdim=True)
            m = Categorical(action_probs)
            action = m.sample().item()

            next_state, reward, done, _= env.step(action)
            log_probs.append(log_action_probs[0, action])
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)

            state = next_state

        # Compute returns and advantages
        
        R = torch.zeros(1, 1).to(device)
        if not done:
            _, R, _, _ = local_model(torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device), h_0, c_0)
        gae = torch.zeros(1, 1).to(device)

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R
        print("R prima del for: ", R)

        for value, reward, log_prob, entropy in list(zip(values, rewards, log_probs, entropies))[::-1]:
            print("value dentro il for: ", value)
            gae = gae * 0.9 * 1 + reward + 0.9 * next_value - value
            next_value = value
            actor_loss += +log_prob * gae
            R = 0.9 * R + reward
            critic_loss += 0.5 * (R - value).pow(2)
            entropy_loss += entropy
        
        total_loss = -actor_loss + critic_loss - 0.01 * entropy_loss 
        #all loss separately
        print("R: ", R)
        print("Value: ", value)
        print(f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Entropy Loss: {entropy_loss.item():.4f}")
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()

        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):

            global_param._grad = local_param.grad

        optimizer.step()

        

        # Update global episode counter
        with global_episode.get_lock():
            global_episode.value += 1

        print(f"Worker {name} finished episode {local_episode}, Global Episode: {global_episode.value}, Reward Episode: {R}, Loss: {total_loss.item():.4f}")


class GlobalAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super(GlobalAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

# Main A3C Training Function
def main():
    eval = False
    if(eval == False):
        mp = _mp.get_context('spawn') # Create a new context for multiprocessing --> without, deadlock

        # Create environment
        env, input_dim, action_dim = create_train_env(1, 1, RIGHT_ONLY)
        print("action_dim: ", action_dim)
        print("input_dim: ", input_dim)

        # Create global model and optimizer
        global_model = ActorCritic(input_dim, action_dim).to(device)
        global_model.share_memory()
        optimizer = GlobalAdam(global_model.parameters(), lr=1e-4)
        print("Global model created")
        print(global_model)

        # Multiprocessing variables
        max_episodes = 1000
        print("cpu: ", mp.cpu_count())
        num_workers = 3
        global_episode = mp.Value('i', 0)

        # Start workers
        workers = []
        for _ in range(num_workers):
            worker_process = mp.Process(target=worker, args=(global_model, optimizer, global_episode, max_episodes))
            workers.append(worker_process)
            worker_process.start()
        
        print("Training started...")
        for worker_process in workers:
            worker_process.join()

        print("Training complete!")

    # print("Testing policy...")
    # #do test on 10 episodes rendering the env and printing the total reward
    # env, _, _ = create_train_env(1, 1, RIGHT_ONLY)
    # state, _  = env.reset()
    # env.render()
    # done = False
    # reward_ep = 0
    # total_reward = 0
    # for _ in range(10):
    #     while not done:
    #         state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    #         action_probs, _ = global_model(state)
    #         action = torch.argmax(action_probs, dim=-1)
    #         next_state, reward, done, _ , _= env.step(action.item())
    #         reward_ep += reward
    #         state = next_state
    #     print(f"Episode reward: {reward_ep}")
    #     total_reward += reward_ep
    # total_reward /= 10
    # print(f"Average total reward: {total_reward}")


    


if __name__ == "__main__":
    main()
