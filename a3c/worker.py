import torch
import torch.multiprocessing as _mp
import numpy as np
from torch.distributions import Categorical
from environment.env import create_train_env
from utils.constants import *
from models.model import ActorCritic
from models.icm import ICM
from utils.utils import save

device = 'cpu'
x_norm = 3161
# Worker Process
def worker(global_model, optimizer, global_episode, max_episodes, logger, categorical = True, renderer = False, global_icm = None, save_path = SAVE_PATH):
    """
    A single worker process for training the reinforcement learning model.

    Args:
        global_model (torch.nn.Module): Shared global model.
        optimizer (torch.optim.Optimizer): Shared optimizer.
        global_episode (multiprocessing.Value): Shared counter for global episodes.
        max_episodes (int): Maximum number of episodes to train.
        logger (Logger): Logger to log training details.
        categorical (bool): Whether to use Categorical sampling for actions. Defaults to True.
        renderer (bool): Whether to render the environment. Defaults to False.
        global_icm (torch.nn.Module, optional): Shared Intrinsic Curiosity Module (ICM) model.
        save_path (str): Directory to save model checkpoints.
    """
    #name = _mp.current_process().name
    env, state_dim, action_dim = create_train_env(render = renderer)
    local_model = ActorCritic(state_dim, action_dim).to(device)
    local_model.load_state_dict(global_model.state_dict())
    local_model.train()

    if global_icm is not None:
        curiosity_model = ICM(state_dim, action_dim).to(device)
        curiosity_model.load_state_dict(global_icm.state_dict())
        curiosity_model.train()

    

    state, _ = env.reset()
    local_episode = int(global_episode.value / NUM_WORKERS)
    local_steps = 0
    done = True
    rewards_done = 0

    if optimizer is None:
        optimizer = torch.optim.Adam(global_model.parameters(), lr=LR)
        if global_icm is not None:
            optimizer_icm = torch.optim.Adam(global_icm.parameters(), lr=LR)
    
    while local_episode < max_episodes:
        log_probs = []
        values = []
        rewards = []
        entropies = []
        forward_losses = []
        inverse_losses = []
        local_episode += 1

        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
            c_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        local_model.load_state_dict(global_model.state_dict())
        if global_icm is not None:
            curiosity_model.load_state_dict(global_icm.state_dict())

        for _ in range(NUM_LOCAL_STEPS):
            local_steps += 1
         
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)            
            logits, value, h_0 , c_0 = local_model(state, h_0, c_0) 
            # print("Critic value (before storing):", value)
            action_probs = torch.softmax(logits, dim=-1)
            log_action_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(action_probs * log_action_probs).sum(-1, keepdim=True)

            if categorical==True:
                m = Categorical(action_probs)
                action = m.sample().item()
            else:
                action = torch.argmax(logits).item()
            
            next_state, reward, done, _, info= env.step(action)
            rewards_done += reward

            if global_icm is not None:
                action_one_onehot = torch.zeros(1, action_dim).to(device)
                action_one_onehot[0, action] = 1
                next_state_icm = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0).to(device)
                s_t1_pred, a_t0_pred, s_t1 = curiosity_model(state, next_state_icm, action_one_onehot)
                inverse_loss = torch.nn.CrossEntropyLoss()(s_t1_pred, torch.tensor([action]).to(device)) / action_dim
                mse_loss = torch.nn.MSELoss(reduction='none')  
                forward_loss = mse_loss(a_t0_pred, s_t1).sum(-1, keepdim=True) / 2 
                # print("Forward Loss: ", forward_loss)
                # print("Inverse Loss: ", inverse_loss)
                intrinsic_reward = ETA * forward_loss
                reward += intrinsic_reward.item()
                reward = max(min(reward, 50), -5)


            
            log_probs.append(log_action_probs[0, action])
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
            if global_icm is not None:
                forward_losses.append(forward_loss)
                inverse_losses.append(inverse_loss)

            state = next_state
            if local_steps > NUM_GLOBAL_STEPS:
                done = True
        
            if done:
                local_steps = 0
                logger.log_reward_distance(rewards_done, info['x_pos']/x_norm)
                rewards_done = 0
                state, _ = env.reset()
                break
                        
        R = torch.zeros(1, 1).to(device)
        if not done:
            _, R, _, _ = local_model(torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device), h_0, c_0)
        
        gae = torch.zeros(1, 1).to(device)
        policy_loss = 0
        value_loss = 0
        curiosity_loss = 0
        total_reward = sum(rewards)
        next_value = R
        if global_icm is None:
            for value, reward, log_prob, entropy in list(zip(values, rewards, log_probs, entropies))[::-1]:
                gae = gae * GAMMA * TAU + reward + GAMMA * next_value.detach() - value.detach()
                next_value = value
                R = GAMMA * R + reward
                value_loss += 0.5 * (R - value).pow(2) 
                policy_loss -= log_prob * gae - ENTROPY_COEFF * entropy
        else:
            for value, reward, log_prob, entropy, inverse_loss, forward_loss in list(zip(values, rewards, log_probs, entropies, inverse_losses, forward_losses))[::-1]:
                gae = gae * GAMMA * TAU + reward + GAMMA * next_value.detach() - value.detach()
                next_value = value
                R = GAMMA * R + reward
                value_loss += 0.5 * (R - value).pow(2) 
                policy_loss -= log_prob * gae - ENTROPY_COEFF * entropy
                curiosity_loss += (1-BETA) * inverse_loss + BETA * forward_loss
        total_loss = LAMBDA * (policy_loss + value_loss * VALUE_LOSS_COEF) + WEIGHT_CURIOSITY * curiosity_loss

        if SHARED_OPTIMIZER == True:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), MAX_GRAD_NORM)
            if global_icm is not None:
                torch.nn.utils.clip_grad_norm_(curiosity_model.parameters(), MAX_GRAD_NORM)

            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                if global_param.grad is not None:
                    break
                global_param._grad = local_param.grad
            
            if global_icm is not None:
                for global_icm_param, local_icm_param in zip(global_icm.parameters(), curiosity_model.parameters()):
                    if global_icm_param.grad is not None:
                        break
                    global_icm_param._grad = local_icm_param.grad

            optimizer.step()
        else:
            optimizer.zero_grad()
            total_loss.backward(retain_graph=(global_icm is not None))
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), MAX_GRAD_NORM)
            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                if global_param.grad is not None:
                    break
                global_param._grad = local_param.grad
            optimizer.step()
            if global_icm is not None:
                optimizer_icm.zero_grad()
                curiosity_loss.backward()
                torch.nn.utils.clip_grad_norm_(curiosity_model.parameters(), MAX_GRAD_NORM)
                for local_icm_param, global_icm_param in zip(curiosity_model.parameters(), global_icm.parameters()):
                    if global_icm_param.grad is not None:
                        break
                    global_icm_param._grad = local_icm_param.grad
                optimizer_icm.step()

        with global_episode.get_lock():
            global_episode.value += 1

        if global_episode.value % SAVE_EPISODE_INTERVAL == 0:
            save_path_a3c = save_path + "/a3c" + "_episode_" + str(global_episode.value)+".pt"
            if global_icm is not None:
                save_path_icm = save_path + "/icm" + "_episode_" + str(global_episode.value)+".pt"

            if global_icm is not None:
                    torch.save(global_model.state_dict(),
                            save_path_a3c)
                    torch.save(global_icm.state_dict(),
                            save_path_icm)
            else:
                    torch.save(global_model.state_dict(),
                            save_path_a3c)
                
        if global_icm is None:   
            logger.log_episode(global_episode.value, total_reward, policy_loss.item(), value_loss.item(), total_loss.item(), 0)
        else:
            logger.log_episode(global_episode.value, total_reward, policy_loss.item(), value_loss.item(), total_loss.item(), curiosity_loss.item())