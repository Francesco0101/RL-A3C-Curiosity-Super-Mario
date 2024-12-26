import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import os 
import shutil
from constants import *

class MetricLogger:
    def __init__(self, save_dir, init_ep, icm=False):

        if icm:
            save_dir = save_dir + "curiosity/"
        else:
            save_dir = save_dir + "no_curiosity/"
        
        if REWARD_TYPE == "dense":
            save_dir = save_dir + "dense/"
        elif REWARD_TYPE == "sparse":
            save_dir = save_dir + "sparse/"
        else:
            save_dir = save_dir + "no_reward/"
    
        prefix = save_dir + "log_"
        save_dir = prefix + str(0)

        if init_ep == 0:
            i = 0
            exist = os.path.exists(save_dir)
            while exist:
                i += 1
                save_dir = prefix + str(i)
             
                exist = os.path.exists(save_dir)
            os.makedirs(save_dir, exist_ok=True)
        else:
            i = 0
            exist = os.path.exists(save_dir)
            while exist:
                i += 1
                save_dir = prefix + str(i)
                exist = os.path.exists(save_dir)
            save_dir = prefix + str(i-1)

        self.save_log = save_dir +f"/log_{WORLD}_{STAGE}.txt"
        self.save_reward_distance = save_dir + f"/reward_distance_{WORLD}_{STAGE}.txt"
        if init_ep ==0:
            with open(self.save_log, "w") as f:
                f.write(
                    f"{'Episode':>8}{'Policy Loss':>15}{'Value Loss':>15}"
                    f"{'Total Reward':>15}{'Total Loss':>15}{'Forward Loss':>15}{'Inverse Loss':>15}{'Time Delta':>15}"
                    f"{'Time':>20}\n"
            )
            with open(self.save_reward_distance, "w") as f:
                f.write(
                    f"{'Reward':>15}{'Distance':>15}\n"
                )
        self.policy_losses_plot = save_dir+ "/policy_loss_plot.jpg"
        self.value_losses_plot = save_dir+ "/value_loss_plot.jpg"
        self.rewards_plot = save_dir+ "/reward_plot.jpg"
        self.total_losses_plot = save_dir+ "/total_loss_plot.jpg"
        self.forward_losses_plot = save_dir+ "/forward_loss_plot.jpg"
        self.inverse_losses_plot = save_dir+ "/inverse_loss_plot.jpg"
        
        # History metrics
        self.ep_rewards = []
        self.ep_policy_losses = []
        self.ep_value_losses = []
        self.ep_total_losses = []
        self.ep_forward_losses = []
        self.ep_inverse_losses = []
        
        # Moving averages
        self.moving_avg_rewards = []
        self.moving_avg_policy_losses = []
        self.moving_avg_value_losses = []
        self.moving_avg_total_losses = []
        self.moving_avg_forward_losses = []
        self.moving_avg_inverse_losses = []

        # Timing
        self.record_time = time.time()

    def log_episode(self, global_episode, total_reward, policy_loss, value_loss, total_loss, forward_loss, inverse_loss):
        """Log metrics at the end of an episode."""
        self.ep_rewards.append(total_reward)
        self.ep_policy_losses.append(policy_loss)
        self.ep_value_losses.append(value_loss)
        self.ep_total_losses.append(total_loss)
        self.ep_forward_losses.append(forward_loss)
        self.ep_inverse_losses.append(inverse_loss)

       
        # Moving averages over last 100 episodes
        mean_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_policy_loss = np.round(np.mean(self.ep_policy_losses[-100:]), 3)
        mean_value_loss = np.round(np.mean(self.ep_value_losses[-100:]), 3)
        mean_total_loss = np.round(np.mean(self.ep_total_losses[-100:]), 3)
        mean_forward_loss = np.round(np.mean(self.ep_forward_losses[-100:]), 3)
        mean_inverse_loss = np.round(np.mean(self.ep_inverse_losses[-100:]), 3)
      
        self.moving_avg_rewards.append(mean_reward)
        self.moving_avg_policy_losses.append(mean_policy_loss)
        self.moving_avg_value_losses.append(mean_value_loss)
        self.moving_avg_total_losses.append(mean_total_loss)
        self.moving_avg_forward_losses.append(mean_forward_loss)
        self.moving_avg_inverse_losses.append(mean_inverse_loss)
       

        # Timing
        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        if( global_episode % LOG_EPISODE_INTERVAL == 0):
            print(
                f"Global Episode {global_episode} - " 
                f"policy Loss {policy_loss:.4f} - value Loss {value_loss:.4f} - "
                f"Total Reward {total_reward:.2f} - " 
                f"Total Loss {total_loss:.4f} - "
                f"Forward Loss {forward_loss:.4f} - "
                f"Inverse Loss {inverse_loss:.4f} - "
                f"Time Delta {time_since_last_record:.3f} - "
                f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
            )
            self.plot_metrics()

        with open(self.save_log, "a") as f:
            f.write(
                f"{global_episode:8d}{policy_loss:15.4f}{value_loss:15.4f}"
                f"{total_reward:15.2f}{total_loss:15.4f}{forward_loss:15.4f}{inverse_loss:15.4f}{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )
            
    def log_reward_distance(self, reward, distance):
        with open(self.save_reward_distance, "a") as f:
            f.write(
                f"{reward:15.4f}{distance:15.4f}\n"
            )

    def plot_metrics(self):
        """Save the plots for all metrics."""
        for metric, name in zip(
            ["rewards","policy_losses", "value_losses", "total_losses", "forward_losses", "inverse_losses"],
            ["rewards_plot", "policy_losses_plot", "value_losses_plot","total_losses_plot", "forward_losses_plot", "inverse_losses_plot"],
        ):
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{name}"))

