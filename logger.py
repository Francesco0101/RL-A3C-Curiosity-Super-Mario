import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import os 
import shutil

class MetricLogger:
    def __init__(self, save_dir, init_ep):
        #if the folder exists reset it
        self.save_log = save_dir +"/log.txt"
        if init_ep == 0:
            path = Path(save_dir)
            if path.exists():
                shutil.rmtree(path)
            os.makedirs(save_dir, exist_ok=True)
            with open(self.save_log, "w") as f:
                f.write(
                    f"{'Episode':>8}{'Policy Loss':>15}{'Value Loss':>15}"
                    f"{'Total Reward':>15}{'Total Loss':>15}{'Time Delta':>15}"
                    f"{'Time':>20}\n"
                )
        self.policy_losses_plot = save_dir+ "/policy_loss_plot.jpg"
        self.value_losses_plot = save_dir+ "/value_loss_plot.jpg"
        self.rewards_plot = save_dir+ "/reward_plot.jpg"
        self.total_losses_plot = save_dir+ "/total_loss_plot.jpg"
        
        # History metrics
        self.ep_rewards = []
        self.ep_policy_losses = []
        self.ep_value_losses = []
        self.ep_total_losses = []
        
        # Moving averages
        self.moving_avg_rewards = []
        self.moving_avg_policy_losses = []
        self.moving_avg_value_losses = []
        self.moving_avg_total_losses = []

        # Timing
        self.record_time = time.time()

    def log_episode(self, global_episode, total_reward, policy_loss, value_loss, total_loss):
        """Log metrics at the end of an episode."""
        self.ep_rewards.append(total_reward)
        self.ep_policy_losses.append(policy_loss)
        self.ep_value_losses.append(value_loss)
        self.ep_total_losses.append(total_loss)

       
        # Moving averages over last 100 episodes
        mean_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_policy_loss = np.round(np.mean(self.ep_policy_losses[-100:]), 3)
        mean_value_loss = np.round(np.mean(self.ep_value_losses[-100:]), 3)
        mean_total_loss = np.round(np.mean(self.ep_total_losses[-100:]), 3)
      
        self.moving_avg_rewards.append(mean_reward)
        self.moving_avg_policy_losses.append(mean_policy_loss)
        self.moving_avg_value_losses.append(mean_value_loss)
        self.moving_avg_total_losses.append(mean_total_loss)
       

        # Timing
        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        if( global_episode % 500 == 0):
            print(
                f"Global Episode {global_episode} - " 
                f"policy Loss {policy_loss:.4f} - value Loss {value_loss:.4f} - "
                f"Total Reward {total_reward:.2f} - " 
                f"Total Loss {total_loss:.4f} - "
                f"Time Delta {time_since_last_record:.3f} - "
                f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
            )

        with open(self.save_log, "a") as f:
            f.write(
                f"{global_episode:8d}{policy_loss:15.4f}{value_loss:15.4f}"
                f"{total_reward:15.2f}{total_loss:15.4f}{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

    def plot_metrics(self):
        """Save the plots for all metrics."""
        for metric, name in zip(
            ["rewards","policy_losses", "value_losses", "total_losses"],
            ["rewards_plot", "policy_losses_plot", "value_losses_plot","total_losses_plot"],
        ):
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{name}"))

