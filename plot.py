
# import matplotlib.pyplot as plt
# import numpy as np

# def moving_average(data, window_size):
#     """
#     Calculate the moving average of a list of numbers.
#     :param data: List of numbers.
#     :param window_size: Size of the moving window.
#     :return: List of moving average values.
#     """
#     return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# def plot(path_loss, path_distance, window_size=10):
#     with open(path_loss, 'r') as f:
#         lines = f.readlines()
#         lines = lines[1:]
#         ep_rewards = []
#         ep_total_losses = []

#         # Extract rewards and total losses from the file
#         for line in lines:
#             line = line.split()
#             ep_rewards.append(float(line[3]))
#             ep_total_losses.append(float(line[4]))

#         # Calculate moving averages
#         rewards_ma = moving_average(ep_rewards, window_size)
#         total_losses_ma = moving_average(ep_total_losses, window_size)

#         # Plot the moving average of rewards
#         plt.plot(rewards_ma, label='Total Reward (Moving Average)')
#         plt.legend()
#         plt.savefig('rewards_ma.png')
#         plt.close()

#         # Plot the moving average of total losses
#         plt.plot(total_losses_ma, label='Total Loss (Moving Average)')
#         plt.legend()
#         plt.savefig('total_losses_ma.png')
#         plt.close()

#     print("/////////////////////////////////////////////////////////////////////////////")

#     with open(path_distance, 'r') as f:
#         lines = f.readlines()
#         lines = lines[1:]
#         distances = []

#         # Extract distances from the file
#         for line in lines:
#             line = line.split()
#             distances.append(float(line[1]))

#         # Calculate moving average of distances
#         distances_ma = moving_average(distances, window_size)

#         # Plot the moving average of distances
#         plt.plot(distances_ma, label='Distance (Moving Average)')
#         plt.legend()
#         plt.savefig('distance_ma.png')
#         plt.close()

import matplotlib.pyplot as plt
import numpy as np

def moving_average_and_std(data, window_size):
    """
    Calculate the moving average and standard deviation for a list of numbers.
    :param data: List of numbers.
    :param window_size: Size of the moving window.
    :return: Tuple of moving averages and standard deviations.
    """
    averages = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    stds = [
        np.std(data[i:i + window_size]) for i in range(len(data) - window_size + 1)
    ]
    return averages, stds

def plot(path, window_size=10):
    path_loss = path + "/log.txt"
    path_distance = path + "/reward_distance.txt"

    save_path_loss = path + "/total_loss.jpg"
    save_path_distance = path + "/distance_plot.jpg"
    save_path_reward = path + "/reward_plot.jpg"

    with open(path_loss, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]
        ep_total_losses = []

        # Extract rewards and total losses from the file
        for line in lines:
            line = line.split()
            ep_total_losses.append(float(line[4]))

        # Calculate moving averages and standard deviations
   
        total_losses_ma, total_losses_std = moving_average_and_std(ep_total_losses, window_size)

        # Plot the moving average and standard deviation for total losses
        plt.plot(total_losses_ma, label='Total Loss (Moving Average)', color='orange')
        plt.fill_between(
            range(len(total_losses_ma)),
            total_losses_ma - total_losses_std,
            total_losses_ma + total_losses_std,
            color='orange',
            alpha=0.2,
            label='Total Loss ± Std Dev'
        )
        plt.legend()
        plt.savefig(save_path_loss)
        plt.close()


    with open(path_distance, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]
        distances = []
        rewards = []

        # Extract distances from the file
        for line in lines:
            line = line.split()
            distances.append(float(line[1]))
            rewards.append(float(line[0]))

        # Calculate moving average and standard deviation for distances
        distances_ma, distances_std = moving_average_and_std(distances, window_size)
        rewards_ma, rewards_std = moving_average_and_std(rewards, window_size)
        # Plot the moving average and standard deviation for distances
        plt.plot(distances_ma, label='Distance (Moving Average)', color='green')
        plt.fill_between(
            range(len(distances_ma)),
            distances_ma - distances_std,
            distances_ma + distances_std,
            color='green',
            alpha=0.2,
            label='Distance ± Std Dev'
        )
        plt.legend()
        plt.savefig(save_path_distance)
        plt.close()
        plt.plot(rewards_ma, label='Reward (Moving Average)', color='red')
        plt.fill_between(
            range(len(rewards_ma)),
            rewards_ma - rewards_std,
            rewards_ma + rewards_std,
            color='red',
            alpha=0.2,
            label='Reward ± Std Dev'
        )
        plt.legend()
        plt.savefig(save_path_reward)
        plt.close()


plot('log/curiosity/no_reward/1_1/log_0', 100)
