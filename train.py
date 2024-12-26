import torch
import torch.multiprocessing as _mp
from env import create_train_env
from model import ActorCritic
from worker import worker
from shared_optim import GlobalAdam
from constants import *
import warnings
from pathlib import Path
from logger import MetricLogger
import shutil
import os
from icm import ICM

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


mp = _mp.get_context('spawn') # Create a new context for multiprocessing --> without, deadlock


def train(init_ep = 0, icm = False):
    print(init_ep)
    os.environ['OMP_NUM_THREADS'] = '1'

    # Create environment
    _, input_dim, action_dim = create_train_env(action_type = ACTION_TYPE)
    print("action_dim: ", action_dim)
    print("input_dim: ", input_dim)

    save_path = SAVE_PATH
    if icm == True:
        save_path = save_path + "/curiosity/"
    else:
        save_path = save_path + "/no_curiosity/"
    
    if REWARD_TYPE == "dense":
        save_path = save_path + "/dense/"
    elif REWARD_TYPE == "sparse":
        save_path = save_path + "/sparse/"
    else:
        save_path = save_path + "/no_reward/"

    save_path = save_path + f"{WORLD}_{STAGE}/"

    new_save_path = save_path + "save_" + str(0)
    if init_ep == 0:
        save = 0
        exist = os.path.exists(new_save_path)
        while exist:
            save += 1
            new_save_path = save_path + "save_" + str(save)
            exist = os.path.exists(new_save_path)
        os.makedirs(new_save_path, exist_ok=True)
    else:
        save = 0
        exist = os.path.exists(new_save_path)
        while exist:
            save += 1
            new_save_path = save_path + "save_" + str(save)
            exist = os.path.exists(new_save_path)
        new_save_path = save_path + "save_" + str(save-1)

    # Create global model and optimizer
    global_model = ActorCritic(input_dim, action_dim).to(device)
    global_model.share_memory()

    if icm == True:
        global_icm = ICM(input_dim, action_dim).to(device)
        global_icm.share_memory()
    else:
        global_icm = None

    if init_ep != 0:
        global_model.load_state_dict(torch.load(f"{new_save_path}/a3c_{WORLD}_{STAGE}_episode_{init_ep}.pt"))

    if icm == True:
        optimizer = GlobalAdam(list(global_model.parameters()) + list(global_icm.parameters()), lr = LR)
    else:
        optimizer = GlobalAdam(global_model.parameters(), lr = LR)
    print("Global model created")
    print(global_model)
    # Multiprocessing variables
    print("cpu: ", mp.cpu_count())
    if init_ep == 0:
        global_episode = mp.Value('i', 0)
    else:
        global_episode = mp.Value('i', init_ep)
    logger = MetricLogger(LOG_PATH, init_ep, icm)

    categorical_workers = NUM_WORKERS - ARGMAX_WORKERS

    # Start workers
    workers = []
    for i in range(NUM_WORKERS):
        if i < categorical_workers:
            if i == 0:
                worker_process = mp.Process(target=worker, args=(global_model, optimizer, global_episode, MAX_EPISODES, logger, True, False, global_icm, new_save_path))
            else:
                worker_process = mp.Process(target=worker, args=(global_model, optimizer, global_episode, MAX_EPISODES, logger, True, False, global_icm, new_save_path))
        else:
            if i == categorical_workers:
                worker_process = mp.Process(target=worker, args=(global_model, optimizer, global_episode, MAX_EPISODES, logger, False, False, global_icm, new_save_path))
            else:
                worker_process = mp.Process(target=worker, args=(global_model, optimizer, global_episode, MAX_EPISODES, logger, False, False, global_icm, new_save_path))
        workers.append(worker_process)
        worker_process.start()
    
    print("Training started...")
    for worker_process in workers:
        worker_process.join()

    print("Training complete!")
    logger.plot_metrics()

if __name__ == "__main__":
    init_ep = 0 #cambiare a mano per continuare il training
    icm = True
    train(init_ep, icm)
