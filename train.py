import torch
import torch.multiprocessing as _mp
from environment.env import create_train_env
from models.model import ActorCritic
from a3c.worker import worker
from optimizer.shared_optim import GlobalAdam
from utils.constants import *
import warnings
from utils.logger import MetricLogger
import os
from models.icm import ICM

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

device = 'cpu'


mp = _mp.get_context('spawn')


def train(init_ep = 0, icm = False):
    """
    Function to train the A3C model with or without Intrinsic Curiosity Module (ICM).
    Handles environment setup, model initialization, and training using multiple worker processes.
    
    Parameters:
    init_ep (int): Initial episode to resume from. Default is 0.
    icm (bool): Whether to use Intrinsic Curiosity Module. Default is False (no curiosity).
    """
    print(init_ep)
    os.environ['OMP_NUM_THREADS'] = '1'

    _, input_dim, action_dim = create_train_env()
    print("action_dim: ", action_dim)
    print("input_dim: ", input_dim)

    save_path = SAVE_PATH
    if icm == True:
        save_path = save_path + "curiosity/"
    else:
        save_path = save_path + "no_curiosity/"
    
    if REWARD_TYPE == "dense":
        save_path = save_path + "dense/"
    elif REWARD_TYPE == "sparse":
        save_path = save_path + "sparse/"
    else:
        save_path = save_path + "no_reward/"

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

    global_model = ActorCritic(input_dim, action_dim).to(device)
    global_model.share_memory()
    if icm == True:
        global_icm = ICM(input_dim, action_dim).to(device)
        global_icm.share_memory()
    else:
        global_icm = None

    if init_ep != 0:
        global_model.load_state_dict(torch.load(f"{new_save_path}/a3c_episode_{init_ep}.pt"))
        if icm == True:
            global_icm.load_state_dict(torch.load(f"{new_save_path}/icm_episode_{init_ep}.pt"))
    if SHARED_OPTIMIZER == True:
        if icm == True:
            optimizer = GlobalAdam(list(global_model.parameters()) + list(global_icm.parameters()), lr = LR)
        else:
            optimizer = GlobalAdam(global_model.parameters(), lr = LR)
    else:
        optimizer = None
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
    init_ep = 0
    icm = True
    train(init_ep, icm)
