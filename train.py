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
    save_path = Path(SAVE_PATH)
    if init_ep == 0:
        if save_path.exists():
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
    # Create global model and optimizer
    global_model = ActorCritic(input_dim, action_dim).to(device)
    global_model.share_memory()

    if icm == True:
        global_icm = ICM(input_dim, action_dim).to(device)
        global_icm.share_memory()
    else:
        global_icm = None

    if init_ep != 0:
        global_model.load_state_dict(torch.load(f"checkpoints/a3c_1_1_episode_{init_ep}.pt"))

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
    logger = MetricLogger(LOG_PATH, init_ep)

    categorical_workers = NUM_WORKERS - ARGMAX_WORKERS

    # Start workers
    workers = []
    for i in range(NUM_WORKERS):
        if i < categorical_workers:
            if i == 0:
                worker_process = mp.Process(target=worker, args=(global_model, optimizer, global_episode, MAX_EPISODES, logger, True, True, global_icm))
            else:
                worker_process = mp.Process(target=worker, args=(global_model, optimizer, global_episode, MAX_EPISODES, logger, True, False, global_icm))
        else:
            if i == categorical_workers:
                worker_process = mp.Process(target=worker, args=(global_model, optimizer, global_episode, MAX_EPISODES, logger, False, True, global_icm))
            else:
                worker_process = mp.Process(target=worker, args=(global_model, optimizer, global_episode, MAX_EPISODES, logger, False, False, global_icm))
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
