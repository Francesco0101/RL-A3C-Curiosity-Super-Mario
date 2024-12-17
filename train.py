import torch
import torch.multiprocessing as _mp
from env import create_train_env
from model import ActorCritic
from worker import worker
from shared_optim import GlobalAdam
from constants import *
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'




def train():
    mp = _mp.get_context('spawn') # Create a new context for multiprocessing --> without, deadlock

    # Create environment
    _, input_dim, action_dim = create_train_env()
    print("action_dim: ", action_dim)
    print("input_dim: ", input_dim)
    save_path = Path(SAVE_PATH)
    if not save_path.exists():
        save_path.mkdir()
    # Create global model and optimizer
    global_model = ActorCritic(input_dim, action_dim).to(device)
    global_model.share_memory()
    optimizer = GlobalAdam(global_model.parameters(), lr = LR)
    print("Global model created")
    print(global_model)

    # Multiprocessing variables
    print("cpu: ", mp.cpu_count())
    global_episode = mp.Value('i', 0)

    # Start workers
    workers = []
    for _ in range(NUM_WORKERS):
        worker_process = mp.Process(target=worker, args=(global_model, optimizer, global_episode, MAX_EPISODES))
        workers.append(worker_process)
        worker_process.start()
    
    print("Training started...")
    for worker_process in workers:
        worker_process.join()

    print("Training complete!")

if __name__ == "__main__":
    train()
