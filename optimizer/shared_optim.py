import torch

class GlobalAdam(torch.optim.Adam):
    """
    A custom implementation of the Adam optimizer that supports shared memory,
    enabling parallel training across multiple processes. 
    Inherits from:
        torch.optim.Adam: The base implementation of the Adam optimizer.

    Args:
        params: Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate for the optimizer.
    """
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

