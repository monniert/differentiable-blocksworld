from bisect import bisect_right
from collections import Counter

from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, _LRScheduler
from utils.logger import print_log


def create_scheduler(cfg, optimizer):
    kwargs = cfg["training"].get("scheduler", {}) or {}
    name = kwargs.pop("name", None)
    scheduler = get_scheduler(name)(optimizer, **kwargs)
    print_log(f"Scheduler '{name}' init: kwargs={kwargs}")
    return scheduler


def get_scheduler(name):
    if name is None:
        name = 'multi_step'
    return {
        "multi_step": MultiStepLR,
        "cosine_annealing": CosineAnnealingLR,
        "exponential": ExponentialLR,
    }[name]


class MultiStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float or list of float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        warmup (int): Nb of epochs of gradually increasing LR
    """

    def __init__(self, optimizer, milestones=None, gamma=0.1, last_epoch=-1, warmup=0):
        self.milestones = Counter(milestones or [])
        self.gamma = [gamma] * len(optimizer.param_groups) if isinstance(gamma, (float,)) else gamma
        self.warmup = warmup
        super(MultiStepLR, self).__init__(optimizer, last_epoch)
        if warmup > 0:
            for group in optimizer.param_groups:
                group['lr'] /= self.warmup

    def get_lr(self):
        if not self._get_lr_called_within_step:
            print_log("To get the last learning rate computed by the scheduler, "
                      "please use `get_last_lr()`.", level='warning')

        if self.warmup > self.last_epoch:
            return [lr / self.warmup * (self.last_epoch + 1) for lr in self.base_lrs]
        elif self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return [group['lr'] * gamma ** self.milestones[self.last_epoch]
                    for group, gamma in zip(self.optimizer.param_groups, self.gamma)]

    def _get_closed_form_lr(self):
        if self.warmup > self.last_epoch:
            return [lr / self.warmup * (self.last_epoch + 1) for lr in self.base_lrs]
        else:
            milestones = sorted(self.milestones.elements())
            return [base_lr * gamma ** bisect_right(milestones, self.last_epoch)
                    for base_lr, gamma in zip(self.base_lrs, self.gamma)]
