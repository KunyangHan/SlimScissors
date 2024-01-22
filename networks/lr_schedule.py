import math
import numpy as np

class LR_Scheduler(object):
    """Learning Rate Scheduler
    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`
        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=None, warmup_epochs=0, target_lr=1e-8, start_lr=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.target_lr = target_lr
        self.iters_per_epoch = iters_per_epoch
        # self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.N = iters_per_epoch * (num_epochs - warmup_epochs)
        assert start_lr < base_lr
        self.start_lr = start_lr

    def __call__(self, optimizer, i, epoch, best_pred=0, ten_time_group=[1]):
        T = epoch * self.iters_per_epoch + i
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = (self.lr - self.start_lr) * T / self.warmup_iters + self.start_lr
        else:
            T -= self.warmup_iters
            if self.mode == 'cos':
                lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
            elif self.mode == 'poly':
                lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
            elif self.mode == 'step':
                lr = self.lr * (0.1 ** (epoch // self.lr_step))
            elif self.mode == 'linear':
                lr = self.target_lr + (1 - T / self.N) * (self.lr - self.target_lr)
            elif self.mode == 'keep':
                lr = self.lr
            else:
                raise NotImplemented

        lr = max(lr, 1e-8)
        self._adjust_learning_rate(optimizer, lr, ten_time_group)


    def _adjust_learning_rate(self, optimizer, lr, ten_time_group):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            assert optimizer.param_groups[0]['lr'] < optimizer.param_groups[-1]['lr']
            # optimizer.param_groups[0]['lr'] = lr
            for i in range(0, len(optimizer.param_groups)):
                if i in ten_time_group:
                    optimizer.param_groups[i]['lr'] = lr * 10
                else:
                    optimizer.param_groups[i]['lr'] = lr


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule