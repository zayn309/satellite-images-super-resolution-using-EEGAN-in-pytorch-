from torch.optim.lr_scheduler import _LRScheduler

class LR_Scheduler(_LRScheduler):
    def __init__(self, optimizers: list,config):
        self.config = config
        self.initial_lr = self.config.LR_Scheduler.initial_lr
        self.final_lr = self.config.LR_Scheduler.final_lr
        self.decay_factor = self.config.LR_Scheduler.decay_factor
        self.current_lr = self.initial_lr
        self.optimizers = optimizers
        super(LR_Scheduler, self).__init__(optimizers[0])

    def get_lr(self):
        return [self.current_lr] * len(self.optimizers[0].param_groups)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.current_lr > self.final_lr:
            self.current_lr *= self.decay_factor
        else:
            self.current_lr = self.final_lr

        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.current_lr
