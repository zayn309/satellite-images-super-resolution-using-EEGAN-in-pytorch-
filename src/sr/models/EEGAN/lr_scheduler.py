from torch.optim.lr_scheduler import _LRScheduler

class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizers, initial_lr=2e-4, final_lr=1e-5, decay_factor=0.9):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay_factor = decay_factor
        self.current_lr = initial_lr
        self.optimizers = optimizers
        super(CustomLRScheduler, self).__init__(optimizers[0])

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
                
