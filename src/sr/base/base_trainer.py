import torch
from abc import abstractmethod
from pathlib import Path
from sr.utils.utils import dict2str

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model_G,model_D , opt_G , opt_D , config , logger , data_loader  , metrics ,  data_loader_valid = None, lr_scheduler=None ):
        
        self.config = config
        self.metrics = metrics
        # setup GPU device if available, move model into configured device
        self.logger = logger
        self.device, device_ids = self._prepare_device(config.n_gpu)
        self.model_G = model_G.to(self.device)
        self.model_D = model_D.to(self.device)
        
        if len(device_ids) > 1:
            self.model_G = torch.nn.DataParallel(model_G, device_ids=device_ids)
            self.model_D = torch.nn.DataParallel(model_D, device_ids=device_ids)
            

        self.opt_G = opt_G
        self.opt_D = opt_D
        

        self.data_loader = data_loader
        self.data_loader_valid = data_loader_valid
        
        self.epochs = self.config.train.niter
        self.current_epoch = 1
        self.save_period = self.config.logger.save_checkpoint_freq
        
        self.checkpoint_dir = Path(config.train.save_dir)
        
        self.lr_scheduler = lr_scheduler

        if config.train.resume is not None:
            self._resume_checkpoint(config.train.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        self.logger.info(f'\n ===> started training <===\n' +
                 f'generator: \n {str(self.model_G)}\n' +
                 f'Discrimininator: \n {str(self.model_D)}\n' +
                 f'optimizers: {self.config.optimizer.type}\n' +
                 f'the training will last for {self.epochs} starting from epoch {self.current_epoch}\n'
                 f'the training will be done on {self.device} device')

        for epoch in range(self.current_epoch, self.epochs + 1):
            print(f'started epoch numner {epoch}')
            result = self._train_epoch(epoch)
            #result = {'psnr': 30.55, 'ssim' : 0.6565 , 'mse': 2.46452, 'vgg Loss': 502.012}
            # save logged informations into log dict
            
            log = {'results of epoch ': epoch}
            log.update(result)
            log = dict2str(log)
            # print logged informations to the screen
            
            self.logger.info(log)

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _prepare_device(self, n_gpu_use): # working 
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch): # working 
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict_G': self.model_G.state_dict(),
            'state_dict_D': self.model_D.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path): # working
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.current_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config'].arch != self.config.arch:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model_G.load_state_dict(checkpoint['state_dict_G'])
        self.model_D.load_state_dict(checkpoint['state_dict_D'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config'].optimizer.type != self.config.optimizer.type:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.opt_G.load_state_dict(checkpoint['opt_G'])
            self.opt_D.load_state_dict(checkpoint['opt_D'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.current_epoch))
