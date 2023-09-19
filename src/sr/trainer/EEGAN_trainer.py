import torch
import torch.nn as nn
from sr.base.base_trainer import BaseTrainer
from sr.models.EEGAN.generator import EEGAN_generator
from sr.models.EEGAN.Discriminator import Discriminator_VGG_128
from sr.data_loader.data_loaders import SR_dataLoader
from sr.metrics.metrics import Metrics
from sr.models.EEGAN.loss import (CharbonnierLoss, ContentLoss)
from sr.models.EEGAN.lr_scheduler import LR_Scheduler
from sr.utils.utils import plot_examples

# model_G,model_D , opt_G , opt_D , config , logger , data_loader = None , metrics = None,  data_loader_valid = None 
class EEGAN_Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config ,logger):

        generator = EEGAN_generator(config.network_G.in_nc,config.network_G.features,scale_factor=config.scale)
        discriminator = Discriminator_VGG_128(config.network_D.in_nc,config.network_D.features)
        
        opt_G = torch.optim.Adam(generator.parameters(),
                                      config.optimizer.args.lr_G, betas=(config.optimizer.args.beta1_G,config.optimizer.args.beta2_G))
        opt_D = torch.optim.Adam(discriminator.parameters(),
                                      config.optimizer.args.lr_D, betas=(config.optimizer.args.beta1_D,config.optimizer.args.beta2_D))
        
        data_loader = SR_dataLoader(config)
        
        val_data_loader = None
        if self.config.data_loader.args.validation_split:
            val_data_loader = data_loader.split_validation()
        
        metrics = Metrics(config)
        
        lr_scheduler = LR_Scheduler(optimizers=[opt_G, opt_D],config=config)
        
        super().__init__(generator,discriminator,opt_G,opt_D,config,
                         logger,data_loader,metrics, val_data_loader,lr_scheduler=lr_scheduler)
        
        self.len_epoch = len(self.data_loader)
        
        self.plot_freq = self.config.train.plote_examples_freq
        
        self.G_scaler = torch.cuda.amp.GradScaler()
        self.D_scaler = torch.cuda.amp.GradScaler()
        
        # Losses
        self.loss_terms = self.config.loss.terms
        self.alpha = self.config.loss.alpha
        self.lamda = self.config.loss.lamda
        
        self.consistencyLoss = CharbonnierLoss().to(self.device)
        self.contentLoss = ContentLoss(self.device,config=self.config,logger=self.logger)
        self.BCE = nn.BCEWithLogitsLoss()
        
    def _train_epoch(self, epoch):
        # L(θG, θD) = Losscont(θG) + αLossadv(θG, θD)+λLosscst(θG)
        self.model_G.train()
        self.model_D.train()
        total_loss = 0
        log = {}
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # train the the disc with the adv loss
            
            """
                Then we feed IBase and IHR into D to determine their
                authenticities, real or fake. We train the discriminator by
                minimizing the adversarial loss, which can encourage G to
                generate the reconstructed image ISR close to the ground truth
                IHR. This process can be formatted as follows:
                Lossadv(θD) = −logD(IHR) − log(1 − D(G(ILR)))
            """
            with torch.cuda.amp.autocast():
                I_base , I_sr,_,_ = self.model_G(data)
                disc_real = self.model_D(target)
                disc_fake = self.model_D(I_base.detach())
                disc_loss_real = self.BCE(
                    disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
                )
                disc_loss_fake = self.BCE(disc_fake, torch.zeros_like(disc_fake))
                loss_disc = self.alpha * (disc_loss_fake + disc_loss_real)

            self.opt_D.zero_grad()
            self.D_scaler.scale(loss_disc).backward()
            self.D_scaler.step(self.opt_D)
            self.D_scaler.update()
            
            # train the generator with the consistency loss and the content loss
            
            with torch.cuda.amp.autocast():
                content_loss =  self.contentLoss(I_base,target)
                consistency_loss = self.consistencyLoss(I_sr, target)            
                loss_gen = content_loss + self.lamda * consistency_loss
            
            self.opt_G.zero_grad()
            self.G_scaler.scale(loss_gen).backward()
            self.G_scaler.step(self.opt_G)
            self.G_scaler.update()

            total_loss = loss_disc + loss_gen
            
            log.update({"total_loss" : total_loss,
                        "vgg_loss" : content_loss,
                        "cst_loss" : consistency_loss,
                        "adv_loss" : loss_disc
                        })
            
            if batch_idx == self.len_epoch:
                break
        

        if self.current_epoch % self.config.train.val_freq == 0:
            val_log = self._valid_epoch(epoch,train=False, log=log)
            log.update(val_log)
        if self.current_epoch % self.config.train.train_val_freq == 0:
            val_log = self._valid_epoch(epoch, train = True, log=log)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.current_epoch % self.plot_freq == 0:
            self.plot_examples()
        
        return log

    def _valid_epoch(self, epoch , train = False, log = None):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model_G.eval()
        
        if log is not None:
            if not train:
                log.update({key + "_val": 0 for key in self.config.metrics})
            else:
                log.update({key + "_train": 0 for key in self.config.metrics})
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader) if not train else  enumerate(self.data_loader):
                _, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                results = self.metrics.result(target,output)
                for key in results:
                    log[key] = log[key] + results[key]
        
        for key in self.config.metrics:
            if train:
                log[key] = log[key] / len(self.data_loader)
            else:
                log[key] = log[key] / len(self.data_loader_valid)                
        return log
    
    def plot_examples(self):
        random_batch = next(iter(self.data_loader))
        self.model_G.eval()
        I_base, I_sr, I_learned_lap, I_lap = self.model_G(random_batch)
        plot_examples(I_base,I_lap,I_learned_lap,I_sr,self.config)
        