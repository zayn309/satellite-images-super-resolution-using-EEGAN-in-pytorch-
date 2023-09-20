import torch.nn as nn
import torch


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps ** 2))
        return loss


class ContentLoss(nn.Module):
    def __init__(self,DEVICE,config, logger):
        super(ContentLoss, self).__init__()
        self.DEVICE = DEVICE
        self.config = config
        self.vgg = None
        try:
            self.logger.info("===> loading the vgg19 model <===")
            self.vgg = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True).features[:36].eval().to(self.DEVICE)
            self.logger.info("===> the vgg19 loaded successfully <===")
        except :
            self.logger.exception("the vgg model is not loaded properly, cheack you internet connection!")
        
        self.loss = CharbonnierLoss().eval().to(self.DEVICE)
        self.BATCH_SIZE = self.config.data_loader.args.batch_size
        self.I_HR_WIDTH = self.config.hr_width
        
        
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, IBase, IHR):
        channel_one_IBase = torch.cat((IBase[:,0,:,:].unsqueeze(1), torch.zeros(self.BATCH_SIZE, 2, self.I_HR_WIDTH, self.I_HR_WIDTH).to(self.DEVICE)), dim = 1)
        channel_two_IBase = torch.cat((torch.zeros(self.BATCH_SIZE, 1, self.I_HR_WIDTH, self.I_HR_WIDTH).to(self.DEVICE),IBase[:,1,:,:].unsqueeze(1),torch.zeros(self.BATCH_SIZE, 1, self.I_HR_WIDTH, self.I_HR_WIDTH).to(self.DEVICE)),dim = 1)
        channel_three_IBase = torch.cat((torch.zeros(self.BATCH_SIZE, 2, self.I_HR_WIDTH, self.I_HR_WIDTH).to(self.DEVICE),IBase[:,2,:,:].unsqueeze(1)),dim = 1)
        channel_four_IBase = torch.cat((torch.zeros(self.BATCH_SIZE, 2, self.I_HR_WIDTH, self.I_HR_WIDTH).to(self.DEVICE),IBase[:,3,:,:].unsqueeze(1)),dim = 1)
        
        channel_one_IHR = torch.cat((IHR[:,0,:,:].unsqueeze(1), torch.zeros(self.BATCH_SIZE, 2, self.I_HR_WIDTH, self.I_HR_WIDTH).to(self.DEVICE)), dim = 1)
        channel_two_IHR = torch.cat((torch.zeros(self.BATCH_SIZE, 1, self.I_HR_WIDTH, self.I_HR_WIDTH).to(self.DEVICE),IHR[:,1,:,:].unsqueeze(1),torch.zeros(self.BATCH_SIZE, 1, self.I_HR_WIDTH, self.I_HR_WIDTH).to(self.DEVICE)),dim = 1)
        channel_three_IHR = torch.cat((torch.zeros(self.BATCH_SIZE, 2, self.I_HR_WIDTH, self.I_HR_WIDTH).to(self.DEVICE),IHR[:,2,:,:].unsqueeze(1)),dim = 1)
        channel_four_IHR = torch.cat((torch.zeros(self.BATCH_SIZE, 2, self.I_HR_WIDTH, self.I_HR_WIDTH).to(self.DEVICE),IHR[:,3,:,:].unsqueeze(1)),dim = 1)
        #channel one
        IBase_features = self.vgg(channel_one_IBase).detach()
        IHR_features = self.vgg(channel_one_IHR).detach()
        loss_content = self.loss(IBase_features,IHR_features)
        #channel two
        IBase_features = self.vgg(channel_two_IBase).detach()
        IHR_features = self.vgg(channel_two_IHR).detach()
        loss_content = loss_content + self.loss(IBase_features,IHR_features)
        #channel three
        IBase_features = self.vgg(channel_three_IBase).detach()
        IHR_features = self.vgg(channel_three_IHR).detach()
        loss_content = loss_content + self.loss(IBase_features,IHR_features)
        #channel four
        IBase_features = self.vgg(channel_four_IBase).detach()
        IHR_features = self.vgg(channel_four_IHR).detach()
        loss_content = loss_content + self.loss(IBase_features,IHR_features)
        
        loss_content = loss_content / 4
        return loss_content
    
