import torch.nn as nn
import torch
from sr.utils.utils import apply_pca

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
    def __init__(self,DEVICE, logger):
        super(ContentLoss, self).__init__()
        self.DEVICE = DEVICE
        self.vgg = None
        self.logger = logger
        try:
            self.logger.info("===> loading the vgg19 model <===")
            self.vgg = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True).features[:36].eval().to(self.DEVICE)
            self.vgg = self.vgg.double()
            self.logger.info("===> the vgg19 loaded successfully <===")
        except :
            self.logger.exception("the vgg model is not loaded properly, cheack you internet connection!")
        
        self.loss = CharbonnierLoss().eval().to(self.DEVICE)
        
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, IBase, IHR):
        I_Base_compressed = apply_pca(IBase,self.DEVICE)
        IHR_compressed = apply_pca(IHR,self.DEVICE)
        IBase_features = self.vgg(I_Base_compressed)
        IHR_features = self.vgg(IHR_compressed)
        loss = self.loss(IBase_features,IHR_features)
        return loss
    