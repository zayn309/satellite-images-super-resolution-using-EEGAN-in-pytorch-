import torch.nn as nn
import torch

class ContentLoss(nn.Module):
    def __init__(self,config):
        super(ContentLoss, self).__init__()
        self.config = config
        self.vgg = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True).features[:36].eval().to(self.config['DEVICE'])
        self.loss = nn.MSELoss()
        self.epsilon = 1e-3
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, IBase, IHR):

        channel_one_IBase = torch.cat((IBase[:,0,:,:].unsqueeze(0), torch.zeros(1, 2, 256, 256).to(self.config['DEVICE'])), dim=1)
        channel_two_IBase = torch.cat((torch.zeros(1, 1, 256, 256).to(self.config['DEVICE']),IBase[:,1,:,:].unsqueeze(0),torch.zeros(1, 1, 256, 256).to(self.config['DEVICE'])),dim = 1)
        channel_three_IBase = torch.cat((torch.zeros(1, 2, 256, 256).to(self.config['DEVICE']),IBase[:,2,:,:].unsqueeze(0)),dim = 1)
        channel_four_IBase = torch.cat((torch.zeros(1, 2, 256, 256).to(self.config['DEVICE']),IBase[:,3,:,:].unsqueeze(0)),dim = 1)


        channel_one_IHR = torch.cat((IHR[:,0,:,:].unsqueeze(0), torch.zeros(1, 2, 256, 256).to(self.config['DEVICE'])), dim=1)
        channel_two_IHR = torch.cat((torch.zeros(1, 1, 256, 256).to(self.config['DEVICE']),IHR[:,1,:,:].unsqueeze(0),torch.zeros(1, 1, 256, 256).to(self.config['DEVICE'])),dim = 1)
        channel_three_IHR = torch.cat((torch.zeros(1, 2, 256, 256).to(self.config['DEVICE']),IHR[:,2,:,:].unsqueeze(0)),dim = 1)
        channel_four_IHR = torch.cat((torch.zeros(1, 2, 256, 256).to(self.config['DEVICE']),IHR[:,3,:,:].unsqueeze(0)),dim = 1)
        #channel one
        IBase_features = self.vgg(channel_one_IBase)
        IHR_features = self.vgg(channel_one_IHR)
        loss_content = torch.mean((IBase_features - IHR_features)**2 + self.epsilon**2)
        #channel two
        IBase_features = self.vgg(channel_two_IBase)
        IHR_features = self.vgg(channel_two_IHR)
        loss_content = loss_content + torch.mean((IBase_features - IHR_features)**2 + self.epsilon**2)
        #channel three
        IBase_features = self.vgg(channel_three_IBase)
        IHR_features = self.vgg(channel_three_IHR)
        loss_content = loss_content + torch.mean((IBase_features - IHR_features)**2 + self.epsilon**2)
        #channel four
        IBase_features = self.vgg(channel_four_IBase)
        IHR_features = self.vgg(channel_four_IHR)
        loss_content = loss_content + torch.mean((IBase_features - IHR_features)**2 + self.epsilon**2)
        
        loss_content = loss_content / 4
        return loss_content