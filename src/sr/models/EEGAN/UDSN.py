import torch
import torch.nn as nn
from sr.models.EEGAN.Gen_components import Dense_block, UpsamplingBlock


class UDSN(nn.Module):
    def __init__(self,in_channels,features, scale_factor):
        super(UDSN, self).__init__()
        self.dense1 = Dense_block(in_channels, features)
        self.dense2 = Dense_block(features,features)
        self.dense3 = Dense_block(features,features)
        self.dense4 = Dense_block(features,features)
        self.dense5 = Dense_block(features,features)
        self.dense6 = Dense_block(features,features)
        self.up_scale = UpsamplingBlock(features, features ,scale_factor )
        
    def forward(self, x):
        out1 = self.dense1(x)
        out2 = self.dense2(out1) + out1
        out3 = self.dense3(out2) + out1 + out2  # Add out1 and out2
        out4 = self.dense4(out3) + out1 + out2 + out3  # Add out1, out2, and out3
        out5 = self.dense5(out4) + out1 + out2 + out3 + out4  # Add out1, out2, out3, and out4
        out6 = self.dense6(out5) + out1 + out2 + out3 + out4 + out5  # Add out1, out2, out3, out4, and out5
        up_scaled_image = self.up_scale(out6)
        
        return up_scaled_image