from sr.models.EEGAN.Gen_components import Dense_block, UpsamplingBlock
import torch
import torch.nn as nn


class UDSN(nn.Module):
    def __init__(self,in_channels,features, scale_factor = 4 ):
        super(UDSN, self).__init__()
        self.dense1 = Dense_block(in_channels, features)
        self.dense2 = Dense_block(features,features)
        self.dense3 = Dense_block(features,features)
        self.dense4 = Dense_block(features,features)
        self.dense5 = Dense_block(features,features)
        self.dense6 = Dense_block(features,features)
        self.up_scale = UpsamplingBlock(features, features ,scale_factor )
        
    def forward(self,x):
        #result = torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(matrix1, matrix2), matrix3), matrix4), matrix5), matrix6)

        out1 = self.dense1(x)
        
        out2 = torch.matmul(self.dense2(out1) ,out1)
        
        out3 = torch.matmul(torch.matmul(self.dense3(out2),out2),out1)
        
        out4 = torch.matmul(torch.matmul(torch.matmul(self.dense4(out3),out3),out2),out1)
        
        out5 = torch.matmul(torch.matmul(torch.matmul(torch.matmul(self.dense5(out4),out4),out3),out2),out1)
        
        out6 = torch.matmul(torch.matmul(torch.matmul(torch.matmul(torch.matmul(self.dense5(out5),out5),out4),out3),out2),out1)
        
        up_scaled_image = self.up_scale(out6)
        
        return up_scaled_image