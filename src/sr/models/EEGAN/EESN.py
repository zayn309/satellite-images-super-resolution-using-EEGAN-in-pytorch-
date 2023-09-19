from sr.base.base_model import BaseModel
import torch
import torch.nn as nn
import kornia
from sr.models.EEGAN.Gen_components import  BeginEdgeConv,EESN_dense,MaskConv,FinalConv

class EESN(BaseModel):
    def __init__(self, scale_factor):
        super(EESN, self).__init__()
        self.beginEdgeConv = BeginEdgeConv() #  Output 64*64*64 input 3*64*64
        self.denseNet = EESN_dense() 
        self.maskConv = MaskConv() # Output 256*64*64
        self.finalConv = FinalConv(256,64,scale_factor) # Output 4*256*256

    def forward(self, x):
        x_lap = kornia.filters.laplacian(x, 3) # see kornia laplacian kernel
        x1 = self.beginEdgeConv(x_lap)
        x2 = self.denseNet(x1)
        x3 = self.maskConv(x1)
        x4 = x3*x2 + x2
        x_learned_lap = self.finalConv(x4)

        return x_learned_lap ,x_lap
