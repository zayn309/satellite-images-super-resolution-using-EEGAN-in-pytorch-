import torch
import torch.nn as nn 
from sr.models.EEGAN.UDSN import UDSN
from sr.models.EEGAN.EESN import EESN
from sr.base.base_model import BaseModel

class EEGAN_generator(BaseModel):
  def __init__(self, in_nc,features, scale_factor):
    super(EEGAN_generator, self).__init__()
    self.UDSN = UDSN(in_channels=in_nc,features=features,scale_factor=scale_factor)
    self.EESN = EESN(scale_factor=scale_factor)

  def forward(self, x):
    I_base = self.UDSN(x) 
    I_learned_lap ,I_lap = self.EESN(I_base) # EESN net
    I_sr = I_learned_lap + I_base - I_lap

    return I_base, I_sr, I_learned_lap, I_lap
  
def test_generator():
  batch_size = 8
  in_channels = 4
  height, width = 64,64
  features = 64
  input_tensor = torch.randn(batch_size, in_channels, height, width)
  model = EEGAN_generator(in_channels,features)
  A , B, C, D = model(input_tensor)
  print(A.shape ,B.shape,C.shape,D.shape)
  