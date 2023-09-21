from sr.base.base_model import BaseModel
import torch
import torch.nn as nn
from functools import partial
import functools
import torch.nn.functional as F
from kornia.filters import laplacian
from sr.utils.utils import (initialize_weights, make_layer)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpsamplingBlock, self).__init__()
        self.conv_prelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(out_channels)
        )
        self.sub_pixel_conv = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv_prelu(x)
        x = self.sub_pixel_conv(x)
        return x


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x
    
class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
    
class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.up = UpsamplingBlock(nf,nf,4)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # out = self.conv_last(self.lrelu(self.HRconv(fea)))
        out = self.lrelu(self.upconv1(fea))
        out = self.lrelu(self.upconv2(fea))
        out = self.lrelu(self.upconv1(fea))
        out = self.up(fea)

        return out

# UDSN components
class ConvLayer(nn.Module):  
    def __init__(self, in_channels,out_channels, kernel_size, strides=[1, 1, 1], padding=1): 
        assert len(strides) == 3
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides[0],
            padding=padding
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,  
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides[1],
            padding=padding
        )
        self.conv3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides[2],
            padding=padding
        )
        
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x, act=True ):
        if isinstance(x, list):
            if act:
                return [
                    self.activation(self.conv1(x[0])),
                    self.activation(self.conv2(x[1])),
                    self.activation(self.conv3(x[2]))
                ]
            else:
                return [
                    self.conv1(x[0]),
                    self.conv2(x[1]),
                    self.conv3(x[2])
                ]
        else:
            if act:
                return [
                    self.activation(self.conv1(x)),
                    self.activation(self.conv2(x)),
                    self.activation(self.conv3(x))
                ]
            else:
                return [
                    self.conv1(x),
                    self.conv2(x),
                    self.conv3(x)
                ]

class Basic_block(nn.Module):
    def __init__(self, in_channels, features):
        super(Basic_block, self).__init__()
        
        self.conv1 = ConvLayer(in_channels ,features ,kernel_size= 3)
        
        self.activation = nn.LeakyReLU(0.2)
        
        self.conv2 = ConvLayer(features*3, in_channels, kernel_size = 1, padding = 0)
        
        
    def forward(self,x):
        
        residual = x
        
        # conv f3
        out = self.conv1.forward(x)
        
        # concatenation
        out = torch.cat(out,dim = 1)
        
        # conv f1
        out = self.conv2.forward(out,act = False)
        
        if (residual is not None):
            if isinstance(residual, list):
                out[0] = self.activation(out[0] + residual[0])
                out[1] = self.activation(out[1] + residual[1])
                out[2] = self.activation(out[2] + residual[2])
            else:
                out[0] = self.activation(out[0] + residual)
                out[1] = self.activation(out[1] + residual)
                out[2] = self.activation(out[2] + residual)
        
        return out
    
class Dense_block(nn.Module):
    def __init__(self,in_channels, features):
        super(Dense_block,self).__init__()
        
        self.block1 = nn.Sequential(
                            nn.Conv2d(
                                in_channels = in_channels,
                                out_channels = features,
                                kernel_size = 3,
                                stride = 1,
                                padding = 1
                            ),
                            nn.LeakyReLU(0.2)
                        )
        self.block2 = Basic_block(features,features)
        self.block3 = Basic_block(features,features)
        self.block4 = Basic_block(features,features)
        self.block5 =  nn.Sequential(
                            nn.Conv2d(
                                in_channels = features*3,
                                out_channels = features,
                                kernel_size = 3,
                                stride = 1,
                                padding = 1
                            ),
                            nn.LeakyReLU(0.2)
                        )
        
    def forward(self,x):
        x = self.block1(x)
        residual = x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.cat(x,dim = 1)
        x = self.block5(x)
        return x + residual
  
# EESN components
class BeginEdgeConv(nn.Module):
    def __init__(self):
        super(BeginEdgeConv, self).__init__()

        self.conv_layer1 = nn.Conv2d(4, 64, 3, 1, 1, bias=True)
        self.conv_layer2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_layer3 = nn.Conv2d(64, 128, 3, 2, 1, bias=True)
        self.conv_layer4 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.conv_layer5 = nn.Conv2d(128, 256, 3, 2, 1, bias=True)
        self.conv_layer6 = nn.Conv2d(256, 64, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv_layer1, self.conv_layer2, self.conv_layer3,
                            self.conv_layer4, self.conv_layer5, self.conv_layer6], 0.1)
        

    def forward(self, x):
      x = self.lrelu(self.conv_layer1(x))
      x = self.lrelu(self.conv_layer2(x))
      x = self.lrelu(self.conv_layer3(x))
      x = self.lrelu(self.conv_layer4(x))
      x = self.lrelu(self.conv_layer5(x))
      x = self.lrelu(self.conv_layer6(x))

      return x

class EESNRRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(EESNRRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        out = self.lrelu(self.conv_last(self.lrelu(self.HRconv(fea))))

        return out

class MaskConv(nn.Module): ## mask branch
    def __init__(self):
        super(MaskConv, self).__init__()

        self.conv_layer1 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_layer2 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)
        self.conv_layer3 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        initialize_weights([self.conv_layer1, self.conv_layer2, self.conv_layer3], 0.1)

    def forward(self, x):
        x = self.lrelu(self.conv_layer1(x))
        x = self.lrelu(self.conv_layer2(x))
        x = self.lrelu(self.conv_layer3(x))
        x = torch.sigmoid(x)

        return x

class FinalConv(nn.Module):
    def __init__(self):
        super(FinalConv, self).__init__()

        self.upconv1 = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(128, 64, 3, 1, 1, bias=True)
        # self.conv_last = nn.Conv2d(64, 4, 3, 1, 1, bias=True)
        self.up = UpsamplingBlock(64,64,4)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        

    def forward(self, x):
#         x = self.lrelu(self.upconv1(F.interpolate(x, scale_factor=2, mode='nearest')))
#         x = self.lrelu(self.upconv2(F.interpolate(x, scale_factor=2, mode='nearest')))
#         x = self.conv_last(self.lrelu(self.HRconv(x)))
        x = self.lrelu(self.upconv1(x))
        x = self.lrelu(self.upconv2(x))
        x = self.lrelu(self.HRconv(x))
        x = self.up(x)
        return x

class EESN(nn.Module):
    def __init__(self):
        super(EESN, self).__init__()
        self.beginEdgeConv = BeginEdgeConv() #  Output 64*64*64 input 3*64*64
        self.denseNet = EESNRRDBNet(64, 256, 64, 5) # RRDB densenet with 64 in kernel, 256 out kernel and 64 intermediate kernel, output: 256*64*64
        self.maskConv = MaskConv() # Output 256*64*64
        self.finalConv = FinalConv() # Output 3*256*256

    def forward(self, x):
        x_lap = laplacian(x, 3) # see kornia laplacian kernel
        x1 = self.beginEdgeConv(x_lap)
        x2 = self.denseNet(x1)
        x3 = self.maskConv(x1)
        x4 = x3*x2 + x2
        x_learned_lap = self.finalConv(x4)

        return x_learned_lap, x_lap
    

class ESRGAN_EESN(BaseModel):
    def __init__(self, in_nc, out_nc, nf, nb):
        super(ESRGAN_EESN, self).__init__()
        self.netRG = RRDBNet(in_nc, out_nc, nf, nb)
        self.netE = EESN()

    def forward(self, x):
        x_base = self.netRG(x) # add bicubic according to the implementation by author but not stated in the paper
        x5, x_lap = self.netE(x_base) # EESN net
        x_sr = x5 + x_base - x_lap

        return x_base, x_sr, x5, x_lap
    def __str__(self):
        return f'{super().__str__()} '