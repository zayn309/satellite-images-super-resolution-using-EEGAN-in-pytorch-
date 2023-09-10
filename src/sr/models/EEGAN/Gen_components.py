import torch
import torch.nn as nn

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


# EESN components

class BeginEdgeConv(nn.Module):
    def __init__(self, features=[128, 256, 64]):
        super(BeginEdgeConv, self).__init__()
        layers = []
        in_channels = [4] + features[:-1]
        out_channels = features

        for in_ch, out_ch in zip(in_channels, out_channels):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2 if out_ch != out_channels[-1] else 1 , padding=1))
            layers.append(nn.PReLU(num_parameters=out_ch))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class EESN_dense(nn.Module):
    def __init__(self, in_channels = 64, features = 64):
        super(EESN_dense, self).__init__()

        self.block1 = Dense_block(in_channels, features)
        self.block2 = Dense_block(features, features)
        self.block3 = Dense_block(features, features)

        self.conv3 = nn.Conv2d(features, features*2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(features*2,features*4, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x =  self.leaky_relu(self.conv3(x))
        x =  self.leaky_relu(self.conv4(x))
        return x

class MaskConv(nn.Module): ## mask branch
    def __init__(self):
        super(MaskConv, self).__init__()

        self.conv_layer1 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_layer2 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)
        self.conv_layer3 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)



    def forward(self, x):
        x = self.lrelu(self.conv_layer1(x))
        x = self.lrelu(self.conv_layer2(x))
        x = self.lrelu(self.conv_layer3(x))
        x = torch.sigmoid(x)

        return x

class FinalConv(nn.Module):
    def __init__(self, in_channels = 256, out_channels = 64, scale_factor = 4):
        super(FinalConv, self).__init__()
        self.up = UpsamplingBlock(in_channels, out_channels, scale_factor)
        
    def forward(self, x):
        return self.up(x)
    
