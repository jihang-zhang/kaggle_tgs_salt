from torch import nn
from torch.nn import functional as F
import torch

from torchvision import models
import pretrainedmodels
from .oc_module.pyramid_oc_block import Pyramid_OC_Module

def conv3x3(in_, out, bias=True):
    return nn.Conv2d(in_, out, 3, padding=1, bias=bias)


class ConvBnRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out, bias=False)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class SpatialGate2d(nn.Module):
    def __init__(self, c):
        super(SpatialGate2d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c // 2, c, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.avg_pool2d(x, (x.size()[2], x.size()[3]))
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ChannelGate2d(nn.Module):
    def __init__(self, c):
        super(ChannelGate2d, self).__init__()
        self.conv = nn.Conv2d(c, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class Gate2d(nn.Module):
    def __init__(self, c):
        super(Gate2d, self).__init__()
        self.spatial_gate = SpatialGate2d(c)
        self.channel_gate = ChannelGate2d(c)

    def forward(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1*x + g2*x

        return x

class Decoder(nn.Module):
    def __init__(self, in_, mid, out, stack):
        super(Decoder, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_, mid, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )
        self.conv1 = ConvBnRelu(mid + stack, out)
        self.spatial_gate = SpatialGate2d(out)
        self.channel_gate = ChannelGate2d(out)

    def forward(self, x, e=None):
        x = self.deconv(x)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1*x + g2*x

        return x

class Superior(nn.Module):
    def __init__(self, num_classes=1, dropout_2d=0.5):
        super(Superior, self).__init__()

        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.encoder = models.resnet34(pretrained=True)

        self.conv1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            Gate2d(64)
        )

        self.conv2 = nn.Sequential(
            self.encoder.layer1,
            Gate2d(64)
        )
        self.conv3 = nn.Sequential(
            self.encoder.layer2,
            Gate2d(128)
        )
        self.conv4 = nn.Sequential(
            self.encoder.layer3,
            Gate2d(256)
        )
        self.conv5 = nn.Sequential(
            self.encoder.layer4,
            Gate2d(512)
        )

        self.context = Pyramid_OC_Module(in_channels=512, out_channels=512, dropout=0.05, sizes=([1, 2, 3, 6]))

        self.fuse_oc = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.center = nn.Sequential(
            ConvBnRelu(512, 512),
            ConvBnRelu(512, 256),
            nn.MaxPool2d(2, 2)
        )

        self.decoder5 = Decoder(256, 256, 64, 512)
        self.decoder4 = Decoder( 64, 128, 64, 256)
        self.decoder3 = Decoder( 64,  64, 64, 128)
        self.decoder2 = Decoder( 64,  64, 64,  64)
        self.decoder1 = Decoder( 64,  64, 64,   0)

        self.logit_pixel_8x8     = nn.Sequential(
            nn.Dropout2d(p=dropout_2d),
            ConvBnRelu(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )
        self.logit_pixel_16x16   = nn.Sequential(
            nn.Dropout2d(p=dropout_2d),
            ConvBnRelu(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )
        self.logit_pixel_32x32   = nn.Sequential(
            nn.Dropout2d(p=dropout_2d),
            ConvBnRelu(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )
        self.logit_pixel_64x64   = nn.Sequential(
            nn.Dropout2d(p=dropout_2d),
            ConvBnRelu(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )
        self.logit_pixel_128x128 = nn.Sequential(
            nn.Dropout2d(p=dropout_2d),
            ConvBnRelu(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )

        self.fuse_image = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True)
        )
        self.logit_image = nn.Linear(64, 1)

        self.logit = nn.Sequential(
            ConvBnRelu(448, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x = self.conv1(x)
        e2 = self.conv2(x)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)

        context = self.context(e5)
        fuse_oc = self.fuse_oc(context)

        f = self.center(context)
        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2)

        logit_pixel_8x8     = self.logit_pixel_8x8(d5)
        logit_pixel_16x16   = self.logit_pixel_16x16(d4)
        logit_pixel_32x32   = self.logit_pixel_32x32(d3)
        logit_pixel_64x64   = self.logit_pixel_64x64(d2)
        logit_pixel_128x128 = self.logit_pixel_128x128(d1)

        e = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)
        e = F.dropout2d(e, p=self.dropout_2d)
        fuse_image = self.fuse_image(e)
        logit_image = self.logit_image(fuse_image).view(batch_size, -1)

        hypercolumns = torch.cat((
            d1,
            F.upsample(d2, scale_factor= 2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor= 4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor= 8, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
            F.upsample(fuse_image.view(batch_size,-1,1,1,), scale_factor=128, mode='nearest'),
            F.upsample(fuse_oc, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)

        final = F.dropout2d(hypercolumns, p=self.dropout_2d)
        logit = self.logit(final)

        return logit, logit_pixel_8x8, logit_pixel_16x16, logit_pixel_32x32, logit_pixel_64x64, logit_pixel_128x128, logit_image
