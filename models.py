import torch.nn as nn
import torch
import torchvision
from ASPP import ASPP

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class Decod_block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.block = nn.Sequential(ConvRelu(in_channels, middle_channels), nn.ConvTranspose2d(
            middle_channels, out_channels, kernel_size=4, stride=2, padding=1),nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)

    # https://distill.pub/2016/deconv-checkerboard/


class Decoder(nn.Module):
    def __init__(self, num_classes=2, num_filters=32):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.center = Decod_block(512, num_filters*8*2, num_filters*8)
        self.dec5 = Decod_block(512 + num_filters * 8,
                                num_filters * 8 * 2, num_filters * 8)
        self.dec4 = Decod_block(512 + num_filters * 8,
                                num_filters * 8 * 2, num_filters * 8)
        self.dec3 = Decod_block(256 + num_filters * 8,
                                num_filters * 4 * 2, num_filters * 2)
        self.dec2 = Decod_block(128 + num_filters * 2,
                                num_filters * 2 * 2, num_filters)
        self.aspp = ASPP(16, nn.BatchNorm2d)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, down1, down2, down3, down4, down5):
        center = self.center(self.pool(down5))

        #dec5 = self.aspp(self.pool(down5))
        dec5 = self.dec5(torch.cat([center, down5], 1))
        dec4 = self.dec4(torch.cat([dec5, down4], 1))
        dec3 = self.dec3(torch.cat([dec4, down3], 1))
        dec2 = self.dec2(torch.cat([dec3, down2], 1))
        dec1 = self.dec1(torch.cat([dec2, down1], 1))

        return self.final(dec1)


class VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.vgg16_bn(pretrained=pretrained).features
        self.fc_cls = nn.Linear(512, 2)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            self.encoder[0], self.encoder[1], self.relu, self.encoder[3], self.encoder[4], self.relu)

        self.conv2 = nn.Sequential(
            self.encoder[7], self.encoder[8], self.relu, self.encoder[10], self.encoder[11], self.relu
        )

        self.conv3 = nn.Sequential(
            self.encoder[14],
            self.encoder[15],
            self.relu,
            self.encoder[17],
            self.encoder[18],
            self.relu,
            self.encoder[20],
            self.encoder[21],
            self.relu,
        )

        self.conv4 = nn.Sequential(
            self.encoder[24],
            self.encoder[25],
            self.relu,
            self.encoder[27],
            self.encoder[28],
            self.relu,
            self.encoder[30],
            self.encoder[31],
            self.relu,
        )

        self.conv5 = nn.Sequential(
            self.encoder[34],
            self.encoder[35],
            self.relu,
            self.encoder[37],
            self.encoder[38],
            self.relu,
            self.encoder[40],
            self.encoder[41],
            self.relu,
        )

    def forward(self, x):
        down1 = self.conv1(x)
        down2 = self.conv2(self.pool(down1))
        down3 = self.conv3(self.pool(down2))
        down4 = self.conv4(self.pool(down3))
        down5 = self.conv5(self.pool(down4))
        vector = torch.flatten(nn.AdaptiveAvgPool2d(output_size=(1, 1))(down5), 1)
        _cls = self.fc_cls(vector)

        return down1, down2, down3, down4, down5, _cls, vector


class Model(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=True, mode='pair'):
        super().__init__()

        self.base = VGG16(pretrained=pretrained)
        self.decoder = Decoder(num_classes=num_classes,
                               num_filters=num_filters)

    def forward(self, x, y = None):
        if y is not None:
            down1_x, down2_x, down3_x, down4_x, down5_x, x_cls, vector_x = self.base(x)
            down1_y, down2_y, down3_y, down4_y, down5_y, y_cls, vector_y = self.base(y)

            segm_x = self.decoder(down1_x, down2_x, down3_x, down4_x, down5_x)
            segm_y = self.decoder(down1_y, down2_y, down3_y, down4_y, down5_y)
            return x_cls, vector_x, segm_x, y_cls, vector_y, segm_y
        
        elif y is None:
            down1_x, down2_x, down3_x, down4_x, down5_x, x_cls, vector_x = self.base(x)
            segm_x = self.decoder(down1_x, down2_x, down3_x, down4_x, down5_x)
            
            return x_cls, segm_x

        else:
            raise NotImplementedError





