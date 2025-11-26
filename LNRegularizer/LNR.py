import torch
import torch.nn as nn


def main():
    pass


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, device: torch.device):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu_layer = nn.ReLU(inplace=True)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu_layer(out)
        out = self.conv2(out)
        out = self.relu_layer(out)
        return out


class LNR(nn.Module):
    def __init__(self, device: torch.device):
        super(LNR, self).__init__()

        self.pool_layer = nn.MaxPool2d(2, 2)

        # Down Convolution block 1
        self.d_conv1 = Conv_Block(6, 8, device)

        # Down Convolution block 2
        self.d_conv2 = Conv_Block(8, 16, device)

        # Down Convolution block 3
        self.d_conv3 = Conv_Block(16, 32, device)

        # Down Convolution block 4
        self.d_conv4 = Conv_Block(32, 64, device)

        # Bottlneck
        self.b_neck = Conv_Block(64, 128, device)

        # Up Convolution block 1
        self.up_conv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.u_conv1 = Conv_Block(128, 64, device)

        # Up Convolution block 2
        self.up_conv2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.u_conv2 = Conv_Block(64, 32, device)

        # Up Convolution block 3
        self.up_conv3 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.u_conv3 = Conv_Block(32, 16, device)

        # Up Convolution block 4
        self.up_conv4 = nn.ConvTranspose2d(16, 8, 2, 2)
        self.u_conv4 = Conv_Block(16, 8, device)

        # Final 1x1 convolution
        self.out_layer = nn.Conv2d(8, 2, 1)

        # Image switching layers
        self.softmax = nn.Softmax2d()
        self.device = device
        self.to(self.device)

    def forward(self, x):
        # Convert 2 3 channel images to a 6 channel paired image
        size = x.size()
        resized_in = torch.zeros(
            int(size[0] / 2), size[1] * 2, size[2], size[3], device=self.device
        )
        for i in range(0, size[0], 2):
            resized_in[int(i / 2)] = torch.cat((x[i], x[i + 1]), dim=0)
        resized_in.requires_grad_()

        skip_block_1 = self.d_conv1(resized_in)
        out = self.pool_layer(skip_block_1)

        skip_block_2 = self.d_conv2(out)
        out = self.pool_layer(skip_block_2)

        skip_block_3 = self.d_conv3(out)
        out = self.pool_layer(skip_block_3)

        skip_block_4 = self.d_conv4(out)
        out = self.pool_layer(skip_block_4)

        out = self.b_neck(out)

        out = self.up_conv1(out)
        out = self.u_conv1(torch.cat((skip_block_4, out), dim=1))

        out = self.up_conv2(out)
        out = self.u_conv2(torch.cat((skip_block_3, out), dim=1))

        out = self.up_conv3(out)
        out = self.u_conv3(torch.cat((skip_block_2, out), dim=1))

        out = self.up_conv4(out)
        out = self.u_conv4(torch.cat((skip_block_1, out), dim=1))

        out = self.out_layer(out)

        out = self.softmax(out)
        out = torch.round(out)

        first_filter = out[:, 0:1]
        second_filter = out[:, 1:]
        out = torch.cat(
            (
                resized_in[:, 0:3] * first_filter + resized_in[:, 0:3] * second_filter,
                resized_in[:, 3:] * first_filter + resized_in[:, 3:] * second_filter,
            )
        )  # Swap the original images

        # cat the filters to get the mask
        mask = torch.cat((first_filter, second_filter))

        return out, mask
