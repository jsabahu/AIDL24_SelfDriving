# Source: https://github.com/davidtvs/PyTorch-ENet/blob/master/models/enet.py
import torch
import torch.nn as nn
from typing import Tuple

# TODO add hyperparameters handler


class InitialBlock(nn.Module):
    """The initial block is composed of two branches:
    1. a main branch which performs a regular convolution with stride 2;
    2. an extension branch which performs max-pooling.

    Doing both operations in parallel and concatenating their results
    allows for efficient downsampling and expansion. The main branch
    outputs 13 feature maps while the extension branch outputs 3, for a
    total of 16 feature maps after concatenation."""

    def __init__(self, in_channels: int, out_channels: int, relu: bool = True):
        super().__init__()
        activation = nn.ReLU if relu else nn.PReLU

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat((main, ext), 1)
        out = self.batch_norm(out)
        return self.out_activation(out)


class RegularBottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.

    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
    ``channels``, also called an expansion;
    4. dropout as a regularizer."""

    def __init__(
        self,
        channels: int,
        internal_ratio: int = 4,
        kernel_size: int = 3,
        padding: int = 0,
        dilation: int = 1,
        asymmetric: bool = False,
        dropout_prob: float = 0,
        relu: bool = True,
    ):
        super().__init__()
        # Check internal_scale parameter is inside range [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError(
                "Value out of range. Expected value in the "
                "interval [1, {0}], got internal_scale={1}.".format(
                    channels, internal_ratio
                )
            )

        activation = nn.ReLU if relu else nn.PReLU
        internal_channels = channels // internal_ratio

        # Main branch - shortcut connection

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution, and,
        # finally, a regularizer (spatial dropout). Number of channels is constant.

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            activation(),
        )

        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm2d(internal_channels),
                activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    padding=(0, padding),
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm2d(internal_channels),
                activation(),
            )
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm2d(internal_channels),
                activation(),
            )

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            activation(),
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.

    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
    unpooling later.

    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
    by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        internal_ratio: int = 4,
        return_indices: bool = False,
        dropout_prob: float = 0,
        relu: bool = True,
    ):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(
                "Value out of range. Expected value in the "
                "interval [1, {0}], got internal_scale={1}. ".format(
                    in_channels, internal_ratio
                )
            )
        activation = nn.ReLU if relu else nn.PReLU
        internal_channels = in_channels // internal_ratio

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(2, stride=2, return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=2, stride=2, bias=False
            ),
            nn.BatchNorm2d(internal_channels),
            activation(),
        )

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(internal_channels),
            activation(),
        )

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(),
        )

        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w).to(main.device)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out), max_indices


class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.

    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.

    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        internal_ratio: int = 4,
        dropout_prob: float = 0,
        relu: bool = True,
    ):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(
                "Value out of range. Expected value in the "
                "interval [1, {0}], got internal_scale={1}. ".format(
                    in_channels, internal_ratio
                )
            )

        activation = nn.ReLU if relu else nn.PReLU
        internal_channels = in_channels // internal_ratio

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Remember that the stride is the same as the kernel_size, just like the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            activation(),
        )

        # Transposed convolution
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels, internal_channels, kernel_size=2, stride=2, bias=False
        )
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()

        # 1x1 expansion convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(
        self, x: torch.Tensor, max_indices: torch.Tensor, output_size: torch.Size
    ) -> torch.Tensor:
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices, output_size=output_size)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext
        return self.out_activation(out)


class ENet(nn.Module):
    """Generate the ENet model."""

    def __init__(
        self,
        num_classes: int,
        encoder_relu: bool = False,
        decoder_relu: bool = True,
        binary_output: bool = True,
    ):
        super().__init__()
        self.binary_output = binary_output

        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            16, 64, dropout_prob=0.01, return_indices=True, relu=encoder_relu
        )
        self.regular1_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu
        )

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            64, 128, dropout_prob=0.1, return_indices=True, relu=encoder_relu
        )
        self.regular2_1 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu
        )

        # Stage 3 - Decoder
        self.upsample3_0 = UpsamplingBottleneck(
            128, 64, dropout_prob=0.1, relu=decoder_relu
        )
        self.regular3_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu
        )

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(
            64, 16, dropout_prob=0.1, relu=decoder_relu
        )
        self.regular4_1 = RegularBottleneck(
            16, padding=1, dropout_prob=0.1, relu=decoder_relu
        )

        self.transposed_conv = nn.ConvTranspose2d(
            16, num_classes, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial block
        input_size = x.size()
        x = self.initial_block(x)

        # Stage 1 - Encoder
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)

        # Stage 2 - Encoder
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)

        # Stage 3 - Decoder
        x = self.upsample3_0(x, max_indices2_0, output_size=stage2_input_size)
        x = self.regular3_1(x)

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular4_1(x)
        x = self.transposed_conv(x, output_size=input_size)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)

        if self.binary_output:
            # Apply sigmoid to get probabilities
            x = torch.sigmoid(x)
            # Apply threshold to get binary output
            x = (x >= 0.5).float()

        return x


if __name__ == "__main__":
    # Example usage
    # from torchviz import make_dot
    model = ENet(num_classes=1)
    input_image = torch.randn(1, 3, 720, 1280)
    output = model(input_image)
    # make_dot(output, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    print("Output Shape:", output.shape)
