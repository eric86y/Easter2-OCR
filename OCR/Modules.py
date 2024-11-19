import torch
from torch import nn


"""
An experimental PyTorch implementation of the OCR Architecture
    -   original OCR-Architecture: https://github.com/kartikgill/Easter2
    -   Note: This is an early experimental version that needs some optimization
    -   On the padding behaviour of the Conv-layers in Pytorch,
        see here: https://github.com/pytorch/pytorch/issues/67551. 
        I opted for manually padding the output.
"""


class GlobalContext(nn.Module):
    def __init__(self, in_channels, out_channels, mean_pool: bool = True):
        super(GlobalContext, self).__init__()
        self.mean_pool = mean_pool
        self.pool = nn.AvgPool1d(kernel_size=out_channels)
        self.linear1 = nn.Linear(in_channels, out_channels // 8)
        self.linear2 = nn.Linear(out_channels // 8, out_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        pool = self.pool(data)

        if self.mean_pool:
            pool = torch.mean(pool, -1)
        else:
            pool = pool[:, :, 0]

        pool = self.linear1(pool)
        pool = self.relu(pool)
        pool = self.linear2(pool)
        pool = self.sigmoid(pool)
        pool = torch.unsqueeze(pool, -1)
        pool = torch.multiply(pool, data)

        return pool


class EasterUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride,
        dropout,
        bn_eps=1e-5,
        bn_decay=0.997,
        mean_pool: bool = True,
    ):
        super(EasterUnit, self).__init__()
        self.bn_eps = bn_eps
        self.bn_decay = bn_decay
        self.dropout = dropout
        self.mean_pool = mean_pool

        self.conv1d_1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=1, groups=1
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=1, groups=1
        )
        self.conv1d_3 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel, stride=stride, groups=1
        )
        self.conv1d_4 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel, stride=stride, groups=1
        )
        self.conv1d_5 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel, stride=stride, groups=1
        )

        self.bn_1 = nn.BatchNorm1d(
            num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay
        )
        self.bn_2 = nn.BatchNorm1d(
            num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay
        )
        self.bn_3 = nn.BatchNorm1d(
            num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay
        )
        self.bn_4 = nn.BatchNorm1d(
            num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay
        )
        self.bn_5 = nn.BatchNorm1d(
            num_features=out_channels, eps=self.bn_eps, momentum=self.bn_decay
        )

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu_3 = nn.ReLU()

        self.drop_1 = nn.Dropout(p=self.dropout)
        self.drop_2 = nn.Dropout(p=self.dropout)
        self.drop_3 = nn.Dropout(p=self.dropout)
        self.global_context = GlobalContext(
            in_channels=out_channels,
            out_channels=out_channels,
            mean_pool=self.mean_pool,
        )

    def forward(self, inputs):
        old, data = inputs
        old = self.conv1d_1(old)
        old = self.bn_1(old)

        this = self.conv1d_2(data)
        this = self.bn_2(this)
        old = torch.add(old, this)

        # First Block
        data = self.conv1d_3(data)
        pad_val = old.shape[-1] - data.shape[-1]
        data = nn.ZeroPad1d(padding=(0, pad_val))(data)
        data = self.bn_3(data)
        data = self.relu_1(data)
        data = self.drop_1(data)

        # Second Block
        data = self.conv1d_4(data)
        pad_val = old.shape[-1] - data.shape[-1]
        data = nn.ZeroPad1d(padding=(0, pad_val))(data)
        data = self.bn_4(data)
        data = self.relu_2(data)
        data = self.drop_2(data)

        # Third Block
        data = self.conv1d_5(data)
        pad_val = old.shape[-1] - data.shape[-1]
        data = nn.ZeroPad1d(padding=(0, pad_val))(data)
        data = self.bn_5(data)
        data = self.global_context(data)
        data = torch.add(old, data)
        data = self.relu_3(data)
        data = self.drop_3(data)

        return data, old


class Easter2(nn.Module):
    def __init__(
        self,
        input_width: int = 2000,
        input_height: int = 80,
        bn_eps=1e-5,
        bn_decay=0.997,
        vocab_size: int = 77,
        mean_pooling=True,
    ):
        super(Easter2, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.vocab_size = vocab_size
        self.bn_eps = bn_eps
        self.bn_decay = bn_decay
        self.zero_pad = nn.ZeroPad1d(padding=(0, 1))
        self.dropout = 0.2
        self.mean_pooling = mean_pooling

        self.conv1d_1 = nn.Conv1d(
            self.input_height, 128, kernel_size=3, stride=2, groups=1
        )  # NxCxL (N=BatchSize, C = number of channels, L = length of the signal)
        self.conv1d_2 = nn.Conv1d(128, 128, kernel_size=3, stride=2, groups=1)
        self.conv1d_3 = nn.Conv1d(256, 512, kernel_size=11, stride=1, dilation=2)
        self.conv1d_4 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding="same")
        self.conv1d_5 = nn.Conv1d(
            512, self.vocab_size, kernel_size=1, stride=1, padding="same"
        )

        self.bn_1 = nn.BatchNorm1d(
            num_features=128, eps=self.bn_eps, momentum=self.bn_decay
        )
        self.bn_2 = nn.BatchNorm1d(
            num_features=128, eps=self.bn_eps, momentum=self.bn_decay
        )
        self.bn_3 = nn.BatchNorm1d(
            num_features=512, eps=self.bn_eps, momentum=self.bn_decay
        )
        self.bn_4 = nn.BatchNorm1d(
            num_features=512, eps=self.bn_eps, momentum=self.bn_decay
        )

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu_3 = nn.ReLU()
        self.relu_4 = nn.ReLU()

        self.easter1 = EasterUnit(128, 128, 5, 1, 0.2, mean_pool=True)
        self.easter2 = EasterUnit(128, 256, 5, 1, 0.2, mean_pool=True)
        self.easter3 = EasterUnit(256, 256, 7, 1, 0.2, mean_pool=True)
        self.easter4 = EasterUnit(256, 256, 9, 1, 0.3, mean_pool=True)

        self.drop_1 = nn.Dropout(p=self.dropout)
        self.drop_2 = nn.Dropout(p=self.dropout)
        self.drop_3 = nn.Dropout(p=0.4)
        self.drop_4 = nn.Dropout(p=0.4)

        print(
            f"Created Easter Network with Inputs: {self.input_width}, {self.input_height}"
        )

    def forward(self, inputs):
        """
        Note: The model inputs should correspond to BxHxW (B = batch size, H = image_height = self.input_height,
        W = image width = self.image_width) in oder to to match the specification of NxCxL of nn.conv1d
        """
        x = self.conv1d_1(inputs)
        x = self.zero_pad(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.drop_1(x)
        x = self.conv1d_2(x)
        x = self.zero_pad(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.drop_2(x)

        old = x
        data, old = self.easter1((old, x))
        data, old = self.easter2((old, data))
        data, old = self.easter3((old, data))
        data, old = self.easter4((old, data))

        x = self.conv1d_3(data)

        x = nn.ZeroPad1d(padding=(10, 10))(x)
        x = self.bn_3(x)
        x = self.relu_3(x)
        x = self.drop_3(x)

        x = self.conv1d_4(x)
        x = self.bn_4(x)
        x = self.relu_4(x)
        x = self.drop_4(x)
        x = self.conv1d_5(x)

        return x
