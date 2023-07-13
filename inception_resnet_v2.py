import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d_bn(nn.Module):
    def __init__(
        self,
        in_size: int,
        filters: int,
        kernel_size,
        stride_len: int = 1,
        padding="same",
        activation: bool = True,
        use_bias: bool = False,
        name=None,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_size,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=stride_len,
            padding=padding,
            bias=use_bias,
        )
        self.bn1 = nn.BatchNorm2d(
            filters,
            eps=0.001,
            momentum=0.1,
            affine=True,
        )

        self.act = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.act:
            x = F.relu(x)

        return x


class InceptionResNetA(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()

        self.branch_0 = Conv2d_bn(320, 32, kernel_size=1)
        self.branch_1 = nn.Sequential(
            Conv2d_bn(in_size=320, filters=32, kernel_size=1),
            Conv2d_bn(in_size=32, filters=32, kernel_size=3),
        )
        self.branch_2 = nn.Sequential(
            Conv2d_bn(in_size=320, filters=32, kernel_size=1),
            Conv2d_bn(in_size=32, filters=48, kernel_size=3),
            Conv2d_bn(in_size=48, filters=64, kernel_size=3),
        )
        self.conv = Conv2d_bn(
            in_size=128, filters=320, kernel_size=1, activation=False, use_bias=True
        )
        self.scale = scale

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        mixed = torch.cat((x0, x1, x2), dim=1)
        r = self.conv(mixed)
        x = x + self.scale * r

        return x


class InceptionResNetB(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.branch_0 = Conv2d_bn(in_size=1088, filters=192, kernel_size=1)
        self.branch_1 = nn.Sequential(
            Conv2d_bn(in_size=1088, filters=128, kernel_size=1),
            Conv2d_bn(in_size=128, filters=160, kernel_size=(1, 7)),
            Conv2d_bn(in_size=160, filters=192, kernel_size=(7, 1)),
        )
        self.conv = Conv2d_bn(
            in_size=384, filters=1088, kernel_size=1, activation=False, use_bias=True
        )
        self.scale = scale

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)

        mixed = torch.cat((x0, x1), dim=1)
        r = self.conv(mixed)
        x = x + r * self.scale

        return x


class InceptionResNetC(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

        self.branch_0 = Conv2d_bn(in_size=2080, filters=192, kernel_size=1)
        self.branch_1 = nn.Sequential(
            Conv2d_bn(in_size=2080, filters=192, kernel_size=1),
            Conv2d_bn(in_size=192, filters=224, kernel_size=(1, 3)),
            Conv2d_bn(in_size=224, filters=256, kernel_size=(3, 1)),
        )
        self.conv = Conv2d_bn(
            in_size=448, filters=2080, kernel_size=1, activation=False
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)

        mixed = torch.cat((x0, x1), dim=1)
        r = self.conv(mixed)
        x = x + r * self.scale

        return x


class StemBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv2d_bn(
                in_size=3, filters=32, kernel_size=3, stride_len=2, padding="valid"
            ),
            Conv2d_bn(in_size=32, filters=32, kernel_size=3, padding="valid"),
            Conv2d_bn(in_size=32, filters=64, kernel_size=3),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Sequential(
            Conv2d_bn(in_size=64, filters=80, kernel_size=1, padding="valid"),
            Conv2d_bn(in_size=80, filters=192, kernel_size=3, padding="valid"),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        return x


class Mixed_5b(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch_0 = Conv2d_bn(in_size=192, filters=96, kernel_size=1)
        self.branch_1 = nn.Sequential(
            Conv2d_bn(in_size=192, filters=48, kernel_size=1),
            Conv2d_bn(in_size=48, filters=64, kernel_size=5),
        )
        self.branch_2 = nn.Sequential(
            Conv2d_bn(in_size=192, filters=64, kernel_size=1),
            Conv2d_bn(in_size=64, filters=96, kernel_size=3),
            Conv2d_bn(in_size=96, filters=96, kernel_size=3),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d_bn(in_size=192, filters=64, kernel_size=1),
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x = torch.cat((x0, x1, x2, x3), dim=1)

        return x


class Mixed_6a(nn.Module):
    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = Conv2d_bn(
            in_size=320, filters=384, kernel_size=3, stride_len=2, padding="valid"
        )
        self.branch1 = nn.Sequential(
            Conv2d_bn(in_size=320, filters=256, kernel_size=1),
            Conv2d_bn(in_size=256, filters=256, kernel_size=3, padding=1),
            Conv2d_bn(
                in_size=256, filters=384, kernel_size=3, stride_len=2, padding="valid"
            ),
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat((x0, x1, x2), dim=1)

        return x


class Mixed_7a(nn.Module):
    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            Conv2d_bn(1088, 256, kernel_size=1),
            Conv2d_bn(256, 384, kernel_size=3, stride_len=2, padding="valid"),
        )

        self.branch1 = nn.Sequential(
            Conv2d_bn(1088, 256, kernel_size=1),
            Conv2d_bn(256, 288, kernel_size=3, stride_len=2, padding="valid"),
        )

        self.branch2 = nn.Sequential(
            Conv2d_bn(1088, 256, kernel_size=1),
            Conv2d_bn(256, 288, kernel_size=3, padding=1),
            Conv2d_bn(288, 320, kernel_size=3, stride_len=2, padding="valid"),
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x = torch.cat((x0, x1, x2, x3), 1)
        return x


class InceptionResNetV2(nn.Module):
    def __init__(self, use_cuda: bool = False) -> None:
        super().__init__()
        self.stem = StemBlock()
        self.mix1 = Mixed_5b()

        incept1 = [InceptionResNetA(0.17)] * 10
        self.incept1 = nn.Sequential(*incept1)

        self.mix2 = Mixed_6a()

        incept2 = [InceptionResNetB(0.1)] * 20
        self.incept2 = nn.Sequential(*incept2)

        self.mix3 = Mixed_7a()

        incept3 = [InceptionResNetC(0.2)] * 10
        self.incept3 = nn.Sequential(*incept3)

        self.conv1 = Conv2d_bn(in_size=2080, filters=1536, kernel_size=1)
        self.pool1 = nn.AvgPool2d(kernel_size=6)
        self.fc = nn.Sequential(
            nn.Linear(1536, 1024), nn.ReLU(), nn.Linear(1024, 2), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.mix1(x)
        x = self.incept1(x)
        x = self.mix2(x)
        x = self.incept2(x)
        x = self.mix3(x)
        x = self.incept3(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = x.view(-1, 1536)
        x = self.fc(x)

        return x

    def predict_tumor(self, x):
        with torch.no_grad():
            x = self.forward(x)
            x = x[0, 0].item()
            # if x>0.5:
            #     return x
            # else:
            #     return 0.5
            return x

    def predict_no_grad(self, x):
        with torch.no_grad():
            x = self.forward(x)
            return x
