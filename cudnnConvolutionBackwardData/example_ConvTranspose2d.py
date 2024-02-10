import torch
from torch import nn


class MyModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upsample = nn.ConvTranspose2d(
            2, 2, 3, stride=1, padding=0, bias=False,
            groups=2
        )
        self.upsample.weight.data = torch.tensor(
            [[[[1.0, 1.0, 1.0],
               [1.0, 1.0, 1.0],
               [1.0, 1.0, 1.0]]],
             [[[2.0, 2.0, 2.0],
               [2.0, 2.0, 2.0],
               [2.0, 2.0, 2.0]]]
             ]
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


if __name__ == '__main__':
    model = MyModel().to("cuda")
    x_ = torch.ones(8).view(1, 2, 2, 2).to("cuda")
    print(x_)
    y = model(x_)
    print(y)
