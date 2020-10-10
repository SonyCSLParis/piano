import torch
from torch import nn


class GRUGate(nn.Module):
    def __init__(self, d_model):
        super(GRUGate, self).__init__()
        self.Wr = nn.Linear(d_model, d_model, bias=False)
        self.Ur = nn.Linear(d_model, d_model, bias=False)
        self.Wz = nn.Linear(d_model, d_model, bias=False)
        self.Uz = nn.Linear(d_model, d_model, bias=False)
        self.Wg = nn.Linear(d_model, d_model, bias=False)
        self.Ug = nn.Linear(d_model, d_model, bias=False)
        self.bg = nn.Parameter(torch.randn((d_model,), requires_grad=True) + 1,
                               requires_grad=True)

    def forward(self, x, y):
        """

        :param x:
        :param y: output from the residual branch
        :return:
        """
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = torch.tanh(self.Wg(y) + self.Ug(r * x))
        return (1 - z) * x + z * h
