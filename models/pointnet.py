import torch
from dataclasses import dataclass
import torch.nn as nn


class SharedLinear(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.share_fc = nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.share_fc(x)))


class Tnet(nn.Module):
    """Model so the transformation is nice"""

    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim
        self.features = nn.Sequential(
                SharedLinear(dim, 64),
                SharedLinear(64, 128),  # B, 128, N
                SharedLinear(128, 1024)
                )

        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.b1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512, 256, bias=False)
        self.b2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU(inplace=True)

        self.lm_head = nn.Linear(256, dim*dim)  # transformation matrix

    def forward(self, x: torch.Tensor):
        B, C, N = x.size()
        x = x.transpose(2, 1)  # B, C, N
        x = self.features(x)
        x = torch.max(x, 2, keepdim=True)[0]  # this inseatd of nn.Maxpool (2x faster)

        x = x.view(-1, 1024)
        x = self.relu1(self.b1(self.fc1(x)))
        x = self.relu2(self.b2(self.fc2(x)))

        x = self.lm_head(x)

        iden = torch.eye(self.dim, requires_grad=True).repeat(B, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x.view(-1, self.dim, self.dim) + iden
        return x


@dataclass
class PointNetConfig:
    n_category: int = 3
    dropout: float = 0.3


class Pointnet(nn.Module):

    def __init__(self, config: PointNetConfig):
        super().__init__()
        self.config = config
        self.tnet1 = Tnet(3)

        self.shared_mlp1 = nn.Sequential(
                SharedLinear(3, 64),
                SharedLinear(64, 64),
                )

        self.tnet2 = Tnet(64)

        self.shared_mlp2 = nn.Sequential(
                SharedLinear(64, 64),
                SharedLinear(64, 128),
                SharedLinear(128, 1024),
                )

        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512, 256, bias=False)
        self.dropout = nn.Dropout(p=config.dropout)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU(inplace=True)

        self.lm_head = nn.Linear(256, config.n_category)

    def forward(self, x):
        x = x[:, :, :3]
        print(x.shape)
        tranform3x3 = self.tnet1(x)
        x = torch.bmm(x, tranform3x3).transpose(2, 1)
        x = self.shared_mlp1(x)

        feat_tr = self.tnet2(x.transpose(2, 1))
        x = torch.bmm(x.transpose(2, 1), feat_tr).transpose(2, 1)
        x = self.shared_mlp2(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.dropout(self.fc2(x))))
        logits = self.lm_head(x)

        return logits, feat_tr
