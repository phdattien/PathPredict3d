import torch
import torch.nn as nn

# download the modelnet10 dataset
import os
import pickle
# import open3d as o3d
import trimesh
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pathlib import Path
# import h5py


def normalize_points(pts: np.ndarray):
    centroid = np.mean(pts, axis=0)
    pc = pts - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


# def visualize_pcd(xyz: np.ndarray):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)
#     o3d.visualization.draw_geometries([pcd])


class ModelNet10Loader(Dataset):
    def __init__(self, root: str = 'data', split: str = 'train', n_points: int = 1024):
        # Download dataset for point cloud classification
        DATA_DIR = Path(__file__).resolve().parent / 'data'
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
        model_folder = DATA_DIR / 'ModelNet10'
        if not model_folder.exists():
            www = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
            zipfile = os.path.basename(www)
            os.system('wget %s; unzip %s' % (www, zipfile))
            os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
            os.system('rm %s' % (zipfile))

        data_fs = list(model_folder.glob('[!README]*'))
        assert split in {'train', 'val'}
        save_file = DATA_DIR / f'modelnet_{split}_{n_points}.dat'

        self.point_labels = {k: v.stem for k, v in enumerate(data_fs)}

        if not save_file.exists():
            self.points_ls = []
            self.labels_ls = []
            # create the dataset
            for i, f in tqdm(enumerate(data_fs), total=len(data_fs), desc="processing folders"):
                f_split = f / split
                for f_pts in f_split.glob('*.off'):
                    points = np.asarray(trimesh.load(str(f_pts)).sample(n_points))
                    self.points_ls.append(points)
                    self.labels_ls.append(i)
            with save_file.open('wb') as f:
                pickle.dump([self.points_ls, self.labels_ls], f)
        else:
            with save_file.open('rb') as f:
                self.points_ls, self.labels_ls = pickle.load(f)

    def __len__(self):
        return len(self.labels_ls)

    def __getitem__(self, idx: int):
        points, label = self.points_ls[idx], self.labels_ls[idx]
        points = normalize_points(points)
        points = torch.tensor(points, dtype=torch.float32)
        return points, label

    def to_class(self, index: int):
        return self.point_labels[index]


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

    def __init__(self, dim: int, n_points: int):
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
        x = x.transpose(2, 1)  # B, C, N
        x = self.features(x)
        x = torch.max(x, 2, keepdim=True)[0]  # this inseatd of nn.Maxpool (2x faster)

        x = x.view(-1, 1024)
        x = self.relu1(self.b1(self.fc1(x)))
        x = self.relu2(self.b2(self.fc2(x)))

        x = self.lm_head(x)

        iden = torch.eye(self.dim, requires_grad=True).repeat(B, 1, 1)
        if x.is_cuda:  # FIX
            iden = iden.cuda()
        x = x.view(-1, self.dim, self.dim) + iden
        return x


class Pointnet(nn.Module):

    def __init__(self, n_points: int, k: int):
        super().__init__()
        self.tnet1 = Tnet(3, n_points)

        self.shared_mlp1 = nn.Sequential(
                SharedLinear(3, 64),
                SharedLinear(64, 64),
                )

        self.tnet2 = Tnet(64, n_points)

        self.shared_mlp2 = nn.Sequential(
                SharedLinear(64, 64),
                SharedLinear(64, 128),
                SharedLinear(128, 1024),
                )

        self.maxpool = nn.MaxPool1d(kernel_size=1024)
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256, bias=False)
        self.dropout = nn.Dropout(p=0.3)
        self.bn2 = nn.BatchNorm1d(256)
        self.lm_head = nn.Linear(256, k)

    def forward(self, x):
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


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


B = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # load the dataset only bathub
    torch.manual_seed(13337)

    train_dataset = ModelNet10Loader(split='train')
    trainDataLoader = DataLoader(train_dataset, batch_size=B, shuffle=True)

    model = Pointnet(1024, 10)
    model.apply(inplace_relu)
    # print(model)
    # return
    # model.compile()
    # model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for points, labels in trainDataLoader:
        for i in range(50):
            optimizer.zero_grad()
            # points = points.to(device)
            # labels = labels.to(device)
            logits, tranform = model(points)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            print(f"iter: {i} loss: {loss}")
        break

    # --------------- visualisation ---------------
    # train_features, train_labels = next(iter(trainDataLoader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # points = train_features[-1]
    # label = train_labels[0]
    # print(f"Label: {label}")
    # visualize_pcd(points)


if __name__ == "__main__":
    main()
