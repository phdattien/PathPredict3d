import torch
import torch.nn as nn

# download the modelnet10 dataset
import os
import pickle
import open3d as o3d
import trimesh
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pathlib import Path
# import h5py


class Tnet(nn.Module):
    """Model so the transformation is nice"""

    def __init__(self, dim: int, n_points: int):
        super().__init()
        self.features = nn.Sequential(
                nn.Conv1d(dim, 64, kernel_size=1),   # B, 64, N
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),

                nn.Conv1d(64, 128, kernel_size=1),   # B, 128, N
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                nn.Conv1d(128, 1024, kernel_size=1),  # B, 1024, N
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                )

        self.maxpool = nn.MaxPool1d(kernel_size=n_points)

        self.transformation = nn.Sequential(
                nn.Linear(1024, 512),  # B, 512, N
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),

                nn.Linear(512, 256),  # B, 256, N
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                )

        self.fc = self.nn.Linear(256, dim*2)  # transformation matrix

    def forward(self, x: torch.Tensor):
        device = x.device
        B, D, N = x.size()
        ...


class Pointnet(nn.Module):

    def __init__(self):
        super().__init__()


def normalize_points(pts: np.ndarray):
    centroid = np.mean(pts, axis=0)
    pc = pts - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


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

        # folders = os.listdir(model_folder)
        # if 'train' not in folders:
        #     # process the dataset into two files train/test
        #     print(111)

    def __len__(self):
        return len(self.labels_ls)

    def __getitem__(self, idx: int):
        points, label = self.points_ls[idx], self.labels_ls[idx]
        points = normalize_points(points)
        return points, label

    def to_class(self, index: int):
        return self.point_labels[index]


def visualize_pcd(xyz: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])


B = 4


def main():
    # load the dataset only bathub
    train_dataset = ModelNet10Loader(split='train')
    trainDataLoader = DataLoader(train_dataset, batch_size=B, shuffle=True)

    for points, labels in trainDataLoader:
        print(labels)
        print(points.shape)
        print(points.size())
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
