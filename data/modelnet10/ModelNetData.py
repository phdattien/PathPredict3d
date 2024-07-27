import torch

# download the modelnet10 dataset
import os
import pickle
# import open3d as o3d
import trimesh
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path


def normalize_points(pts: np.ndarray):
    centroid = np.mean(pts, axis=0)
    pc = pts - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ModelNet10Loader(Dataset):
    def __init__(self, root: str = None, split: str = 'train', n_points: int = 1024):
        # Download dataset for point cloud classification
        if root is None:
            DATA_DIR = Path(__file__).resolve().parent
        else:
            DATA_DIR = root

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
        assert split in {'train', 'test'}
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
