import torch.utils.data as data
import torch
import numpy as np
import pickle
from pathlib import Path
import trimesh
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / 'data'


def normalize_points(pts: np.ndarray):
    centroid = np.mean(pts, axis=0)
    pc = pts - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ModelNet40(data.Dataset):
    def __init__(self, n_points: int = 1024, transforms = None, normal: bool = True, split: str = "train"):
        super().__init__()

        self.n_points = n_points
        self.transforms = transforms

        modelnet40_folder = DATA_DIR / "ModelNet40"
        assert modelnet40_folder.exists(), "ModelNet40 dataset not dowloaded"

        object_folder = list(modelnet40_folder.iterdir())
        self.point_labels = {k: v.stem for k, v in enumerate(object_folder)}

        if normal:
            self._resampled_folder = DATA_DIR / f'modelnet40_normal_resampled_{n_points}'
        else:
            self._resampled_folder = DATA_DIR / f'modelnet40_resampled_{n_points}'

        # TODO uncomment
        if not self._resampled_folder.exists():
            self._resampled_folder.mkdir(parents=True, exist_ok=True)
            # create dataset
            for split in ['train', 'test']:
                self.points_ls = []
                self.labels_ls = []
                for i, f in tqdm(enumerate(object_folder), total=len(object_folder), desc=f'Creating {split}'):
                    f_split = f / split
                    for f_pts in f_split.glob('*off'):
                        mesh = trimesh.load(str(f_pts))
                        points, faces = mesh.sample(n_points, return_index=True)
                        points = np.asarray(points, dtype=np.float32)
                        normals = np.asarray(mesh.face_normals[faces], dtype=np.float32)
                        res = np.concatenate((points, normals), axis=1)
                        self.points_ls.append(res)
                        self.labels_ls.append(i)
                    with (self._resampled_folder / f"{split}.dat").open('wb') as f:
                        pickle.dump([self.points_ls, self.labels_ls], f)
        else:
            with (self._resampled_folder / f"{split}.dat").open('rb') as f:
                self.points_ls, self.labels_ls = pickle.load(f)

    def __len__(self):
        return len(self.points_ls)

    def __getitem__(self, idx: int):
        points, label = self.points_ls[idx], self.labels_ls[idx]

        pt_idxs = np.arange(0, self.n_points)
        np.random.shuffle(pt_idxs)

        point_set = points[pt_idxs, :]
        point_set[:, 0:3] = normalize_points(point_set[:, 0:3])

        if self.transforms is not None:
            point_set = self.transforms(point_set)

        return point_set, label

    def to_class(self, index: int):
        return self.point_labels[index]


if __name__ == "__main__":
    import torch
    from torchvision import transforms
    import utils

    transforms = transforms.Compose(
        [
            utils.PointcloudToTensor(),
            utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            utils.PointcloudScale(),
            utils.PointcloudTranslate(),
            utils.PointcloudJitter(),
        ]
    )
    dset = ModelNet40(1024, split='test')
    print(dset[0][0])
    print(dset[0][1])

    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
