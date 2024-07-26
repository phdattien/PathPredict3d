import torch
import sys
from dataclasses import dataclass
import torch.nn as nn

# download the modelnet10 dataset
import os
import time
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


def get_lr(it: int):
    """it: epoch iteration"""
    global LR
    if it == 0:
        return LR
    scale = 0.5 ** (it // 20)
    return LR * scale


def get_loss(logits: torch.tensor,
             labels: torch.tensor,
             tranforms: torch.tensor,
             reg_rate: int):
    # add orthogonal regulization

    loss = F.cross_entropy(logits, labels)
    IDEN = torch.eye(64)[None, :, :]
    IDEN = IDEN.to(device)
    reg = torch.linalg.norm(IDEN - torch.bmm(tranforms, tranforms.transpose(2, 1)), dim=(1, 2))
    reg = torch.mean(reg)
    return loss + reg * reg_rate


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, dataLoader):
    """Function estimate loss on data"""
    model.eval()
    size = len(dataLoader)
    losses = torch.zeros(size)
    accs = torch.zeros(size)

    for i, (points, labels) in enumerate(dataLoader):
        points = points.to(device)
        labels = labels.to(device)
        logits, tranforms = model(points)

        losses[i] = F.cross_entropy(logits, labels).item()
        accs[i] = calculate_acc(logits, labels).item()

    model.train()
    return losses.mean(), accs.mean()


def calculate_acc(logits: torch.tensor, labels: torch.tensor):
    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(labels.view_as(pred)).sum()
    return correct / len(labels)


BATCH_SIZE = 32
DECAY_RATE = 1e-4
REG_LOSS_RATE = 0.001
LR = 0.001
NUM_POINTS = 1024  # how many point in
MODEL = 'pointnet'
DATA = 'ModelNet10'  # which dataset using
NUM_CATEGORY = 10
DROPOUT = 0.3

MODEL_NAME = 'pointNet_1024_model_net10'
RESUME = True  # keep training same model

# wandb logging
wandb_log = True
wandb_project = 'pointcls'
wandb_run_name = 'pointNet'
wandb_id = 'pointnet3'
wandb_resume = 'allow'

# training config
device = 'cuda'

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging


OUT_DIR = Path(__file__).resolve().parent / 'out'
if not OUT_DIR.exists():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

ITER_NUM = 0
BEST_VAL_ACC = 0.0
MAX_ITER = 200

# =====================MODEL CREATION / RESUMING TRAINING ====================
# model creation (nanoGPT inspo)
model_args = dict(n_category=NUM_CATEGORY, dropout=DROPOUT)
conf = PointNetConfig(**model_args)
model = Pointnet(conf)

if RESUME:
    try:
        model_path = OUT_DIR / f"{MODEL_NAME}.pt"
        print(f"Resume training from model {model_path}")
        ckpt = torch.load(str(model_path), map_location=device)
        state_dict = ckpt['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        ITER_NUM = ckpt['iter_num']
        BEST_VAL_ACC = ckpt['best_val_acc']
        print(BEST_VAL_ACC)
        print("Succes of resuming model")

    except Exception as e:
        print(f"Model does not exist or wrong model arguments {e}")
        sys.exit(1)
model.to(device)

sys.exit(0)


# ================== LOGGING wandb =================================
if wandb_log:
    import wandb
    # TODO: add config to be able to know what we used
    wandb.init(project=wandb_project, name=wandb_run_name, id=wandb_id,
               resume=wandb_resume, config=config)


# ==================== DATA LOADING ==============================
torch.manual_seed(13337)
train_dataset = ModelNet10Loader(split='train')
val_dataset = ModelNet10Loader(split='test')
trainDataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

valDataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ===================== OPTIMIZOR =====================================

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY_RATE)
if RESUME:
    optimizer.load_state_dict(ckpt['optimizer'])


# ============================TRAINING LOOP====================================
if RESUME:
    print(f"Start training from epoch: {ITER_NUM}")
else:
    print("Start training from scratch")

model = torch.compile(model)
model.train()
while True:
    lr = get_lr(ITER_NUM)
    # set the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    running_loss = 0.0
    running_acc = 0.0
    t0 = time.time()

    for j, (points, labels) in enumerate(trainDataLoader):
        optimizer.zero_grad()
        points = points.to(device)
        labels = labels.to(device)

        logits, tranforms = model(points)
        loss = get_loss(logits, labels, tranforms, REG_LOSS_RATE)

        running_loss += loss.item()
        running_acc += calculate_acc(logits, labels).item()

        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        # print(f"epoch {i}: loss {batch_loss:4f}, acc {batch_acc}, time {dt*1000:.2f}ms, points/sec {B*1024/dt}")

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    # evaluate | logging | saving check points
    train_loss = running_loss / len(trainDataLoader)
    train_acc = running_acc / len(trainDataLoader)
    val_loss, val_acc = estimate_loss(model, valDataLoader)

    print(f"epoch {ITER_NUM}: train loss {train_loss:4f}, train acc {train_acc:4f}, val loss {val_loss:4f}, val acc {val_acc:4f} time {dt*1000:.2f}ms")

    ITER_NUM += 1
    if wandb_log:
        wandb.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "lr": lr
            })

    if val_acc > BEST_VAL_ACC:
        BEST_VAL_ACC = val_acc
        state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'model_args': model_args,
                'iter_num': ITER_NUM,
                'best_val_acc': BEST_VAL_ACC
                }
        print(f"Saving, checkpoint with acc {BEST_VAL_ACC:4f} to {str(OUT_DIR)}")
        torch.save(state, str(OUT_DIR / f"{MODEL_NAME}.pt"))

    if ITER_NUM > MAX_ITER:
        break

    # end of epoch, do evaluation

# --------------- visualisation ---------------
# train_features, train_labels = next(iter(trainDataLoader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# points = train_features[-1]
# label = train_labels[0]
# print(f"Label: {label}")
# visualize_pcd(points)
