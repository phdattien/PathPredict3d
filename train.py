# inspiration from nanoGPT and https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/train_classification.py
import torch
import sys

import provider
import time
# import open3d as o3d
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.pointnet import Pointnet, PointNetConfig
from data.modelnet10.ModelNetData import ModelNet10Loader

from pathlib import Path


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

MODEL_NAME = 'pointnet_augment'  # name the model, will be placed in out dir
RESUME = False  # keep training same model

# wandb logging
wandb_log = True  # wheter we want keep logging
wandb_project = 'pointcls'
wandb_run_name = 'pointNet_augment'
wandb_resume = 'allow'  # in case we fail a run find ID of the run and resume
# wandb_id = 'pointnet3'

# training config
device = 'cuda'

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
print(config)


# all models will be save in out file
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
        print("Succes of resuming model")

    except Exception as e:
        print(f"Model does not exist or wrong model arguments {e}")
        sys.exit(1)
model.to(device)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {param_count}")


# ================== LOGGING wandb =================================
if wandb_log:
    import wandb
    # TODO: add config to be able to know what we used
    wandb.init(project=wandb_project, name=wandb_run_name,
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
    print(f"Start training from epoch: {ITER_NUM} with best acc: {BEST_VAL_ACC}")
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

        points = points.data.numpy()
        points = provider.rotate_point_cloud(points)
        points = provider.jitter_point_cloud(points)
        points = torch.Tensor(points)

        points = points.to(device)
        labels = labels.to(device)

        logits, tranforms = model(points)
        loss = get_loss(logits, labels, tranforms, REG_LOSS_RATE)

        running_loss += loss.item()
        running_acc += calculate_acc(logits, labels).item()

        loss.backward()
        optimizer.step()

    # evaluate | logging | saving check points
    train_loss = running_loss / len(trainDataLoader)
    train_acc = running_acc / len(trainDataLoader)
    val_loss, val_acc = estimate_loss(model, valDataLoader)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

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
