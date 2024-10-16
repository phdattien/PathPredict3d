# inspiration from nanoGPT and https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/train_classification.py
import torch

import wandb
import time
from models.pointnet import Pointnet
# import open3d as o3d
from datetime import datetime
import torch.nn.functional as F
from torchvision import transforms

from models.pointnet2_cls_ssg import PointNet2ClassificationSSG

from data_utils.ModelNet40Loader import ModelNet40
from pathlib import Path
import argparse
from data_utils import utils


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, help='training on ModelNet/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--wb', action='store_true', default=True, help='use weight and biases')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
    parser.add_argument('--wandb_project', type=str, default="pointcls", help='Name of wb project')
    parser.add_argument('--debug', action='store_true', default=False, help='Debugging')
    return parser.parse_args()


def get_loss(logits: torch.tensor,
             labels: torch.tensor):
    loss = F.cross_entropy(logits, labels)
    return loss


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, dataLoader, device):
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


def get_transforms():
    train_transforms = transforms.Compose(
        [
            utils.PointcloudToTensor(),
            utils.PointcloudScale(),
            utils.PointcloudRotate(),
            utils.PointcloudRotatePerturbation(),
            utils.PointcloudTranslate(),
            utils.PointcloudJitter(),
            utils.PointcloudRandomInputDropout(),
        ]
    )
    return train_transforms


def main(args):
    torch.manual_seed(13337)
    out_dir = Path(__file__).resolve().parent / 'out'
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb logging
    wandb_run_name = 'pointnet2' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # wandb_id = 'pointnet3'
    wandb.init(project=args.wandb_project, name=wandb_run_name,
               config=args, mode=('online' if not args.debug else 'disabled'),
               save_code=True)

    '''DATA LOADING'''
    transforms = get_transforms()
    # transforms = None
    train_dataset = ModelNet40(split='train', transforms=transforms)
    test_dataset = ModelNet40(split='test', transforms=None)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    valDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    model = PointNet2ClassificationSSG(args)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {param_count}")

    try:
        model_path = out_dir / "poitnet2_ckpt.pt"
        print(f"Resume training from model {model_path}")
        ckpt = torch.load(str(model_path), map_location=device)
        state_dict = ckpt['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        start_epoch = ckpt['epoch']
        best_val_acc = ckpt['best_val_acc']
        print("Succes of resuming model")

    except:  # noqa
        print("Training from scratch")
        start_epoch = 0
        best_val_acc = 0.0

    model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_step = 0
    best_val_acc = 0.0

    # training loop
    print('Start training...')
    # model = torch.compile(model)

    data_iter = iter(trainDataLoader)
    points, labels = next(data_iter)

    running_loss = 0.0
    running_acc = 0.0
    for epoch in range(start_epoch, args.epoch):
        t0 = time.time()
        model.train()

        # for j, (points, labels) in enumerate(trainDataLoader):
        # debug
        optimizer.zero_grad()

        points = points.to(device)
        labels = labels.to(device)

        logits = model(points)
        loss = get_loss(logits, labels)

        running_loss += loss.item()
        acc = calculate_acc(logits, labels).item() # debug
        running_acc += calculate_acc(logits, labels).item()

        loss.backward()
        optimizer.step()

        global_step += 1
        # debug

        scheduler.step()

        # evaluate | logging | saving check points
        train_loss = running_loss / len(trainDataLoader)
        train_acc = running_acc / len(trainDataLoader)
        # val_loss, val_acc = estimate_loss(model, valDataLoader, device) # debug

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        print(f"epoch {epoch}: train loss {loss:.4f}, train acc {acc:.4f}, time {dt*1000:.2f}ms") # debug
        # print(f"epoch {epoch}: train loss {train_loss:4f}, train acc {train_acc:4f}, val loss {val_loss:4f}, val acc {val_acc:4f} time {dt*1000:.2f}ms")
        # wandb.log({
        #     "train/loss": train_loss,
        #     "train/acc": train_acc,
        #     "val/loss": val_loss,
        #     "val/acc": val_acc,
        #     "lr": scheduler.get_last_lr()
        #     })

        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     state = {
        #             'model': model.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'model_args': args,
        #             'epoch': epoch+1,
        #             'best_val_acc': best_val_acc
        #             }
        #     print(f"Saving, checkpoint with acc {best_val_acc:4f} to {str(out_dir)}")
        #     torch.save(state, str(out_dir / "poitnet2_ckpt.pt"))

    print('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
