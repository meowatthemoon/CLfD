# Based on contrastive learning with multi-gpu of : https://github.com/Spijkervet/SimCLR

import argparse
import os
import ssl

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from alignment import validate
from clfd import CLfD  # the model
# Dataset
from PickPlaceContextDatasets import PickPlaceContextContrastiveDataset
from PouringDatasets import PouringContrastiveDataset
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from nt_xent import NT_Xent
from utils import load_optimizer, plot_loss, save_model, yaml_config_hook

ssl._create_default_https_context = ssl._create_unverified_context

def train(args, train_loader, model, criterion, optimizer, writer):
    model.train()
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)


        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch",
                              loss.item(), args.global_step)
            args.global_step += 1
            
        loss_epoch += loss.item()
    return loss_epoch / len(train_loader)


def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Dataset
    if args.dataset == "POURING":
        train_dataset = PouringContrastiveDataset(type = "train", image_size = args.image_size, normalize=args.normalize)
    elif args.dataset == "PICKPLACECONTEXT":
        train_dataset = PickPlaceContextContrastiveDataset(type = "train", image_size=args.image_size, n_view_points = args.n_view_points, normalize=args.normalize)
    else:
        raise NotImplementedError

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    # Model
    model = CLfD(backbone = args.network, out_features = args.out_features, projection_dim =args.projection_dim)
    if args.start_epoch != 0:
        chkpt_path = os.path.join(args.model_path, f"model_{args.start_epoch}.tar")
        model.load_state_dict(torch.load(chkpt_path, map_location=args.device.type))
    model = model.to(args.device)


    # Optimizer
    optimizer, scheduler = load_optimizer(args, model)

    # Criterion
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    # If Multi-gpu
    if args.nodes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    # Train
    args.global_step = 0
    args.current_epoch = 0
    tr_losses = []
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]["lr"]
        train_loss_epoch = train(args, train_loader, model,criterion, optimizer, writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % args.save_every == 0:
            alignment_error_perc = validate(method = "contrastive", dataset_name = args.dataset, image_size = args.image_size, device = args.device, normalize = args.normalize, model = model, normalizer=train_dataset.normalizer)
            print(f"Epoch [{epoch}/{args.epochs}]\t ALIGNMENT VALIDATION ERROR: {alignment_error_perc}%")
            with open(os.path.join(args.model_path,"alignment_error.txt"), "a") as f:
                f.write(f"[{epoch}] Encoder achieved NEW alignment error of {alignment_error_perc}%\n\n")
            save_model(args, model, f"model_{epoch}.tar")

        if args.nr == 0:
            writer.add_scalar("Loss/train", train_loss_epoch, epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(f"Epoch [{epoch}/{args.epochs}]\t Train Loss: {train_loss_epoch}\t lr: {round(lr, 5)}")
            args.current_epoch += 1
            tr_losses.append(train_loss_epoch)

    if args.nr == 0:
        save_model(args, model, f"model_last.tar")

        # Plot Losses
        plot_loss(losses=tr_losses, filename=os.path.join(args.model_path,f"train.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLfD")
    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    name = f"CLFD__{args.network}__{args.dataset}__{args.epochs}__{args.batch_size * args.nodes}__{args.optimizer}__{args.n_view_points}v__{args.out_features}f__{args.normalize}"
    path = os.path.join("../results/", name)
    args.model_path = path
    
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory at {path}")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    args.image_size = 299 if args.network == "tcn" else 224

    if args.nodes > 1:
        print(f"Training with {args.nodes} nodes, waiting until all nodes join before starting training")
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)

