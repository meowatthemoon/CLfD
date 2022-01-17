import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_optimizer(args, model):
    scheduler = None
    if args.optimizer == "Adam": # Recommended
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)
    else:
        print(args.optimizer)
        raise NotImplementedError

    return optimizer, scheduler


def plot_loss(losses, filename, x=None, window=5):
    N = len(losses)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(losses[max(0, t - window):(t + 1)])
    if x is None:
        x = [i for i in range(N)]
    plt.clf()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(x, running_avg)
    plt.savefig(filename)


def save_model(args, model, model_name):
    out = os.path.join(args.model_path, model_name)
    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg
