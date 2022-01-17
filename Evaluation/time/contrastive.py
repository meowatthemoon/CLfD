import os
import time

import numpy as np
import torch
from clfd import CLfD  # the model
from PickPlaceContextDatasets import PickPlaceContextContrastiveDataset

from nt_xent import NT_Xent


def main():
    n_iterations = 1000
    network = "resnet18"
    out_features = 32
    batch_size = 50
    normalize = True
    workers = 4

    projection_dim = 64
    temperature = 0.5
    world_size = 1

    # Create model
    model = CLfD(backbone = network, out_features = out_features, projection_dim = projection_dim)
    if network == "tcn":
        image_size = 299
    elif network == "resnet18":
        image_size = 224
    else:
        raise NotImplementedError
    model.to("cuda")

    # Create Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Criterion
    criterion = NT_Xent(batch_size, temperature, world_size)

    times = []
    while len(times) < 1000:
        train_dataset = PickPlaceContextContrastiveDataset(type = "train", image_size= image_size, normalize=normalize)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=workers
        )
        for (x_i, x_j), _ in train_loader:
            start_time = time.time()

            optimizer.zero_grad()
            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)
            
            # positive pair, with encoding
            h_i, h_j, z_i, z_j = model(x_i, x_j)

            loss = criterion(z_i, z_j)
            loss.backward()

            optimizer.step()

            step_time = time.time() - start_time
            times.append(step_time)

            if len(times) >= n_iterations:
                break

    times = np.asarray(times)
    mean = np.mean(times)
    print(f"Contrastive Method average Execution Time on {network} is {mean}")

if __name__ ==  "__main__":
    main()
