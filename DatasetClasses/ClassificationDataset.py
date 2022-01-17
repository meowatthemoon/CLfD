import os
import socket

import torch
from skimage import io
from skimage.transform import resize
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms


class PickOrPlaceClassificationDataset(Dataset):
    def __init__(self, type: str, normalize_input: bool, view_point_type: str):
        if (
            socket.gethostname() == "socialab-167"
            or socket.gethostname() == "socialab-VirtualBox"
            or socket.gethostname() == "pop-os"
        ):
            seen_base_folder: str = (
                f"/home/{os.getlogin()}/Documents/PhD/Datasets/PickOrPlaceFront"
            )
            unseen_base_folder: str = (
                f"/home/{os.getlogin()}/Documents/PhD/Datasets/PickOrPlaceRight"
            )
        elif socket.gethostname() == "LAPTOP-199MMMNT":
            seen_base_folder: str = (
                "C:\\Users\\35192\\Documents\\PhDv3\\Datasets\\PickOrPlaceFront"
            )
            unseen_base_folder: str = (
                "C:\\Users\\35192\\Documents\\PhDv3\\Datasets\\PickOrPlaceRight"
            )
        else:
            raise NotImplementedError

        self.normalize = normalize_input

        self.image_size = 224

        if self.normalize:
            if view_point_type == "seen":
                self.normalizer = transforms.Normalize(
                    mean=[0.4584, 0.4460, 0.3116], std=[0.3630, 0.2435, 0.1649]
                )
            elif view_point_type == "unseen":
                self.normalizer = transforms.Normalize(
                    mean=[0.7415, 0.6189, 0.3654], std=[0.1319, 0.1060, 0.1424]
                )
            elif view_point_type == "both":
                self.normalizer = transforms.Normalize(
                    mean=[0.6005, 0.5328, 0.3386], std=[0.3071, 0.2064, 0.1562]
                )
            else:
                raise NotImplementedError

        self.seenframes_folder = os.path.join(seen_base_folder, type)
        self.unseenframes_folder = os.path.join(unseen_base_folder, type)
        self.image_paths = []
        self.classes = []

        if view_point_type == "seen" or view_point_type == "both":
            for pick_image in os.listdir(os.path.join(self.seenframes_folder, "pick")):
                self.image_paths.append(
                    os.path.join(self.seenframes_folder, "pick", pick_image)
                )
                self.classes.append(0)
            for place_image in os.listdir(
                os.path.join(self.seenframes_folder, "place")
            ):
                self.image_paths.append(
                    os.path.join(self.seenframes_folder, "place", place_image)
                )
                self.classes.append(1)
        if view_point_type == "unseen" or view_point_type == "both":
            for pick_image in os.listdir(
                os.path.join(self.unseenframes_folder, "pick")
            ):
                self.image_paths.append(
                    os.path.join(self.unseenframes_folder, "pick", pick_image)
                )
                self.classes.append(0)
            for place_image in os.listdir(
                os.path.join(self.unseenframes_folder, "place")
            ):
                self.image_paths.append(
                    os.path.join(self.unseenframes_folder, "place", place_image)
                )
                self.classes.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.image_paths[idx]
        image = io.imread(image_name)
        image_resized = resize(
            image, (self.image_size, self.image_size), anti_aliasing=True
        )
        image_tensor = torch.tensor(image_resized, dtype=torch.float).transpose(2, 0)

        if self.normalize:
            image_tensor = self.normalizer(image_tensor)

        y = torch.tensor(self.classes[idx])

        return image_tensor, y


# https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
def get_mean_and_std(train_dataloader, val_dataloader):
    channels_sum = 0
    channels_squared_sum = 0
    num_batches = 0
    for x, _ in train_dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(x, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(x ** 2, dim=[0, 2, 3])
        num_batches += 1

    for x, _ in val_dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(x, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(x ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == "__main__":
    view_point_type = "both"
    train_dataset = PickOrPlaceClassificationDataset(
        type="train", normalize_input=False, view_point_type=view_point_type
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=200, shuffle=False, drop_last=True, num_workers=8
    )
    val_dataset = PickOrPlaceClassificationDataset(
        type="val", normalize_input=False, view_point_type=view_point_type
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False, drop_last=True, num_workers=8
    )
    mean, std = get_mean_and_std(
        train_dataloader=train_loader, val_dataloader=val_loader
    )
    print(f"PickOrPlace Classification Dataset for {view_point_type} Mean = {mean}")
    print(f"PickOrPlace Classification Dataset for {view_point_type} Std = {std}")
