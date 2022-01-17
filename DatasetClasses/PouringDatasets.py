import os
import socket

import numpy as np
from skimage import io
from skimage.transform import resize
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PouringContrastiveDataset(Dataset):

    def __init__(self, type: str, image_size: int, normalize : bool):
        if socket.gethostname() == "socialab-167" or socket.gethostname() == 'socialab-VirtualBox' or socket.gethostname() == "pop-os":
            frames_folder: str = f"/home/{os.getlogin()}/Documents/PhD/Datasets/multiview_pouring/pouring_frames"
        elif socket.gethostname() == "LAPTOP-199MMMNT":
            frames_folder: str = "C:\\Users\\35192\\Documents\\PhDv3\\Datasets\\multiview_pouring\\pouring_frames"
        else:
            raise NotImplementedError
            
        self.normalize = normalize
        self.normalizer = transforms.Normalize(mean=[0.6703, 0.5933, 0.5220], std=[0.2398, 0.2295, 0.2304])
        
        self.frames_folder = os.path.join(frames_folder, type)
        self.anchor_paths = []
        self.positive_paths = []
        self.image_size = image_size
        for video in os.listdir(self.frames_folder):
            video_frames = sorted(os.listdir(os.path.join(self.frames_folder, video)))
            for idx in range(0, len(video_frames), 2):
                anchor = video_frames[idx]
                positive = video_frames[idx + 1]
                self.anchor_paths.append(os.path.join(self.frames_folder, video, anchor))
                self.positive_paths.append(os.path.join(self.frames_folder, video, positive))
                                    
    
    def __len__(self):
        return len(self.anchor_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anchor_name = self.anchor_paths[idx]
        positive_name = self.positive_paths[idx]
        anchor_image = io.imread(anchor_name)
        anchor_image_resized = resize(anchor_image, (self.image_size, self.image_size),
                                      anti_aliasing=True)
        positive_image = io.imread(positive_name)
        positive_image_resized = resize(positive_image, (self.image_size, self.image_size),
                                        anti_aliasing=True)

        anchor_tensor = torch.tensor(anchor_image_resized, dtype=torch.float).transpose(2, 0)
        positive_tensor = torch.tensor(positive_image_resized, dtype=torch.float).transpose(2, 0)
        if self.normalize:
            anchor_tensor = self.normalizer(anchor_tensor)
            positive_tensor = self.normalizer(positive_tensor)

        return (anchor_tensor, positive_tensor), torch.tensor([1])


class PouringTripletDataset(Dataset):
    def __init__(self, type: str, image_size: int, normalize : bool):
        if socket.gethostname() == "socialab-167" or socket.gethostname() == 'socialab-VirtualBox' or socket.gethostname() == "pop-os":
            frames_folder: str = f"/home/{os.getlogin()}/Documents/PhD/Datasets/multiview_pouring/pouring_frames"
        elif socket.gethostname() == "LAPTOP-199MMMNT":
            frames_folder: str = "C:\\Users\\35192\\Documents\\PhDv3\\Datasets\\multiview_pouring\\pouring_frames"
        else:
            raise NotImplementedError

        self.normalize = normalize
        self.normalizer = transforms.Normalize(mean=[0.6730, 0.5960, 0.5247], std=[0.2404, 0.2301, 0.2310])
        
        self.frames_folder = os.path.join(frames_folder, type)
        self.anchor_paths = []
        self.positive_paths = []
        self.negative_paths = []
        self.image_size = image_size
        self.negative_frame_margin = 30 * 2

        for video in os.listdir(self.frames_folder):
            video_frames = sorted(os.listdir(os.path.join(self.frames_folder, video)))
            video_length = len(video_frames)
            for idx in range(0, video_length, 2):
                anchor = video_frames[idx]
                positive = video_frames[idx + 1]

                # Negatives
                range1 = np.arange(0, max(0, idx - self.negative_frame_margin))
                range2 = np.arange(min(idx + self.negative_frame_margin, video_length), video_length)
                ranges = np.concatenate([range1, range2])
                #print(f"{idx}/{video_length} (0, {max(0, idx - self.negative_frame_margin)}) ({min(idx + self.negative_frame_margin, video_length)}, {video_length})")
                if len(ranges) == 0:
                    continue
                choice = np.random.choice(ranges)
                negative = video_frames[choice]
                self.negative_paths.append(os.path.join(self.frames_folder, video, negative))
                self.anchor_paths.append(os.path.join(self.frames_folder, video, anchor))
                self.positive_paths.append(os.path.join(self.frames_folder, video, positive))

    def __len__(self):
        return len(self.anchor_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anchor_name = self.anchor_paths[idx]
        positive_name = self.positive_paths[idx]
        negative_name = self.negative_paths[idx]
        anchor_image = io.imread(anchor_name)
        anchor_image_resized = resize(anchor_image, (self.image_size, self.image_size), anti_aliasing=True)
        positive_image = io.imread(positive_name)
        positive_image_resized = resize(positive_image, (self.image_size, self.image_size), anti_aliasing=True)
        negative_image = io.imread(negative_name)
        negative_image_resized = resize(negative_image, (self.image_size, self.image_size), anti_aliasing=True)

        anchor_tensor = torch.tensor(anchor_image_resized, dtype=torch.float).transpose(2, 0)
        positive_tensor = torch.tensor(positive_image_resized, dtype=torch.float).transpose(2, 0)
        negative_tensor = torch.tensor(negative_image_resized, dtype=torch.float).transpose(2, 0)

        if self.normalize:
            anchor_tensor = self.normalizer(anchor_tensor)
            positive_tensor = self.normalizer(positive_tensor)
            negative_tensor = self.normalizer(negative_tensor)

        return (anchor_tensor, positive_tensor, negative_tensor), torch.tensor([1])

# https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
def get_mean_and_std(train_dataloader, val_dataloader):
    channels_sum=0
    channels_squared_sum=0
    num_batches = 0
    for (x_i, x_j), _ in train_dataloader:
        # Mean over batch, height and width, but not over the channels
        data = torch.cat((x_i, x_j), 0)
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    for (x_i, x_j), _ in val_dataloader:
        # Mean over batch, height and width, but not over the channels
        data = torch.cat((x_i, x_j), 0)
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == "__main__":
    train_dataset = PouringContrastiveDataset(type = 'train', image_size = 299, normalize = False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=True,
        num_workers=8
    )

    val_dataset = PouringContrastiveDataset(type = 'val', image_size = 299, normalize = False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=True,
        num_workers=8
    )

    mean, std = get_mean_and_std(train_dataloader = train_loader, val_dataloader = val_loader)
    print(f"Pouring Dataset Mean = {mean}")
    print(f"Pouring Dataset Std = {std}")