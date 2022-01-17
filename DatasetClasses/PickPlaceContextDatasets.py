import json
import os
import random
import socket

#from Environment import Position

import numpy as np
from skimage import io
from skimage.transform import resize
import torch
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms

class Position:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"x:{self.x}, y:{self.y}, z:{self.z}"

class PickPlaceContextContrastiveDataset(Dataset):

    def __init__(self, type : str, image_size: int, normalize : bool, n_view_points: int = 2):
        if socket.gethostname() == "socialab-167" or socket.gethostname() == 'socialab-VirtualBox' or socket.gethostname() == "pop-os":
            frames_folder: str = f"/home/{os.getlogin()}/Documents/PhD/Datasets/MultiContextPickAndPlace"
        elif socket.gethostname() == "LAPTOP-199MMMNT":
            frames_folder: str = "C:\\Users\\35192\\Documents\\PhDv3\\Datasets\\MultiContextPickAndPlace"
        else:
            raise NotImplementedError
        self.normalize = normalize
        self.normalizer = transforms.Normalize(mean=[0.6094, 0.5346, 0.3273], std=[0.3029, 0.2001, 0.1377])
        
        self.frames_folder = os.path.join(frames_folder, type)
        self.anchor_paths = []
        self.positive_paths = []
        self.image_size = image_size

        cameras = ["camera_top", "camera_front", "camera_overhead", "camera_right", "camera_wrist"]
        viewpoints = cameras[:n_view_points]

        #anchor_folder = "camera_top"
        #positive_folder = "camera_front"

        for demo in os.listdir(self.frames_folder):
            for camera_idx, anchor_folder in enumerate(viewpoints[:-1]):
                for positive_folder in viewpoints[camera_idx+1:]:
                    anchor_frames = sorted(os.listdir(os.path.join(self.frames_folder, demo, anchor_folder)))
                    positive_frames = sorted(os.listdir(os.path.join(self.frames_folder, demo, positive_folder)))
                    for idx in range(0, len(anchor_frames)):
                        anchor = anchor_frames[idx]
                        positive = positive_frames[idx]
                        self.anchor_paths.append(os.path.join(self.frames_folder, demo, anchor_folder, anchor))
                        self.positive_paths.append(
                            os.path.join(self.frames_folder, demo, positive_folder, positive))

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

class PickPlaceContextTripletDataset(Dataset):
    def __init__(self, type : str, image_size: int, normalize : bool, n_view_points: int = 2):
        if socket.gethostname() == "socialab-167" or socket.gethostname() == 'socialab-VirtualBox' or socket.gethostname() == "pop-os":
            frames_folder: str = f"/home/{os.getlogin()}/Documents/PhD/Datasets/MultiContextPickAndPlace"
        elif socket.gethostname() == "LAPTOP-199MMMNT":
            frames_folder: str = "C:\\Users\\35192\\Documents\\PhDv3\\Datasets\\MultiContextPickAndPlace"
        else:
            raise NotImplementedError
        self.normalize = normalize
        self.normalizer = transforms.Normalize(mean=[0.6093, 0.5346, 0.3272], std=[0.3031, 0.2003, 0.1379]) # real
        
        
        self.frames_folder = os.path.join(frames_folder, type)
        self.anchor_paths = []
        self.positive_paths = []
        self.negative_paths = []
        self.image_size = image_size
        self.negative_frame_margin = 30 * 2

        cameras = ["camera_top", "camera_front", "camera_overhead", "camera_right", "camera_wrist"]
        viewpoints = cameras[:n_view_points]

        for demo in os.listdir(self.frames_folder):
            for camera_idx, anchor_folder in enumerate(viewpoints[:-1]):
                for positive_folder in viewpoints[camera_idx+1:]:
                    anchor_frames = sorted(os.listdir(os.path.join(self.frames_folder, demo, anchor_folder)))
                    positive_frames = sorted(os.listdir(os.path.join(self.frames_folder, demo, positive_folder)))
                    demo_length = len(positive_frames)
                    for idx in range(0, len(anchor_frames)):
                        anchor = anchor_frames[idx]
                        positive = positive_frames[idx]

                        # Negatives
                        range1 = np.arange(0, max(0, idx - self.negative_frame_margin))
                        range2 = np.arange(min(idx + self.negative_frame_margin, demo_length), demo_length)
                        ranges = np.concatenate([range1, range2])
                        if len(ranges) == 0:
                            continue

                        negative = anchor_frames[np.random.choice(ranges)]

                        self.anchor_paths.append(os.path.join(self.frames_folder, demo, anchor_folder, anchor))
                        self.positive_paths.append(os.path.join(self.frames_folder, demo, positive_folder, positive))
                        if random.uniform(0,1) < 0.5:
                            self.negative_paths.append(os.path.join(self.frames_folder, demo, anchor_folder, negative))
                        else:
                            self.negative_paths.append(os.path.join(self.frames_folder, demo, positive_folder, negative))

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

        #print(f"{anchor_name} | {positive_name} | {negative_name}")
        if self.normalize:
            anchor_tensor = self.normalizer(anchor_tensor)
            positive_tensor = self.normalizer(positive_tensor)
            negative_tensor = self.normalizer(negative_tensor)

        return (anchor_tensor, positive_tensor, negative_tensor), torch.tensor([1])

class PickPlaceContextRegressionDataset(Dataset):
    def __init__(self, type : str, image_size: int, normalize : bool, n_view_points: int = 2):
        if socket.gethostname() == "socialab-167" or socket.gethostname() == 'socialab-VirtualBox' or socket.gethostname() == "pop-os":
            frames_folder: str = f"/home/{os.getlogin()}/Documents/PhD/Datasets/MultiContextPickAndPlace"
        elif socket.gethostname() == "LAPTOP-199MMMNT":
            frames_folder: str = "C:\\Users\\35192\\Documents\\PhDv3\\Datasets\\MultiContextPickAndPlace"
        else:
            raise NotImplementedError
        self.normalize = normalize
        self.normalizer = transforms.Normalize(mean=[0.6094, 0.5346, 0.3273], std=[0.3029, 0.2001, 0.1377])
        
        self.frames_folder = os.path.join(frames_folder, type)
        self.image_size = image_size
        self.image_paths = []
        self.box_xs = []
        self.box_ys = []
        self.stack_xs = []
        self.stack_ys = []

        camera = "camera_top"

        for demo in os.listdir(self.frames_folder):
            image_path = os.path.join(self.frames_folder, demo, camera, "000.jpg")
            
            demo = json.load(open(os.path.join(self.frames_folder, demo, "demo.json"), "r"))
            
            box_positions = [Position(x=pos[0], y=pos[1], z=pos[2]) for pos in demo["context"]["box_positions"]]
            box_x = box_positions[0].x
            box_y = box_positions[0].y

            stack_position = Position(x=demo["context"]["stack_position"][0], y=demo["context"]["stack_position"][1], z=demo["context"]["stack_position"][2])
            stack_x = stack_position.x
            stack_y = stack_position.y

            self.image_paths.append(image_path)
            self.box_xs.append(box_x)
            self.box_ys.append(box_y)
            self.stack_xs.append(stack_x)
            self.stack_ys.append(stack_y)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.image_paths[idx]
        image = io.imread(image_name)
        image_resized = resize(image, (self.image_size, self.image_size), anti_aliasing=True)

        image_tensor = torch.tensor(image_resized, dtype=torch.float).transpose(2, 0)
        
        if self.normalize:
            image_tensor = self.normalizer(image_tensor)

        box_x = self.box_xs[idx]
        box_y = self.box_ys[idx]
        stack_x = self.stack_xs[idx]
        stack_y = self.stack_ys[idx]

        return image_tensor, torch.tensor([box_x, box_y, stack_x, stack_y])

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
    train_dataset = PickPlaceContextContrastiveDataset(type = 'train', image_size = 224, normalize = False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=True,
        num_workers=8
    )

    val_dataset = PickPlaceContextContrastiveDataset(type = 'val', image_size = 224, normalize = False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=True,
        num_workers=8
    )

    mean, std = get_mean_and_std(train_dataloader = train_loader, val_dataloader = val_loader)
    print(f"PickPlaceContext Dataset Mean = {mean}")
    print(f"PickPlaceContext Dataset Std = {std}")
