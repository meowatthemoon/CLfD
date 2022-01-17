import argparse
import os

import numpy as np
import torch
from clfd import CLfD
from PickPlaceContextDatasets import \
    PickPlaceContextContrastiveDataset  # , PickPlaceContextTripletDataset
from PouringDatasets import \
    PouringContrastiveDataset  # , PouringTripletDataset
from skimage import io
from skimage.transform import resize
from tcn import TCN
from torchvision import transforms
from tqdm import tqdm

from utils import yaml_config_hook


def mse(a, b):
    return  ((a - b)**2).mean(axis=0)

def compute_average_alignment(seqname_to_embeddings,):
    all_alignments = []
    for view_embeddings in tqdm(seqname_to_embeddings):
        embeddings_view_i = np.asarray(view_embeddings[0])
        embeddings_view_j = np.asarray(view_embeddings[1])

        length = embeddings_view_j.shape[0]
        for view_1_idx in range(length):
            min_dist = float("inf")
            min_idx = -1
            for view_2_idx in range(length):
                dist = mse(embeddings_view_i[view_1_idx], embeddings_view_j[view_2_idx])
                if dist < min_dist:
                    min_dist = dist
                    min_idx = view_2_idx
            alignment_error = abs(view_1_idx - min_idx) / length
            all_alignments.append(alignment_error)

    average_alignment = np.mean(all_alignments)
    return average_alignment

def validate(method, dataset_name, image_size, device, normalize, model, normalizer):
    if dataset_name == "PICKPLACECONTEXT":
        dataset_dir = '../Datasets/MultiContextPickAndPlace/val'
    elif dataset_name == "POURING":
        dataset_dir = '../Datasets/multiview_pouring/pouring_frames/val'
    else:
        raise NotImplementedError

    with torch.no_grad():
        # Loop through test video pairs, extract embeddings
        embeddings = []
        for video_pair in tqdm(os.listdir(dataset_dir)):
            if dataset_name == "PICKPLACECONTEXT":
                view1_path = os.path.join(dataset_dir, video_pair, 'camera_top')
                view2_path = os.path.join(dataset_dir, video_pair, 'camera_front')

                view1_frames = sorted(os.listdir(view1_path))
                view2_frames = sorted(os.listdir(view2_path))
            else:
                view1_path = os.path.join(dataset_dir, video_pair)
                view2_path = view1_path

                all_frames = sorted(os.listdir(view1_path))
                view1_frames = [all_frames[i] for i in range(0, len(all_frames), 2)]
                view2_frames = [all_frames[i] for i in range(1, len(all_frames), 2)]
            
            length = len(view1_frames)
            assert len(view1_frames) == len(view2_frames)

            embeddings_view_1 = []
            embeddings_view_2 = []

            for frame_idx in range(0, length):
                image_1_path = os.path.join(view1_path, view1_frames[frame_idx])
                image_2_path = os.path.join(view2_path, view2_frames[frame_idx])

                image_1 = io.imread(image_1_path)
                image_2 = io.imread(image_2_path)
                
                image_1_resized = resize(image_1, (image_size, image_size), anti_aliasing=True)
                image_2_resized = resize(image_2, (image_size, image_size), anti_aliasing=True)

                tensor_1 = torch.tensor(image_1_resized, dtype=torch.float).transpose(2, 0)
                tensor_2 = torch.tensor(image_2_resized, dtype=torch.float).transpose(2, 0)

                tensor_1 = tensor_1.unsqueeze(0).to(device)
                tensor_2 = tensor_2.unsqueeze(0).to(device)


                if normalize:
                    tensor_1 = normalizer(tensor_1)
                    tensor_2 = normalizer(tensor_2)

                # Forward tensors, obtaining embeddings
                if method == "contrastive":
                    (f1, _, h1, _) = model(tensor_1, tensor_1)
                    (f2, _, h2, _) = model(tensor_2, tensor_2)
                else:
                    f1 = model(tensor_1)
                    f2 = model(tensor_2)
                f1 = f1.cpu().detach().numpy()
                f2 = f2.cpu().detach().numpy()

                embeddings_view_1.append(np.squeeze(f1, 0)) # np.ndarray
                embeddings_view_2.append(np.squeeze(f2, 0)) # np.ndarray
            embeddings.append([embeddings_view_1, embeddings_view_2])
    
    alignment_error = compute_average_alignment(embeddings)
    alignment_error_perc = alignment_error * 100
    return alignment_error_perc    

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="CLfD")
    config = yaml_config_hook("al_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define dataset, normalizer and models folder
    if args.method == "contrastive":
        model = CLfD(backbone = args.network, out_features = args.out_features, projection_dim =args.projection_dim)
        name = f"CLfD__{args.network}__{args.dataset}__{args.epochs}__{args.batch_size}__Adam__{args.n_view_points}v__{args.out_features}f__True"
    elif args.method == "triplet":
        model = TCN(out_features = args.out_features)
        name = f"TCN__{args.dataset}__{args.epochs}__{args.batch_size}__{args.margin}__Adam__True"
    else:
        raise NotImplementedError
    model_path = os.path.join("../results/", name)

    for model_name in os.listdir(model_path):
        if model_name[-3:] != "tar":
            continue
        print(f"{model_name} : Loading model...")
        chkpt_path = os.path.join(model_path, model_name)
        model.load_state_dict(torch.load(chkpt_path, map_location=args.device.type))
        model = model.to(args.device)

        print(f"{model_name} : Generating Embeddings")

        alignment_error_perc = validate(args, model)

        print(f"{model_name} : {alignment_error_perc}")
