import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from clfd import CLfD
from Environment import ArmActionType, Color, Environment, Position
from PIL import Image
from skimage import io
from skimage.transform import resize
from torch.autograd import Variable
from torchvision import transforms

import DDPG
import PER_buffer
import utils


def features_from_tensor(encoder : CLfD, tensor : torch.tensor) -> np.ndarray:
    h,_,z,_ = encoder(tensor, tensor)
    h = h.detach()
    return torch.squeeze(h, 0).cpu().detach().numpy()

def get_demo_state_features(encoder, directory, normalizer):
    features = []
    for image_name in sorted(os.listdir(directory)):
        image_path = os.path.join(directory, image_name)

        t = tensor_from_path(path = image_path, normalizer = normalizer)
        
        f = features_from_tensor(encoder = encoder, tensor = t)
        features.append(f)
    return np.asarray(features)

def get_reward(obs_f : np.ndarray, demo_f : np.ndarray) -> float:
    diff = demo_f - obs_f
    sqrd = diff ** 2
    mean = sqrd.mean(axis=0)
    reward =  -1 * mean
    return reward

def tensor_from_path(path : str, normalizer) -> torch.tensor:
    """
    Loads image into a cuda tensor.
    """
    image = io.imread(path)
    image_resized = resize(image, (224, 224), anti_aliasing=True)
    t = torch.tensor(image_resized, dtype=torch.float).transpose(2, 0)
    t = normalizer(t)
    t = torch.unsqueeze(t, 0)
    t = t.cuda()
    
    return t

def load_rewards(file):
    if not os.path.exists(file):
        return []
    return json.load(open(file, "r"))

def save_rewards(file, reward_history):
    with open(file, 'w') as outfile:
        json.dump(reward_history, outfile, indent=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = "tcn"
if not os.path.exists("data"):
    os.mkdir("data")
rewards_file = "data/rewards.json"

#Fixed Parameters
test_freq = 2000
train_steps = 100000
episodes = 10000

#DDPG parameters
batch_size = 64
gamma = 0.99
tau = 0.001
buffer_size = 20000

#PER parameters
prioritized_replay_alpha=0.6
prioritized_replay_beta0=0.4
prioritized_replay_beta_iters=None
prioritized_replay_eps=1e-6

# Create encoder, load weights, set to eval
normalizer = transforms.Normalize(mean=[0.6094, 0.5346, 0.3273], std=[0.3029, 0.2001, 0.1377])
encoder = CLfD(backbone = network,out_features = 32, projection_dim = 64)
encoder.load_state_dict(torch.load(f"pretrained_clfd.tar", map_location=device.type))
encoder = encoder.to(device)
encoder.eval()

# Load demonstration
demonstration_path = "/home/andre/Documents/PhD/Datasets/MultiContextPickAndPlace/train/0003"
demo_json = json.load(open(os.path.join(demonstration_path, "demo.json"), "r"))
demo_transitions = demo_json["transitions"]
demo_context = demo_json["context"]
demo_state_features = get_demo_state_features(encoder = encoder, directory = os.path.join(demonstration_path, "camera_front"), normalizer = normalizer)
goal_features = demo_state_features[59]

# Define box and stack positions
demo_box = demo_context["box_positions"][0]
demo_stack = demo_context["stack_position"]
box_positions = [Position(x = demo_box[0], y = demo_box[1], z = demo_box[2])]
stack_position = Position(x = demo_stack[0], y = demo_stack[1], z = demo_stack[2])

# Launch environment
env = Environment(arm_action_type=ArmActionType.ABS_JOINT_VELOCITY,
                    num_boxes=1,
                    goal_box_idx=0,
                    background_color=Color.Transparent,
                    box_positions=box_positions,
                    stack_position=stack_position)


# environment information
s_dim = 32 + 7 + 7
a_dim = 7
a_max = math.pi

#Create DDPG policy
policy = DDPG.DDPG(s_dim, a_dim, a_max)
if os.path.exists("data/actor.tar"):
    policy.load_data()

# PER
replay_buffer = PER_buffer.PrioritizedReplayBuffer(buffer_size, prioritized_replay_alpha)
if os.path.exists("data/buffer.pkl"):
    with open("data/buffer.pkl", 'rb') as inp:
        replay_buffer = pickle.load(inp)
if prioritized_replay_beta_iters is None:
    prioritized_replay_beta_iters = train_steps
#Create annealing schedule
beta_schedule = utils.LinearSchedule(prioritized_replay_beta_iters, initial_p=prioritized_replay_beta0, final_p=1.0)

reward_history = load_rewards(rewards_file)
total_time = 0
for episode in range(len(reward_history), episodes):
    obs = env.reset()
    episode_r = 0
    episode_t = 0

    # Create current_state (s) from obs
    Image.fromarray(obs.front_rgb).save("/tmp/new_state.jpg")
    state_image_tensor = tensor_from_path(path = "/tmp/new_state.jpg", normalizer = normalizer)
    state_features = features_from_tensor(encoder = encoder, tensor = state_image_tensor)
    state_joints = np.asarray(obs.joint_positions) # / math.pi
    state_vels = np.asarray(obs.joint_velocities)

    s = np.concatenate((state_features, state_joints, state_vels),axis = 0)

    for step in range(1, 60):
        # Given current state, get action
        a = policy.get_action(np.array(s))
        # Apply exploration noise to action
        a = (a + np.random.normal(0, 0.1, size=a_dim)).clip([-a_max], [a_max])

        # Step
        obs, _, done = env.step(np.array(list(a) + [1.0]))

        # Create new_state from obs
        Image.fromarray(obs.front_rgb).save("/tmp/new_state.jpg")
        new_state_image_tensor = tensor_from_path(path = "/tmp/new_state.jpg", normalizer = normalizer)
        new_state_features = features_from_tensor(encoder = encoder, tensor = new_state_image_tensor)
        new_state_joints = np.asarray(obs.joint_positions) # / math.pi
        new_state_vels = np.asarray(obs.joint_velocities)
        s_new = np.concatenate((new_state_features, new_state_joints, new_state_vels), axis = 0)

        # Reward
        r = get_reward(obs_f = new_state_features, demo_f = demo_state_features[step])#)# goal_features
        #print(r)

        # Store data in replay buffer
        done = float(done or step == 59 )#or r > -0.7
        replay_buffer.add(s, a, r, s_new, done)

        # Update state and episode statistics
        s = s_new
        episode_r += r
        total_time += 1
        episode_t += 1
        if done:
            break


    print(f"Episode: {(episode + 1)} Episode Reward:{episode_r}")

    # Set beta value used for importance sampling weights
    beta_value = beta_schedule.value(total_time)

    # Train DDPG
    policy.train(replay_buffer, True, beta_value, prioritized_replay_eps, episode_t, batch_size, gamma)

    reward_history.append(episode_r)

    if (episode + 1) % 50 == 0:
        break

save_rewards(rewards_file, reward_history)
with open("data/buffer.pkl", 'wb') as outp:
    pickle.dump(replay_buffer, outp, pickle.HIGHEST_PROTOCOL)
policy.save_data()
env.close()
        