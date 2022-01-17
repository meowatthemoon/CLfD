import math
import os
from typing import List, Tuple

from Environment import ArmActionType, Color, Environment
from PIL import Image


def get_path(environment: Environment, position: List[float]):
    try:
        path = environment.robot.robot.get_path(position=position, euler=[0, math.radians(180), 0])
        return path
    except Exception as e:
        print(f'Could not find path : {e}')
        return None

def generate_demonstration(env: Environment, view_point : str) -> Tuple[List, List, bool]:
    env.reset()

    box_pos = env.context.boxes[env.context.goal_box_idx].get_position()
    box_pos_list = [box_pos.x, box_pos.y, box_pos.z]
    stack_pos = env.context.stack.get_position()

    above_box_pos_list = [box_pos.x, box_pos.y, box_pos.z+0.15]

    stack_pos_list = [stack_pos.x, stack_pos.y, stack_pos.z + 0.1]

    # 1 - Get path to box position
    path_to_box = get_path(environment=env, position=box_pos_list)
    if path_to_box is None:
        return None, None, False

    # 2 - Execute the path, in each step get an observation
    pick_images = []
    done = False
    while not done:
        done = path_to_box.step()
        env.env.step()

        obs = env.get_observation()
        if view_point == "Front":
            pick_images.append(obs.front_rgb)
        elif view_point == "Right":
            pick_images.append(obs.right_shoulder_rgb)
        else:
            raise NotImplementedError

    # 3 - Close the gripper, get an observation
    action = env.robot.robot.get_joint_positions() + [0.0]
    new_state, reward, done = env.step(action)

    # 4 - Move up
    path_above = get_path(environment=env, position=above_box_pos_list)
    if path_above is None:
        return None, None, False

    # 5 - Execute the path, in each step get an observation
    place_images = []
    done = False
    while not done:
        done = path_above.step()
        env.env.step()

        obs = env.get_observation()
        if view_point == "Front":
            place_images.append(obs.front_rgb)
        elif view_point == "Right":
            place_images.append(obs.right_shoulder_rgb)
        else:
            raise NotImplementedError

    # 6 - Get path to stack position
    path_to_stack = get_path(environment=env, position=stack_pos_list)
    if path_to_stack is None:
        return None, None, False

    # 7 - Execute the path, in each step get an observation
    done = False
    while not done:
        done = path_to_stack.step()
        env.env.step()

        new_state = env.get_observation()

    # 8 - Open the gripper, get an observation
    action = env.robot.robot.get_joint_positions() + [1.0]
    new_state, reward, done = env.step(action)

    # 9 - Get a final observation
    for i in range(20):
        new_state, reward, done = env.step(action)
        if env.success():
            break

    return pick_images, place_images, env.success()

if __name__ == "__main__":
    view_point = "Right" # "Front" for seen or "Right" for unseen

    # Configurations
    base_path = f"../Datasets/PickOrPlace{view_point}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for type in ["train", "val"]:
        pick_path = os.path.join(base_path, type, "pick")
        place_path = os.path.join(base_path, type, "place")
        if not os.path.exists(pick_path):
            os.makedirs(pick_path)
        if not os.path.exists(place_path):
            os.makedirs(place_path)
        
        n_images = 10000 if type == "train" else 1000

        while len(os.listdir(pick_path)) < n_images:
            env = Environment(arm_action_type=ArmActionType.ABS_JOINT_POSITION, 
                              num_boxes=1,
                              goal_box_idx=0, 
                              background_color=Color.Transparent)
            pick_images, place_images, success = generate_demonstration(env = env, view_point = view_point)
            env.close()

            if not success:
                continue

            n = min(len(pick_images), len(place_images))
            pick_images = pick_images[:n]
            place_images = place_images[-n:]
            for img_idx in range(n):
                idx = len(os.listdir(pick_path))
                Image.fromarray(pick_images[img_idx]).save(os.path.join(pick_path,  f"{img_idx+idx:05}.jpg"))
                #print(os.path.join(pick_path,  f"{img_idx+idx:05}.jpg"))
                Image.fromarray(place_images[img_idx]).save(os.path.join(place_path,  f"{img_idx+idx:05}.jpg"))
                #print(os.path.join(place_path,  f"{img_idx+idx:05}.jpg"))
            #break
            