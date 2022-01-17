import math
import os
from typing import List, Tuple

from pyrep.errors import ConfigurationPathError

from Environment import Environment
from Demonstration import Demonstration
from CombinedDemonstration import CombinedDemonstration


def combine_demonstrations(demonstrations: List[Demonstration]) -> CombinedDemonstration:
    combined_demo = CombinedDemonstration(demonstrations[0])

    for demonstration in demonstrations:
        combined_demo.visual_states_from_demonstration(demonstration)
    return combined_demo


def get_path(environment: Environment, position: List[float]):
    try:
        path = environment.robot.robot.get_path(position=position, euler=[0, math.radians(180), 0])
        return path
    except ConfigurationPathError as e:
        print(f'Could not find path : {e}')
        return None


def get_demo_index(path: str) -> str:
    folders = os.listdir(path)
    index = -1
    for i in range(len(folders) + 1):
        if f"{i:04}" not in folders:
            return f"{i:04}"
    return f"{index:04}"


def generate_demonstration(env: Environment) -> Tuple[Demonstration, bool]:
    initial_state = env.reset()

    box_pos = env.context.boxes[env.context.goal_box_idx].get_position()
    box_pos_list = [box_pos.x, box_pos.y, box_pos.z]
    stack_pos = env.context.stack.get_position()

    above_box_pos_list = [box_pos.x, box_pos.y, box_pos.z+0.15]

    stack_pos_list = [stack_pos.x, stack_pos.y, stack_pos.z + 0.1]

    demonstration = Demonstration(context=env.context, initial_observation=initial_state)

    # 1 - Get path to box position
    path_to_box = get_path(environment=env, position=box_pos_list)
    if path_to_box is None:
        return None, False

    # 2 - Execute the path, in each step get an observation
    done = False
    while not done:
        done = path_to_box.step()
        env.env.step()

        new_state = env.get_observation()
        demonstration.store_observation(new_state)

    # 3 - Close the gripper, get an observation
    action = env.robot.robot.get_joint_positions() + [0.0]
    new_state, reward, done = env.step(action)
    demonstration.store_observation(new_state)

    # 4 - Move up
    path_above = get_path(environment=env, position=above_box_pos_list)
    if path_above is None:
        return None, False

    # 5 - Execute the path, in each step get an observation
    done = False
    while not done:
        done = path_above.step()
        env.env.step()

        new_state = env.get_observation()
        demonstration.store_observation(new_state)

    # 6 - Get path to stack position
    path_to_stack = get_path(environment=env, position=stack_pos_list)
    if path_to_stack is None:
        return None, False

    # 7 - Execute the path, in each step get an observation
    done = False
    while not done:
        done = path_to_stack.step()
        env.env.step()

        new_state = env.get_observation()
        demonstration.store_observation(new_state)

    # 8 - Open the gripper, get an observation
    action = env.robot.robot.get_joint_positions() + [1.0]
    new_state, reward, done = env.step(action)
    demonstration.store_observation(new_state)

    # 9 - Get a final observation
    for i in range(20):
        new_state, reward, done = env.step(action)
        demonstration.store_observation(new_state)
        if env.success():
            break

    return demonstration, env.success()


def replay_demo(env: Environment, source_demo: Demonstration) -> Tuple[Demonstration, bool]:
    initial_state = env.reset()
    new_demonstration = Demonstration(context=env.context, initial_observation=initial_state)

    for i, observation in enumerate(source_demo.observations[1:]):
        action = observation.joint_positions.tolist() + [observation.gripper_open]
        env.step(action)

        new_state = env.get_observation()
        new_demonstration.store_observation(new_state)

    return new_demonstration, env.success()
