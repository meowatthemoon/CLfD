import json
import os

from PIL import Image

from Environment import Observation
from Environment import Context


class Demonstration:
    def __init__(self, context: Context, initial_observation: Observation):
        self.context = context
        self.observations = [initial_observation]
        self.box_positions = [[context.boxes[box_id].position.x,
                               context.boxes[box_id].position.y,
                               context.boxes[box_id].position.z] for box_id in range(len(context.boxes))]
        self.stack_position = [context.stack.position.x,
                               context.stack.position.y,
                               context.stack.position.z]
        self.background_color = self.context.background_color.value

    def store_observation(self, observation: Observation):
        self.observations.append(observation)

    def store_observation(self, observation: Observation):
        self.observations.append(observation)

    def save_demonstration(self, full_path: str):
        camera_folders = ["camera_top", "camera_right", "camera_overhead", "camera_wrist", "camera_front"]
        context_data = {"plane_color": self.background_color,
                        "box_positions": self.box_positions,
                        "stack_position": self.stack_position}
        transitions_data = []

        # Create demonstration folder
        os.makedirs(full_path)

        # Create camera folders
        for camera_folder in camera_folders:
            os.mkdir(os.path.join(full_path, camera_folder))

        # Loop through each transition
        for observation_index, observation in enumerate(self.observations):
            # Save visual data
            Image.fromarray(observation.top_rgb).save(
                os.path.join(full_path, "camera_top", f"{observation_index:03}.jpg"))
            Image.fromarray(observation.right_shoulder_rgb).save(
                os.path.join(full_path, "camera_right", f"{observation_index:03}.jpg"))
            Image.fromarray(observation.overhead_rgb).save(
                os.path.join(full_path, "camera_overhead", f"{observation_index:03}.jpg"))
            Image.fromarray(observation.wrist_rgb).save(
                os.path.join(full_path, "camera_wrist", f"{observation_index:03}.jpg"))
            Image.fromarray(observation.front_rgb).save(
                os.path.join(full_path, "camera_front", f"{observation_index:03}.jpg"))

            observation_data = {
                "joint_velocities": observation.joint_velocities.tolist(),
                "joint_positions": observation.joint_positions.tolist(),
                "joint_forces": observation.joint_forces.tolist(),
                "gripper_open": observation.gripper_open,
                "gripper_pose": observation.gripper_pose.tolist(),
                "gripper_joint_positions": observation.gripper_joint_positions.tolist(),
                "gripper_touch_forces": observation.gripper_touch_forces.tolist()
            }
            transitions_data.append(observation_data)

        non_visual_data = {"context": context_data, "transitions": transitions_data}

        # Save non-visual data
        with open(os.path.join(full_path, "demo.json"), 'w') as outfile:
            json.dump(non_visual_data, outfile, indent=4)
            outfile.close()
