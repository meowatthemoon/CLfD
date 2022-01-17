import json
import os

from PIL import Image

from Demonstration import Demonstration


class CombinedDemonstration:
    def __init__(self, demonstration: Demonstration):
        self.box_positions = demonstration.box_positions
        self.stack_pos = demonstration.stack_position

        self.non_visual_states = self.__non_visual_sates_from_demonstration(demonstration)

        self.visual_states = {}

    def __non_visual_sates_from_demonstration(self, demonstration):
        self.visual_states = {}
        non_visual_states = []
        for observation in demonstration.observations:
            observation_data = {
                "joint_velocities": observation.joint_velocities.tolist(),
                "joint_positions": observation.joint_positions.tolist(),
                "joint_forces": observation.joint_forces.tolist(),
                "gripper_open": observation.gripper_open,
                "gripper_pose": observation.gripper_pose.tolist(),
                "gripper_joint_positions": observation.gripper_joint_positions.tolist(),
                "gripper_touch_forces": observation.gripper_touch_forces.tolist()
            }
            non_visual_states.append(observation_data)
        return non_visual_states

    def visual_states_from_demonstration(self, demonstration):
        color = demonstration.context.background_color.name

        image_sequence = []
        for observation in demonstration.observations:
            image_data = {
                "right_shoulder_rgb": observation.right_shoulder_rgb,
                "overhead_rgb": observation.overhead_rgb,
                "wrist_rgb": observation.wrist_rgb,
                "front_rgb": observation.front_rgb,
                "top_rgb": observation.top_rgb
            }
            image_sequence.append(image_data)

        self.visual_states[str(color)] = image_sequence

    def save_demonstration(self, full_path: str):
        camera_folders = ["camera_top", "camera_right", "camera_overhead", "camera_wrist", "camera_front"]
        context_data = {"box_positions": self.box_positions,
                        "stack_position": self.stack_pos}

        # Create demonstration folder
        os.makedirs(full_path)

        # Save non-visual data
        non_visual_data = {"context": context_data, "transitions": self.non_visual_states}
        with open(os.path.join(full_path, "demo.json"), 'w') as outfile:
            json.dump(non_visual_data, outfile, indent=4)
            outfile.close()

        # Save visual data
        for color in self.visual_states.keys():
            visual_path = os.path.join(full_path, color)
            os.mkdir(visual_path)
            for camera_folder in camera_folders:
                os.mkdir(os.path.join(visual_path, camera_folder))

            for observation_index, image_data in enumerate(self.visual_states[color]):
                Image.fromarray(image_data["right_shoulder_rgb"]).save(
                    os.path.join(visual_path, "camera_right", f"{observation_index:03}.jpg"))
                Image.fromarray(image_data["overhead_rgb"]).save(
                    os.path.join(visual_path, "camera_overhead", f"{observation_index:03}.jpg"))
                Image.fromarray(image_data["wrist_rgb"]).save(
                    os.path.join(visual_path, "camera_wrist", f"{observation_index:03}.jpg"))
                Image.fromarray(image_data["front_rgb"]).save(
                    os.path.join(visual_path, "camera_front", f"{observation_index:03}.jpg"))
                Image.fromarray(image_data["top_rgb"]).save(
                    os.path.join(visual_path, "camera_top", f"{observation_index:03}.jpg"))
