import os

from Environment import ArmActionType
from Environment import Color
from Environment import Environment
from Demonstrator import combine_demonstrations, generate_demonstration, \
    get_demo_index, replay_demo

if __name__ == "__main__":
    # Configurations
    base_path = "../Datasets/MultiContextPickAndPlace"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    colors = [Color.Transparent, Color.Black, Color.Blue, Color.White, Color.Yellow]

    while len(os.listdir(base_path)) < 150:
        env = Environment(arm_action_type=ArmActionType.ABS_JOINT_POSITION, num_boxes=1,
                          goal_box_idx=0, background_color=Color.Transparent)

        box_positions = [env.context.boxes[0].position]
        stack_position = env.context.stack.position

        source_demo, success = generate_demonstration(env=env)
        env.close()

        if source_demo is None or not success:
            continue

        """
        demonstrations = [source_demo]

        for color_idx, color in enumerate(colors[1:]):
            # print(f"Color #{color_idx + 1} / {len(colors[1:])} : {color}")
            env = Environment(arm_action_type=ArmActionType.ABS_JOINT_POSITION,
                              num_boxes=1,
                              goal_box_idx=0,
                              background_color=color,
                              box_positions=box_positions,
                              stack_position=stack_position)

            replayed_demo, success = replay_demo(env=env, source_demo=source_demo)
            env.close()

            if replayed_demo is None or not success:
                print(f"Failed demo {color}: {replayed_demo} or {success}")
                break

            demonstrations.append(replayed_demo)

        # If all demos were successfully executed, combine them and save them
        if len(demonstrations) == len(colors):
            combined_demo = combine_demonstrations(demonstrations)
            combined_demo.save_demonstration(os.path.join(base_path, get_demo_index(base_path)))
        else:
            print("Didn't save")
        """
        source_demo.save_demonstration(os.path.join(base_path, get_demo_index(base_path)))
