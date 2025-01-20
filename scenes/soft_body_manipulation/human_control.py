from sofa_env.utils.human_input import XboxController
from sofa_env.wrappers.trajectory_recorder import TrajectoryRecorder
from sofa_env.wrappers.realtime import RealtimeWrapper
from sofa_env.scenes.soft_body_manipulation.soft_body_env import SoftBodyManipulationEnv, ObservationType, ActionType, RenderMode, Phase
import cv2
from typing import Tuple
import numpy as np
import time
from collections import deque
from pathlib import Path
import argparse
from sofa_env.scenes.soft_body_manipulation.sofa_objects.gripper1 import Gripper1, GRIPPER_PLUGIN_LIST
from sofa_env.scenes.soft_body_manipulation.sofa_objects.gripper2 import Gripper2, GRIPPER_PLUGIN_LIST
from sofa_env.scenes.grasp_lift_touch.sofa_objects.tool_controller1 import KeyboardController
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup human input behavior.")
    parser.add_argument("-rv", "--record_video", action="store_true", help="Record video of the trajectory.")
    parser.add_argument("-rt", "--record_trajectory", action="store_true", help="Record the full trajectory.")
    parser.add_argument("-i", "--info", action="store", type=str, help="Additional info to store in the metadata.")
    args = parser.parse_args()

    controller = KeyboardController()#XboxController()
    if not controller.is_alive():
        raise RuntimeError("Could not find keyboard controller.")

    image_shape = (1024, 1024)
    image_shape_to_save = (256, 256)

    env = SoftBodyManipulationEnv(
        render_mode=RenderMode.HUMAN,
        observation_type=ObservationType.STATE,
        action_type=ActionType.CONTINUOUS,
        start_in_phase=Phase.GRASP,
        end_in_phase=Phase.DONE,
        image_shape=image_shape,
        individual_agents=False,
        individual_rewards=False,
        time_step=1 / 30,
        frame_skip=1,
    )
    env = RealtimeWrapper(env)
    
    if args.record_video:
        video_folder = Path("/home/saketh/SOFA/v23.06.00/bin/lapgym/sofa_env/sofa_env/scenes/soft_body_manipulation/videos")
        video_folder.mkdir(exist_ok=True)
        video_name = time.strftime("%Y%m%d-%H%M%S")
        video_path = video_folder / f"{video_name}.mp4"
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            1 / (env.time_step / env.frame_skip),
            image_shape[::-1],
        )
    else:
        video_writer = None

    if args.record_trajectory:

        def store_rgb_obs(self: TrajectoryRecorder, shape: Tuple[int, int] = image_shape_to_save):
            observation = self.env.render()
            observation = cv2.resize(
                observation,
                shape,
                interpolation=cv2.INTER_AREA,
            )
            self.trajectory["rgb"].append(observation)

        metadata = {
            "frame_skip": env.frame_skip,
            "time_step": env.time_step,
            "observation_type": env.observation_type.name,
            "reward_amount_dict": env.reward_amount_dict,
            "user_info": args.info,
        }

        env = TrajectoryRecorder(
            env,
            log_dir="trajectories",
            metadata=metadata,
            store_info=True,
            save_compressed_keys=["observation", "terminal_observation", "rgb", "info"],
            after_step_callbacks=[store_rgb_obs],
            after_reset_callbacks=[store_rgb_obs],
        )

    reset_obs = env.reset()
    if video_writer is not None:
        video_writer.write(env.render()[:, :, ::-1])

    done = False

    instrument = 0
    up = True

    fps_list = deque(maxlen=100)
    
    while not done:
        start = time.perf_counter()
        #lx, ly, rx, ry, lt, rt = controller.read()
        w, a, s, d, up, down, left, right, q, z, x, c, escape = controller.read()
        action = np.zeros_like(env.action_space.sample())
        #action[0] = ly
        #action[1] = lx
        #action[5] = rx
        #action[6] = -ry
        #action[2 + instrument * 5] = controller.right_bumper - controller.left_bumper
        #action[3 + instrument * 5] = rt - lt
        #action[4 + instrument * 5] = controller.b - controller.a

        #if controller.y:
        #    if up:
        #        instrument = 0 if instrument == 1 else 1
        #    up = Falsezqa
        #else:
        #    up = True
        #print(action)
        #action = np.array([0.05, 0.05, 0.0, 0.0, 0.55,0.0, 0.0, 0.0, 0.0, 0.0])
        #action = controller.do_action(action)
        minus = -1 if z else 1
        action[0] = w
        action[1] = a
        action[2] = s
        action[3] = d
        action[4] = q
        #action[5] = up
        #action[6] = down
        #action[7] = left
        #action[8] = right
        #action[9] = c
        
        action[0] = minus*action[0]
        action[1] = minus*action[1]
        action[2] = minus*action[2]
        action[3] = minus*action[3]
        action[4] = minus*action[4]
        #action[5] = minus*action[5]
        #action[6] = minus*action[6]
        #action[7] = minus*action[7]
        #action[8] = minus*action[8]
        #action[9] = minus*action[9]
        
        #print(action)        
        
        obs, reward, done, info = env.step(action)
        print(info["final_phase"])
        if video_writer is not None:
            video_writer.write(env.render()[:, :, ::-1])

        #if controller.x:
        #    cv2.imwrite("exit_image.png", env.render()[:, :, ::-1])
        #    break
        if escape:
            cv2.imwrite("/home/saketh/SOFA/v23.06.00/bin/lapgym/sofa_env/sofa_env/scenes/soft_body_manipulation/start_image.png", env.render()[:, :, ::-1])
            break
        end = time.perf_counter()
        fps = 1 / (end - start)
        fps_list.append(fps)
        #print(f"FPS Mean: {np.mean(fps_list):.5f}    STD: {np.std(fps_list):.5f}")
        #controller.print_tool_state()


    if video_writer is not None:
        video_writer.release()
