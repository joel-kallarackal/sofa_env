from collections import defaultdict
import gymnasium.spaces as spaces
import numpy as np
import cv2
from enum import Enum, unique
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List, Callable
from functools import reduce

import Sofa
import Sofa.Core

from sofa_env.base import RenderMode, SofaEnv, RenderFramework
from sofa_env.scenes.soft_body_manipulation.sofa_objects.liver import Liver
from sofa_env.utils.camera import world_to_pixel_coordinates
from sofa_env.utils.math_helper import distance_between_line_segments, farthest_point_sampling

from sofa_env.scenes.soft_body_manipulation.sofa_objects.point_of_interest import PointOfInterest
from sofa_env.scenes.soft_body_manipulation.sofa_objects.gripper2 import Gripper2
from sofa_env.scenes.soft_body_manipulation.sofa_objects.intestine import Intestine
from sofa_env.scenes.soft_body_manipulation.sofa_objects.gripper1 import Gripper1
from sofa_env.utils.math_helper import farthest_point_sampling
from sofa_env.utils.motion_planning import create_linear_motion_action_plan

HERE = Path(__file__).resolve().parent
SCENE_DESCRIPTION_FILE_PATH = HERE / "scene_description_1.py"


@unique
class ObservationType(Enum):
    """Observation type specifies whether the environment step returns RGB images or a defined state"""

    RGB = 0
    STATE = 1
    RGBD = 2


@unique
class Phase(Enum):
    GRASP = 0
    TOUCH = 1
    DONE = 2
    ANY = 3

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(self._name_)


@unique
class CollisionEffect(Enum):
    """Collision effect specifies how collisions are punished. Proportial to the number of collisions, constant for collision/not collison, or ending the episode."""

    PROPORTIONAL = 0
    CONSTANT = 1
    FAILURE = 2


@unique
class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


class SoftBodyManipulationEnv(SofaEnv):
    """Grasp Lift and Touch Environment

    The goal of this environment is to grasp the infundibulum of a intestine, lift the intestine to expose a target point,
    and touch the target point with a electrogripper2y hook.

    Notes:
        - We only consider a grasp successful if it was established on the infundibulum of the intestine.

    Args:
        scene_path (Union[str, Path]): Path to the scene description script that contains this environment's ``createScene`` function.
        image_shape (Tuple[int, int]): Height and Width of the rendered images.
        observation_type (ObservationType): Whether to return RGB images or an array of states as the observation.
        time_step (float): size of simulation time step in seconds (default: 0.01).
        frame_skip (int): number of simulation time steps taken (call ``_do_action`` and advance simulation) each time step is called (default: 1).
        settle_steps (int): How many steps to simulate without returning an observation after resetting the environment.
        render_mode (RenderMode): Create a window (``RenderMode.HUMAN``), run headless (``RenderMode.HEADLESS``), or do not create a render buffer at all (``RenderMode.NONE``).
        render_framework (RenderFramework): choose between pyglet and pygame for rendering
        start_in_phase (Phase): The phase the environment starts in for the reset.
        end_in_phase (Phase): The phase the environment ends in for the done signal.
        tool_collision_distance (float): Distance in mm at which the tool is considered to be in collision with each other.
        goal_tolerance (float): Distance between gripper2 and target point in mm at which the goal is considered to be reached.
        create_scene_kwargs (Optional[dict]): A dictionary to pass additional keyword arguments to the ``createScene`` function.
        maximum_state_velocity (Union[np.ndarray, float]): Velocity in deg/s for pts and mm/s for d in state space which are applied with a normalized action of value 1.
        discrete_action_magnitude (Union[np.ndarray, float]): Discrete change in state space in deg/s for pts and mm/s for d.
        individual_agents (bool): Wether the instruments are individually (dictionary action space).
        individual_rewards (bool): Whether the rewards are assigned individually to each instrument (dictionary reward).
        action_type (ActionType): Discrete or continuous actions to define the action space of the environment.
        on_reset_callbacks (Optional[List[Callable]]): Functions that are called as the last step of the ``env.reset()`` function.
        reward_amount_dict (dict): Dictionary to weigh the components of the reward function.
        collision_punish_mode (CollisionEffect): How to punish collisions.
        losing_grasp_ends_episode (bool): Whether losing the grasp ends the episode.
        num_intestine_tracking_points (int): Number of points on the intestine to track for state observations.
        max_intestine_liver_overlap (float): Amount of overlap between the intestine and the liver to accept before blocking the done signal.
        intestine_force_scaling_factor (float): Scaling factor for the force measured in the intestine (see reward function).
        limit_grasping_attempts_in_reset (bool): Whether to limit the number of grasping attempts in the reset function, if the starting phase is past the grasping phase.
    """

    def __init__(
        self,
        scene_path: Union[str, Path] = SCENE_DESCRIPTION_FILE_PATH,
        image_shape: Tuple[int, int] = (64, 64),
        observation_type: ObservationType = ObservationType.RGB,
        time_step: float = 0.1,
        frame_skip: int = 1,
        settle_steps: int = 10,
        render_mode: RenderMode = RenderMode.HEADLESS,
        render_framework: RenderFramework = RenderFramework.PYGLET,
        start_in_phase: Phase = Phase.GRASP,
        end_in_phase: Phase = Phase.DONE,
        tool_collision_distance: float = 5.0,
        goal_tolerance: float = 5.0,
        create_scene_kwargs: Optional[dict] = None,
        maximum_state_velocity: Union[np.ndarray, float] = np.array([15.0, 15.0, 25.0, 25.0, 20.0]),
        discrete_action_magnitude: Union[np.ndarray, float] = np.array([10.0, 10.0, 15.0, 15.0, 12.0]),
        individual_agents: bool = False,
        individual_rewards: bool = False,
        action_type: ActionType = ActionType.CONTINUOUS,
        on_reset_callbacks: Optional[List[Callable]] = None,
        reward_amount_dict: dict =  {
            Phase.ANY: {
                "intestine_is_grasped": 20.0,
                "new_grasp_on_intestine": 10.0,
                "lost_grasp_on_intestine": -10.0,
                "active_grasping_springs": 1.0,
                "delta_active_grasping_springs": 1.0,
                "dynamic_force_on_intestine": -0.003,
                "successful_task": 1000.0,
                "failed_task": -1.0,
                "gripper1_action_violated_state_limits": -1.0,
                "gripper1_action_violated_cartesian_workspace": -1.0,
                "phase_change": 10.0,
            },
            Phase.GRASP: {
                "distance_gripper1_graspable_region": -0.01,
                "delta_distance_gripper1_graspable_region": -1.0,
            },
            Phase.TOUCH: {
                "img_to_img":10.0,
            },
        },
        collision_punish_mode: CollisionEffect = CollisionEffect.CONSTANT,
        losing_grasp_ends_episode: bool = False,
        num_intestine_tracking_points: int = 10,
        randomize_intestine_tracking_points: bool = True,
        max_intestine_liver_overlap: float = 0.1,
        intestine_force_scaling_factor: float = 1e-9,
        limit_grasping_attempts_in_reset: bool = False,
    ):
        # Pass image shape to the scene creation function
        if not isinstance(create_scene_kwargs, dict):
            create_scene_kwargs = {}
        create_scene_kwargs["image_shape"] = image_shape

        if render_mode == RenderMode.NONE:
            raise ValueError("This environment does not support render mode NONE, use HEADLESS instead. We need to render the scene to check if the target is visible to the camera which is part of the LIFT phase of the task.")

        super().__init__(
            scene_path=scene_path,
            time_step=time_step,
            frame_skip=frame_skip,
            render_mode=render_mode,
            render_framework=render_framework,
            create_scene_kwargs=create_scene_kwargs,
        )
        self.goal_img = cv2.resize(cv2.imread('/home/sofa/SOFA_v23.06.00/bin/lapgym/sofa_env/sofa_env/scenes/soft_body_manipulation/exit_image_2.png'), (512, 512))
        # Initialize the BRIEF extractor
        self.star = cv2.xfeatures2d.StarDetector_create()
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.kp1, self.desc1 = self.brief.compute(self.goal_img, self.star.detect(self.goal_img, None))
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # In which phase of the task to start. Enables skipping the grasping / lifting
        self.starting_phase = start_in_phase
        self.limit_grasping_attempts_in_reset = limit_grasping_attempts_in_reset
        # Buffer for storing the previous rewards/errors
        self.reward_buffer = [None] * 10  # Initialize with None to indicate unpopulated values
        self.current_step = 0  # Track the current step for indexing

        # In which phase of the task to end. Enables skipping lifting / touching
        if end_in_phase <= start_in_phase:
            raise ValueError(f"The end phase should be after the selected start phase. {start_in_phase=}    {end_in_phase=}")
        else:
            self.ending_phase: Phase = end_in_phase

        # How many simulation steps to wait before starting the episode
        self._settle_steps = settle_steps

        # Observation type of the env observation_space
        self.observation_type = observation_type

        # How close the tool axis have to be to each other to be considered in collision
        self.tool_collision_distance = tool_collision_distance

        # How are collisions punished? (proportional to the number of contacts, constant value, ending an episode early)
        self.collision_punish_mode = collision_punish_mode

        # How close the tool needs to be to the target to be considered successful
        self.goal_tolerance = goal_tolerance

        # All parameters of the reward function that are not passed default to 0.0
        self.reward_amount_dict = defaultdict(dict,{phase: defaultdict(float, **reward_dict) for phase, reward_dict in reward_amount_dict.items()})
        # Split the reward parts for the individual agents
        self.individual_rewards = individual_rewards
        self.agent_specific_rewards = defaultdict(float)

        # Does losing a grasp end an episode early?
        self.losing_grasp_ends_episode = losing_grasp_ends_episode

        # How much to scale down the forces read from the intestine MechanicalObject, before passing them to the reward function
        self.intestine_force_scaling_factor = intestine_force_scaling_factor

        # Infos per episode
        self.episode_info = defaultdict(float)

        # Infos from the reward
        self.reward_info = {}
        self.reward_features = {}

        # Callback functions called on reset
        self.on_reset_callbacks = on_reset_callbacks if on_reset_callbacks is not None else []

        ###########################
        # Set up observation spaces
        ###########################

        # How many points on the intestine surface to include in the state observations
        self.num_intestine_tracking_points = num_intestine_tracking_points
        self.randomize_intestine_tracking_points = randomize_intestine_tracking_points
        if self.observation_type == ObservationType.STATE:
            # pose of the gripper1 -> 7
            # pose of the gripper2 -> 7
            # ptsda_state gripper1 -> 5
            # ptsda_state gripper2 -> 5
            # target position -> 3
            # gripper1 has grasped -> 1
            # phase -> 1
            # tracking points on the intestine surface
            observations_size = 44#7 + 7 + 5 + 5 + 3 + 1 + 1 + 3 * self.num_intestine_tracking_points
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observations_size,), dtype=np.float32)
        elif self.observation_type == ObservationType.RGB:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (3,), dtype=np.uint8)

        elif self.observation_type == ObservationType.RGBD:
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape + (4,), dtype=np.uint8)

        else:
            raise ValueError(f"Unknown observation type {self.observation_type}.")

        ######################
        # Set up action spaces
        ######################
        action_dimensionality = 5
        self._maximum_state_velocity = maximum_state_velocity
        self._discrete_action_magnitude = discrete_action_magnitude
        if individual_agents:
            self._do_action = self._do_action_dict
            if action_type == ActionType.CONTINUOUS:
                self.action_space = spaces.Dict(
                    {
                        "gripper1": spaces.Box(low=-1.0, high=1.0, shape=(action_dimensionality,), dtype=np.float32),
                    }
                )
                self._scale_action = self._scale_continuous_action
            else:
                self.action_space = spaces.Dict(
                    {
                        "gripper1": spaces.Discrete(action_dimensionality * 2 + 1)
                    }
                )

                self._scale_action = self._scale_discrete_action
                if isinstance(discrete_action_magnitude, np.ndarray):
                    if not len(discrete_action_magnitude) == action_dimensionality:
                        raise ValueError("If you want to use individual discrete action step sizes per action dimension, please pass an array of length {action_dimensionality} as discrete_action_magnitude. Received {discrete_action_magnitude=} with lenght {len(discrete_action_magnitude)}.")

                # [step, 0, 0, ...], [-step, 0, 0, ...], [0, step, 0, ...], [0, -step, 0, ...]
                action_list = []
                for i in range(action_dimensionality * 2):
                    action = [0.0] * action_dimensionality
                    step_size = discrete_action_magnitude if isinstance(discrete_action_magnitude, float) else discrete_action_magnitude[int(i / 2)]
                    action[int(i / 2)] = (1 - 2 * (i % 2)) * step_size
                    action_list.append(action)

                # Noop action
                action_list.append([0.0] * action_dimensionality)

                self._discrete_action_lookup = np.array(action_list)
                self._discrete_action_lookup *= self.time_step
                self._discrete_action_lookup.flags.writeable = False
        else:
            self._do_action = self._do_action_array
            if action_type == ActionType.CONTINUOUS:
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
                self._scale_action = self._scale_continuous_action
            else:
                raise NotImplementedError("GraspLiftTouchEnv currently only supports continuous actions for non-individual agents.")

    def _do_action(self, action) -> None:
        """Only defined to satisfy ABC."""
        pass
    def _update_reward_buffer(self, new_error: float):
        """Updates the reward buffer with the latest error."""
        self.reward_buffer[self.current_step % 10] = new_error  # Circular buffer update
        self.current_step += 1

    def _do_action_dict(self, action: Dict[str, np.ndarray]) -> None:
        """Apply action to the simulation."""
        self.gripper1.do_action(self._scale_action(action["gripper1"]))

    def _do_action_array(self, action: np.ndarray) -> None:
        """Apply action to the simulation."""
        self.gripper1.do_action(self._scale_action(action[:5]))

    def _scale_discrete_action(self, action: int) -> np.ndarray:
        """Maps action indices to a motion delta."""
        return self._discrete_action_lookup[action]

    def _scale_continuous_action(self, action: np.ndarray) -> np.ndarray:
        """
        Policy is output is clipped to [-1, 1] e.g. [-0.3, 0.8, 1].
        We want to scale that to the maximum velocities defined
        in ``maximum_state_velocity`` in [angle, angle, angle, mm, angle | activation] / step.
        and further to per second (because delta T is not 1 second).
        """
        return self.time_step * self._maximum_state_velocity *action

    def _init_sim(self) -> None:
        super()._init_sim()
        self.gripper1: Gripper1 = self.scene_creation_result["interactive_objects"]["gripper1"]
        self.intestine: Intestine = self.scene_creation_result["intestine"]
        self.contact_listener: Dict[str, Sofa.Core.ContactListener] = self.scene_creation_result["contact_listener"]

        self._distance_normalization_factor = 1.0 / np.linalg.norm(self.gripper1.cartesian_workspace["high"] - self.gripper1.cartesian_workspace["low"])

        #seeds = self.seed_sequence.spawn(3)
        #self.gripper1.seed(seed=seeds[0])

        # Select indices on the intestine surface as tracking points for state observations
        self.intestine_surface_mechanical_object = self.intestine.deformable_object.collision_model_node.MechanicalObject
        intestine_surface_points = self.intestine_surface_mechanical_object.position.array()
        first_intestine_tracking_point_index = self.rng.integers(low=0, high=len(intestine_surface_points), endpoint=False)
        tracking_point_indices = farthest_point_sampling(
            intestine_surface_points,
            self.num_intestine_tracking_points,
            starting_point_index=first_intestine_tracking_point_index,
            return_indices=True,
        )
        self.intestine_tracking_indices = tracking_point_indices

    def step(self, action: Any) -> Tuple[Union[np.ndarray, dict], float, bool, dict]:
        """Step function of the environment that applies the action to the simulation and returns observation, reward, done signal, and info."""

        image_observation = super().step(action)
        observation = self._get_observation(image_observation)
        reward = self._get_reward(image=image_observation)
        terminated = self.active_phase == self.ending_phase
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _get_info(self) -> dict:
        """Assemble the info dictionary."""

        self.info = {}
        self.info["final_phase"] = self.active_phase.value

        for key, value in self.reward_info.items():
            # shortens 'reward_delta_gripper1_distance_to_torus_tracking_points'
            # to 'ret_del_gri_dis_to_tor_tra_poi'
            words = key.split("_")[1:]
            shortened_key = reduce(lambda x, y: x + "_" + y[:3], words, "ret")
            self.episode_info[shortened_key] += value
        #print(self.episode_info)

        return {**self.info, **self.reward_info, **self.episode_info, **self.reward_features, **self.agent_specific_rewards}

    def _get_observation(self, image_observation: Union[np.ndarray, None]) -> Union[np.ndarray, dict]:
        if self.observation_type == ObservationType.RGB:
            return image_observation.copy()
        elif self.observation_type == ObservationType.RGBD:
            observation = self.observation_space.sample()
            observation[:, :, :3] = image_observation.copy()
            observation[:, :, 3:] = self.get_depth()
            return observation
        else:
            state_dict = {}
            state_dict["gripper1_pose"] = self.gripper1.get_physical_pose()
            state_dict["gripper1_ptsda"] = self.gripper1.get_ptsda_state()
            state_dict["gripper1_has_grasped"] = np.asarray(float(self.gripper1.grasp_established))[None]  # 1 -> [1]
            state_dict["phase"] = np.asarray(float(self.active_phase.value))[None]  # 1 -> [1]
            state_dict["intestine_tracking_points"] = self.intestine_surface_mechanical_object.position.array()[self.intestine_tracking_indices].ravel()
            return np.concatenate(tuple(state_dict.values()))

    def reset(self, seed: Union[int, np.random.SeedSequence, None] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Union[np.ndarray, None], Dict]:
        super().reset(seed)
        
        # Seed the instruments
        if self.unconsumed_seed:
            seeds = self.seed_sequence.spawn(1)
            self.gripper1.seed(seed=seeds[0])

            # Select indices on the intestine surface as tracking points for state observations
            self.intestine_surface_mechanical_object = self.intestine.deformable_object.collision_model_node.MechanicalObject
            intestine_surface_points = self.intestine_surface_mechanical_object.position.array()
            if self.randomize_intestine_tracking_points:
                first_intestine_tracking_point_index = self.rng.integers(low=0, high=len(intestine_surface_points), endpoint=False)
            else:
                first_intestine_tracking_point_index = 0
            tracking_point_indices = farthest_point_sampling(
                intestine_surface_points,
                self.num_intestine_tracking_points,
                starting_point_index=first_intestine_tracking_point_index,
                return_indices=True,
            )
            self.intestine_tracking_indices = tracking_point_indices
            self.unconsumed_seed = False




        # Reset the phase to the one set on environment creation
        self.active_phase = self.starting_phase

        # Reset Point of Interest

        # Reset tools
        self.gripper1.reset_gripper()

        # Save the indices that describe indices on the intestine that might be suitable for grasping
        self.intestine_indices_in_grasping_roi = self.intestine.graspable_region_box.indices.array().copy()
        self.intestine_indices_in_grasping_roi_collision = self.intestine.graspable_region_box_collision.indices.array().copy()
        
        if self.active_phase >= Phase.TOUCH:#change_
            if self.active_phase >= Phase.DONE:
                raise ValueError(f"Was asked to reset the environment to {self.active_phase}. This is not possible.")


        # Execute any callbacks passed to the env.
        for callback in self.on_reset_callbacks:
            callback(self)

        # Animate several timesteps without actions until simulation settles
        for _ in range(self._settle_steps):
            self.sofa_simulation.animate(self._sofa_root_node, self._sofa_root_node.getDt())

        # Fill reward features dict with initial values to correctly calculate "delta" features
        grasp_region_center = np.mean(self.intestine.deformable_object.node.MechanicalObject.position.array()[self.intestine_indices_in_grasping_roi], axis=0)

        self.reward_features = {
            "distance_gripper1_graspable_region": np.linalg.norm(self.gripper1.get_grasp_center_position() - grasp_region_center),
            "active_grasping_springs": self.gripper1.get_number_of_springs(),
            "gripper1_distance_to_trocar": np.linalg.norm(self.gripper1.remote_center_of_motion[:3] - self.gripper1.get_pose()[:3]),
        }
        self.reward_features["intestine_is_grasped"] = self.gripper1.grasp_established
        # Restrict grasps to ones that are performed on the infundibulum of the intestine
        #if np.any(np.in1d(self.gripper1.grasping_force_field["shaft_1"].points.array(), self.intestine_indices_in_grasping_roi_collision)) and np.any(np.in1d(self.gripper1.grasping_force_field["jaw_1"].points.array(), self.intestine_indices_in_grasping_roi_collision)):
        #    self.reward_features["intestine_is_grasped"] = self.gripper1.grasp_established
        #else:
        #    self.reward_features["intestine_is_grasped"] = False

        # Clear the episode info values
        for key in self.episode_info:
            self.episode_info[key] = 0.0

        # Clear the agent specific rewards
        for key in self.agent_specific_rewards:
            self.agent_specific_rewards[key] = 0.0

        return self._get_observation(self._maybe_update_rgb_buffer()),{}

    def _get_reward(self, image: np.ndarray) -> float:
        """Retrieve the reward features and scale them with the ``reward_amount_dict``."""
        reward = 0.0
        self.reward_info = {}

        # Calculate the features that describe the reward of the current step
        previous_reward_features = self.reward_features
        reward_features = self._get_reward_features(image, previous_reward_features=self.reward_features)
        self.reward_features = reward_features.copy()
        previous_phase = self.active_phase

        if reward_features["intestine_is_grasped"]:
            if not self.active_phase == Phase.DONE:
                self.active_phase = Phase.TOUCH
        else:
            self.active_phase = Phase.GRASP

        # Figuring out whether the task is done
        if self.active_phase == Phase.TOUCH:
            # Replace the img_to_img reward logic with exponential decay logic
            current_error = abs(reward_features.get("img_to_img", 1000))  # Default to a high error
            self._update_reward_buffer(current_error)

            # Parameters for exponential decay
            alpha, beta = 1.0, 0.5  # Error weights
            gamma, delta = 1.0, 0.3  # Gradient weights

            # Compute normalization factors based on available steps
            num_steps = min(self.current_step, 10)  # Limit steps to available frames
            Z_delta = alpha * (1 - np.exp(-beta * num_steps)) / (1 - np.exp(-beta))
            Z_gradient = gamma * (1 - np.exp(-delta * num_steps)) / (1 - np.exp(-delta))

            # Compute exponential decay reward
            img_to_img_reward = 0.0
            for i in range(num_steps):
                index = (self.current_step - 1 - i) % 10  # Circular buffer indexing
                past_error = self.reward_buffer[index]
                if past_error is None:  # Skip uninitialized entries
                    continue

                if i == 0:  # Current error
                    weight_error = alpha * np.exp(-beta * i) / Z_delta
                    img_to_img_reward += weight_error * (-past_error)
                else:  # Gradient (rate of change)
                    prev_index = (index - 1) % 10
                    if self.reward_buffer[prev_index] is not None:  # Ensure previous value exists
                        gradient = past_error - self.reward_buffer[prev_index]
                        weight_gradient = gamma * np.exp(-delta * i) / Z_gradient
                        img_to_img_reward += weight_gradient * (-gradient)

            reward_features["img_to_img"] = img_to_img_reward

            # Check for task success based on error threshold
            if current_error < 15.0:
                self.active_phase = Phase.DONE

        # Giving the successful task reward, if the ending phase is reached
        if self.active_phase == self.ending_phase:
            reward_features["successful_task"] = True
        else:
            reward_features["successful_task"] = False

        # Dictionary of the active phase overwrites the values of the ANY phase (original dict is not changed)
        reward_amount_dict = self.reward_amount_dict[Phase.ANY]
        l1 = self.reward_amount_dict[self.active_phase].keys()
        l2 = self.reward_amount_dict[Phase.ANY].keys()
        for key in l1:
            if key in l2:
                reward_amount_dict[key] = self.reward_amount_dict[self.active_phase][key]
            else:
                reward_amount_dict[key] = self.reward_amount_dict[self.active_phase][key]

        # Figure out if the episode is failed because of losing a grasp
        if self.losing_grasp_ends_episode and reward_features["lost_grasp_on_intestine"]:
            reward_features["failed_task"] = True
            self.active_phase = Phase.DONE
        else:
            reward_features["failed_task"] = False

        # Check if there was a positive or negative change of the phase
        phase_change = self.active_phase.value - previous_phase.value
        reward_features["phase_change"] = phase_change

        # Figure out how collisions should be punished (and if the episode is failed because of them)
        collision_reward_features = dict(filter(lambda item: "collision" in item[0], reward_features.items()))
        if self.collision_punish_mode == CollisionEffect.CONSTANT:
            for key, value in collision_reward_features.items():
                reward_features[key] = min(1, value)
        elif self.collision_punish_mode == CollisionEffect.FAILURE:
            reward_features["failed_task"] = reward_features["failed_task"] | any(np.fromiter(collision_reward_features.values(), dtype=int) > 0)

        for key, value in reward_features.items():
            # Normalize distance and velocity features with the size of the workspace
            if "distance" in key or "velocity" in key:
                value = value  # self._distance_normalization_factor * value

            # Aggregate the features with their specific factors
            self.reward_info[f"reward_{key}"] = reward_amount_dict[key] * value

            # Add them to the reward
            reward += self.reward_info[f"reward_{key}"]

            # Aggregate the agent-specific rewards, if necessary
            if self.individual_rewards:
                if "success" in key or "failed" in key:
                    self.agent_specific_rewards["gripper1_reward"] += self.reward_info[f"reward_{key}"]
                else:
                    self.agent_specific_rewards["gripper1_reward"] += self.reward_info[f"reward_{key}"]

        self.reward_info["reward"] = reward
        self.reward_info["successful_task"] = float(reward_features["successful_task"])
        return float(reward)


    def _get_reward_features(self, image: np.ndarray, previous_reward_features: dict) -> dict:
        reward_features = {}

        # Collisions

        # State and workspace limits
        reward_features["gripper1_action_violated_state_limits"] = self.gripper1.last_set_state_violated_state_limits
        reward_features["gripper1_action_violated_cartesian_workspace"] = self.gripper1.last_set_state_violated_workspace_limits

        # gripper2

        # gripper1
        # Target visibility

        # Distance to grasp region
        gripper1_position = self.gripper1.get_grasp_center_position()
        grasp_region_center = np.mean(self.intestine.deformable_object.node.MechanicalObject.position.array()[self.intestine_indices_in_grasping_roi], axis=0)
        reward_features["distance_gripper1_graspable_region"] = np.linalg.norm(gripper1_position - grasp_region_center)
        reward_features["delta_distance_gripper1_graspable_region"] = reward_features["distance_gripper1_graspable_region"] - previous_reward_features["distance_gripper1_graspable_region"]

        # Grasping
        # Restrict grasps to ones that are performed on the infundibulum of the intestine
        reward_features["intestine_is_grasped"] = self.gripper1.grasp_established
        reward_features["new_grasp_on_intestine"] = reward_features["intestine_is_grasped"] and not previous_reward_features["intestine_is_grasped"]
        #if np.any(np.in1d(self.gripper1.grasping_force_field["shaft_1"].points.array(), self.intestine_indices_in_grasping_roi_collision)) and np.any(np.in1d(self.gripper1.grasping_force_field["jaw_1"].points.array(), self.intestine_indices_in_grasping_roi_collision)):
        #    reward_features["intestine_is_grasped"] = self.gripper1.grasp_established
        #    reward_features["new_grasp_on_intestine"] = reward_features["intestine_is_grasped"] and not previous_reward_features["intestine_is_grasped"]
        #else:
        #    reward_features["intestine_is_grasped"] = False
        #    reward_features["new_grasp_on_intestine"] = reward_features["intestine_is_grasped"] and not previous_reward_features["intestine_is_grasped"]

        reward_features["lost_grasp_on_intestine"] = not reward_features["intestine_is_grasped"] and previous_reward_features["intestine_is_grasped"]
        reward_features["active_grasping_springs"] = self.gripper1.get_number_of_springs()
        reward_features["delta_active_grasping_springs"] = reward_features["active_grasping_springs"] - previous_reward_features["active_grasping_springs"]

        # Lifting
        distance_to_trocar = np.linalg.norm(self.gripper1.remote_center_of_motion[:3] - gripper1_position)
        reward_features["gripper1_distance_to_trocar"] = distance_to_trocar
        reward_features["gripper1_pulls_intestine_out"] = reward_features["intestine_is_grasped"] * (previous_reward_features["gripper1_distance_to_trocar"] - reward_features["gripper1_distance_to_trocar"])


        # Excerting dynamic force on the intestine
        reward_features["dynamic_force_on_intestine"] = self.intestine.get_internal_force_magnitude() * self.intestine_force_scaling_factor
        #print('GOAL_IMAGE_SHAPE',self.goal_img.shape)
        #print(image.shape)
        '''
        kp2, desc2 = self.brief.compute(image, self.star.detect(image, None))
        matches = self.bf.match(self.desc1, desc2)
        avg_distance = sum(m.distance for m in matches) / len(matches) if matches else 1000
        reward_features["img_to_img"] = -1*avg_distance*self.gripper1.grasp_established
        '''
        try:
            kp2, desc2 = self.brief.compute(image, self.star.detect(image, None))
            matches = self.bf.match(self.desc1, desc2)
            avg_distance = sum(m.distance for m in matches) / len(matches) if matches else 1000
            reward_features["img_to_img"] = -1*avg_distance*self.gripper1.grasp_established
        except:
            cv2.imwrite("/home/saketh/SOFA/v23.06.00/bin/lapgym/sofa_env/sofa_env/scenes/soft_body_manipulation/error_image_2.png",image)
            reward_features["img_to_img"] = previous_reward_features["img_to_img"]
        
        
        #print(reward_features)

        return reward_features


if __name__ == "__main__":
    import time

    env = SoftBodyManipulationEnv(
        render_mode=RenderMode.HUMAN,
        start_in_phase=Phase.TOUCH,
        end_in_phase=Phase.DONE,
        image_shape=(512, 512),
    )

    env.reset()

    while True:
        start = time.time()
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            #print(env.active_phase)
        env.reset()
        end = time.time()
