import numpy as np

from enum import Enum, unique

import Sofa
import Sofa.Core

from sofa_env.scenes.grasp_lift_touch.sofa_objects.cauter import Cauter
from sofa_env.scenes.grasp_lift_touch.sofa_objects.gripper import Gripper


@unique
class Tool(Enum):
    GRIPPER = 0
    CAUTER = 1


def switchTool(tool: Tool):
    return Tool.CAUTER if tool == Tool.GRIPPER else Tool.GRIPPER


class ToolController(Sofa.Core.Controller):
    def __init__(self, gripper: Gripper, cauter: Cauter) -> None:
        super().__init__()
        self.name = "ToolController"
        self.gripper = gripper
        self.cauter = cauter
        self.action = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0])
        self.active_tool = Tool.GRIPPER

def onKeypressedEvent(self, event):
    key = event["key"]
    self.bool1 = True
    zero_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    if ord(key) == 1:  # Tab
        self.bool1 = not self.bool1  # Toggle the value of self.bool
        self.active_tool = switchTool(tool=self.active_tool)
    if ord(key) == 32:  # Space
        self.print_tool_state()
    if ord(key) == 19:  # up
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    elif ord(key) == 21:  # down
        action = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    elif ord(key) == 18:  # left
        action = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    elif ord(key) == 20:  # right
        action = np.array([0.0, -1.0, 0.0, 0.0, 0.0])
    elif key == "T":
        action = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    elif key == "G":
        action = np.array([0.0, 0.0, -1.0, 0.0, 0.0])
    elif key == "V":
        action = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    elif key == "D":
        action = np.array([0.0, 0.0, 0.0, -1.0, 0.0])
    elif key == "K":
        action = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    elif key == "L":
        action = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
    else:
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    if self.bool1:
        self.action = np.concatenate((zero_array, action))
        print('current',self.action)
    else:
        self.action = np.concatenate((action, zero_array))
        print('current',self.action)
    return 

    def print_tool_state(self):
        ptsd_state_gripper: np.ndarray = self.gripper.ptsd_state
        ptsd_state_cauter: np.ndarray = self.cauter.ptsd_state

        print("==================================================")
        print("Gripper Tool State")
        print(f"pan: {ptsd_state_gripper[0]}, tilt: {ptsd_state_gripper[1]}, spin: {ptsd_state_gripper[2]}, depth: {ptsd_state_gripper[3]}")
        print("Cauter Tool State")
        print(f"pan: {ptsd_state_cauter[0]}, tilt: {ptsd_state_cauter[1]}, spin: {ptsd_state_cauter[2]}, depth: {ptsd_state_cauter[3]}")
        print("==================================================")

    def do_action(self, action: np.ndarray):
        if self.active_tool == Tool.GRIPPER:
            self.gripper.do_action(action=action)
        if self.active_tool == Tool.CAUTER:
            self.cauter.do_action(action=action)
