import threading
import evdev
from typing import List

class KeyboardController:
    """Class to interface a keyboard controller."""

    def __init__(self) -> None:
        self.w = False
        self.a = False
        self.s = False
        self.d = False
        self.up = False
        self.down = False
        self.left = False
        self.right = False
        self.q = False
        self.z = False
        self.x = False
        self.c = False
        # self.space = False
        # self.shift = False
        # self.ctrl = False
        # self.alt = False
        # self.enter = False
        self.escape = False

        self.devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        # self.keyboard = []
        # while len(self.keyboard)<1:
        #     self.keyboard = [dev for dev in self.devices if dev.name == 'ITE Tech. Inc. ITE Device(8910) Keyboard']
        
        # self.keyboard = self.keyboard[0]

        self._monitor_thread = threading.Thread(target=self._monitor_keyboard, daemon=True)
        self._monitor_thread.start()

    def read(self) -> List[float]:
        """Reads the current state of the controller."""
        w = float(self.w)
        a = float(self.a)
        s = float(self.s)
        d = float(self.d)
        up = float(self.up)
        down = float(self.down)
        left = float(self.left)
        right = float(self.right)
        q = float(self.q)
        z = float(self.z)
        x = float(self.x)
        c = float(self.c)
        # space = float(self.space)
        # shift = float(self.shift)
        # ctrl = float(self.ctrl)
        # alt = float(self.alt)
        # enter = float(self.enter)
        escape = float(self.escape)

        # return [w, a, s, d, up, down, left, right, space, shift, ctrl, alt, enter, escape]
        return [w, a, s, d, up, down, left, right,q, z, x, c, escape]

    def is_alive(self) -> bool:
        return self._monitor_thread.is_alive()

    def _monitor_keyboard(self):
        """This function is run in a separate thread and constantly monitors the keyboard."""
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        keyboard = [dev for dev in devices if dev.name == 'Dell KB216 Wired Keyboard'][0]
        
        for event in keyboard.read_loop():
            if event.type == evdev.ecodes.EV_KEY:
                key_code = event.code
                key_state = event.value == evdev.KeyEvent.key_down

                if key_code == evdev.ecodes.KEY_W:
                    self.w = key_state
                elif key_code == evdev.ecodes.KEY_A:
                    self.a = key_state
                elif key_code == evdev.ecodes.KEY_S:
                    self.s = key_state
                elif key_code == evdev.ecodes.KEY_D:
                    self.d = key_state
                elif key_code == evdev.ecodes.KEY_UP:
                    self.up = key_state
                elif key_code == evdev.ecodes.KEY_DOWN:
                    self.down = key_state
                elif key_code == evdev.ecodes.KEY_LEFT:
                    self.left = key_state
                elif key_code == evdev.ecodes.KEY_RIGHT:
                    self.right = key_state
                elif key_code == evdev.ecodes.KEY_Q:
                    self.q = key_state
                elif key_code == evdev.ecodes.KEY_Z:
                    self.z = key_state
                elif key_code == evdev.ecodes.KEY_X:
                    self.x = key_state
                elif key_code == evdev.ecodes.KEY_C:
                    self.c = key_state
                # elif key_code == evdev.ecodes.KEY_SPACE:
                #     self.space = key_state
                # elif key_code == evdev.ecodes.KEY_LEFTSHIFT or key_code == evdev.ecodes.KEY_RIGHTSHIFT:
                #     self.shift = key_state
                # elif key_code == evdev.ecodes.KEY_LEFTCTRL or key_code == evdev.ecodes.KEY_RIGHTCTRL:
                #     self.ctrl = key_state
                # elif key_code == evdev.ecodes.KEY_LEFTALT or key_code == evdev.ecodes.KEY_RIGHTALT:
                #     self.alt = key_state
                # elif key_code == evdev.ecodes.KEY_ENTER:
                #     self.enter = key_state
                elif key_code == evdev.ecodes.KEY_ESC:
                    self.escape = key_state

# controller = KeyboardController()

# Continuously print the state of the keyboard
# while controller.is_alive():
#     keyboard_state = controller.read()
#     print(keyboard_state)
             