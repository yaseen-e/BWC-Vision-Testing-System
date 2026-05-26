from enum import Enum, auto

class SystemState(Enum):
    STARTUP = auto()
    WAIT_FOR_COMMAND = auto()
    PRESS_BUTTON = auto()
    READ_DISPLAY = auto()
    REPORT_TO_LABVIEW = auto()
    ERROR = auto()
    SHUTDOWN = auto()
