from enum import Enum
from typing import Optional

class LabViewCommand(Enum):
    READ = "READ"
    SHUTDOWN = "SHUTDOWN"
    UP = "UP"
    LEFT = "LEFT"
    SELECT = "SELECT"
    RIGHT = "RIGHT"
    BACK = "BACK"
    DOWN = "DOWN"
    MENU = "MENU"

LABVIEW_COMMANDS: tuple[str, ...] = tuple(command.value for command in LabViewCommand)

def parse_labview_command(command: str) -> Optional[LabViewCommand]:
    """Normalize a LabVIEW TCP/IP command into a known command token."""
    if not command:
        return None

    normalized = command.strip().upper()
    try:
        return LabViewCommand(normalized)
    except ValueError:
        return None
