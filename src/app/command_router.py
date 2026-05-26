from src.network.labview_protocol import LabViewCommand
from src.motion import servo_driver

LABVIEW_BUTTON_COMMANDS = {
    LabViewCommand.UP: servo_driver.Button.UP,
    LabViewCommand.LEFT: servo_driver.Button.LEFT,
    LabViewCommand.SELECT: servo_driver.Button.SELECT,
    LabViewCommand.RIGHT: servo_driver.Button.RIGHT,
    LabViewCommand.BACK: servo_driver.Button.BACK,
    LabViewCommand.DOWN: servo_driver.Button.DOWN,
    LabViewCommand.MENU: servo_driver.Button.MENU,
}

def get_button_for_command(command: LabViewCommand) -> servo_driver.Button | None:
    return LABVIEW_BUTTON_COMMANDS.get(command)
