import collections

# Simulated sequences for the 5 modes (first five rows). Concluding with READ (RUN_OCR equivalent).
# These mimic what LabVIEW would send over TCP/IP during automated mode testing.
_SIMULATED_COMMANDS = collections.deque(
    [
        # HEAT_PUMP
        "BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "DOWN", "DOWN", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
        # HYBRID_STANDARD
        "BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
        # HYBRID_PLUS
        "BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "DOWN", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
        # ELECTRIC(ONLY)
        "BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "DOWN", "DOWN", "DOWN", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
        # VACATION
        "BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "DOWN", "DOWN", "DOWN", "DOWN", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
        # Finally shutdown test when done
        "SHUTDOWN"
    ]
)

def get_next_command(simulated: bool = False) -> str:
    """Receive next command from LabVIEW."""
    # TODO: listen to LabVIEW via Serial or TCP/IP socket
    if simulated:
        if _SIMULATED_COMMANDS:
            return _SIMULATED_COMMANDS.popleft()
        return ""  # No more simulated commands
    return ""

def send_report(ocr_result: str) -> None:
    """Send data back to LabVIEW."""
    # TODO: send ocr_result back to LabVIEW via Serial or TCP/IP socket
    pass
