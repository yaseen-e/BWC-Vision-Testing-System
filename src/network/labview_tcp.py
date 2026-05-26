def get_next_command(simulated: bool = False) -> str:
    """Receive next command from LabVIEW."""
    # TODO: listen to LabVIEW via Serial or TCP/IP socket
    if simulated:
        return "UP"  # Simulated command received from LabVIEW
    return ""

def send_report(ocr_result: str) -> None:
    """Send data back to LabVIEW."""
    # TODO: send ocr_result back to LabVIEW via Serial or TCP/IP socket
    pass
