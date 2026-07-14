import collections
import socket
from xmlrpc import server


# Simulated sequences for the 5 modes (first five rows). Concluding with READ (RUN_OCR equivalent).
# These mimic what LabVIEW would send over TCP/IP during automated mode testing.
_SIMULATED_COMMANDS = collections.deque(
    [
        # MENU TRAVERSAL
        # "RIGHT", "SELECT", "RUN_OCR", "DOWN", "RUN_OCR", "SHUTDOWN",
        # LOCATION
        # "MENU", "RIGHT", "DOWN", "SELECT", "DOWN", "DOWN", "SELECT", "DOWN", "DOWN", "SELECT", "RUN_OCR", "LEFT", "RUN_OCR", "SHUTDOWN",
        # DELETE SCHEDULE
        # "MENU", "RIGHT", "RIGHT", "SELECT",
        # "SELECT", 
        # "SELECT",
        # "RUN_OCR", "SHUTDOWN", # For user schedule 1 name
        # "DOWN", "SELECT", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", 
        # "SELECT", "SELECT", "RUN_OCR", "SHUTDOWN",

        # CHECK OUT ACTIVE FAULTS LIST
        "RIGHT", "SELECT", "RUN_OCR", "BACK", "LEFT",

        # Test schedule 1 name
        "MENU", "RIGHT", "RIGHT", "SELECT", "SELECT", "SELECT", "RUN_OCR",
        "BACK", "BACK", "BACK", "BACK",

        # Test System Status 3 rois
        "MENU", "DOWN", "SELECT", "RUN_OCR", "DOWN", "RUN_OCR", "SHUTDOWN",

        # HEAT_PUMP
        "BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "DOWN", "DOWN", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
        # HYBRID_STANDARD
        "BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
        # HYBRID_PLUS
        "BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "DOWN", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
        # ELECTRIC
        "BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "DOWN", "DOWN", "DOWN", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
        # VACATION
        "BACK", "MENU", "SELECT", "LEFT", "LEFT", "SELECT", "DOWN", "DOWN", "DOWN", "DOWN", "SELECT", "RIGHT", "RIGHT", "RUN_OCR",
        # Finally shutdown test when done
        "SHUTDOWN"
    ]
)


_SERVER_SOCKET = None
conn = None

def start_tcp_server() -> bool:
    """Start TCP server to listen for LabVIEW commands."""
    HOST = '0.0.0.0' # Listen on all interfaces
    PORT = 5000
    global _SERVER_SOCKET, conn

    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        server.settimeout(0.1)
        _SERVER_SOCKET = server
        conn = None
        print(f"[NETWORK] TCP server listening on {HOST}:{PORT}")
        print("[NETWORK] Waiting for connection...")
        return True
    except Exception as exc:
        _SERVER_SOCKET = None
        conn = None
        print(f"[ERROR] Failed to start TCP server on {HOST}:{PORT}: {exc}")
        return False


def _accept_connection_if_needed() -> bool:
    """Accept the first LabVIEW connection without blocking the main loop."""
    global conn
    if conn is not None or _SERVER_SOCKET is None:
        return conn is not None

    try:
        new_conn, addr = _SERVER_SOCKET.accept()
        new_conn.settimeout(0.1)
        conn = new_conn
        print(f"[NETWORK] Connected by {addr}")
        return True
    except socket.timeout:
        return False
    except Exception as exc:
        print(f"[WARNING] TCP accept failed: {exc}")
        return False


_COMMAND_RESPONSES = {
    "UP": "Button_Pressed",
    "SELECT": "Button_Pressed",
    "DOWN": "Button_Pressed",
    "LEFT": "Button_Pressed",
    "RIGHT": "Button_Pressed",
    "BACK": "Button_Pressed",
    "MENU": "Button_Pressed",
    "RUN_OCR": "Running_OCR123",
    "SEND_OCR_RESULT": "Sending_OCR123",
}


def get_next_command(simulated: bool = False) -> str:
    """Receive next command from LabVIEW."""
    # TODO: listen to LabVIEW via Serial or TCP/IP socket
    if simulated:
        if _SIMULATED_COMMANDS:
            command = _SIMULATED_COMMANDS.popleft()
            print(f"[SIMULATION] Replaying command: {command}")
            return command
        print("[SIMULATION] No more simulated commands.")
        return ""  # No more simulated commands

    #if there is a Labview command, capture it with data
    if not _accept_connection_if_needed():
        return ""

    try:
        data = conn.recv(1024)  #1024 byte string limit

        if data:
            command = data.decode().strip()
            print("Received:", command)
            response = _COMMAND_RESPONSES.get(command, "Unknown Command\n")
            conn.sendall(response.encode())
            return data.decode().strip()
        
        else:
            return ""
        
    #ensures that if no data is found in data = conn.recv(1024) line then the program #doesn’t timeout
    except socket.timeout:
        pass 
    except Exception as exc:
        print(f"[WARNING] TCP receive failed: {exc}")
        return ""


#def send_report(ocr_result: str) -> None:
def send_report(ocr_result: str, simulated: bool = False) -> None:
    """Send data back to LabVIEW."""
    # TODO: send ocr_result back to LabVIEW via Serial or TCP/IP socket
    if simulated:
        print(f"[SIMULATION] OCR report ready:\n{ocr_result}")
        return

    if conn is None:
        print("[WARNING] Cannot send OCR report: no LabVIEW connection yet.")
        return

    try:
        response = f"OCR ready\n{ocr_result}\n"
        conn.sendall(response.encode())
        print("[NETWORK] Sent OCR report to LabVIEW.")
    except Exception as exc:
        print(f"[WARNING] Failed to send OCR report: {exc}")
