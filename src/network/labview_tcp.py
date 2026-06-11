import collections
import socket
from xmlrpc import server


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

def start_tcp_server() -> None:
    """Start TCP server to listen for LabVIEW commands."""
    HOST = '0.0.0.0' # Listen on all interfaces
    PORT = 5000
    global conn

    #Listening and waiting for tcp connection from client
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print("Waiting for connection...")

    #connection from client found
    conn, addr = server.accept()
    print(f"Connected by {addr}")

    #Timeout needed to ensure Pi has time to send a OCR ready response independent #from a Labview Command
    conn.settimeout(0.1) 

    # initialize command
    TCP_Command = ""



#def get_next_command(simulated: bool = False) -> str:
def get_next_command() -> str:
    """Receive next command from LabVIEW."""
    # TODO: listen to LabVIEW via Serial or TCP/IP socket
    #if there is a Labview command, capture it with data
    try:
        data = conn.recv(1024)  #1024 byte string limit

        if data:
                #decodes \n to end string
                command = data.decode().strip()
                print("Received:", command)
        

                # Example command handling
                if command == "UP":
                    response = "UP Button Pressed\n"

                elif command == "SELECT":
                    response = "SELECT Button Pressed\n"

                elif command == "DOWN":
                    response = "DOWN Button Pressed\n"

                elif command == "LEFT":
                    response = "LEFT Button Pressed\n"

                elif command == "RIGHT":
                    response = "RIGHT Button Pressed\n"

                elif command == "BACK":
                    response = "BACK Button Pressed\n"

                elif command == "MENU":
                    response = "MENU Button Pressed\n"

                elif command == "RUN_OCR":
                    response = "Running OCR\n"

                elif command == "SEND_OCR_RESULT":
                    response = "Sending OCR\n"

                else:
                    response = "Unknown Command\n"

                conn.sendall(response.encode())
        
    #ensures that if no data is found in data = conn.recv(1024) line then the program #doesn’t timeout
    except socket.timeout:
        pass 

    #if simulated:
    #    if _SIMULATED_COMMANDS:
    #        return _SIMULATED_COMMANDS.popleft()
     #   return ""  # No more simulated commands
    #return ""

    if data:
        return data
    else:
        return ""

#def send_report(ocr_result: str) -> None:
def send_report(conn: str) -> None:
    """Send data back to LabVIEW."""
    # TODO: send ocr_result back to LabVIEW via Serial or TCP/IP socket
    response = "OCR ready\n"
    conn.sendall(response.encode())
    pass
