"""
Bradford White Corporation (BWC) Water Heater Vision Testing System
Team 14 - Senior Project
./src/main.py - Main Event Loop
"""

import time
import traceback
from enum import Enum, auto

# Placeholder imports for your future modules
# from vision import vision_engine
# from motion import servo_driver

class SystemState(Enum):
    STARTUP = auto()
    WAIT_FOR_COMMAND = auto()
    PRESS_BUTTON = auto()
    READ_DISPLAY = auto()
    REPORT_TO_LABVIEW = auto()
    ERROR = auto()
    SHUTDOWN = auto()

def main():
    """Main Event Loop (The Orchestrator)"""
    
    # initial state on startup
    current_state = SystemState.STARTUP
    last_command = ""
    ocr_result = ""
    error_message = ""
    
    print("--- Starting BWC Water Heater Vision Testing System ---")

    try:
        while True:
            match current_state:
                
                case SystemState.STARTUP:
                    print("[INFO] Homing servos, warming up camera...")
                    # TODO: initialize hardware (home servos, warm up camera, etc.)
                    # servo_driver.home_all()
                    # vision_engine.warm_up()
                    current_state = SystemState.WAIT_FOR_COMMAND
                    
                case SystemState.WAIT_FOR_COMMAND:
                    last_command = "CMD_PRESS_TEMP_UP" # simulated command received from LabVIEW
                    print(f"[NETWORK] Received command from LabVIEW: {last_command}")
                    # TODO: listen to LabVIEW via Serial or TCP/IP socket      
                    if last_command: # string is true if not empty = command received
                        # --- COMMAND ROUTING LOGIC ---
                        if last_command.startswith("CMD_PRESS"):
                            current_state = SystemState.PRESS_BUTTON
                        elif last_command == "CMD_READ_ONLY":
                            current_state = SystemState.READ_DISPLAY
                        elif last_command == "CMD_SHUTDOWN":
                            current_state = SystemState.SHUTDOWN
                        else:
                            print(f"[WARNING] Unknown command from LabVIEW: {last_command}")
                            # just ignore it and wait for a valid one
                            last_command = ""
                        
                case SystemState.PRESS_BUTTON:
                    print(f"[ACTION] Executing command: {last_command}")
                    # TODO: press the button
                    # servo_driver.press_button(last_command)
                    current_state = SystemState.READ_DISPLAY
                    
                case SystemState.READ_DISPLAY:
                    print("[ACTION] Reading UI display...")
                    # TODO: capture image
                    # TODO: process image with Tesseract OCR
                    ocr_result = "120F" # simulated Tesseract output
                    current_state = SystemState.REPORT_TO_LABVIEW
                    
                case SystemState.REPORT_TO_LABVIEW:
                    print(f"[NETWORK] Reporting data to LabVIEW: {ocr_result}")
                    # TODO: send ocr_result back to LabVIEW via Serial or TCP/IP socket
                    # Serial.write(ocr_result)
                    current_state = SystemState.WAIT_FOR_COMMAND
                    
                case SystemState.ERROR:
                    print(f"[FATAL] System Faulted: {error_message}")
                    break
                    
                case SystemState.SHUTDOWN:
                    print("[INFO] LabVIEW requested shutdown. Parking servos, exiting.")
                    # TODO: park servos, release camera resources, etc.
                    # servo_driver.home_all()
                    # vision_engine.shutdown()
                    break
    
    # alternative for case default - this catches ANY Python crash (divide by zero, camera disconnected, etc.)
    except Exception as e:
        error_message = str(e)
        print("\n[EMERGENCY] Unhandled exception caught!")
        traceback.print_exc() # prints exact line number of crash\

        # servo_driver.home_all() 
        
        print("[EMERGENCY] System parked safely. Exiting.")

if __name__ == "__main__":
    main()
