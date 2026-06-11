import socket

HOST = '0.0.0.0' # Listen on all interfaces
PORT = 5000

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

ocr_result = True # Placeholder for OCR result status, should be set to True when OCR is ready
ocr_sent = False
command = ""

while True:

#checking if ocr_result is ready from camera vision system
    if ocr_result and not ocr_sent:
        response = "OCR ready\n"
        conn.sendall(response.encode())
        ocr_sent = True

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

    


#conn.close() 
