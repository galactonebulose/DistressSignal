import cv2
import requests
import time

# Replace with your actual server URL
server_url = 'http://34.0.6.49:5000/process_image'

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Cannot open camera")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to capture frame")
            break

        # Encode image to JPEG
        _, img_encoded = cv2.imencode('.jpg', frame)

        # Prepare POST request
        files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

        try:
            response = requests.post(server_url, files=files, timeout=5)
            print("Server Response:", response.json())
        except requests.exceptions.RequestException as e:
            print("Error sending frame:", e)

        # Optional delay (adjust FPS)
        time.sleep(0.1)  # 10 FPS

except KeyboardInterrupt:
    print("\n Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()