import cv2
import socket
import struct
import pickle
import pyrealsense2 as rs
import numpy as np
import dlib
from utils import *
from PIL import Image
from feature import *
import time
import math
import collections

server_ip = '220.149.82.236'
server_port = 4444

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
pipeline.start(config)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

center_init = []
left_init = []
top_init = []
right_init = []
gaze_width_lst =[]
gaze_height_lst = []
target_image_path = f"video6_images.png"
prev_frame_time = 0

frame_num = 0

radius = 15
circle_color = (0, 0, 255)  # Red color
circle_thickness = 2  # Thickness of the circle

#FHD
monitor_pixels = [
    ("17inch", (2417, 1360)),
    ("21inch", (2986, 1680)),
    ("23inch", (3271, 1840)),
    ("27inch", (1920, 1080))
]

selected_monitor = "27inch"
image_width, image_height = next((width, height) for name, (width, height) in monitor_pixels if name == selected_monitor)

target_size = next(size for name, size in monitor_pixels if name == selected_monitor)
monitor_width, monitor_height = target_size

# Load and resize image
image_path = 'C:/Users/jungmin/Desktop/연구실/EYETRACKING/ZIP/gaze-tracking-pipeline/image.png'
test_image = cv2.imread(image_path)
resized_image = cv2.resize(test_image, target_size)

avg_gaze_width = 0
avg_gaze_height = 0
prev_x = 0
prev_y = 0
fps_deque = collections.deque(maxlen=60)

def calculate_diff(x, y, prev_x, prev_y):
    return math.sqrt((x - prev_x)**2 + (y - prev_y)**2)

def apply_geometric_correction(cali_x, cali_y, D, pixel_to_cm_x, pixel_to_cm_y):
    cali_x_cm = cali_x * pixel_to_cm_x
    cali_y_cm = cali_y * pixel_to_cm_y
    corrected_x_cm = (cali_x_cm * D) / np.sqrt((D**2) + (cali_x_cm**2))
    corrected_y_cm = (cali_y_cm * D) / np.sqrt((D**2) + (cali_y_cm**2))
    corrected_x = corrected_x_cm / pixel_to_cm_x
    corrected_y = corrected_y_cm / pixel_to_cm_y
    return corrected_x, corrected_y

gaze_points_file = "gaze_points.txt"
with open(gaze_points_file, "w") as gaze_file:
    try:
        while True:
            frame_num += 1
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
            display = resized_image.copy()

            for (x, y, w, h) in faces:
                dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
                right_eye = landmarks[36:42]
                left_eye = landmarks[42:48]
                face_image = frame[y:y + h, x:x + w]
                right_eye_bbox = get_bounding_box(right_eye)
                left_eye_bbox = get_bounding_box(left_eye)
                ER_image = frame[right_eye_bbox[2]:right_eye_bbox[3], right_eye_bbox[0]:right_eye_bbox[1]]
                EL_image = frame[left_eye_bbox[2]:left_eye_bbox[3], left_eye_bbox[0]:left_eye_bbox[1]]
                face_grid = getfaceGrid(frame, x, y, w, h)
                data = {
                    'face_image': face_image,
                    'ER_image': ER_image,
                    'EL_image': EL_image,
                    'face_grid': face_grid
                }
                # Sending data from local to server
                data_dict = pickle.dumps(data)
                client_socket.sendall(struct.pack(">L", len(data_dict)) + data_dict)
                # Receiving data from server to local
                data_size = struct.unpack(">L", client_socket.recv(4))[0]
                data = b""
                while len(data) < data_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    data += packet
                gaze_value = pickle.loads(data)
                gaze_point = gaze_value

                # Draw gaze point on display image
                if gaze_point is not None and gaze_point.size > 0:
                    # Ensure gaze_point is a 1D array with two elements (x, y)
                    gaze_point = np.array(gaze_point).flatten()

                    if gaze_point.size >= 2:
                        x_gaze = int(gaze_point[0])
                        y_gaze = int(gaze_point[1])
                        cv2.circle(display, (x_gaze, y_gaze), radius, circle_color, circle_thickness)


                gaze_file.write(f"{frame_num}: {gaze_point}\n")

            new_frame_time = time.time()
            fps_deque.append(1 / (new_frame_time - prev_frame_time))
            prev_frame_time = new_frame_time
            if frame_num % 60 == 0:
                print(f'FPS: {np.mean(fps_deque):5.2f}')

            cv2.imshow('Processed Video', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        client_socket.close()
        cv2.destroyAllWindows()
