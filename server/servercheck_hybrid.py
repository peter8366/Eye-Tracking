import socket
import collections
import time
from argparse import ArgumentParser
import struct
import pickle

import albumentations as A
import cv2
import mediapipe as mp
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import traceback
import zlib

from model import Model
from mpii_face_gaze_preprocessing import normalize_single_image
from utils import get_camera_matrix, get_face_landmarks_in_ccs, gaze_2d_to_3d, ray_plane_intersection, plane_equation, get_monitor_dimensions, get_point_on_screen
from visualization import Plot3DScene
from webcam import WebcamSource

# 서버에서부터 데이터 받는 함수 
def recv_data(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

# 서버 IP 주소와 포트 번호
server_ip = '0.0.0.0'
server_port = 4444

# 모델 저장 위치 
model_path = "./pretrained_model/p00.ckpt"
#model_path = "./pretrained_model/epoch=49-step=47500.ckpt"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 소켓 생성, binding
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(10)

#모니터 정보 -27인치 FHD 기준 
monitor_mm = (597.7, 336.2)
monitor_pixels = (1920,1080)
# monitor_pixels = (3840, 2160) - 4k
# monitor_pixel = (2560,1440) - QHD

#모니터 옵션  - 모니터 픽셀은 4k 기준 

#23인치 
# monitor_mm = (509.7,286.8)
# monitor_pixels  = (3271, 1840)

# #21인치
# monitor_mm = (464.1, 261.2)
# monitor_pixels = (2986, 1680)

#17인치
# monitor_mm = (376.4,211.6)
# monitor_pixels = (2417, 1360)


# 필요없을시 지우기 
#parameter
# smoothing_buffer = collections.deque(maxlen=3)
# rvec_buffer = collections.deque(maxlen=3)
# tvec_buffer = collections.deque(maxlen=3)
# gaze_vector_buffer = collections.deque(maxlen=10)
# rvec, tvec = None, None
# plot_3d_scene = Plot3DScene(face_model, monitor_mm[0], monitor_mm[1], 20) if visualize_3d else None

def main():
    # 모델 불러오기 
    model = Model.load_from_checkpoint(model_path).to(device)
    model.eval()
    print("서버가 시작되었습니다. 연결을 기다립니다...")

    try:
        while True:
            client_socket, addr = server_socket.accept()
            print(f"클라이언트 연결됨: {addr}")
        
            while True: 
                # 데이터 크기 수신 (4바이트)
                data_size_data = recv_data(client_socket, 4)
                if data_size_data is None:
                    print("데이터 수신 오류 또는 연결 끊김.")
                    break

                data_size = struct.unpack(">L", data_size_data)[0]

                # 데이터 수신
                data = recv_data(client_socket, data_size)
                if data is None:
                    print("데이터 수신 오류 또는 연결 끊김.")
                    break

                # 데이터를 역직렬화(pickle)
                serialized_data  = zlib.decompress(data)
                
                received_data = pickle.loads(serialized_data)

                # person_idx, full_face_image, left_eye_image, right_eye_image, face_cente, rotation_matrixr 수신
                person_idx = torch.Tensor(received_data['person_idx']).unsqueeze(0).long().to(device)
                full_face_image = torch.Tensor(received_data['full_face_image']).to(device)
                left_eye_image = torch.Tensor(received_data['left_eye_image']).to(device)
                right_eye_image = torch.Tensor(received_data['right_eye_image']).to(device)
                face_center = torch.Tensor(received_data['face_center']).to(device)
                rotation_matrix = torch.Tensor(received_data['rotation_matrix']).cpu()    
                
                # 모델을 이용한 예측
                with torch.no_grad():
                    output = model(person_idx, full_face_image, right_eye_image, left_eye_image).squeeze(0).cpu().numpy()

                # Gaze vector 계산
                gaze_vector_3d_normalized = gaze_2d_to_3d(output)              
                gaze_vector = np.dot(np.linalg.inv(rotation_matrix.reshape(3,3)), gaze_vector_3d_normalized)
                
                #3d 공간에서의 평면방정식 구함
                plane = plane_equation(np.eye(3), np.asarray([[0], [0], [0]]))  
                plane_w = plane[0:3]
                plane_b = plane[3]


                # 얼굴 중심과의 교차점 계산
                result = ray_plane_intersection(face_center.cpu().numpy().reshape(3), gaze_vector, plane_w, plane_b)

                # 화면 좌표로 변환
                point_on_screen = get_point_on_screen(monitor_mm, monitor_pixels, result)

                # gaze point를 클라이언트로 전송
                response_data = pickle.dumps(point_on_screen)
                client_socket.sendall(struct.pack(">L", len(response_data)) + response_data)
            
    except KeyboardInterrupt:
        print("서버 종료 중...")
    finally:
        server_socket.close()

if __name__ == "__main__":
    main()