import socket
import cv2
import pickle
import struct
import dlib
import torch
from utils import *
from pytorch.ITrackerModel import ITrackerModel
import scipy.io as sio
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 서버 IP 주소와 포트 번호
server_ip = '0.0.0.0'
server_port = 4444
imSize = (224,224)
MEAN_PATH = '/data1/kunwoolee/EYE_Tracking/Eye_Tracking_for_Everyone/pytorch'

def loadMetadata(filename, silent = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


faceMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'))['image_mean']
eyeLeftMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
eyeRightMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']

# 소켓 생성 및 바인딩
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(10)



class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """       
        return tensor.sub(self.meanImg)




transformEyeR = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(imSize),
    transforms.ToTensor(),
    SubtractMean(meanImg=eyeRightMean),
])
transformFace = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(imSize),
    transforms.ToTensor(),
    SubtractMean(meanImg=faceMean),
])
transformEyeL = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(imSize),
    transforms.ToTensor(),
    SubtractMean(meanImg=eyeLeftMean),
])

# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

    
def process_data(face_image, ER_image, EL_image, face_grid):
    # 이미지를 224x224로 리사이즈하고 Tensor로 변환
    face_tensor = transformFace(face_image).unsqueeze(0).float()
    ER_tensor = transformEyeR(ER_image).unsqueeze(0).float()
    EL_tensor = transformEyeL(EL_image).unsqueeze(0).float()

    # face_grid는 추가적인 전처리가 필요할 수 있음, 여기서는 단순히 Tensor로 변환
    grid_tensor = torch.tensor(face_grid).unsqueeze(0).float()


    
    # Gaze 좌표 반환
    return face_tensor, ER_tensor, EL_tensor,grid_tensor




def main():
    model = ITrackerModel()
    model.to(device)
    imsize=(224,224)
    print("Load Pretrained Model..\n")
    saved = load_checkpoint()
    state = saved['state_dict']
    model.load_state_dict(state)
    best_prec1 = saved['best_prec1'] 
    model.eval()
    print("서버가 시작되었습니다. 연결을 기다립니다...")
    try:
        while True:
            client_socket, addr = server_socket.accept()
            print(f"클라이언트 연결됨: {addr}")
            
            while True:
                # 클라이언트로부터 프레임 수신
                data_size = struct.unpack(">L", client_socket.recv(4))[0]
                data = b""
                while len(data) < data_size:
                    data += client_socket.recv(4096)
                data = pickle.loads(data)
                
                face, ER , EL, grid = process_data(data['face_image'], data['ER_image'], data['EL_image'], data['face_grid'])

                # 프레임 처리 (예: 흑백 변환)
                #processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = face.to(device)
                ER = ER.to(device)
                EL = EL.to(device)
                grid = grid.to(device)
                
                gaze = model(face, ER, EL ,grid)
                gaze = gaze.detach().cpu().numpy()
                # 처리된 프레임을 클라이언트로 전송
                data = pickle.dumps(gaze)
                client_socket.sendall(struct.pack(">L", len(data)) + data)
    finally:
        client_socket.close()
        server_socket.close()
        
if __name__ == "__main__":
    main()