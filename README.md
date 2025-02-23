# Eye-Tracking

---

위의 코드를 실행 시키기 위해서는 GPU를 사용하여 gaze point를 inference 할 서버에서의 환경 1개와
로컬에서 webcam을 통해서 비디오를 입력받아서 처리할 로컬 환경 1개가 필요하다. 


각각의 /local과 /server은 각 환경에서 실행하는 코드들이다. 
각 폴더 안의 requirements.txt를 이용하여 환경을 설정하면 된다. 

pip install -r requirements.txt

데이터의 경우에는 각각의 eyetracking for everyone(https://github.com/CSAILVision/GazeCapture) 과 efficiency in real-time webcam gaze tracking(https://github.com/pperle/gaze-tracking-pipeline)에서 사용한 방법과 같은 방법을 사용하였다. 

실행 방법 

1. 서버에서 코드를 시작하여 서버와의 소켓 통신  python servercheck_hybrid.py
2. 서버에서 코드를 실행하여 eyetracking 시작   python main_hybrid.py
