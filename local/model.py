import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchvision import models
import os
from torchmetrics.classification import Accuracy
from torchmetrics import Accuracy
from torch import optim


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation layer

    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc = nn.Sequential(  # Excitation (similar to attention)
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Model(LightningModule):
    """
    Model from https://github.com/pperle/gaze-tracking
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # subject_biases는 각 피험자에 대한 pitch와 yaw의 오프셋
        self.subject_biases = nn.Parameter(torch.zeros(15 * 2, 2))  # pitch and yaw offset for the original and mirrored participant

        # 로컬에서 VGG16 가중치를 로드하도록 수정
        self.cnn_face = nn.Sequential(
            self.load_vgg16_weights('C:/Users/jungmin/Desktop/연구실/EYETRACKING/ZIP/gaze-tracking-pipeline/pretrained_model').features[:9],  # 첫 번째 9개 convolutional layers만 사용
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(11, 11)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.cnn_eye = nn.Sequential(
            self.load_vgg16_weights('C:/Users/jungmin/Desktop/연구실/EYETRACKING/ZIP/gaze-tracking-pipeline/pretrained_model').features[:9],  # 첫 번째 9개 convolutional layers만 사용
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(4, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 11)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.fc_face = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 128, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.cnn_eye2fc = nn.Sequential(
            SELayer(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            SELayer(256),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            SELayer(128),
        )

        self.fc_eye = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 6 * 128, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        self.fc_eyes_face = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

    def load_vgg16_weights(self, weights_path):
        """
        로컬 경로에서 VGG16 가중치를 로드하는 함수.
        weights_path: VGG16 가중치가 있는 로컬 경로
        """
        vgg16_model = models.vgg16(pretrained=False)  # pretrained=False로 설정하여 자동 다운로드 방지
        weight_file = os.path.join(weights_path, 'vgg16-397923af.pth')  # VGG16 가중치 파일 경로
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"VGG16 weights not found at {weight_file}")
        
        state_dict = torch.load(weight_file, map_location="cpu")
        vgg16_model.load_state_dict(state_dict)  # 로컬 가중치 로드
        return vgg16_model

    def forward(self, person_idx: torch.Tensor, full_face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor):
        out_cnn_face = self.cnn_face(full_face)
        out_fc_face = self.fc_face(out_cnn_face)

        out_cnn_right_eye = self.cnn_eye(right_eye)
        out_cnn_left_eye = self.cnn_eye(left_eye)
        out_cnn_eye = torch.cat((out_cnn_right_eye, out_cnn_left_eye), dim=1)

        cnn_eye2fc_out = self.cnn_eye2fc(out_cnn_eye)  # feature fusion
        out_fc_eye = self.fc_eye(cnn_eye2fc_out)

        fc_concatenated = torch.cat((out_fc_face, out_fc_eye), dim=1)
        t_hat = self.fc_eyes_face(fc_concatenated)  # subject-independent term

        return t_hat + self.subject_biases[person_idx].squeeze(1)  # t_hat + subject-dependent bias term



# class Model(LightningModule):
#     """
#     Model from https://github.com/pperle/gaze-tracking
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__()

#         self.subject_biases = nn.Parameter(torch.zeros(15 * 2, 2))  # pitch and yaw offset for the original and mirrored participant

#         self.cnn_face = nn.Sequential(
#             models.vgg16(pretrained=True, progress=True).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
#             nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 5)),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(11, 11)),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),
#         )

#         self.cnn_eye = nn.Sequential(
#             models.vgg16(pretrained=True).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
#             nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(4, 5)),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 11)),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),
#         )

#         self.fc_face = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(6 * 6 * 128, 256),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(256),
#             nn.Linear(256, 64),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(64),
#         )

#         self.cnn_eye2fc = nn.Sequential(
#             SELayer(256),

#             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding='same'),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(256),

#             SELayer(256),

#             nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),

#             SELayer(128),
#         )

#         self.fc_eye = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(4 * 6 * 128, 512),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(512),
#         )

#         self.fc_eyes_face = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(576, 256),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(256),
#             nn.Dropout(p=0.5),
#             nn.Linear(256, 2),
#         )

#     def forward(self, person_idx: torch.Tensor, full_face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor):
#         out_cnn_face = self.cnn_face(full_face)
#         out_fc_face = self.fc_face(out_cnn_face)

#         out_cnn_right_eye = self.cnn_eye(right_eye)
#         out_cnn_left_eye = self.cnn_eye(left_eye)
#         out_cnn_eye = torch.cat((out_cnn_right_eye, out_cnn_left_eye), dim=1)

#         cnn_eye2fc_out = self.cnn_eye2fc(out_cnn_eye)  # feature fusion
#         out_fc_eye = self.fc_eye(cnn_eye2fc_out)

#         fc_concatenated = torch.cat((out_fc_face, out_fc_eye), dim=1)
#         t_hat = self.fc_eyes_face(fc_concatenated)  # subject-independent term

#         return t_hat + self.subject_biases[person_idx].squeeze(1)  # t_hat + subject-dependent bias term


