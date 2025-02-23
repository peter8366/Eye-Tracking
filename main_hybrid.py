import cv2
import socket
import struct
import pickle
import time
import pyrealsense2 as rs
import mediapipe as mp
import numpy as np
import random
import collections
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import get_camera_matrix, get_face_landmarks_in_ccs, gaze_2d_to_3d, ray_plane_intersection, plane_equation, get_monitor_dimensions, get_point_on_screen
from mpii_face_gaze_preprocessing import normalize_single_image
from collections import deque
from screeninfo import get_monitors
import logging
from logging.handlers import RotatingFileHandler
import os
import zlib
from math import sqrt


face_model_all: np.ndarray = np.array([
    [0.000000, -3.406404, 5.979507],
    [0.000000, -1.126865, 7.475604],
    [0.000000, -2.089024, 6.058267],
    [-0.463928, 0.955357, 6.633583],
    [0.000000, -0.463170, 7.586580],
    [0.000000, 0.365669, 7.242870],
    [0.000000, 2.473255, 5.788627],
    [-4.253081, 2.577646, 3.279702],
    [0.000000, 4.019042, 5.284764],
    [0.000000, 4.885979, 5.385258],
    [0.000000, 8.261778, 4.481535],
    [0.000000, -3.706811, 5.864924],
    [0.000000, -3.918301, 5.569430],
    [0.000000, -3.994436, 5.219482],
    [0.000000, -4.542400, 5.404754],
    [0.000000, -4.745577, 5.529457],
    [0.000000, -5.019567, 5.601448],
    [0.000000, -5.365123, 5.535441],
    [0.000000, -6.149624, 5.071372],
    [0.000000, -1.501095, 7.112196],
    [-0.416106, -1.466449, 6.447657],
    [-7.087960, 5.434801, 0.099620],
    [-2.628639, 2.035898, 3.848121],
    [-3.198363, 1.985815, 3.796952],
    [-3.775151, 2.039402, 3.646194],    
    [-4.465819, 2.422950, 3.155168],
    [-2.164289, 2.189867, 3.851822],
    [-3.208229, 3.223926, 4.115822],
    [-2.673803, 3.205337, 4.092203],
    [-3.745193, 3.165286, 3.972409],
    [-4.161018, 3.059069, 3.719554],
    [-5.062006, 1.934418, 2.776093],
    [-2.266659, -7.425768, 4.389812],
    [-4.445859, 2.663991, 3.173422],
    [-7.214530, 2.263009, 0.073150],
    [-5.799793, 2.349546, 2.204059],
    [-2.844939, -0.720868, 4.433130],
    [-0.711452, -3.329355, 5.877044],
    [-0.606033, -3.924562, 5.444923],
    [-1.431615, -3.500953, 5.496189],
    [-1.914910, -3.803146, 5.028930],
    [-1.131043, -3.973937, 5.189648],
    [-1.563548, -4.082763, 4.842263],
    [-2.650112, -5.003649, 4.188483],
    [-0.427049, -1.094134, 7.360529],
    [-0.496396, -0.475659, 7.440358],
    [-5.253307, 3.881582, 3.363159],
    [-1.718698, 0.974609, 4.558359],
    [-1.608635, -0.942516, 5.814193],
    [-1.651267, -0.610868, 5.581319],
    [-4.765501, -0.701554, 3.534632],
    [-0.478306, 0.295766, 7.101013],
    [-3.734964, 4.508230, 4.550454],
    [-4.588603, 4.302037, 4.048484],
    [-6.279331, 6.615427, 1.425850],
    [-1.220941, 4.142165, 5.106035],
    [-2.193489, 3.100317, 4.000575],
    [-3.102642, -4.352984, 4.095905],
    [-6.719682, -4.788645, -1.745401],
    [-1.193824, -1.306795, 5.737747],
    [-0.729766, -1.593712, 5.833208],
    [-2.456206, -4.342621, 4.283884],
    [-2.204823, -4.304508, 4.162499],
    [-4.985894, 4.802461, 3.751977],
    [-1.592294, -1.257709, 5.456949],
    [-2.644548, 4.524654, 4.921559],
    [-2.760292, 5.100971, 5.015990],
    [-3.523964, 8.005976, 3.729163],
    [-5.599763, 5.715470, 2.724259],
    [-3.063932, 6.566144, 4.529981],
    [-5.720968, 4.254584, 2.830852],
    [-6.374393, 4.785590, 1.591691],
    [-0.672728, -3.688016, 5.737804],
    [-1.262560, -3.787691, 5.417779],
    [-1.732553, -3.952767, 5.000579],
    [-1.043625, -1.464973, 5.662455],
    [-2.321234, -4.329069, 4.258156],
    [-2.056846, -4.477671, 4.520883],
    [-2.153084, -4.276322, 4.038093],
    [-0.946874, -1.035249, 6.512274],
    [-1.469132, -4.036351, 4.604908],
    [-1.024340, -3.989851, 4.926693],
    [-0.533422, -3.993222, 5.138202],
    [-0.769720, -6.095394, 4.985883],
    [-0.699606, -5.291850, 5.448304],
    [-0.669687, -4.949770, 5.509612],
    [-0.630947, -4.695101, 5.449371],
    [-0.583218, -4.517982, 5.339869],
    [-1.537170, -4.423206, 4.745470],
    [-1.615600, -4.475942, 4.813632],
    [-1.729053, -4.618680, 4.854463],
    [-1.838624, -4.828746, 4.823737],
    [-2.368250, -3.106237, 4.868096],
    [-7.542244, -1.049282, -2.431321],
    [0.000000, -1.724003, 6.601390],
    [-1.826614, -4.399531, 4.399021],
    [-1.929558, -4.411831, 4.497052],
    [-0.597442, -2.013686, 5.866456],
    [-1.405627, -1.714196, 5.241087],
    [-0.662449, -1.819321, 5.863759],
    [-2.342340, 0.572222, 4.294303],
    [-3.327324, 0.104863, 4.113860],
    [-1.726175, -0.919165, 5.273355],
    [-5.133204, 7.485602, 2.660442],
    [-4.538641, 6.319907, 3.683424],
    [-3.986562, 5.109487, 4.466315],
    [-2.169681, -5.440433, 4.455874],
    [-1.395634, 5.011963, 5.316032],
    [-1.619500, 6.599217, 4.921106],
    [-1.891399, 8.236377, 4.274997],
    [-4.195832, 2.235205, 3.375099],
    [-5.733342, 1.411738, 2.431726],
    [-1.859887, 2.355757, 3.843181],
    [-4.988612, 3.074654, 3.083858],
    [-1.303263, 1.416453, 4.831091],
    [-1.305757, -0.672779, 6.415959],
    [-6.465170, 0.937119, 1.689873],
    [-5.258659, 0.945811, 2.974312],
    [-4.432338, 0.722096, 3.522615],
    [-3.300681, 0.861641, 3.872784],
    [-2.430178, 1.131492, 4.039035],
    [-1.820731, 1.467954, 4.224124],
    [-0.563221, 2.307693, 5.566789],
    [-6.338145, -0.529279, 1.881175],
    [-5.587698, 3.208071, 2.687839],
    [-0.242624, -1.462857, 7.071491],
    [-1.611251, 0.339326, 4.895421],
    [-7.743095, 2.364999, -2.005167],
    [-1.391142, 1.851048, 4.448999],
    [-1.785794, -0.978284, 4.850470],
    [-4.670959, 2.664461, 3.084075],
    [-1.333970, -0.283761, 6.097047],
    [-7.270895, -2.890917, -2.252455],
    [-1.856432, 2.585245, 3.757904],
    [-0.923388, 0.073076, 6.671944],
    [-5.000589, -6.135128, 1.892523],
    [-5.085276, -7.178590, 0.714711],
    [-7.159291, -0.811820, -0.072044],
    [-5.843051, -5.248023, 0.924091],
    [-6.847258, 3.662916, 0.724695],
    [-2.412942, -8.258853, 4.119213],
    [-0.179909, -1.689864, 6.573301],
    [-2.103655, -0.163946, 4.566119],
    [-6.407571, 2.236021, 1.560843],
    [-3.670075, 2.360153, 3.635230],
    [-3.177186, 2.294265, 3.775704],
    [-2.196121, -4.598322, 4.479786],
    [-6.234883, -1.944430, 1.663542],
    [-1.292924, -9.295920, 4.094063],
    [-3.210651, -8.533278, 2.802001],
    [-4.068926, -7.993109, 1.925119],
    [0.000000, 6.545390, 5.027311],
    [0.000000, -9.403378, 4.264492],
    [-2.724032, 2.315802, 3.777151],
    [-2.288460, 2.398891, 3.697603],
    [-1.998311, 2.496547, 3.689148],
    [-6.130040, 3.399261, 2.038516],
    [-2.288460, 2.886504, 3.775031],
    [-2.724032, 2.961810, 3.871767],
    [-3.177186, 2.964136, 3.876973],
    [-3.670075, 2.927714, 3.724325],
    [-4.018389, 2.857357, 3.482983],
    [-7.555811, 4.106811, -0.991917],
    [-4.018389, 2.483695, 3.440898],
    [0.000000, -2.521945, 5.932265],
    [-1.776217, -2.683946, 5.213116],
    [-1.222237, -1.182444, 5.952465],
    [-0.731493, -2.536683, 5.815343],
    [0.000000, 3.271027, 5.236015],
    [-4.135272, -6.996638, 2.671970],
    [-3.311811, -7.660815, 3.382963],
    [-1.313701, -8.639995, 4.702456],
    [-5.940524, -6.223629, -0.631468],
    [-1.998311, 2.743838, 3.744030],
    [-0.901447, 1.236992, 5.754256],
    [0.000000, -8.765243, 4.891441],
    [-2.308977, -8.974196, 3.609070],
    [-6.954154, -2.439843, -0.131163],
    [-1.098819, -4.458788, 5.120727],
    [-1.181124, -4.579996, 5.189564],
    [-1.255818, -4.787901, 5.237051],
    [-1.325085, -5.106507, 5.205010],
    [-1.546388, -5.819392, 4.757893],
    [-1.953754, -4.183892, 4.431713],
    [-2.117802, -4.137093, 4.555096],
    [-2.285339, -4.051196, 4.582438],
    [-2.850160, -3.665720, 4.484994],
    [-5.278538, -2.238942, 2.861224],
    [-0.946709, 1.907628, 5.196779],
    [-1.314173, 3.104912, 4.231404],
    [-1.780000, 2.860000, 3.881555],
    [-1.845110, -4.098880, 4.247264],
    [-5.436187, -4.030482, 2.109852],
    [-0.766444, 3.182131, 4.861453],
    [-1.938616, -6.614410, 4.521085],
    [0.000000, 1.059413, 6.774605],
    [-0.516573, 1.583572, 6.148363],
    [0.000000, 1.728369, 6.316750],
    [-1.246815, 0.230297, 5.681036],
    [0.000000, -7.942194, 5.181173],
    [0.000000, -6.991499, 5.153478],
    [-0.997827, -6.930921, 4.979576],
    [-3.288807, -5.382514, 3.795752],
    [-2.311631, -1.566237, 4.590085],
    [-2.680250, -6.111567, 4.096152],
    [-3.832928, -1.537326, 4.137731],
    [-2.961860, -2.274215, 4.440943],
    [-4.386901, -2.683286, 3.643886],
    [-1.217295, -7.834465, 4.969286],
    [-1.542374, -0.136843, 5.201008],
    [-3.878377, -6.041764, 3.311079],
    [-3.084037, -6.809842, 3.814195],
    [-3.747321, -4.503545, 3.726453],
    [-6.094129, -3.205991, 1.473482],
    [-4.588995, -4.728726, 2.983221],
    [-6.583231, -3.941269, 0.070268],
    [-3.492580, -3.195820, 4.130198],
    [-1.255543, 0.802341, 5.307551],
    [-1.126122, -0.933602, 6.538785],
    [-1.443109, -1.142774, 5.905127],
    [-0.923043, -0.529042, 7.003423],
    [-1.755386, 3.529117, 4.327696],
    [-2.632589, 3.713828, 4.364629],
    [-3.388062, 3.721976, 4.309028],
    [-4.075766, 3.675413, 4.076063],
    [-4.622910, 3.474691, 3.646321],
    [-5.171755, 2.535753, 2.670867],
    [-7.297331, 0.763172, -0.048769],
    [-4.706828, 1.651000, 3.109532],
    [-4.071712, 1.476821, 3.476944],
    [-3.269817, 1.470659, 3.731945],
    [-2.527572, 1.617311, 3.865444],
    [-1.970894, 1.858505, 3.961782],
    [-1.579543, 2.097941, 4.084996],
    [-7.664182, 0.673132, -2.435867],
    [-1.397041, -1.340139, 5.630378],
    [-0.884838, 0.658740, 6.233232],
    [-0.767097, -0.968035, 7.077932],
    [-0.460213, -1.334106, 6.787447],
    [-0.748618, -1.067994, 6.798303],
    [-1.236408, -1.585568, 5.480490],
    [-0.387306, -1.409990, 6.957705],
    [-0.319925, -1.607931, 6.508676],
    [-1.639633, 2.556298, 3.863736],
    [-1.255645, 2.467144, 4.203800],
    [-1.031362, 2.382663, 4.615849],
    [-4.253081, 2.772296, 3.315305],
    [-4.530000, 2.910000, 3.339685],
    [0.463928, 0.955357, 6.633583],
    [4.253081, 2.577646, 3.279702],
    [0.416106, -1.466449, 6.447657],
    [7.087960, 5.434801, 0.099620],
    [2.628639, 2.035898, 3.848121],
    [3.198363, 1.985815, 3.796952],
    [3.775151, 2.039402, 3.646194],
    [4.465819, 2.422950, 3.155168],
    [2.164289, 2.189867, 3.851822],
    [3.208229, 3.223926, 4.115822],
    [2.673803, 3.205337, 4.092203],
    [3.745193, 3.165286, 3.972409],
    [4.161018, 3.059069, 3.719554],
    [5.062006, 1.934418, 2.776093],
    [2.266659, -7.425768, 4.389812],
    [4.445859, 2.663991, 3.173422],
    [7.214530, 2.263009, 0.073150],
    [5.799793, 2.349546, 2.204059],
    [2.844939, -0.720868, 4.433130],
    [0.711452, -3.329355, 5.877044],
    [0.606033, -3.924562, 5.444923],
    [1.431615, -3.500953, 5.496189],
    [1.914910, -3.803146, 5.028930],
    [1.131043, -3.973937, 5.189648],
    [1.563548, -4.082763, 4.842263],
    [2.650112, -5.003649, 4.188483],
    [0.427049, -1.094134, 7.360529],
    [0.496396, -0.475659, 7.440358],
    [5.253307, 3.881582, 3.363159],
    [1.718698, 0.974609, 4.558359],
    [1.608635, -0.942516, 5.814193],
    [1.651267, -0.610868, 5.581319],
    [4.765501, -0.701554, 3.534632],
    [0.478306, 0.295766, 7.101013],
    [3.734964, 4.508230, 4.550454],
    [4.588603, 4.302037, 4.048484],
    [6.279331, 6.615427, 1.425850],
    [1.220941, 4.142165, 5.106035],
    [2.193489, 3.100317, 4.000575],
    [3.102642, -4.352984, 4.095905],
    [6.719682, -4.788645, -1.745401],
    [1.193824, -1.306795, 5.737747],
    [0.729766, -1.593712, 5.833208],
    [2.456206, -4.342621, 4.283884],
    [2.204823, -4.304508, 4.162499],
    [4.985894, 4.802461, 3.751977],
    [1.592294, -1.257709, 5.456949],
    [2.644548, 4.524654, 4.921559],
    [2.760292, 5.100971, 5.015990],
    [3.523964, 8.005976, 3.729163],
    [5.599763, 5.715470, 2.724259],
    [3.063932, 6.566144, 4.529981],
    [5.720968, 4.254584, 2.830852],
    [6.374393, 4.785590, 1.591691],
    [0.672728, -3.688016, 5.737804],
    [1.262560, -3.787691, 5.417779],
    [1.732553, -3.952767, 5.000579],
    [1.043625, -1.464973, 5.662455],
    [2.321234, -4.329069, 4.258156],
    [2.056846, -4.477671, 4.520883],
    [2.153084, -4.276322, 4.038093],
    [0.946874, -1.035249, 6.512274],
    [1.469132, -4.036351, 4.604908],
    [1.024340, -3.989851, 4.926693],
    [0.533422, -3.993222, 5.138202],
    [0.769720, -6.095394, 4.985883],
    [0.699606, -5.291850, 5.448304],
    [0.669687, -4.949770, 5.509612],
    [0.630947, -4.695101, 5.449371],
    [0.583218, -4.517982, 5.339869],
    [1.537170, -4.423206, 4.745470],
    [1.615600, -4.475942, 4.813632],
    [1.729053, -4.618680, 4.854463],
    [1.838624, -4.828746, 4.823737],
    [2.368250, -3.106237, 4.868096],
    [7.542244, -1.049282, -2.431321],
    [1.826614, -4.399531, 4.399021],
    [1.929558, -4.411831, 4.497052],
    [0.597442, -2.013686, 5.866456],
    [1.405627, -1.714196, 5.241087],
    [0.662449, -1.819321, 5.863759],
    [2.342340, 0.572222, 4.294303],
    [3.327324, 0.104863, 4.113860],
    [1.726175, -0.919165, 5.273355],
    [5.133204, 7.485602, 2.660442],
    [4.538641, 6.319907, 3.683424],
    [3.986562, 5.109487, 4.466315],
    [2.169681, -5.440433, 4.455874],
    [1.395634, 5.011963, 5.316032],
    [1.619500, 6.599217, 4.921106],
    [1.891399, 8.236377, 4.274997],
    [4.195832, 2.235205, 3.375099],
    [5.733342, 1.411738, 2.431726],
    [1.859887, 2.355757, 3.843181],
    [4.988612, 3.074654, 3.083858],
    [1.303263, 1.416453, 4.831091],
    [1.305757, -0.672779, 6.415959],
    [6.465170, 0.937119, 1.689873],
    [5.258659, 0.945811, 2.974312],
    [4.432338, 0.722096, 3.522615],
    [3.300681, 0.861641, 3.872784],
    [2.430178, 1.131492, 4.039035],
    [1.820731, 1.467954, 4.224124],
    [0.563221, 2.307693, 5.566789],
    [6.338145, -0.529279, 1.881175],
    [5.587698, 3.208071, 2.687839],
    [0.242624, -1.462857, 7.071491],
    [1.611251, 0.339326, 4.895421],
    [7.743095, 2.364999, -2.005167],
    [1.391142, 1.851048, 4.448999],
    [1.785794, -0.978284, 4.850470],
    [4.670959, 2.664461, 3.084075],
    [1.333970, -0.283761, 6.097047],
    [7.270895, -2.890917, -2.252455],
    [1.856432, 2.585245, 3.757904],
    [0.923388, 0.073076, 6.671944],
    [5.000589, -6.135128, 1.892523],
    [5.085276, -7.178590, 0.714711],
    [7.159291, -0.811820, -0.072044],
    [5.843051, -5.248023, 0.924091],
    [6.847258, 3.662916, 0.724695],
    [2.412942, -8.258853, 4.119213],
    [0.179909, -1.689864, 6.573301],
    [2.103655, -0.163946, 4.566119],
    [6.407571, 2.236021, 1.560843],
    [3.670075, 2.360153, 3.635230],
    [3.177186, 2.294265, 3.775704],
    [2.196121, -4.598322, 4.479786],
    [6.234883, -1.944430, 1.663542],
    [1.292924, -9.295920, 4.094063],
    [3.210651, -8.533278, 2.802001],
    [4.068926, -7.993109, 1.925119],
    [2.724032, 2.315802, 3.777151],
    [2.288460, 2.398891, 3.697603],
    [1.998311, 2.496547, 3.689148],
    [6.130040, 3.399261, 2.038516],
    [2.288460, 2.886504, 3.775031],
    [2.724032, 2.961810, 3.871767],
    [3.177186, 2.964136, 3.876973],
    [3.670075, 2.927714, 3.724325],
    [4.018389, 2.857357, 3.482983],
    [7.555811, 4.106811, -0.991917],
    [4.018389, 2.483695, 3.440898],
    [1.776217, -2.683946, 5.213116],
    [1.222237, -1.182444, 5.952465],
    [0.731493, -2.536683, 5.815343],
    [4.135272, -6.996638, 2.671970],
    [3.311811, -7.660815, 3.382963],
    [1.313701, -8.639995, 4.702456],
    [5.940524, -6.223629, -0.631468],
    [1.998311, 2.743838, 3.744030],
    [0.901447, 1.236992, 5.754256],
    [2.308977, -8.974196, 3.609070],
    [6.954154, -2.439843, -0.131163],
    [1.098819, -4.458788, 5.120727],
    [1.181124, -4.579996, 5.189564],
    [1.255818, -4.787901, 5.237051],
    [1.325085, -5.106507, 5.205010],
    [1.546388, -5.819392, 4.757893],
    [1.953754, -4.183892, 4.431713],
    [2.117802, -4.137093, 4.555096],
    [2.285339, -4.051196, 4.582438],
    [2.850160, -3.665720, 4.484994],
    [5.278538, -2.238942, 2.861224],
    [0.946709, 1.907628, 5.196779],
    [1.314173, 3.104912, 4.231404],
    [1.780000, 2.860000, 3.881555],
    [1.845110, -4.098880, 4.247264],
    [5.436187, -4.030482, 2.109852],
    [0.766444, 3.182131, 4.861453],
    [1.938616, -6.614410, 4.521085],
    [0.516573, 1.583572, 6.148363],
    [1.246815, 0.230297, 5.681036],
    [0.997827, -6.930921, 4.979576],
    [3.288807, -5.382514, 3.795752],
    [2.311631, -1.566237, 4.590085],
    [2.680250, -6.111567, 4.096152],
    [3.832928, -1.537326, 4.137731],
    [2.961860, -2.274215, 4.440943],
    [4.386901, -2.683286, 3.643886],
    [1.217295, -7.834465, 4.969286],
    [1.542374, -0.136843, 5.201008],
    [3.878377, -6.041764, 3.311079],
    [3.084037, -6.809842, 3.814195],
    [3.747321, -4.503545, 3.726453],
    [6.094129, -3.205991, 1.473482],
    [4.588995, -4.728726, 2.983221],
    [6.583231, -3.941269, 0.070268],
    [3.492580, -3.195820, 4.130198],
    [1.255543, 0.802341, 5.307551],
    [1.126122, -0.933602, 6.538785],
    [1.443109, -1.142774, 5.905127],
    [0.923043, -0.529042, 7.003423],
    [1.755386, 3.529117, 4.327696],
    [2.632589, 3.713828, 4.364629],
    [3.388062, 3.721976, 4.309028],
    [4.075766, 3.675413, 4.076063],
    [4.622910, 3.474691, 3.646321],
    [5.171755, 2.535753, 2.670867],
    [7.297331, 0.763172, -0.048769],
    [4.706828, 1.651000, 3.109532],
    [4.071712, 1.476821, 3.476944],
    [3.269817, 1.470659, 3.731945],
    [2.527572, 1.617311, 3.865444],
    [1.970894, 1.858505, 3.961782],
    [1.579543, 2.097941, 4.084996],
    [7.664182, 0.673132, -2.435867],
    [1.397041, -1.340139, 5.630378],
    [0.884838, 0.658740, 6.233232],
    [0.767097, -0.968035, 7.077932],
    [0.460213, -1.334106, 6.787447],
    [0.748618, -1.067994, 6.798303],
    [1.236408, -1.585568, 5.480490],
    [0.387306, -1.409990, 6.957705],
    [0.319925, -1.607931, 6.508676],
    [1.639633, 2.556298, 3.863736],
    [1.255645, 2.467144, 4.203800],
    [1.031362, 2.382663, 4.615849],
    [4.253081, 2.772296, 3.315305],
    [4.530000, 2.910000, 3.339685]
], dtype=float)

# 기준점을 (0,0)으로 설정 
face_model_all -= face_model_all[1]     
# 축 고정 
face_model_all *= np.array([1, -1, -1])  
face_model_all *= 10   

# Mediapipe Face Mesh에서 주요 랜드마크 (왼쪽 눈, 오른쪽 눈, 입, 얼굴 중심점 )
landmarks_ids = [33, 133, 362, 263, 61, 291, 1]  
face_model = np.asarray([face_model_all[i] for i in landmarks_ids])

# Face mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

#필요없을 시 삭제
# plane = plane_equation(np.eye(3), np.asarray([[0], [0], [0]]))
# plane_w = plane[0:3]
# plane_b = plane[3]


# FPS 측정 
fps_deque = collections.deque(maxlen=60)  
prev_frame_time = 0


smoothing_buffer = collections.deque(maxlen=3)
rvec_buffer = collections.deque(maxlen=3)
tvec_buffer = collections.deque(maxlen=3)
# gaze_vector_buffer = collections.deque(maxlen=10)
rvec, tvec = None, None
gaze_points = collections.deque()
calibrated_gaze_points = collections.deque()
gaze_data = collections.defaultdict(list)


#calibration matrix 불러오기 
calibration_matrix_path = "./gaze-tracking-pipeline/calibration_matrix2.yaml" 
camera_matrix, dist_coefficients = get_camera_matrix(calibration_matrix_path)


# transform 함수 
transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

# 각 물체의 위치 
points = [
    ("lamp", (60, 50)),
    ("water", (132, 68)),
    ("boat", (230, 38)),
    ("ballon", (353, 64)),
    ("shopping bag", (440, 64)),
    ("hat", (565, 49)),
    ("trash can", (660, 64)),
    ("labtop", (63,165)), 
    ("train",(240,110)),
    ("car", (277,170)),
    ("electric fan", (418,165)),
    ("truck", (540,140)),
    ("switch", (656,173)),
    ("bread", (78,258)),
    ("galsses", (186,243)),
    ("bag", (325,258)),
    ("chair", (436,275)),
    ("yacht", (535,240)),
    ("melon", (659,268)),
    ("van", (115,335)),
    ("mouse", (263,345)),
    ("cake", (366,351)),
    ("umbrella", (530,330)),
    ("airplane", (640,336))
]

WINDOW_NAME = 'laser pointer preview'

#모니터 사양
monitor_mm = (597.7, 336.2)
original_width, original_height = 720, 405

#FHD
monitor_pixels = [
    ("17inch", (2417, 1360)),
    ("21inch", (2986, 1680)),
    ("23inch", (3271, 1840)),
    ("27inch", (1920, 1080))
]


#4k
# monitor_pixels = [
#     ("17inch", (2417, 1360)),
#     ("21inch", (2986, 1680)),
#     ("23inch", (3271, 1840)),
#     ("27inch", (3840, 2160))
# ]

selected_monitor = "27inch"
image_width, image_height = next((width, height) for name, (width, height) in monitor_pixels if name == selected_monitor)

target_size = next(size for name, size in monitor_pixels if name == selected_monitor)
monitor_width, monitor_height = target_size


#이미지 load
image_path = 'C:/Users/jungmin/Desktop/연구실/EYETRACKING/ZIP/gaze-tracking-pipeline/image.png'   
test_image = cv2.imread(image_path)

# 이미지 resize
resized_image = cv2.resize(test_image,target_size)

# 이미지 크기에 맞춰서 물체 위치 조정 
scaled_points = []
for name, (x, y) in points:
    new_x = int(x * monitor_width / original_width)
    new_y = int(y * monitor_height / original_height)
    scaled_points.append((name, (new_x, new_y)))

random.seed(42)

#calibration 초기화 

leftup_init = []
rightup_init = []
leftdown_init = []
rightdown_init = []
midleft_init = []
midright_init = []

# 빨간 점 띄우기 
randomized_points = random.sample(scaled_points, len(scaled_points))
total_points = len(scaled_points)
total_frames = len(scaled_points) * 150
selected_index = 0

initial_delay = 5 
red_dot_duration = 5
start_time = time.time()  
last_update_time = start_time
gaze_record_start_time = None

# EAR 계산하는 함수 

def calculate_ear(eye_landmarks):

    # 수직 거리 평균 
    vertical_distance_1 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[3]) 
    vertical_distance_2 = np.linalg.norm(eye_landmarks[4] - eye_landmarks[5]) 
    vertical_distance = (vertical_distance_1 + vertical_distance_2) / 2.0  

    # 수평 거리 평균 
    horizontal_distance = np.linalg.norm(eye_landmarks[0] - eye_landmarks[1]) 

    # EAR 계산
    ear = vertical_distance / horizontal_distance if horizontal_distance != 0 else 0
    return ear


# MSE 계산하는 함수 

def calculate_rmse(gaze_data, scaled_points):
    total_mse = 0
    # 총 객체의 수 24
    n = len(scaled_points)  
    
    for point_name, (x_gt, y_gt) in scaled_points:
        if point_name in gaze_data and len(gaze_data[point_name]) > 0:
            # 평균 gaze point 계산
            avg_x = np.mean([point[0] for point in gaze_data[point_name]])
            avg_y = np.mean([point[1] for point in gaze_data[point_name]])
            
            # MSE 계산
            mse = ((avg_x - x_gt) ** 2 + (avg_y - y_gt) ** 2) / 2
            total_mse += mse

            print(f"{point_name} - predict: ({avg_x}, {avg_y}), ground_truth: ({x_gt}, {y_gt})")
            

    rmse = np.sqrt(total_mse / len(scaled_points))

    return rmse


# accuracy 측정하는 함수 
def calculate_accuracy(gaze_data, scaled_points, tolerance_radius=125):
    correct_count = 0
    # 총 객체의 수 24
    total_count = len(scaled_points)  
    
    # 중간 결과를 저장할 리스트 생성
    results = []
    
    for point_name, (x_gt, y_gt) in scaled_points:
        if point_name in gaze_data and len(gaze_data[point_name]) > 0:
            # 평균 gaze point 계산
            avg_x = np.mean([point[0] for point in gaze_data[point_name]])
            avg_y = np.mean([point[1] for point in gaze_data[point_name]])
            
            # 거리 계산
            distance = np.sqrt((avg_x - x_gt) ** 2 + (avg_y - y_gt) ** 2)
            
            # tolerance 반지름 내에 있는지 확인
            result = f"{point_name} - predict: ({avg_x:.2f}, {avg_y:.2f}), ground_truth: ({x_gt}, {y_gt}), distance: {distance:.2f}, {'O' if distance <= tolerance_radius else 'X'}"
            results.append(result)
            
            if distance <= tolerance_radius:
                # 원 안에 포함되면 정답으로 카운트
                correct_count += 1  
    
    # 각 결과를 한 줄씩 출력
    for result in results:
        print(result)
    
    # 정확도 계산
    accuracy = correct_count / total_count * 100  
    return accuracy

# calibration 하는 점들 위치 
actual_corners = np.array([
    [200, 250],
    [1720, 250],                       
    [150, 880],
    [1770, 880],                   
    [600, 550],                                                                  
    [1320, 530]                 
], dtype="float32")

# Homography를 이용하여 calibration
def calibrate_gaze_point(gaze_point):
    gaze_point =  np.array(gaze_point, dtype="float32").reshape(1, 1, 2)
    
    # 필요 없을 시 삭제제
    # H, _ = cv2.findHomography(gaze_means, actual_corners)   # Homography값 계산 
    
    #gaze_point에 H 적용 
    calibrated_point = cv2.perspectiveTransform(gaze_point, H)  

    return calibrated_point[0][0]


smoothed_x, smoothed_y = 0, 0 

# server setup
server_ip = '220.149.82.236'
server_port = 4444

# RealSense 동작 
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 다운샘플링 비율
downsample_factor = 4  
x_indices = np.arange(0, image_width, downsample_factor)
y_indices = np.arange(0, image_height, downsample_factor)
X, Y = np.meshgrid(x_indices, y_indices)
radius = 15

# 서버 연결 
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

print(f"모니터 크기: {selected_monitor}, 해상도: {image_width}x{image_height}")

# 프레임 저장 폴더 설정
image_folder = './images'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

#gaze_point을 저장용 txt파일 설정
gaze_file = open('gaze_points.txt', 'w')
gaze_point_file = open('gaze_points_original.txt', 'w')
ear_file = open('ear.txt','w')

# 눈 깜빡임을 감지할 임계값 설정 
EAR_THRESHOLD = 0.20
previous_gaze_point = None

# mesh_map.jpg으로 부터의 랜드마크 값 
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

#프레임 
frame_idx = 0
a = 1

try:
    while True:
        
        # 모니터 해상도 출력
        monitors = get_monitors()
        #for monitor in monitors:
            # print(f"모니터: {monitor.name}, 해상도: {monitor.width}x{monitor.height}")

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        
        frame = np.asanyarray(color_frame.get_data())
        height, width, _ = frame.shape

        # 이미지 저장 
        image_filename = os.path.join(image_folder, f"frame_{frame_idx}.jpg")
        cv2.imwrite(image_filename, frame)
        # print(f"Saved: {image_filename}")

        # 얼굴 랜드마크 추출 
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            

            face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])

            # 왼쪽 및 오른쪽 눈의 랜드마크 추출
            left_eye_landmarks = face_landmarks[[33, 133, 160, 144, 158, 153]]  # 왼쪽 눈 랜드마크
            right_eye_landmarks = face_landmarks[[362, 263, 385, 380, 387, 373]]  # 오른쪽 눈 랜드마크

            # 왼쪽 및 오른쪽 눈 EAR 계산
            left_ear = calculate_ear(left_eye_landmarks)
            right_ear = calculate_ear(right_eye_landmarks)
            
            # 평균 EAR 계산
            ear = (left_ear + right_ear) / 2.0
            ear_file.write(f"{frame_idx}: {ear}\n")  # 1부터 시작하는 라벨링


            face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])
            smoothing_buffer.append(face_landmarks)
            face_landmarks = np.asarray(smoothing_buffer).mean(axis=0)

            display = resized_image.copy()


            #if ear < EAR_THRESHOLD:
                #print(f"EAR below threshold: {ear}, using previous gaze_point")
                # gaze_point를 이전값으로 유지
                #gaze_point = gaze_points[0] if gaze_points else None  # 이전 gaze_point가 없으면 None으로 설정
            
            if a ==2:
                pass
            
            else:

                # rvec와 tvec 얻음 
                success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)  # Initial fit
                for _ in range(10):
                    success, rvec, tvec = cv2.solvePnP(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)  # Second fit for higher accuracy

                rvec_buffer.append(rvec)
                rvec = np.asarray(rvec_buffer).mean(axis=0)
                tvec_buffer.append(tvec)
                tvec = np.asarray(tvec_buffer).mean(axis=0)

                # data preprocessing
                face_model_transformed, face_model_all_transformed = get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, frame.shape, results, face_model, face_model_all, landmarks_ids)
                left_eye_center = 0.5 * (face_model_transformed[:, 2] + face_model_transformed[:, 3]).reshape((3, 1))  # center eye
                right_eye_center = 0.5 * (face_model_transformed[:, 0] + face_model_transformed[:, 1]).reshape((3, 1))  # center eye
                face_center = face_model_transformed.mean(axis=1).reshape((3, 1))   # 얼굴의 중앙 좌표 구하기기

                img_warped_left_eye, _, _ = normalize_single_image(image_rgb, rvec, None, left_eye_center, camera_matrix)
                img_warped_right_eye, _, _ = normalize_single_image(image_rgb, rvec, None, right_eye_center, camera_matrix)
                # rotation matrix 구하기 
                img_warped_face, _, rotation_matrix = normalize_single_image(image_rgb, rvec, None, face_center, camera_matrix, is_eye=False)


                # 최종 preprocess 데이터 얻기기
                person_idx = np.array([0])  
                full_face_image = transform(image=img_warped_face)["image"].unsqueeze(0).float().numpy()
                left_eye_image = transform(image=img_warped_left_eye)["image"].unsqueeze(0).float().numpy()
                right_eye_image = transform(image=img_warped_right_eye)["image"].unsqueeze(0).float().numpy()
                face_center = face_center.flatten()  
                rotation_matrix = rotation_matrix.flatten()

                data = {
                    'person_idx': person_idx,
                    'full_face_image': full_face_image,
                    'left_eye_image': left_eye_image,
                    'right_eye_image': right_eye_image,                         
                    'face_center': face_center,
                    'rotation_matrix' : rotation_matrix
                }
                # 데이터를 서버로 전송
                serialized_data = pickle.dumps(data)
                compressed_data = zlib.compress(serialized_data)
                data_size = struct.pack(">L", len(compressed_data))
                client_socket.sendall(data_size + compressed_data)
                

            

            

                # 서버로부터 gaze point 결과 수신
                response_size = struct.unpack(">L", client_socket.recv(4))[0]
                response_data = b""
                while len(response_data) < response_size:
                    response_data += client_socket.recv(4096)

                # gaze point 수신
                gaze_point = pickle.loads(response_data)

            gaze_points.appendleft(gaze_point)

            # alpha 값이 낮을수록 smoothing이 더 강하게 적용
            alpha = 0.2  

            # 현재 프레임의 gaze point
            current_x, current_y = gaze_points[0]

            # 스무딩 기법 적용 
            if len(gaze_points) >= 3:
                
                x1, y1 = gaze_points[1]
                x2, y2 = gaze_points[2]
            
                if abs(current_x-x1) > 10000 or abs(current_y-y1) > 10000:
                    smoothed_x = alpha * current_x + (1 - alpha) * x1
                    smoothed_y = alpha * current_y + (1 - alpha) * y1
                    gaze_points[0] = (smoothed_x, smoothed_y)


            if frame_idx >= 492:

                calibrated_point = calibrate_gaze_point(gaze_point)
                calibrated_gaze_points.appendleft(calibrated_point)

                recent_gaze_points = list(calibrated_gaze_points)[:8]
                recent_gaze_x = np.array([p[0] for p in recent_gaze_points])
                recent_gaze_y = np.array([p[1] for p in recent_gaze_points])

                # mean_x = np.mean(recent_gaze_x)
                # mean_y = np.mean(recent_gaze_y)

                # std를 제한하여 원의 크기 조절 
                max_std_x = image_width * 0.05
                max_std_y = image_height * 0.05  

                std_x = np.clip(np.std(recent_gaze_x), None, max_std_x)
                std_y = np.clip(np.std(recent_gaze_y), None, max_std_x)

                mean_x = int(np.mean(recent_gaze_x))
                mean_y = int(np.mean(recent_gaze_y))

                if smoothed_x == 0 and smoothed_y == 0:  # 첫 프레임인지 확인
                    smoothed_x, smoothed_y = mean_x, mean_y


                # EMA 사용
                beta = 0.2     
                smoothed_x = beta * mean_x + (1 - beta) * smoothed_x 
                smoothed_y = beta * mean_y + (1 - beta) * smoothed_y

                smoothed_x = int(smoothed_x)
                smoothed_y = int(smoothed_y)

                radius = int(max(std_x, std_y))

                circle_color = (255,0,0)
                circle_thickness = 2

                elapsed_time = time.time() - start_time

                if elapsed_time >= initial_delay:
                    # 빨간 점 표시할 좌표 업데이트 (랜덤화된 좌표 리스트를 사용)
                    if time.time() - last_update_time > red_dot_duration:
                        # 빨간 점을 표시할 좌표 갱신
                        selected_index = (selected_index + 1) % total_points
                        last_update_time = time.time()  # 업데이트 시간 갱신
                        gaze_record_start_time = time.time()    #gaze_point 저장을 위한 시간 기록 
                    
                        

                        for point_name, gaze_points_save in gaze_data.items():
                            if len(gaze_points_save) > 0: 
                                avg_x = np.mean([point[0] for point in gaze_points_save])
                                avg_y = np.mean([point[1] for point in gaze_points_save])
                                print(f"{point_name}의 3초 동안의 gaze_point 평균: (X: {avg_x:.2f}, Y: {avg_y:.2f})")
                        
                        point_name = randomized_points[selected_index][0]
                        gaze_data[point_name] = []



                    point_name, coordinates = randomized_points[selected_index]
                    cv2.circle(display, coordinates, radius=25, color=(0, 0, 255), thickness=-1)

                    if time.time() - gaze_record_start_time >= 2:  # 표시 시작 후 2초가 지난 시점부터 기록
                        gaze_data[point_name].append(calibrated_point)
                    
                    if len(gaze_data) >= 24:
                        print("모든 물체에 대해 트래킹이 완료되었습니다.")
                        break  
                
                # gaze_point_file.write(f"{frame_idx}: {gaze_point[0]}, {gaze_point[1]}\n")

                # if elapsed_time < initial_delay:
                #     gaze_file.write(f"{frame_idx}: {calibrated_point[0]}, {calibrated_point[1]} (Red Dot Displayed)\n")  # 1부터 시작하는 라벨링
                # else:
                #     gaze_file.write(f"{frame_idx}: {calibrated_point[0]}, {calibrated_point[1]}\n")  # 1부터 시작하는 라벨링
                
                # text = f"Frame: {frame_idx}"
                # cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # # overlay = cv2.addWeighted(display, 0.7, heatmap, 0.3, 0)
                # # cv2.circle(display, (mean_x, mean_y), radius, circle_color, circle_thickness)
                # cv2.circle(display, (smoothed_x, smoothed_y), radius, circle_color, circle_thickness)
                # cv2.namedWindow('target image', cv2.WINDOW_AUTOSIZE)  # cv2.WINDOW_KEEPRATIO 대신 cv2.WINDOW_NORMAL 사용   기존에 cv2.WINDOW_AUTOSIZE사용용
                # # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                # # cv2.imshow(WINDOW_NAME, overlay)
                cv2.imshow('target image', display)
                
            


            # Calibration 과정 진행 
            elif frame_idx == 491:

                # 왼쪽 위
                x_mean1 = sum(x for x, y in leftup_init) / len(leftup_init)
                y_mean1 = sum(y for x, y in leftup_init) / len(leftup_init)
                leftup_mean = np.array([x_mean1, y_mean1], dtype="float32")

                # 오른쪽 위 
                x_mean2 = sum(x for x, y in rightup_init) / len(rightup_init)
                y_mean2 = sum(y for x, y in rightup_init) / len(rightup_init)
                rightup_mean = np.array([x_mean2, y_mean2], dtype="float32")

                # 왼쪽 아래
                x_mean3 = sum(x for x, y in leftdown_init) / len(leftdown_init)
                y_mean3 = sum(y for x, y in leftdown_init) / len(leftdown_init)
                leftdown_mean = np.array([x_mean3, y_mean3], dtype="float32")

                # 오른쪽 아래
                x_mean4 = sum(x for x, y in rightdown_init) / len(rightdown_init)
                y_mean4 = sum(y for x, y in rightdown_init) / len(rightdown_init)
                rightdown_mean = np.array([x_mean4, y_mean4], dtype="float32")

                # 가운데 왼쪽 
                x_mean5 = sum(x for x, y in midleft_init) / len(midleft_init)
                y_mean5 = sum(y for x, y in midleft_init) / len(midleft_init)
                midleft_mean = np.array([x_mean5, y_mean5], dtype="float32")

                # 오른쪽 아래
                x_mean6 = sum(x for x, y in midright_init) / len(midright_init)
                y_mean6 = sum(y for x, y in midright_init) / len(midright_init)
                midright_mean = np.array([x_mean6, y_mean6], dtype="float32")

                gaze_means = [
                    leftup_mean, rightup_mean, 
                    leftdown_mean, rightdown_mean, 
                    midleft_mean, midright_mean
                ]

                gaze_means = np.array(gaze_means, dtype="float32")
                scaled_calibrated_points = []

                H, _ = cv2.findHomography(gaze_means, actual_corners)   # Homography값 계산 

                # 각 ground truth 좌표에 대해 보정을 적용
                for label, point in scaled_points:
                    calibrated_point = calibrate_gaze_point(point)
                    scaled_calibrated_points.append((label, calibrated_point))
                
                for i, (x, y) in enumerate(gaze_means):
                    print(f"Point {i + 1}: x={x}, y={y}")

            else:


                if frame_idx <= 70:
                    # cv2.circle(display_image,(50,50),radius,color=(0,0,255),thickness= -1)
                    cv2.putText(display,f"ready for calibration",(200,200), cv2.FONT_HERSHEY_SIMPLEX,5 , (0,0,255), 5, cv2.LINE_AA)
                    cv2.imshow('target image', display)

                elif frame_idx <= 140:
                    cv2.circle(display,(200,250),radius,color=(0,0,255),thickness= -1)
                    cv2.putText(display,f"leftup",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1 , (0,255,0), 2, cv2.LINE_AA)
                    cv2.imshow('target image', display)
                    if frame_idx > 90:
                        leftup_init.append(gaze_point)

                elif frame_idx <= 210:
                    cv2.circle(display,(1720, 250),radius,color=(0,0,255),thickness= -1)
                    cv2.putText(display,f"rightup",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1 , (0,255,0), 2, cv2.LINE_AA)
                    cv2.imshow('target image', display)
                    if frame_idx > 160:
                        rightup_init.append(gaze_point)

                elif frame_idx <= 280:
                    cv2.circle(display,(150,880),radius,color=(0,0,255),thickness= -1)
                    cv2.putText(display,f"leftdown",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1 , (0,255,0), 2, cv2.LINE_AA)
                    cv2.imshow('target image', display)
                    if frame_idx > 230:
                        leftdown_init.append(gaze_point)

                elif frame_idx <= 350:
                    cv2.circle(display,(1770,880),radius,color=(0,0,255),thickness= -1)
                    cv2.putText(display,f"rightdown",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1 , (0,255,0), 2, cv2.LINE_AA)
                    cv2.imshow('target image', display)
                    if frame_idx > 300:
                        rightdown_init.append(gaze_point)

                elif frame_idx <= 420:
                    cv2.circle(display,(600, 550),radius,color=(0,0,255),thickness= -1)
                    cv2.putText(display,f"midleft",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1 , (0,255,0), 2, cv2.LINE_AA)
                    cv2.imshow('target image', display)
                    if frame_idx > 370:
                        midleft_init.append(gaze_point)

                elif frame_idx <= 490:
                    cv2.circle(display,(1320, 530),radius,color=(0,0,255),thickness= -1)
                    cv2.putText(display,f"midright",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1 , (0,255,0), 2, cv2.LINE_AA)
                    cv2.imshow('target image', display)
                    if frame_idx > 440:
                        midright_init.append(gaze_point)

            frame_idx+=1


        new_frame_time = time.time()
        fps_deque.append(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        if frame_idx % 60 == 0:
            print(f'FPS: {np.mean(fps_deque):5.2f}')

        text = f"X: {round(float(gaze_point[0]), 4)}, Y: {round(float(gaze_point[1]), 4)}"
        cv2.putText(frame, text, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2, cv2.LINE_AA)
        cv2.imshow('Processed Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


except KeyboardInterrupt:
    print("클라이언트 종료 중...")
finally:
    client_socket.close()
    pipeline.stop()
    accuracy = calculate_accuracy(gaze_data, scaled_points, tolerance_radius = 175)
    print(f"Accuracy: {accuracy:.2f}%")
    final_rmse  = calculate_rmse(gaze_data, scaled_points)
    print(final_rmse)
