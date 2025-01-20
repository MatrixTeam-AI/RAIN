import cv2
import numpy as np

import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose

class Wholebody:
    def __init__(self, onnx_det_path, onnx_pose_path, providers=None):
        
        if(providers is None):
            providers = ort.capi._pybind_state.get_available_providers()
        self.session_det = ort.InferenceSession(path_or_bytes=onnx_det_path, providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose_path, providers=providers)
        
    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        areas = [(det_result[i][3] - det_result[i][1]) * (det_result[i][2] - det_result[i][0]) for i in range(len(det_result))]
        ind = np.argmax(areas) if len(areas) > 0 else -1
        keypoints, scores = inference_pose(self.session_pose, [det_result[ind]] if ind>=0 else [], oriImg)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]
        
        return keypoints, scores
    


