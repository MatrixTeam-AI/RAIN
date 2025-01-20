# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .wholebody import Wholebody
from PIL import Image
from torchvision.transforms import transforms

def draw_pose(pose, H, W, translations, patterns, face_only):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if(not face_only):
        canvas = util.draw_bodypose(canvas, candidate, subset)

        canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces, translations, patterns)

    return canvas


class DWposeDetector:
    def __init__(self, onnx_det_path, onnx_pose_path, providers, damping_step=1, translations=[], detection_size=(768, 768), no_cache=False, face_only=False):

        self.pose_estimation = Wholebody(onnx_det_path, onnx_pose_path, providers)
        self.pose_cache = []
        self.damping_step = damping_step
        self.detection_size = detection_size
        self.no_cache = no_cache
        self.translations = translations
        self.transform = transforms.Compose([transforms.Resize(self.detection_size)])
        self.patterns = None
        self.face_only = face_only
        
    def set_translations(self, translations):
        self.translations = translations

    def set_patterns(self, patterns):
        self.patterns = patterns
    
    def __call__(self, oriImg, direct_no_cache=False, direct_translations=None):
        oriImg = oriImg.copy()
        oriImg = np.array(self.transform(Image.fromarray(oriImg)))
        H, W, C = oriImg.shape
        candidate, subset = self.pose_estimation(oriImg)
        nums, keys, locs = candidate.shape
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        body = candidate[:,:18].copy()
        body = body.reshape(nums*18, locs)
        score = subset[:,:18]
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > 0.3:
                    score[i][j] = int(18*i+j)
                else:
                    score[i][j] = -1

        un_visible = subset < 0.3
        candidate[un_visible] = -1

        foot = candidate[:,18:24]

        faces = candidate[:,24:92]

        hands = candidate[:,92:113]
        hands = np.vstack([hands, candidate[:,113:]])
        if(not self.no_cache and not direct_no_cache):
            faces_mask = np.where(faces==-1, 1, 0)
            fsindex = np.argmax(np.sum(faces_mask,axis=(1,2)))
            if(fsindex != 0):
                c = faces[0,:,:].copy()
                faces[0, :, :] = faces[fsindex, :, :]
                faces[fsindex, :, :] = c
            if(len(self.pose_cache) > 0):            
                faces = np.where(faces==-1, self.pose_cache[-1], faces)

            self.pose_cache.append(faces[0:1,:,:])
            n = min(len(self.pose_cache), self.damping_step)
            window = (np.arange(1, n + 1) / ((n + 1) * n / 2.0)).reshape((n, 1, 1, 1))
            faces = np.sum(np.array(self.pose_cache[-n:]) * window, axis=0)
            
        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)

        return draw_pose(pose, H, W, self.translations if direct_translations is None else direct_translations, self.patterns, self.face_only)
    
    def detect(self, oriImg):
        return self(oriImg)
